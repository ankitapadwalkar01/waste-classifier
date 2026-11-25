from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ConvBNAct(nn.Sequential):
    """Convolution + batch normalization + Hardswish activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    """Light channel-attention module used to recalibrate feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class InvertedResidualBlock(nn.Module):
    """MobileNet-style expansion + depthwise filtering + linear projection."""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, expansion: int = 3, use_se: bool = True):
        super().__init__()
        hidden_channels = in_channels * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(ConvBNAct(in_channels, hidden_channels, kernel_size=1))
        layers.append(ConvBNAct(
            hidden_channels, hidden_channels, kernel_size=3, stride=stride,
            groups=hidden_channels
        ))
        if use_se:
            layers.append(SqueezeExcitation(hidden_channels))
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.output_activation = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.output_activation(out)


class LightweightWasteCNN(nn.Module):
    """Compact student with inverted residuals, depthwise convs, and channel attention."""

    def __init__(self, num_classes: int = 9, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, kernel_size=3, stride=2),
            InvertedResidualBlock(24, 32, stride=1, expansion=2),
            InvertedResidualBlock(32, 48, stride=2, expansion=3),
            InvertedResidualBlock(48, 48, stride=1, expansion=3),
            InvertedResidualBlock(48, 72, stride=2, expansion=3),
            InvertedResidualBlock(72, 72, stride=1, expansion=3),
            InvertedResidualBlock(72, 112, stride=2, expansion=4),
            InvertedResidualBlock(112, 112, stride=1, expansion=4),
            ConvBNAct(112, 160, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_teacher(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # With ImageNet initialisation, fine-tune the final residual stage and classifier.
    # This reduces training time while retaining high-level transfer features.
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      labels: torch.Tensor,
                      temperature: float = 4.0,
                      alpha: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, hard-label CE, and soft-target KL losses."""
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    teacher_soft = nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_log_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)
    soft_loss = nn.functional.kl_div(
        student_log_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)
    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
    return total_loss, hard_loss.detach(), soft_loss.detach()
