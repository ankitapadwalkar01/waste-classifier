from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .models import LightweightWasteCNN
from torchvision import transforms


def inference_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_student(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LightweightWasteCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state"])
    model.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    model.to(device).eval()
    return model


def predict_image(model, image: Image.Image, device: torch.device,
                  image_size: int = IMAGE_SIZE, top_k: int = 3):
    tensor = inference_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    values, indices = probabilities.topk(min(top_k, len(CLASS_NAMES)))
    return [
        {"class": CLASS_NAMES[int(index)], "confidence": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ], tensor


def gradcam(model, tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
    """Compute Grad-CAM for the final convolutional feature block."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.features[-1]

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    forward_handle = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    if class_index is None:
        class_index = int(logits.argmax(dim=1).item())
    score = logits[:, class_index].sum()
    score.backward()
    forward_handle.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
