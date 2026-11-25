import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models import LightweightWasteCNN, count_parameters, distillation_loss


def test_student_output_shape():
    model = LightweightWasteCNN(num_classes=9)
    output = model(torch.randn(2, 3, 160, 160))
    assert output.shape == (2, 9)


def test_student_is_lightweight_relative_to_teacher_family():
    model = LightweightWasteCNN(num_classes=9)
    assert count_parameters(model) < 2_000_000


def test_distillation_loss_is_finite():
    student = torch.randn(4, 9, requires_grad=True)
    teacher = torch.randn(4, 9)
    labels = torch.tensor([0, 1, 2, 3])
    loss, ce, kl = distillation_loss(student, teacher, labels)
    assert torch.isfinite(loss)
    assert torch.isfinite(ce)
    assert torch.isfinite(kl)
    loss.backward()
    assert student.grad is not None
