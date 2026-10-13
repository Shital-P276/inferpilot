import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_balanced_resnet18(num_classes: int):
    """
    Balanced tier for iWildCam: ResNet-18 backbone.
    Returns unwrapped torchvision model for checkpoint compatibility.
    """
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


if __name__ == "__main__":
    model = build_balanced_resnet18(num_classes=182)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 182]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
