import torch
import torch.nn as nn
from torchvision.models import convnext_base


def build_heavy_convnextbase(num_classes: int):
    """
    Heavy tier for iWildCam: ConvNeXt-Base backbone.
    Returns unwrapped torchvision model for checkpoint compatibility.
    """
    m = convnext_base(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


if __name__ == "__main__":
    model = build_heavy_convnextbase(num_classes=182)
    dummy = torch.randn(2, 3, 384, 384)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 182]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
