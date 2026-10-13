import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def build_fast_mobilenetv3small(num_classes: int):
    """
    Fast tier for iWildCam: MobileNetV3-Small backbone.
    Returns unwrapped torchvision model for checkpoint compatibility.
    """
    m = mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


if __name__ == "__main__":
    model = build_fast_mobilenetv3small(num_classes=182)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 182]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
