import torch.nn as nn


class GateNet(nn.Module):
    """
    Tiny 3-way routing gate: predicts which tier (fast/balanced/heavy)
    should handle an image, reading raw pixels directly -- no dependency
    on Fast's own confidence, no hand-crafted stats.
    Deliberately ~16.5x smaller than FastCNN (24K vs 396K params) to
    protect the compute-cost advantage over rule_based, whose Fast
    forward pass does double duty (confidence check + the actual answer).
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 224->112
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 112->56
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 56->28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    import torch
    model = GateNet(num_classes=3)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 3]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")