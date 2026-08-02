import torch.nn as nn


class FastCNN(nn.Module):
    """
    Lightweight custom CNN — the 'Fast' tier.
    4 conv blocks + global average pool + small FC head.
    Designed for low latency / low resource cost, not max accuracy.
    """
    def __init__(self, num_classes: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 224->112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 112->56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),# 56->28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),# 28->14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    # quick sanity check: run this file directly to confirm shapes work
    import torch
    model = FastCNN(num_classes=28)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 28]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")