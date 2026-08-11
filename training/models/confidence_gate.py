import torch.nn as nn


class ConfidenceGateNet(nn.Module):
    """
    Predicts Fast's expected confidence directly from image pixels,
    WITHOUT calling Fast at inference time. Same lightweight backbone
    as the earlier (failed) discrete-classification GateNet, but now
    a single sigmoid-bounded regression output instead of 3-way softmax --
    confidence is a continuous, more learnable signal than the discrete
    best_tier label, which was shown (3 independent experiments) to be
    too entangled with Fast's idiosyncratic per-image correctness to
    learn from pixels alone.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 224->112
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 112->56
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 56->28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # bounds output to [0,1], matching confidence's real range
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.regressor(x).squeeze(1)  # [B, 1] -> [B]


if __name__ == "__main__":
    import torch
    model = ConfidenceGateNet()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2]
    print("Output range check:", out.min().item(), out.max().item())  # should be within [0,1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")