import torch.nn as nn


class CorrectnessGateNet(nn.Module):
    """
    Predicts P(correct) independently for each of the 3 tiers (fast, balanced,
    heavy) directly from image pixels, without calling any tier model at
    inference time. Unlike GateNet (discrete 3-way classification over a single
    best_tier label) or ConfidenceGateNet (regression toward Fast's confidence
    scalar), this predicts per-tier CORRECTNESS directly using each tier's own
    {tier}_correct ground truth column from generate_utility_labels.py's
    output. Three independent sigmoid outputs, not softmax -- these are NOT
    competing probabilities (an image can plausibly have all three tiers
    correct, or none), so they must not be forced to sum to 1.
    """
    def __init__(self, num_tiers: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 224->112
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 112->56
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 56->28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_tiers),
            nn.Sigmoid(),  # per-tier sigmoid, values independently in [0,1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)  # [B, 3], ordered (fast, balanced, heavy)


if __name__ == "__main__":
    import torch
    model = CorrectnessGateNet()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 3]
    print("Output range check:", out.min().item(), out.max().item())  # should be within [0,1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
