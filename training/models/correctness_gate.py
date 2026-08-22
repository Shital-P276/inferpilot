import torch
import torch.nn as nn


class CorrectnessGateNet(nn.Module):
    """
    V1: 3 conv blocks (3->16->32->64), AdaptiveAvgPool, Linear(64,3)->Sigmoid.
    24,003 params, all trainable, trained from scratch on ~4000 images.
    """
    def __init__(self, num_tiers: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_tiers),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


class CorrectnessGateNetV2(nn.Module):
    """
    V2: Frozen MobileNetV3-Small features[:5] backbone (24,224 frozen params)
    + minimal Linear(40,3) head (123 trainable params). Tests whether
    pretrained features alone help, given near-zero head capacity.
    """
    def __init__(self, num_tiers: int = 3, backbone_feature_blocks: int = 5):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        full_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = full_model.features[:backbone_feature_blocks]

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        with torch.no_grad():
            out_channels = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, num_tiers),
            nn.Sigmoid(),
        )

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class CorrectnessGateNetV2b(nn.Module):
    """
    V2b isolates the V2 confound: V2 paired frozen pretrained features with a
    near-zero-capacity head (123 params). V2b uses the SAME frozen
    MobileNetV3-Small features[:5] backbone but adds a trainable head with
    ~24K params (comparable to V1's 24,003), so any AUROC gap between V1 and
    V2b is attributable to the frozen features themselves, not head capacity.

    Head: AdaptiveAvgPool -> Linear(40,180)+BN+ReLU+Drop(0.3) ->
                              Linear(180,90)+BN+ReLU+Drop(0.3) ->
                              Linear(90,3)+Sigmoid
    """
    def __init__(self, num_tiers: int = 3, backbone_feature_blocks: int = 5):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        full_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = full_model.features[:backbone_feature_blocks]

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        with torch.no_grad():
            out_channels = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, 180),
            nn.BatchNorm1d(180),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(180, 90),
            nn.BatchNorm1d(90),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(90, num_tiers),
            nn.Sigmoid(),
        )

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


if __name__ == "__main__":
    import torch

    dummy = torch.randn(2, 3, 224, 224)

    # V1
    m1 = CorrectnessGateNet()
    o1 = m1(dummy)
    n1 = sum(p.numel() for p in m1.parameters())
    print(f"V1: output {o1.shape}, params {n1:,}")

    # V2
    m2 = CorrectnessGateNetV2()
    o2 = m2(dummy)
    n2t = sum(p.numel() for p in m2.parameters())
    n2r = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"V2: output {o2.shape}, total {n2t:,}, trainable {n2r:,}, frozen {n2t-n2r:,}")

    # V2b
    m2b = CorrectnessGateNetV2b()
    o2b = m2b(dummy)
    n2bt = sum(p.numel() for p in m2b.parameters())
    n2br = sum(p.numel() for p in m2b.parameters() if p.requires_grad)
    print(f"V2b: output {o2b.shape}, total {n2bt:,}, trainable {n2br:,}, frozen {n2bt-n2br:,}")
