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


class CorrectnessGateNetV3(nn.Module):
    """
    V3: same frozen MobileNetV3-Small features[:5] backbone as V2b, but the
    LAST block of that slice is unfrozen for fine-tuning (V2b kept it fully
    frozen). Two output heads sharing the same backbone+pooled features:
      - correctness_head: Linear(90,3) + Sigmoid -- same 3-tier correctness
        prediction as V2b (fast/balanced/heavy P(correct))
      - confidence_head: Linear(90,3) + Sigmoid -- NEW auxiliary head,
        predicts each tier's OWN confidence value (fast_confidence,
        balanced_confidence, heavy_confidence) directly from pixels,
        without calling that tier. Motivated by a confirmed finding this
        session: heavy_confidence/balanced_confidence meaningfully separate
        tiers (unlike fast_confidence, which does not) -- giving the model
        an explicit training signal to predict these should transfer useful
        representation learning into the correctness heads too.
    Shared trunk: AdaptiveAvgPool -> Linear(40,180)+BN+ReLU+Drop(0.3) ->
                  Linear(180,90)+BN+ReLU+Drop(0.3) (same as V2b's shared
                  layers), then TWO separate final Linear(90,3)+Sigmoid
                  heads branching off the shared 90-dim representation.
    forward() returns a tuple (correctness_pred, confidence_pred), each [B,3].
    Ablation flags (both default True = published V3; defaults preserve the
    exact architecture AND state-dict keys, so existing V3 checkpoints load
    unchanged):
      - unfreeze_last_block=False -> frozen backbone (V2b-style)
      - dual_head=False -> single correctness head; forward returns one [B,3]
        tensor (V2b-style output contract)
    """
    def __init__(self, num_tiers: int = 3, backbone_feature_blocks: int = 5,
                 unfreeze_last_block: bool = True, dual_head: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        full_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = full_model.features[:backbone_feature_blocks]

        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze the LAST block/layer of the slice for fine-tuning
        # (ablation flag: unfreeze_last_block=False reproduces V2b-style frozen backbone)
        if unfreeze_last_block:
            for param in self.backbone[-1].parameters():
                param.requires_grad = True
        self._unfreeze_last_block = unfreeze_last_block
        self.backbone.eval()

        with torch.no_grad():
            out_channels = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, 180),
            nn.BatchNorm1d(180),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(180, 90),
            nn.BatchNorm1d(90),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.correctness_head = nn.Sequential(
            nn.Linear(90, num_tiers),
            nn.Sigmoid(),
        )
        # Ablation flag: dual_head=False gives a single correctness head
        # (V2b-style output contract: forward returns one [B,3] tensor).
        self._dual_head = dual_head
        if dual_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(90, num_tiers),
                nn.Sigmoid(),
            )
        else:
            self.confidence_head = None

    def forward(self, x):
        # Frozen portion stays in eval, unfrozen last block follows train/eval via train() override
        # Run backbone without torch.no_grad so unfrozen last block can receive gradients
        x = self.backbone(x)
        x = self.pool(x)
        x = self.shared(x)
        correctness_pred = self.correctness_head(x)
        if not self._dual_head:
            return correctness_pred
        confidence_pred = self.confidence_head(x)
        return correctness_pred, confidence_pred

    def train(self, mode=True):
        super().train(mode)
        # Keep FROZEN portion in eval mode always, allow unfrozen last block to follow mode
        for idx, block in enumerate(self.backbone):
            if idx < len(self.backbone) - 1:
                block.eval()
            else:
                if mode and self._unfreeze_last_block:
                    block.train()
                else:
                    block.eval()
        return self


class CorrectnessGateNetV2c_costaware(nn.Module):
    """
    V2c-costaware: same frozen MobileNetV3-Small features[:5] backbone and
    same head architecture as V2b (Linear(40,180)+BN+ReLU+Drop(0.3) ->
    Linear(180,90)+BN+ReLU+Drop(0.3) -> Linear(90,3)), EXCEPT the final
    layer has NO Sigmoid activation -- outputs raw scores (logits), because
    this model is trained with a softmax-based cost-aware loss, not BCE.
    Outputs are NOT calibrated per-tier P(correct) like V1/V2/V2b -- do not
    interpret them that way anywhere downstream.
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


class CorrectnessGateNetV2d_css(nn.Module):
    """
    V2d-css: same frozen MobileNetV3-Small backbone + head architecture as
    V2b/V2c, but with num_tiers+1 outputs (one per tier PLUS one "defer"
    logit), trained with a Cost-Sensitive Softmax Cross-Entropy (CSS) loss
    (Mozannar & Sontag 2020 / Narasimhan et al. 2022 formulation), a
    Bayes-consistent alternative to the ad hoc softmax-utility loss used in
    V2c-costaware, which was found to underperform V2b's simpler post-hoc
    approach (0.9599 vs 0.9686 balanced routing accuracy).
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
            nn.Linear(90, num_tiers + 1),
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

    # V2c-costaware
    m2c = CorrectnessGateNetV2c_costaware()
    o2c = m2c(dummy)
    n2ct = sum(p.numel() for p in m2c.parameters())
    n2cr = sum(p.numel() for p in m2c.parameters() if p.requires_grad)
    print(f"V2c-costaware: output {o2c.shape}, total {n2ct:,}, trainable {n2cr:,}, frozen {n2ct-n2cr:,}")
    print(f"  (no Sigmoid -- raw logits output, range [{o2c.min().item():.3f}, {o2c.max().item():.3f}])")

    # V2d-css
    m2d = CorrectnessGateNetV2d_css()
    o2d = m2d(dummy)
    n2dt = sum(p.numel() for p in m2d.parameters())
    n2dr = sum(p.numel() for p in m2d.parameters() if p.requires_grad)
    print(f"V2d-css: output {o2d.shape}, total {n2dt:,}, trainable {n2dr:,}, frozen {n2dt-n2dr:,}")
    print(f"  (num_tiers+1=4 logits, no Sigmoid -- CSS defer head, range [{o2d.min().item():.3f}, {o2d.max().item():.3f}])")

    # V3
    m3 = CorrectnessGateNetV3()
    o3c, o3f = m3(dummy)
    n3t = sum(p.numel() for p in m3.parameters())
    n3r = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    print(f"V3: correctness {o3c.shape}, confidence {o3f.shape}, total {n3t:,}, trainable {n3r:,}, frozen {n3t-n3r:,}")
    print(f"  (two heads Linear(90,3)+Sigmoid, last backbone block unfrozen)")
