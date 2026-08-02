import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class BalancedMobileNet(nn.Module):
    """
    MobileNetV3-Large, pretrained on ImageNet, fine-tuned for our 28 classes.
    The 'Balanced' tier -- middle ground on accuracy/latency/resource cost.
    """
    def __init__(self, num_classes: int = 28, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.backbone = mobilenet_v3_large(weights=weights)

        # Replace the final classifier layer for our number of classes
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        # Freeze everything except the classifier head
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 3):
        # Unfreeze the last n feature blocks for fine-tuning phase 2
        total_blocks = len(self.backbone.features)
        for i, block in enumerate(self.backbone.features):
            if i >= total_blocks - n_blocks:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    import torch
    model = BalancedMobileNet(num_classes=28)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 28]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} total")