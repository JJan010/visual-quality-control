import torch
from torch import nn


class PatchcoreFeatureExport(nn.Module):
    """Export the two feature maps used by our PatchCore configuration.

    Input:
        RGB images normalized with the checkpoint's ImageNet statistics.
        Shape: (B, 3, 256, 256), dtype: float32.

    Outputs:
        layer2: (B, 512, 32, 32)
        layer3: (B, 1024, 16, 16)
    """

    def __init__(self, feature_extractor: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images)
        return features["layer2"], features["layer3"]
