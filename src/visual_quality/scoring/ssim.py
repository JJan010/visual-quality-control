import torch
from torch import nn
from torch.nn import functional as F


class SSIMScorer(nn.Module):
    """Single-scale RGB SSIM scoring for float32 images in [0, 1].

    Uses an 11x11 Gaussian window, sigma=1.5, and valid convolution.
    Returns one anomaly score per image and a spatial anomaly map.
    """

    def __init__(self):
        super().__init__()

        self.window_size = 11
        self.sigma = 1.5
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

        coordinates = torch.arange(
            self.window_size, dtype=torch.float32
        ) - self.window_size // 2

        gaussian = torch.exp(
            -(coordinates.square()) / (2 * self.sigma ** 2)
        )
        gaussian = gaussian / gaussian.sum()

        window = gaussian[:, None] * gaussian[None, :]
        kernel = window[None, None].repeat(3, 1, 1, 1)

        # Stałe wagi okna, przenoszone na GPU razem z modułem.
        self.register_buffer("kernel", kernel)

    def forward(self, images, reconstructions):
        if images.shape != reconstructions.shape:
            raise ValueError("Obrazy i rekonstrukcje muszą mieć ten sam kształt.")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Oczekiwany kształt: (batch, 3, height, width).")
        if min(images.shape[-2:]) < self.window_size:
            raise ValueError("Obraz jest mniejszy niż okno SSIM.")
        if images.dtype != torch.float32 or reconstructions.dtype != torch.float32:
            raise ValueError("Ten wariant oczekuje torch.float32.")
        if images.device != reconstructions.device or images.device != self.kernel.device:
            raise ValueError("Obrazy, rekonstrukcje i moduł muszą być na jednym urządzeniu.")

        for tensor in (images, reconstructions):
            if not torch.isfinite(tensor).all().item():
                raise ValueError("Wejście zawiera NaN lub Inf.")
            if tensor.min().item() < 0 or tensor.max().item() > 1:
                raise ValueError("Oczekiwane wartości pikseli w [0, 1].")

        def local_average(tensor):
            return F.conv2d(tensor, self.kernel, groups=3)

        mean_x = local_average(images)
        mean_y = local_average(reconstructions)

        variance_x = local_average(images.square()) - mean_x.square()
        variance_y = local_average(reconstructions.square()) - mean_y.square()
        covariance = (
            local_average(images * reconstructions) - mean_x * mean_y
        )

        luminance = (
            (2 * mean_x * mean_y + self.c1)
            / (mean_x.square() + mean_y.square() + self.c1)
        )
        contrast_structure = (
            (2 * covariance + self.c2)
            / (variance_x + variance_y + self.c2)
        )

        # Ograniczamy drobne wyjścia poza zakres wynikające z arytmetyki.
        similarity = (luminance * contrast_structure).clamp(-1, 1)

        anomaly_map = (1 - similarity).mean(dim=1)
        scores = anomaly_map.mean(dim=(1, 2))

        return scores, anomaly_map
