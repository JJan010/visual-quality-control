import torch
from torch import nn
from torch.nn import functional as F


class SeparableGaussianBlur2d(nn.Module):
    """Separowalne wygładzanie dla jednokanałowych map float32.

    Odtwarza filtrowanie z paddingiem same/reflect na podstawie
    istniejącego, separowalnego jądra 2D.
    """

    def __init__(self, reference_kernel: torch.Tensor):
        super().__init__()

        if reference_kernel.ndim != 4:
            raise ValueError("Oczekiwano jądra o czterech wymiarach.")
        if tuple(reference_kernel.shape[:2]) != (1, 1):
            raise ValueError("Ta implementacja obsługuje jeden kanał.")
        if reference_kernel.dtype != torch.float32:
            raise TypeError("Oczekiwano jądra float32.")

        # Rozkład wykonujemy jednorazowo na CPU, z większą precyzją.
        kernel = reference_kernel.detach().to(
            device="cpu", dtype=torch.float64
        )[0, 0]

        height, width = kernel.shape
        if height % 2 == 0 or width % 2 == 0:
            raise ValueError("Wymiary jądra muszą być nieparzyste.")
        if not torch.isfinite(kernel).all() or (kernel < 0).any():
            raise ValueError("Jądro musi być skończone i nieujemne.")

        mass = kernel.sum()
        if mass <= 0:
            raise ValueError("Suma jądra musi być dodatnia.")

        # Sumy kolumn i wierszy pozwalają odtworzyć jądro separowalne.
        horizontal = kernel.sum(dim=0)
        vertical = kernel.sum(dim=1) / mass
        reconstructed = vertical[:, None] * horizontal[None, :]

        torch.testing.assert_close(
            reconstructed,
            kernel,
            rtol=1e-5,
            atol=1e-10,
        )

        self.register_buffer(
            "horizontal",
            horizontal.to(torch.float32).reshape(1, 1, 1, width),
        )
        self.register_buffer(
            "vertical",
            vertical.to(torch.float32).reshape(1, 1, height, 1),
        )
        self.padding = (
            width // 2, width // 2,
            height // 2, height // 2,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Oczekiwano tensora (B, 1, H, W).")

        # Tak jak w oryginale: najpierw odbicie na wszystkich brzegach.
        padded = F.pad(images, self.padding, mode="reflect")

        # Najpierw filtr poziomy, następnie pionowy.
        horizontal_result = F.conv2d(padded, self.horizontal)
        return F.conv2d(horizontal_result, self.vertical)
