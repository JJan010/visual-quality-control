import torch

from visual_quality.scoring.ssim import SSIMScorer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA niedostępna.")

    device = torch.device("cuda")
    torch.manual_seed(42)
    scorer = SSIMScorer().to(device).eval()

    with torch.inference_mode():
        # 1. Obraz identyczny z samym sobą.
        images = torch.rand(2, 3, 64, 64, device=device)
        scores, maps = scorer(images, images)

        assert scores.shape == (2,)
        assert maps.shape == (2, 54, 54)
        assert maps.device.type == "cuda"
        assert torch.allclose(
            maps, torch.zeros_like(maps), atol=1e-5, rtol=0
        )
        print("Identity: OK")

        # 2. Symetria porównania.
        other = torch.rand_like(images)
        scores_xy, maps_xy = scorer(images, other)
        scores_yx, maps_yx = scorer(other, images)

        assert torch.allclose(maps_xy, maps_yx, atol=1e-5, rtol=0)
        assert torch.isfinite(scores_xy).all().item()
        print("Symmetry: OK")

        # 3. Jednolite obrazy: różnią się tylko jasnością.
        dark = torch.full((1, 3, 64, 64), 0.2, device=device)
        light = torch.full_like(dark, 0.7)

        actual, _ = scorer(dark, light)
        expected = 1 - (
            (2 * 0.2 * 0.7 + 0.01 ** 2)
            / (0.2 ** 2 + 0.7 ** 2 + 0.01 ** 2)
        )

        assert abs(actual.item() - expected) < 5e-4
        print(
            f"Constant images: OK | "
            f"actual={actual.item():.6f}, expected={expected:.6f}"
        )

        # 4. Zmiana w drugim obrazie nie może wpływać na pierwszy.
        base = torch.full((2, 3, 64, 64), 0.25, device=device)
        changed = base.clone()
        changed[1, :, 24:40, 24:40] = 0.75

        scores, maps = scorer(base, changed)

        assert abs(scores[0].item()) < 1e-5
        assert scores[1].item() > 0

        # Współrzędne mapy uwzględniają margines 5 pikseli.
        center_error = maps[1, 20:30, 20:30].mean().item()
        distant_error = maps[1, :10, :10].abs().max().item()

        assert center_error > 0.1
        assert distant_error < 1e-5
        print("Local change and batch independence: OK")

    print("Device:", device)
    print("SSIM CHECKS: OK")


if __name__ == "__main__":
    main()
