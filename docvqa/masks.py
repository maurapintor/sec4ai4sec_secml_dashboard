"""Perturbation masks. mask_include_all/mask_bottom_right_corner are verbatim from
pralab/adv-docVQA (attacks/masks.py). mask_white_pixels is not in that file — the repo's
README mentions a background-only mask ("try the one on Colab that doesn't apply perturbation
to all white pixels") but it only ships in the linked Colab notebook, not in this module, so
it's added here to match.
"""

import torch


def mask_include_all(image: torch.Tensor):
    """Creates a mask for the image, including all pixels."""
    return torch.ones_like(image, dtype=torch.bool)


def mask_bottom_right_corner(image: torch.Tensor, ratio=0.15) -> torch.Tensor:
    """Creates a mask for the image on the bottom right corner."""
    mask = torch.zeros_like(image, dtype=torch.bool)
    h, w, _ = image.shape
    size = int(min(w, h) * ratio)
    x_start, y_start = w - size, h - size
    mask[y_start:, x_start:, :] = 1
    return mask


def mask_white_pixels(image: torch.Tensor, threshold: float = 245.0) -> torch.Tensor:
    """Creates a mask that only allows perturbation on near-white pixels (document
    background), leaving text/lines/logos untouched — keeps the forged document
    looking clean at a glance instead of visibly noisy over the content.
    """
    is_white = (image >= threshold).all(dim=-1, keepdim=True)
    return is_white.expand_as(image)


def mask_below_gray_threshold(image: torch.Tensor, threshold: float = 245.0) -> torch.Tensor:
    """Creates a mask that only allows perturbation on non-white pixels — anything
    with actual content: text, lines, logos, but also mid-tones and light grays,
    not just pure black. Leaves the blank white background untouched. The exact
    inverse of mask_white_pixels: noise hides inside existing ink and graphics
    instead of scattering visibly across blank paper.
    """
    gray = image.mean(dim=-1, keepdim=True)
    is_content = gray < threshold
    return is_content.expand_as(image)


def mask_dominant_channel(
    image: torch.Tensor, white_threshold: float = 245.0, color_diff_threshold: float = 15.0
) -> torch.Tensor:
    """Combines mask_below_gray_threshold (skip the white background) with a
    per-pixel channel restriction: on a colorful pixel (e.g. green ink), only
    perturb the channel that's already dominant there (green), so the noise blends
    into the existing color instead of showing up as an off-color speckle. On a
    black/gray/white pixel (channels all close together) there's no dominant color
    to hide behind, so all three channels stay eligible.
    """
    is_content = (image.mean(dim=-1, keepdim=True) < white_threshold).expand_as(image)

    channel_max = image.max(dim=-1, keepdim=True).values
    channel_min = image.min(dim=-1, keepdim=True).values
    is_colorful = (channel_max - channel_min) >= color_diff_threshold
    is_dominant_channel = image >= channel_max

    channel_allowed = torch.where(is_colorful.expand_as(image), is_dominant_channel, True)
    return is_content & channel_allowed
