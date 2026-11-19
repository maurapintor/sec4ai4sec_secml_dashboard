"""Pix2Struct image processing. Verbatim from pralab/adv-docVQA
(models/processing/pix2struct_processor.py), Donut-specific parts dropped.
"""

import functools
import math
import textwrap
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_PATH = "ybelkada/fonts"


def torch_extract_patches(image_tensor, patch_height, patch_width):
    """Extract patches from a given image tensor."""
    image_tensor = image_tensor.unsqueeze(0)
    patches = F.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)


@functools.lru_cache(maxsize=1)
def _default_font_path() -> str:
    # hf_hub_download checks the Hub for updates on every call; resolve the path once.
    return hf_hub_download(DEFAULT_FONT_PATH, "Arial.TTF")


@functools.lru_cache(maxsize=32)
def _header_tensor(header: str) -> torch.Tensor:
    """Rendered question header as a (C, H, W) float tensor. The header doesn't
    depend on the image, so it's rendered once per question instead of every step."""
    header_image = render_text(header)
    return torch.from_numpy(np.array(header_image)).float().permute(2, 0, 1)


def render_text(
    text: str,
    text_size: int = 36,
    text_color: str = "black",
    background_color: str = "white",
    left_padding: int = 5,
    right_padding: int = 5,
    top_padding: int = 5,
    bottom_padding: int = 5,
    font_bytes: Optional[bytes] = None,
    font_path: Optional[str] = None,
) -> Image.Image:
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = "\n".join(wrapper.wrap(text=text))

    if font_bytes is not None and font_path is None:
        import io

        font = io.BytesIO(font_bytes)
    elif font_path is not None:
        font = font_path
    else:
        font = _default_font_path()
    font = ImageFont.truetype(font, encoding="UTF-8", size=text_size)

    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)
    return image


class Pix2StructImageProcessor:
    """Patch extraction & header rendering matching google/pix2struct preprocessing."""

    def __init__(self, patch_size=None, max_patches=2048, is_vqa: bool = True) -> None:
        self.patch_size = patch_size or {"height": 16, "width": 16}
        self.max_patches = max_patches

    def my_render_header(
        self,
        image_tensor: torch.Tensor,
        header: str,
        **kwargs,
    ):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor.permute(2, 0, 1)
        _, h, w = image_tensor.shape

        if kwargs:
            header_image = render_text(header, **kwargs)
            header_tensor = torch.from_numpy(np.array(header_image)).float().permute(2, 0, 1)
        else:
            header_tensor = _header_tensor(header)
        header_tensor = header_tensor.to(image_tensor.device)
        if header_tensor.dim() == 4:
            header_tensor = header_tensor.squeeze(0)

        _, header_h, header_w = header_tensor.shape
        new_width = max(w, header_w)
        new_height = int(h * (new_width / w))
        new_header_height = int(header_h * (new_width / header_w))

        image_resized = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        header_resized = F.interpolate(
            header_tensor.unsqueeze(0),
            size=(new_header_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return torch.cat([header_resized, image_resized], dim=1)

    def extract_flattened_patches(
        self,
        image: torch.Tensor,
        max_patches: int = 2048,
        patch_size: dict = None,
    ) -> torch.Tensor:
        patch_size = patch_size or {"height": 16, "width": 16}
        patch_height, patch_width = patch_size["height"], patch_size["width"]
        _, height, width = image.shape

        scale = math.sqrt(max_patches * (patch_height / height) * (patch_width / width))
        num_rows = max(min(math.floor(scale * height / patch_height), max_patches), 1)
        num_cols = max(min(math.floor(scale * width / patch_width), max_patches), 1)

        resized_height = max(num_rows * patch_height, 1)
        resized_width = max(num_cols * patch_width, 1)

        # MPS has no backward for antialiased bilinear resize, so on Apple GPUs this
        # one op runs on the CPU (autograd follows the device hop). CUDA runs it natively.
        device = image.device
        resize_on = torch.device("cpu") if device.type == "mps" else device
        image = F.interpolate(
            image.to(resize_on).unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0).to(device)

        patches = torch_extract_patches(image, patch_height, patch_width)
        rows, columns, depth = patches.shape[1], patches.shape[2], patches.shape[3]
        patches = patches.reshape([rows * columns, depth])

        row_ids = torch.arange(rows, device=image.device).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns, device=image.device).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])
        row_ids = (row_ids + 1).to(torch.float32)
        col_ids = (col_ids + 1).to(torch.float32)

        result = torch.cat([row_ids, col_ids, patches], -1)
        result = F.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()
        return result

    def my_normalize(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32)
        mean = torch.mean(image_tensor)
        std = torch.std(image_tensor)
        adjusted_stddev = max(std.item(), 1.0 / math.sqrt(image_tensor.numel()))
        return (image_tensor - mean) / adjusted_stddev, mean, std

    def my_preprocess_image(self, image: torch.Tensor, header_text: Optional[str] = None):
        image = self.my_render_header(image, header_text)
        image, mean, std = self.my_normalize(image)
        return image, mean, std
