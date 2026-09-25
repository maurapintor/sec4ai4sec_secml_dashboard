"""Constraints for implementing the attack. Verbatim from pralab/adv-docVQA (attacks/constraints.py)."""

from typing import Union

import torch
from secmlt.optimization.constraints import InputSpaceConstraint


class QuantizationConstraint(InputSpaceConstraint):
    """Constraint for ensuring quantized outputs into specified levels."""

    def __init__(
        self,
        preprocessing=None,
        levels: Union[list, torch.Tensor, int] = 255,
    ) -> None:
        if isinstance(levels, (int, float)):
            if levels < 2:
                raise ValueError("Number of levels must be at least 2.")
            if int(levels) != levels:
                raise ValueError("Pass an integer number of levels.")
            self.levels = torch.linspace(0, 1, int(levels))
        elif isinstance(levels, list):
            self.levels = torch.tensor(levels, dtype=torch.float32)
        elif isinstance(levels, torch.Tensor):
            self.levels = levels.type(torch.float32)
            if len(self.levels) < 2:
                raise ValueError("Number of custom levels must be at least 2.")
        else:
            raise TypeError("Levels must be an integer, list, or torch.Tensor.")
        self.levels = self.levels.sort().values
        super().__init__(preprocessing)

    def _apply_constraint(self, x, *args, **kwargs):
        min_val, max_val = self.levels[0].item(), self.levels[-1].item()
        x_clamped = x.clamp(min_val, max_val)
        idx = torch.round(x_clamped).long()
        return self.levels[idx]


class QuantizationConstraintWithMask(QuantizationConstraint):
    """Quantization constraint with masked components (not quantized), e.g. position encoding."""

    def __init__(self, preprocessing=None, levels=255, mask=None) -> None:
        self.mask = mask
        super().__init__(preprocessing=preprocessing, levels=levels)

    def __call__(self, x):
        """Apply the quantization."""
        transformed_x = x.detach().clone()
        if self.mask is None:
            self.mask = torch.ones_like(x)
        transformed_x = super().__call__(transformed_x)
        return torch.where(self.mask, transformed_x, x)
