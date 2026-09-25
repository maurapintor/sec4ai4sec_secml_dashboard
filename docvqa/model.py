"""Pix2Struct model wrapper. Verbatim from pralab/adv-docVQA
(models/processing/base_processor.py + models/pix2struct.py), Donut-specific parts dropped.
"""

from typing import List, Union

import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration


class BaseDocVQAProcessor(DataProcessing):
    """Base class for DocVQA processors, sharing common text tokenization methods."""

    def __init__(self):
        super().__init__()
        self.processor = None
        self.target_suffix = ""
        self.add_special_tokens = True

    def get_input_ids(self, text: Union[List, str]):
        if text is None:
            return text
        if isinstance(text, str):
            return self.processor.tokenizer(text=text, return_tensors="pt").input_ids

        if isinstance(text, list) and len(text) > 0:
            tokenized_list = []
            for item in text:
                if self.target_suffix:
                    item += self.target_suffix
                tokenized = self.processor.tokenizer(
                    text=item,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                ).input_ids
                tokenized_list.append(tokenized)
                tokenized_list.append(torch.tensor([-1], dtype=torch.long).unsqueeze(0))
            tokenized_list.pop()
            return torch.cat(tokenized_list, dim=1)
        raise ValueError("You have to put at least one element")

    def reconstruct_targets(self, targets, separator=-1):
        res = []
        indices = (targets.flatten() == separator).to(dtype=torch.uint8).nonzero()
        if indices.numel() == 0:
            return targets
        start_index = 0
        for index in indices.squeeze(0):
            res.append(targets[:, start_index:index].squeeze(0))
            start_index = index + 1
        res.append(targets[:, index + 1 :].squeeze(0))
        return res

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented."""


class Pix2StructModelProcessor(BaseDocVQAProcessor):
    """Data processing utility for Pix2Struct model."""

    def __init__(self) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")
        self.target_suffix = ""
        self.add_special_tokens = True

    def _process(self, x, q: str) -> torch.Tensor:
        return self.processor(images=x, text=q, return_tensors="pt")

    def decode(self, x):
        """Decode model outputs."""
        return self.processor.decode(x, skip_special_tokens=True)


class Pix2StructModel(BaseModel):
    """Wrapper for the DocVQA PyTorch model."""

    MAX_OUTPUT_TOKENS = 50

    def __init__(self, device="cpu") -> None:
        self._model: torch.nn.Module = Pix2StructForConditionalGeneration.from_pretrained(
            "google/pix2struct-docvqa-base",
        ).to(device)
        self.model_processor = Pix2StructModelProcessor()
        super().__init__()

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.decision_function(x)[0]

    def _decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """Not implemented."""

    def gradient(self, x: torch.Tensor, y) -> torch.Tensor:
        x = x.to(device=self._get_device())
        x.requires_grad = True
        loss = self.loss_fn(x=x, y=y)
        loss.backward()
        return x.grad

    def loss_fn(self, x: torch.Tensor, y: str):
        x = x.to(device=self._get_device())
        y = y.to(device=self._get_device())
        predictions = self._model(labels=y, flattened_patches=x)
        return predictions, predictions.loss

    def train(self, dataloader: DataLoader):
        """Not implemented."""

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self._get_device())
        return self._decision_function(x)

    def torch_predict(self, image, questions):
        """Run inference and decode the generated answer for each question."""
        device = self._get_device()
        processor = self.model_processor.processor
        inputs = processor(images=[image] * len(questions), text=questions, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.MAX_OUTPUT_TOKENS)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)
