"""Modular gradient-accumulation attack. Adapted from pralab/adv-docVQA
(attacks/modular_attack_grad_accumulation.py) for secml-torch 1.5.

Only change vs. upstream: __call__ is overridden to skip secml-torch's
BasePyTorchClassifier auto-wrapping (added after the 1.2.2 this repo targets),
since our BaseModel subclass (Pix2StructModel) already implements the contract
_run() needs directly.
"""

import logging
from functools import partial
from typing import Literal, Union

import torch
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import Manipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import Constraint
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.trackers.trackers import Tracker
from secmlt.utils.tensor_utils import atleast_kd
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ModularEvasionAttackFixedEps(BaseEvasionAttack):
    """Modular evasion attack for fixed-epsilon attacks."""

    def __init__(
        self,
        y_target: int | None,
        num_steps: int,
        step_size: float,
        loss_function,
        optimizer_cls: str | partial[Optimizer],
        manipulation_function: Manipulation,
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: list[Tracker] | Tracker | None = None,
    ) -> None:
        self.y_target = y_target
        self.num_steps = num_steps
        self.step_size = step_size
        self.trackers = trackers
        self.loss_function = loss_function

        if isinstance(optimizer_cls, str):
            optimizer_cls = OptimizerFactory.create_from_name(optimizer_cls, lr=step_size)
        self.optimizer_cls = optimizer_cls

        self._manipulation_function = manipulation_function
        self.initializer = initializer
        self.gradient_processing = gradient_processing

        super().__init__()

    @property
    def manipulation_function(self) -> Manipulation:
        return self._manipulation_function

    @manipulation_function.setter
    def manipulation_function(self, manipulation_function: Manipulation) -> None:
        self._manipulation_function = manipulation_function

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }

    @classmethod
    def _trackers_allowed(cls) -> Literal[True]:
        return True

    def _init_perturbation_constraints(self) -> list[Constraint]:
        raise NotImplementedError("Must be implemented accordingly")

    def _create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
        return self.optimizer_cls([delta], lr=self.step_size, **kwargs)

    def forward_loss(self, model: BaseModel, x: torch.Tensor, target: torch.Tensor):
        scores = model.decision_function(x)
        target = target.to(scores.device)
        losses = self.loss_function(scores, target)
        return scores, losses

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        init_deltas: torch.Tensor = None,
        optim_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if optim_kwargs is None:
            optim_kwargs = {}
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)
        model._model.eval()
        for param in model._model.parameters():
            param.requires_grad = False
        if init_deltas is not None:
            delta = init_deltas.data
        elif isinstance(self.initializer, BaseEvasionAttack):
            _, delta = self.initializer._run(model, samples, target)
        else:
            delta = self.initializer(samples.data)
        delta.requires_grad = True

        optimizer = self._create_optimizer(delta, **optim_kwargs)
        x_adv, delta = self.manipulation_function(samples, delta)
        x_adv.data, delta.data = self.manipulation_function(samples.data, delta.data)
        best_losses = torch.zeros(samples.shape[0]).fill_(torch.inf)
        best_delta = torch.zeros_like(samples)

        for i in range(self.num_steps):
            optimizer.zero_grad()
            scores, losses = self.forward_loss(model=model, x=x_adv, target=target)
            logger.info("%d. Loss = %s", i, losses)

            grad_before_processing = delta.grad.data
            delta.grad.data = multiplier * self.gradient_processing(delta.grad.data)
            optimizer.step()
            x_adv.data, delta.data = self.manipulation_function(samples.data, delta.data)

            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(i, losses, scores, x_adv, delta, grad_before_processing)

            best_delta.data = torch.where(
                atleast_kd(losses < best_losses, len(samples.shape)),
                delta.data,
                best_delta.data,
            )
            best_losses.data = torch.where(losses < best_losses, losses, best_losses.data)

            if losses < 1e-6:
                break

        x_adv, _ = self.manipulation_function(samples.data, best_delta.data)
        return x_adv, best_delta

    def __call__(self, model, data_loader: DataLoader, stream: bool = False):
        """Run the attack without secml-torch's BasePyTorchClassifier auto-wrap.

        Our BaseModel subclasses (e.g. Pix2StructModel) implement decision_function/
        gradient directly and aren't torch.nn.Module instances, so the generic
        wrapping added in later secml-torch versions doesn't apply here.
        """

        def _run_batches():
            for samples, labels in data_loader:
                if self.trackers is not None:
                    for tracker in self.trackers:
                        tracker.init_tracking()
                try:
                    x_adv, _ = self._run(model, samples, labels)
                finally:
                    if self.trackers is not None:
                        for tracker in self.trackers:
                            tracker.end_tracking()
                yield x_adv, labels

        attacked_batches = _run_batches()
        if stream:
            return attacked_batches

        adversarials, original_labels = [], []
        for x_adv, labels in attacked_batches:
            adversarials.append(x_adv)
            original_labels.append(labels)
        adversarials = torch.vstack(adversarials)
        original_labels = torch.hstack(original_labels)
        return DataLoader(
            TensorDataset(adversarials, original_labels),
            batch_size=data_loader.batch_size,
        )
