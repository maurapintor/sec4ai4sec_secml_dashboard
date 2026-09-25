import base64
import io
import queue
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.fmn import FMN
from secmlt.metrics.classification import Accuracy
from secmlt.trackers.trackers import (
    LossTracker,
    PerturbationNormTracker,
    SampleTracker,
    Tracker,
)

try:
    from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier as _Wrapper
except ImportError:
    from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier as _Wrapper  # type: ignore[no-redef]

try:
    import robustbench  # noqa: F401
    HAS_ROBUSTBENCH = True
except ImportError:
    HAS_ROBUSTBENCH = False

NORMALIZE_PARAMS = {
    "mnist":   {"mean": (0.1307,),                    "std": (0.3081,)},
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465),     "std": (0.2023, 0.1994, 0.2010)},
}

DATASETS = {
    "mnist":   {"class": datasets.MNIST,   "num_classes": 10},
    "cifar10": {"class": datasets.CIFAR10, "num_classes": 10},
}

DATASET_CLASSES = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "mnist":   ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

PRECONFIGURED_MODELS = [
    # ── PyTorch Hub ──────────────────────────────────────────────────────────
    {
        "id": "cifar10_resnet20",
        "label": "ResNet-20 (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_resnet20",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": True,
    },
    {
        "id": "cifar10_resnet56",
        "label": "ResNet-56 (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_resnet56",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": True,
    },
    {
        "id": "cifar10_vgg11_bn",
        "label": "VGG-11 BN (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_vgg11_bn",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": True,
    },
    # ── RobustBench ──────────────────────────────────────────────────────────
    {
        "id": "rb_cifar10_Wong2020",
        "label": "Wong 2020 (RobustBench)",
        "source": "robustbench",
        "model_name": "Wong2020Fast",
        "dataset": "cifar10",
        "threat_model": "Linf",
        "normalize": False,
    },
    {
        "id": "rb_cifar10_Rice2020",
        "label": "Rice 2020 (RobustBench)",
        "source": "robustbench",
        "model_name": "Rice2020Overfitting",
        "dataset": "cifar10",
        "threat_model": "Linf",
        "normalize": False,
    },
]

MODEL_BY_ID: dict = {m["id"]: m for m in PRECONFIGURED_MODELS}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pytorch_model(model_config: dict):
    """Load a raw PyTorch model from config. Returns (model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = model_config["source"]

    if source == "hub":
        model = torch.hub.load(
            model_config["hub_repo"],
            model_config["model_name"],
            trust_repo=True,
            skip_validation=True,
            **model_config.get("model_kwargs", {}),
        )
    elif source == "robustbench":
        if not HAS_ROBUSTBENCH:
            raise ImportError(
                "robustbench is not installed. Run: pip install robustbench"
            )
        from robustbench.utils import load_model as _rb_load
        model = _rb_load(
            model_name=model_config["model_name"],
            dataset=model_config["dataset"],
            threat_model=model_config["threat_model"],
        )
    else:
        raise ValueError(f"Unknown model source: {source!r}")

    model.eval()
    return model.to(device), device


def _make_wrapper(pytorch_model, model_config: dict):
    """Wrap a PyTorch model with optional normalization. Returns (wrapper, preprocessing)."""
    preprocessing = None
    if model_config.get("normalize"):
        p = NORMALIZE_PARAMS[model_config["dataset"]]
        preprocessing = transforms.Normalize(p["mean"], p["std"])
    return _Wrapper(pytorch_model, preprocessing=preprocessing), preprocessing


def _tensor_to_b64(tensor: torch.Tensor) -> str:
    """Convert a (C, H, W) float tensor in [0,1] to a base64-encoded PNG string."""
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get_fooled(pytorch_model, preprocessing, adv_loader, device) -> torch.Tensor:
    pytorch_model.eval()
    results = []
    with torch.no_grad():
        for adv_x, y in adv_loader:
            adv_x = adv_x.to(device)
            if preprocessing is not None:
                adv_x = preprocessing(adv_x)
            preds = pytorch_model(adv_x).argmax(dim=1).cpu()
            results.append(preds != y)
    return torch.cat(results)


class _ProgressTracker(Tracker):
    """Reports live FMN iteration progress back to the SSE queue."""

    def __init__(self, q: queue.Queue, num_steps: int, num_samples: int,
                 progress_start: float, progress_end: float):
        super().__init__(name="progress")
        self.q = q
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.progress_start = progress_start
        self.progress_end = progress_end

    def track(self, iteration, loss, scores, x_adv, delta, grad) -> None:
        fraction = (iteration + 1) / self.num_steps
        self.q.put({
            "type": "progress",
            "message": (
                f"Running FMN — iteration {iteration + 1}/{self.num_steps} "
                f"on {self.num_samples} sample{'s' if self.num_samples != 1 else ''}..."
            ),
            "progress": self.progress_start + (self.progress_end - self.progress_start) * fraction,
        })

    def get(self):
        return None


def _run_fmn(
    model,
    pytorch_model,
    preprocessing,
    loader,
    device,
    perturbation_model,
    num_steps,
    step_size,
    track_iterations=False,
    extra_trackers=None,
):
    history_trackers = None
    trackers = []
    if track_iterations:
        history_trackers = [LossTracker(), PerturbationNormTracker("l2"), SampleTracker()]
        trackers.extend(history_trackers)
    if extra_trackers:
        trackers.extend(extra_trackers)
    attack = FMN(
        perturbation_model=perturbation_model,
        num_steps=num_steps,
        step_size=step_size,
        backend=Backends.NATIVE,
        trackers=trackers or None,
    )
    adv_loader = attack(model, loader)
    adversarial_batches = []
    labels = []
    with torch.no_grad():
        for adv_x, y in adv_loader:
            adversarial_batches.append(adv_x.detach().cpu())
            labels.append(y.detach().cpu())

    adversarial = torch.cat(adversarial_batches)
    labels = torch.cat(labels)
    originals = torch.cat([batch.detach().cpu() for batch, _ in loader])
    distances = torch.linalg.vector_norm(
        (adversarial - originals).flatten(start_dim=1), ord=2, dim=1
    )
    fooled = _get_fooled(
        pytorch_model,
        preprocessing,
        [(adversarial, labels)],
        device,
    )
    history = None
    if history_trackers is not None:
        history = {
            "loss": history_trackers[0].get(),
            "distance": history_trackers[1].get(),
            "samples": history_trackers[2].get(),
        }
    return originals, adversarial, labels, distances, fooled, history


def get_sample_images(dataset_name: str, count: int = 16, start_index: int = 0) -> list:
    """Return sample test images as base64 PNG thumbnails."""
    ds_cfg = DATASETS[dataset_name]
    test_dataset = ds_cfg["class"](
        root="./data", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    classes = DATASET_CLASSES[dataset_name]
    result = []
    for i in range(start_index, min(start_index + count, len(test_dataset))):
        img_tensor, label = test_dataset[i]
        result.append({
            "index": i,
            "label": int(label),
            "label_name": classes[int(label)],
            "image_b64": _tensor_to_b64(img_tensor),
        })
    return result


# ── Evaluation (security curve) ──────────────────────────────────────────────

def run_evaluation(config: dict, q: queue.Queue) -> None:
    """Blocking evaluation — intended to run in a daemon thread."""
    try:
        model_config = MODEL_BY_ID[config["model_id"]]
        dataset_name = model_config["dataset"]
        num_samples = config["num_samples"]
        sample_indices = config.get("sample_indices")
        perturbation_model = config["perturbation_model"]
        num_steps = config["num_steps"]
        step_size = config["step_size"]

        q.put({"type": "progress", "message": f"Loading '{model_config['label']}'..."})
        pytorch_model, device = load_pytorch_model(model_config)
        model, preprocessing = _make_wrapper(pytorch_model, model_config)

        q.put({"type": "progress", "message": f"Loading {dataset_name} test set..."})
        ds_cfg = DATASETS[dataset_name]
        test_dataset = ds_cfg["class"](
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        if sample_indices:
            dataset_indices = sorted({
                int(i) for i in sample_indices if 0 <= int(i) < len(test_dataset)
            })
            if not dataset_indices:
                raise ValueError("No valid sample indices were provided for evaluation")
        else:
            dataset_indices = list(range(min(num_samples, len(test_dataset))))
        num_samples = len(dataset_indices)
        test_loader = DataLoader(
            Subset(test_dataset, dataset_indices), batch_size=num_samples, shuffle=False
        )

        q.put({"type": "progress", "message": "Computing clean accuracy..."})
        clean_acc = float(Accuracy()(model, test_loader))
        q.put({"type": "clean_accuracy", "accuracy": clean_acc})

        q.put({
            "type": "progress",
            "message": f"Running FMN once on {num_samples} samples...",
            "progress": 0.2,
        })
        progress_tracker = _ProgressTracker(
            q, num_steps=num_steps, num_samples=num_samples,
            progress_start=0.2, progress_end=0.85,
        )
        originals, adversarial, labels, distances, fooled, _ = _run_fmn(
            model,
            pytorch_model,
            preprocessing,
            test_loader,
            device,
            perturbation_model,
            num_steps,
            step_size,
            extra_trackers=[progress_tracker],
        )

        q.put({
            "type": "progress",
            "message": "Computing security curve points...",
            "progress": 0.85,
        })
        successful_distances = distances[fooled]
        finite_distances = successful_distances[torch.isfinite(successful_distances)]
        epsilon_values = [0.0]
        if finite_distances.numel() > 0:
            epsilon_values.extend(torch.unique(finite_distances).sort().values.tolist())
        n = len(epsilon_values)
        for i, eps in enumerate(epsilon_values):
            fooled_at_epsilon = (successful_distances <= eps).sum().item()
            robust_acc = 1.0 - fooled_at_epsilon / num_samples
            q.put({
                "type": "result_point",
                "epsilon": eps,
                "accuracy": robust_acc,
                "index": i,
                "total": n,
                "progress": 0.85 + 0.15 * ((i + 1) / n),
            })

        q.put({"type": "done", "message": "Evaluation complete!"})

    except Exception as exc:
        q.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})


# ── Visualization (single-image perturbation viewer) ─────────────────────────

def run_visualization(config: dict, q: queue.Queue) -> None:
    """Run FMN once on one image and stream its minimum-distance result."""
    try:
        model_config = MODEL_BY_ID[config["model_id"]]
        dataset_name = model_config["dataset"]
        image_index = config["image_index"]
        perturbation_model = config["perturbation_model"]
        num_steps = config["num_steps"]
        step_size = config["step_size"]

        classes = DATASET_CLASSES[dataset_name]

        q.put({"type": "progress", "message": f"Loading '{model_config['label']}'..."})
        pytorch_model, device = load_pytorch_model(model_config)
        model, preprocessing = _make_wrapper(pytorch_model, model_config)

        q.put({"type": "progress", "message": f"Loading image {image_index}..."})
        ds_cfg = DATASETS[dataset_name]
        test_dataset = ds_cfg["class"](
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        img_tensor, true_label = test_dataset[image_index]
        true_label = int(true_label)

        # Clean prediction
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            inp = preprocessing(x) if preprocessing is not None else x
            probs = torch.softmax(pytorch_model(inp), dim=1)[0].cpu().numpy()
        clean_pred = int(np.argmax(probs))

        q.put({
            "type": "image",
            "clean": True,
            "epsilon": 0.0,
            "image_b64": _tensor_to_b64(img_tensor),
            "predicted_class": classes[clean_pred],
            "predicted_idx": clean_pred,
            "true_class": classes[true_label],
            "true_idx": true_label,
            "confidence": float(probs[clean_pred]),
            "correct": clean_pred == true_label,
            "probs": [float(p) for p in probs],
            "class_names": classes,
        })

        q.put({
            "type": "progress",
            "message": "Running FMN once to find the minimum-distance example...",
            "progress": 0.25,
        })
        single_loader = DataLoader(
            Subset(test_dataset, [image_index]), batch_size=1, shuffle=False
        )
        originals, adversarial, labels, distances, fooled, history = _run_fmn(
            model,
            pytorch_model,
            preprocessing,
            single_loader,
            device,
            perturbation_model,
            num_steps,
            step_size,
            track_iterations=True,
        )
        iteration_count = history["samples"].shape[-1]
        for iteration in range(iteration_count):
            adv_img = history["samples"][0, ..., iteration]
            with torch.no_grad():
                x = adv_img.unsqueeze(0).to(device)
                inp = preprocessing(x) if preprocessing is not None else x
                probs = torch.softmax(pytorch_model(inp), dim=1)[0].cpu().numpy()
            adv_pred = int(np.argmax(probs))
            distance = float(history["distance"][0, iteration])
            loss = float(history["loss"][0, iteration])

            q.put({
                "type": "image",
                "clean": False,
                "iteration": iteration + 1,
                "total_iterations": iteration_count,
                "epsilon": distance,
                "loss": loss,
                "image_b64": _tensor_to_b64(adv_img.cpu()),
                "predicted_class": classes[adv_pred],
                "predicted_idx": adv_pred,
                "true_class": classes[true_label],
                "true_idx": true_label,
                "confidence": float(probs[adv_pred]),
                "correct": adv_pred == true_label,
                "probs": [float(p) for p in probs],
                "class_names": classes,
            })

        q.put({"type": "done"})

    except Exception as exc:
        q.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})
