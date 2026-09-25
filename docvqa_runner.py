"""Backend for the Document Analysis tab: adversarial forgery against DocVQA (Pix2Struct).

Vendored attack from https://github.com/pralab/adv-docVQA, run through secml-torch.
"""

import base64
import io
import pathlib
import queue
import threading
import traceback
import uuid

import torch
from PIL import Image

from docvqa.attack import attack_pix2struct
from docvqa.masks import (
    mask_below_gray_threshold,
    mask_bottom_right_corner,
    mask_dominant_channel,
    mask_include_all,
    mask_white_pixels,
)
from docvqa.model import Pix2StructModel
from docvqa.processor import Pix2StructImageProcessor

AVAILABLE_MASKS = {
    "dominant_channel": mask_dominant_channel,
    "dark_pixels": mask_below_gray_threshold,
    "white_pixels": mask_white_pixels,
    "include_all": mask_include_all,
    "bottom_right_corner": mask_bottom_right_corner,
}

SAMPLE_DOCS_DIR = pathlib.Path("static/docvqa_samples")
DISPLAY_MAX_SIDE = 900  # keep response payloads reasonable for large scanned documents
THUMB_SIZE = (140, 180)

# ── Lazy model singleton (the ~1.1GB Pix2Struct-DocVQA checkpoint is downloaded once) ──

_model_lock = threading.Lock()
_model: Pix2StructModel | None = None


def _get_model() -> Pix2StructModel:
    global _model
    with _model_lock:
        if _model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = Pix2StructModel(device=device)
    return _model


def _image_processor() -> Pix2StructImageProcessor:
    return Pix2StructImageProcessor()


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _for_display(image: Image.Image) -> Image.Image:
    image = image.copy()
    image.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
    return image


def _normalize_answer(text: str) -> str:
    """Collapse whitespace/case so minor formatting doesn't hide a successful forgery."""
    return "".join(text.split()).lower()


class _IterationTracker:
    """Streams the perturbed document + the model's actual answer after every attack step, for
    the iteration slider on the frontend.

    Duck-typed to the interface the vendored attack loop calls (track/init_tracking/end_tracking)
    rather than subclassing secml-torch's Tracker, since this tracker's needs (running a live
    prediction) don't fit that interface anyway.

    Running the model's prediction at every step is genuinely expensive (a full autoregressive
    generate() call per iteration, on top of the attack's own forward/backward pass), but that's
    the point of this tracker: see the answer evolve, not just the loss.
    """

    def __init__(
        self,
        q: queue.Queue,
        num_steps: int,
        progress_start: float,
        progress_end: float,
        model: Pix2StructModel,
        question: str,
        target: str,
    ):
        self.q = q
        self.num_steps = num_steps
        self.progress_start = progress_start
        self.progress_end = progress_end
        self.model = model
        self.question = question
        self.target = target

    def init_tracking(self) -> None:
        pass

    def end_tracking(self) -> None:
        pass

    def track(self, iteration, loss, scores, x_adv, delta, grad) -> None:  # noqa: ARG002
        image_arr = x_adv[0].clamp(0, 255).detach().cpu().numpy().astype("uint8")
        image = Image.fromarray(image_arr)
        prediction = self.model.torch_predict(image, [self.question])[0]
        fraction = (iteration + 1) / self.num_steps
        self.q.put({
            "type": "iteration",
            "iteration": iteration + 1,
            "total_iterations": self.num_steps,
            "prediction": prediction,
            "fooled": _normalize_answer(prediction) == _normalize_answer(self.target),
            "image_b64": _pil_to_b64(_for_display(image)),
            "progress": self.progress_start + (self.progress_end - self.progress_start) * fraction,
        })


# ── Document store (bundled samples + this session's uploads) ──────────────────────

_documents: dict[str, dict] = {}


def _register_document(doc_id: str, name: str, image: Image.Image, source: str) -> dict:
    image = image.convert("RGB")
    thumb = image.copy()
    thumb.thumbnail(THUMB_SIZE)
    entry = {
        "id": doc_id,
        "name": name,
        "source": source,
        "image": image,
        "thumb_b64": _pil_to_b64(thumb),
    }
    _documents[doc_id] = entry
    return entry


def _load_sample_documents() -> None:
    if not SAMPLE_DOCS_DIR.is_dir():
        return
    for path in sorted(SAMPLE_DOCS_DIR.glob("*.jpg")):
        doc_id = f"sample:{path.stem}"
        if doc_id in _documents:
            continue
        try:
            image = Image.open(path)
            image.load()
        except Exception:  # noqa: BLE001 - corrupt/missing bundled sample shouldn't crash startup
            continue
        _register_document(doc_id, path.stem.replace("_", " "), image, source="sample")


def list_documents() -> list[dict]:
    return [
        {"id": d["id"], "name": d["name"], "source": d["source"], "thumb_b64": d["thumb_b64"]}
        for d in _documents.values()
    ]


def add_uploaded_document(file_bytes: bytes, filename: str) -> dict:
    image = Image.open(io.BytesIO(file_bytes))
    image.load()
    doc_id = f"upload:{uuid.uuid4()}"
    entry = _register_document(doc_id, filename, image, source="upload")
    return {"id": entry["id"], "name": entry["name"], "source": entry["source"], "thumb_b64": entry["thumb_b64"]}


def get_document_image(doc_id: str) -> Image.Image:
    if doc_id not in _documents:
        raise ValueError(f"Unknown document: {doc_id!r}")
    return _documents[doc_id]["image"]


def get_document_preview(doc_id: str) -> dict:
    """Full display-resolution image for on-demand inspection (vs. the small gallery thumbnail)."""
    entry = _documents.get(doc_id)
    if entry is None:
        raise ValueError(f"Unknown document: {doc_id!r}")
    return {
        "id": entry["id"],
        "name": entry["name"],
        "image_b64": _pil_to_b64(_for_display(entry["image"])),
    }


# ── Attack ───────────────────────────────────────────────────────────────────────

def run_docvqa_attack(config: dict, q: queue.Queue) -> None:
    """Run the Pix2Struct forgery attack — intended to run in a daemon thread."""
    try:
        doc_id = config["doc_id"]
        question = config["question"]
        target = config["target"]
        eps = config["eps"]
        steps = config["steps"]
        step_size = config["step_size"]
        perturbation_model = config.get("perturbation_model", "l2")
        mask_function = AVAILABLE_MASKS[config.get("mask", "dominant_channel")]

        image = get_document_image(doc_id)

        q.put({"type": "progress", "message": "Loading Pix2Struct-DocVQA model (first run downloads ~1.1GB)..."})
        model = _get_model()

        q.put({"type": "progress", "message": "Running clean prediction...", "progress": 0.1})
        clean_answer = model.torch_predict(image, [question])[0]
        q.put({
            "type": "image",
            "clean": True,
            "image_b64": _pil_to_b64(_for_display(image)),
            "prediction": clean_answer,
        })

        q.put({
            "type": "progress",
            "message": f"Running Pix2Struct forgery attack ({steps} steps, with a prediction each "
                       f"step)... this can take several minutes on CPU.",
            "progress": 0.25,
        })
        tracker = _IterationTracker(
            q, num_steps=steps, progress_start=0.25, progress_end=0.85,
            model=model, question=question, target=target,
        )
        adv_image = attack_pix2struct(
            model,
            _image_processor(),
            image,
            question,
            target,
            eps=eps,
            steps=steps,
            step_size=step_size,
            is_targeted=True,
            mask_function=mask_function,
            perturbation_model=perturbation_model,
            trackers=[tracker],
        )

        q.put({"type": "progress", "message": "Running prediction on the forged document...", "progress": 0.9})
        adv_answer = model.torch_predict(adv_image, [question])[0]
        q.put({
            "type": "image",
            "clean": False,
            "image_b64": _pil_to_b64(_for_display(adv_image)),
            "prediction": adv_answer,
            "target": target,
            "fooled": _normalize_answer(adv_answer) == _normalize_answer(target),
        })

        q.put({"type": "done", "message": "Attack complete!"})

    except Exception as exc:  # noqa: BLE001
        q.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})


_load_sample_documents()
