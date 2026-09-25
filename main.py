import asyncio
import json
import queue
import threading
import uuid
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from evaluator import (
    DATASETS,
    MODEL_BY_ID,
    PRECONFIGURED_MODELS,
    get_sample_images,
    run_evaluation,
    run_visualization,
)
from docvqa_runner import (
    add_uploaded_document,
    get_document_preview,
    list_documents,
    run_docvqa_attack,
)

app = FastAPI(title="SecML-Torch Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")

# job_id -> {"queue": Queue, "status": str}
jobs: Dict[str, Dict] = {}


class EvalParams(BaseModel):
    model_id: str
    num_samples: int = 20
    sample_indices: List[int] | None = None
    perturbation_model: str = "l2"
    num_steps: int = 50
    step_size: float = 0.04

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if v not in MODEL_BY_ID:
            raise ValueError(f"Unknown model_id: {v!r}")
        return v

    @field_validator("perturbation_model")
    @classmethod
    def validate_norm(cls, v: str) -> str:
        if v not in ("l2", "linf"):
            raise ValueError("perturbation_model must be 'l2' or 'linf'")
        return v

    @field_validator("sample_indices")
    @classmethod
    def validate_sample_indices(cls, v: List[int] | None) -> List[int] | None:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("sample_indices cannot be empty")
        if any(i < 0 for i in v):
            raise ValueError("sample_indices must contain non-negative integers")
        return v


class VisualizeParams(BaseModel):
    model_id: str
    image_index: int = 0
    perturbation_model: str = "l2"
    num_steps: int = 50
    step_size: float = 0.04

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if v not in MODEL_BY_ID:
            raise ValueError(f"Unknown model_id: {v!r}")
        return v

    @field_validator("perturbation_model")
    @classmethod
    def validate_norm(cls, v: str) -> str:
        if v not in ("l2", "linf"):
            raise ValueError("perturbation_model must be 'l2' or 'linf'")
        return v


class DocVQAParams(BaseModel):
    doc_id: str
    question: str
    target: str
    mask: str = "dark_pixels"
    perturbation_model: str = "l2"
    eps: float = 30000.0
    steps: int = 10
    step_size: float = 10.0

    @field_validator("question", "target")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @field_validator("mask")
    @classmethod
    def validate_mask(cls, v: str) -> str:
        if v not in ("dark_pixels", "white_pixels", "include_all", "bottom_right_corner"):
            raise ValueError(
                "mask must be 'dark_pixels', 'white_pixels', 'include_all' or 'bottom_right_corner'"
            )
        return v

    @field_validator("perturbation_model")
    @classmethod
    def validate_norm(cls, v: str) -> str:
        if v not in ("l2", "linf"):
            raise ValueError("perturbation_model must be 'l2' or 'linf'")
        return v


@app.get("/", response_class=HTMLResponse)
async def serve_home() -> str:
    # CIFAR-10-backed tabs (Adversarial Examples / Security Curve) are temporarily
    # unlinked from the nav — slow to work with right now — so Document Analysis
    # is the landing page. Their routes still work directly if visited by URL.
    with open("static/docvqa.html") as f:
        return f.read()


@app.get("/curve", response_class=HTMLResponse)
async def serve_index() -> str:
    with open("static/index.html") as f:
        return f.read()


@app.get("/visualize", response_class=HTMLResponse)
async def serve_visualize() -> str:
    with open("static/visualize.html") as f:
        return f.read()


@app.get("/documents", response_class=HTMLResponse)
async def serve_docvqa() -> str:
    with open("static/docvqa.html") as f:
        return f.read()


@app.get("/api/models")
async def list_models() -> List[Dict]:
    return PRECONFIGURED_MODELS


@app.get("/api/datasets")
async def list_datasets() -> List[str]:
    return list(DATASETS.keys())


@app.get("/api/sample-images")
async def sample_images(
    dataset: str = "cifar10", count: int = 16, start: int = 0
) -> List[Dict]:
    if dataset not in DATASETS:
        raise HTTPException(400, f"Unknown dataset: {dataset!r}")
    return get_sample_images(dataset, count=count, start_index=start)


@app.post("/api/evaluate")
async def start_evaluation(params: EvalParams) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running"}

    config = {
        "model_id": params.model_id,
        "num_samples": params.num_samples,
        "sample_indices": params.sample_indices,
        "perturbation_model": params.perturbation_model,
        "num_steps": params.num_steps,
        "step_size": params.step_size,
    }
    threading.Thread(target=run_evaluation, args=(config, q), daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/visualize")
async def start_visualization(params: VisualizeParams) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running"}

    config = {
        "model_id": params.model_id,
        "image_index": params.image_index,
        "perturbation_model": params.perturbation_model,
        "num_steps": params.num_steps,
        "step_size": params.step_size,
    }
    threading.Thread(target=run_visualization, args=(config, q), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/docvqa/documents")
async def docvqa_documents() -> List[Dict]:
    return list_documents()


@app.get("/api/docvqa/documents/{doc_id}/preview")
async def docvqa_document_preview(doc_id: str) -> Dict:
    try:
        return get_document_preview(doc_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/docvqa/upload")
async def docvqa_upload(file: UploadFile = File(...)) -> Dict:
    contents = await file.read()
    try:
        return add_uploaded_document(contents, file.filename or "uploaded document")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read document: {exc}") from exc


@app.post("/api/docvqa/attack")
async def start_docvqa_attack(params: DocVQAParams) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running"}

    config = {
        "doc_id": params.doc_id,
        "question": params.question,
        "target": params.target,
        "mask": params.mask,
        "perturbation_model": params.perturbation_model,
        "eps": params.eps,
        "steps": params.steps,
        "step_size": params.step_size,
    }
    threading.Thread(target=run_docvqa_attack, args=(config, q), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_results(job_id: str) -> StreamingResponse:
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    q = jobs[job_id]["queue"]
    loop = asyncio.get_event_loop()

    async def event_generator():
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=60))
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("done", "error"):
                jobs[job_id]["status"] = item["type"]
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
