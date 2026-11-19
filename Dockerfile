FROM python:3.11-slim

# git: needed for the RobustBench dependency (installed straight from GitHub).
# build-essential: fallback in case any dependency has no prebuilt wheel for this
#   platform/Python combo — keeps the build resilient without hand-picking libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision come from the CUDA 12.6 index rather than plain PyPI, whose
# default wheel targets the newest CUDA runtime and can silently fall back to CPU
# on hosts with older drivers. CUDA 12.x wheels run on any driver >= 525.
# torch >= 2.6 is required by recent transformers releases.
# Not supported: Blackwell GPUs (RTX 50xx, B200), which need a cu128 build.
#
# /constraints.txt keeps these versions fixed when requirements.txt is installed,
# and pins transformers to the version the dashboard was tested with.
RUN printf "torch==2.7.1\ntorchvision==0.22.1\ntransformers==4.57.6\n" > /constraints.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -c /constraints.txt torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126

# Installed in its own layer so `docker build` only re-downloads/re-installs
# dependencies when requirements.txt actually changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -c /constraints.txt -r requirements.txt

# Bake the Pix2Struct-DocVQA checkpoint (~1.1GB) and the font used to render the
# question header into the image, so a fresh container (e.g. a RunPod pod) can run
# attacks immediately without hitting Hugging Face. Kept above `COPY . .` so code
# edits don't invalidate this layer.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
snapshot_download('google/pix2struct-docvqa-base', ignore_patterns=['*.bin']); \
hf_hub_download('ybelkada/fonts', 'Arial.TTF')"

COPY . .

EXPOSE 9000

CMD ["python", "main.py"]
