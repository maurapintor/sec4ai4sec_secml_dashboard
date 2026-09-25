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

# torch/torchvision are pinned to a cu121 build rather than pulled in via
# requirements.txt, because the default (unpinned) PyPI wheel now targets a much
# newer CUDA runtime than older workstation drivers support — it installs fine but
# torch.cuda.is_available() silently comes back False ("CUDA initialization: the
# NVIDIA driver on your system is too old"). cu121 wheels work with any driver
# >= 530.x (CUDA 12.1 forward-compatible); if your workstation's driver is even
# older, pick an earlier torch/cu11x pair from https://download.pytorch.org/whl/.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Installed in its own layer so `docker build` only re-downloads/re-installs
# dependencies when requirements.txt actually changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face cache for the Pix2Struct-DocVQA checkpoint (~1.1GB, downloaded on
# first use of the Document Analysis tab). Mount this path as a volume so the
# download survives container restarts/recreation instead of happening every time.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 9000

CMD ["python", "main.py"]
