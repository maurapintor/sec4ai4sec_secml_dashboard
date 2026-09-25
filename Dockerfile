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

# Installed in its own layer so `docker build` only re-downloads/re-installs
# dependencies when requirements.txt actually changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face cache for the Pix2Struct-DocVQA checkpoint (~1.1GB, downloaded on
# first use of the Document Analysis tab). Mount this path as a volume so the
# download survives container restarts/recreation instead of happening every time.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

CMD ["python", "main.py"]
