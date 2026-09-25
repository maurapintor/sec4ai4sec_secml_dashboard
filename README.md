# SecML-Torch Dashboard

A web-based dashboard for evaluating the adversarial robustness of PyTorch models using the [SecML-Torch](https://github.com/pralab/secml-torch) FMN (Fast Minimum-Norm) attack. The **Adversarial Examples** page lets you pick a single test image and watch FMN perturb it iteration by iteration, tracking the model's prediction and the attack loss as it converges. The **Security Curve** page runs FMN once over a batch of samples and plots robust accuracy against the minimum L2 distance needed to fool each one, keeping earlier runs visible at reduced opacity for comparison. The **Document Analysis** page runs a targeted adversarial-forgery attack (via [pralab/adv-docVQA](https://github.com/pralab/adv-docVQA), vendored under `docvqa/`) against a Pix2Struct document VQA model, forging a document image so the model answers a chosen question with a chosen target answer.

> [!WARNING]
> Models loaded via `torch.hub.load(..., trust_repo=True)`. Only point this tool at hub repos you trust.

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)

Install dependencies:

```bash
pip install -r requirements.txt
```

RobustBench is installed directly from GitHub; a `models/` directory is created automatically to cache its weights (excluded from version control).

## Running the server

```bash
python main.py
```

Then open `http://localhost:9000` in your browser.

## Pages

| Page | URL | Description |
|---|---|---|
| **Adversarial Examples** | `/` or `/visualize` | Per-image FMN perturbation viewer, with prediction and loss tracked across iterations |
| **Security Curve** | `/curve` | Security evaluation curve (robust accuracy vs. minimum L2 distance) |
| **Document Analysis** | `/documents` | Adversarial forgery against a Document VQA model (Pix2Struct) |

## Usage

### 1. Model

Choose a pre-configured model from the dropdown. Available models:

| Label | Source | Dataset |
|---|---|---|
| ResNet-20 (PyTorch Hub) | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| ResNet-56 (PyTorch Hub) | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| VGG-11 BN (PyTorch Hub) | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| Wong 2020 (RobustBench) | RobustBench | CIFAR-10 |
| Rice 2020 (RobustBench) | RobustBench | CIFAR-10 |

RobustBench models are downloaded and cached in `./models/` on first use.

### 2. Dataset

Dataset and normalization are fixed per model. Samples are taken from the test split, downloaded automatically to `./data/` on first use.

### 3. Attack — FMN

Both pages run the native SecML-Torch **FMN** attack (minimum-norm, L2 by default). Advanced settings (iterations, step size) can be revealed with "Show advanced" on each page.

### 4. Adversarial Examples page

Pick an image from the gallery, then click **Run Attack**. FMN runs once on that image with per-iteration tracking enabled. As results stream in you'll see:

- The original and perturbed image side by side.
- A slider over the FMN iterations — dragging it updates the perturbed image, the model's prediction, and the confidence/probability bars for that iteration.
- A loss-vs-iteration plot with a marker showing the loss at the currently selected iteration.

### 5. Security Curve page

Select a model, choose which test samples to include, and click **Run Evaluation**. FMN runs once over the batch; each sample's minimum successful L2 distance becomes a point on the robust-accuracy curve. Previous runs stay on the chart at reduced opacity so you can compare configurations.

### 6. Exporting results

Click **Download PDF** on the Security Curve page after an evaluation completes. The PDF includes the configuration, clean accuracy, the security evaluation curve, and a table of (distance, accuracy, drop) values.

### 7. Document Analysis page

Open `/documents`. Upload a document image (or pick one of the bundled samples), enter a question about it and the answer you want the model to produce instead, then click **Run Attack**. The page shows the original document with the model's real answer, and the forged document with the model's answer after the attack.

> [!NOTE]
> First run downloads the `google/pix2struct-docvqa-base` checkpoint (~1.1GB) from Hugging Face and can take several minutes on CPU per attack (it runs a full forward/backward pass through the model on every step). Subsequent runs reuse the cached model.

## API

The server exposes a REST + SSE API usable without the UI.

### `GET /api/models`

Returns the list of pre-configured model descriptors.

### `GET /api/datasets`

Returns the list of supported dataset names.

### `GET /api/sample-images`

Returns a page of test-set images as base64-encoded PNG thumbnails.

Query params: `dataset` (default `cifar10`), `count` (default 16), `start` (default 0).

### `POST /api/evaluate`

Start a security-curve evaluation job. Returns a `job_id`.

```json
{
  "model_id": "cifar10_resnet20",
  "num_samples": 20,
  "sample_indices": [0, 1, 2],
  "perturbation_model": "l2",
  "num_steps": 20,
  "step_size": 0.02
}
```

### `POST /api/visualize`

Start a single-image visualization job. Returns a `job_id`.

```json
{
  "model_id": "cifar10_resnet20",
  "image_index": 0,
  "perturbation_model": "l2",
  "num_steps": 40,
  "step_size": 0.02
}
```

### `GET /api/stream/{job_id}`

Server-Sent Events stream shared by both job types. Each event is a JSON object with a `type` field:

| type | Fields | Description |
|---|---|---|
| `progress` | `message`, `progress` (0–1) | Status update |
| `clean_accuracy` | `accuracy` | Clean accuracy on the test subset (evaluation jobs) |
| `result_point` | `epsilon`, `accuracy`, `index`, `total`, `progress` | Robust accuracy at one minimum-distance value (evaluation jobs) |
| `image` | `iteration`, `total_iterations`, `epsilon`, `loss`, `image_b64`, `predicted_class`, `predicted_idx`, `confidence`, `correct`, `probs` | Perturbed image and prediction at one FMN iteration (visualization jobs) |
| `done` | `message` | Job finished |
| `error` | `message`, `traceback` | Unhandled exception |

## Docker (GPU workstation)

The app auto-detects CUDA (`torch.cuda.is_available()`) everywhere a model is
loaded, so the same image runs on CPU or GPU — GPU just needs to be exposed to
the container.

### On the GPU workstation

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   (lets Docker pass GPUs through to containers) and confirm it works:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
2. Copy this project directory to the workstation (or `docker save`/`docker load`
   the image built elsewhere — see below).
3. Build and run:
   ```bash
   docker compose up --build -d
   ```
   or without Compose:
   ```bash
   docker build -t secml-dashboard .
   docker run --gpus all -p 9000:9000 -v hf-cache:/app/.cache/huggingface secml-dashboard
   ```
4. Open `http://<workstation-ip>:9000`.

Notes:
- The `-v hf-cache:/app/.cache/huggingface` volume caches the Pix2Struct-DocVQA
  checkpoint (~1.1GB) so it's downloaded once, not on every container restart.
- CIFAR-10 and the pretrained RobustBench weights (`data/`, `models/`) are baked
  into the image, so Security Curve / Adversarial Examples work offline; only the
  Document Analysis tab needs a network connection on first run, to fetch
  Pix2Struct from Hugging Face.
- If the workstation's Docker predates the `deploy.resources.reservations.devices`
  Compose GPU syntax, drop that block from `docker-compose.yml` and run with
  `docker run --gpus all` directly instead.

### Building elsewhere and shipping the image

If you'd rather build on a machine with faster internet and move the image over:
```bash
docker build -t secml-dashboard .
docker save secml-dashboard | gzip > secml-dashboard.tar.gz
# copy secml-dashboard.tar.gz to the workstation, then:
docker load < secml-dashboard.tar.gz
```
