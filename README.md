# Captcha OCR Server

Self-contained inference service for the 6-char captcha model (4-model ensemble,
~94.6% full-string / 99.1% per-char on held-out test, ~93% first-try on live traffic).

Just the production essentials — no training/eval/debug code.

```
ocr_server/
├── serve.py            # the HTTP server + model (self-contained)
├── models/             # ensemble_1..4.pt  (the deployed ensemble)
├── requirements.txt    # CPU PyTorch + pillow + numpy
├── Dockerfile          # CPU image
└── docker-compose.yml
```

## Run with Docker (recommended)

```bash
docker compose up --build        # builds + starts on http://localhost:8765
# or:
docker build -t captcha-ocr .
docker run -p 8765:8765 captcha-ocr
```

## Run locally (no Docker)

```bash
pip install -r requirements.txt
python serve.py                  # http://127.0.0.1:8765  (auto GPU if available)
```

## API

```
GET  /                 -> {"status":"ok","models":4,"device":"cpu"}
POST /ocr
     {"image":"<base64 jpg/png>"}      -> {"text":"ABC123","conf":0.987}
     {"path":"/abs/path/to.jpg"}       -> {"text":"ABC123","conf":0.987}
     (bad input)                       -> HTTP 400 {"error":"..."}
```

`conf` is the mean per-character confidence (correct reads ≈ 0.99, errors ≈ 0.6 —
threshold on it if you want to fall back to a backup OCR only when unsure).

Example (PowerShell):
```powershell
$img = [Convert]::ToBase64String([IO.File]::ReadAllBytes("captcha.jpg"))
Invoke-RestMethod http://localhost:8765/ocr -Method Post -ContentType application/json -Body (@{image=$img}|ConvertTo-Json)
```

Example (Node / your scraper):
```ts
const res = await axios.post('http://localhost:8765/ocr', { image: imageBuffer.toString('base64') });
const code = res.data.text;        // res.data.conf available for thresholding
```

## Config (env vars)

| var | default | meaning |
|-----|---------|---------|
| `HOST` | `0.0.0.0` | bind address |
| `PORT` | `8765` | port |
| `TTA` | `0` | `1` = test-time augmentation (slower, marginal accuracy) |
| `MODEL_DIR` | `./models` | where `ensemble_*.pt` live |
| `OMP_NUM_THREADS` | `4` | CPU threads for torch |

## Speed

- GPU: ~10-30 ms/req · CPU: ~75 ms/req (4-model ensemble) — the network round-trip to
  the captcha site dominates, not OCR.

## GPU image (optional)

Swap the base to `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, install Python + the
`cu124` torch wheels (see `requirements.txt` comment), and run with `--gpus all`.

## Updating the model

Drop new `ensemble_*.pt` into `models/` (retrained on more verified `true/` data) and
rebuild / restart. The server loads whatever `ensemble_*.pt` it finds.
