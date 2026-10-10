"""
Captcha OCR HTTP server (self-contained, production).

Loads the model ensemble ONCE and serves predictions. Auto-uses GPU if available,
else CPU. No training/eval code — just inference.

Env vars:
  MODEL_DIR  default ./models       directory holding ensemble_*.pt (or best_model.pt)
  HOST       default 0.0.0.0
  PORT       default 8765
  TTA        default 0              "1" enables test-time augmentation (slower, marginal)

API:
  GET  /            -> {"status":"ok","models":N,"device":"cuda|cpu"}
  POST /ocr         body {"image":"<base64 jpg/png>"}  or  {"path":"<file path>"}
                    -> {"text":"ABC123","conf":0.987}   (HTTP 400 with {"error":...} on bad input)
"""
import base64
import glob
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_H, IMG_W, N_POS = 64, 192, 6
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "models"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
USE_TTA = os.environ.get("TTA", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ───────────────────────────────────────────────────────────────────
def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CaptchaNet(nn.Module):
    def __init__(self, num_chars, n_pos=N_POS, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 64),    conv_bn(64, 64),   nn.MaxPool2d(2),
            conv_bn(64, 128),  conv_bn(128, 128), nn.MaxPool2d(2),
            conv_bn(128, 256), conv_bn(256, 256), nn.MaxPool2d(2),
            conv_bn(256, 256), conv_bn(256, 256), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 12))
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 12, 512),
                                    nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(512, num_chars) for _ in range(n_pos)])

    def forward(self, x):
        f = self.shared(self.pool(self.features(x)))
        return torch.stack([h(f) for h in self.heads], dim=1)


# ── Loading / preprocessing ───────────────────────────────────────────────────
def find_ckpts():
    ens = sorted(glob.glob(os.path.join(MODEL_DIR, "ensemble_*.pt")))
    if ens:
        return ens
    bm = os.path.join(MODEL_DIR, "best_model.pt")
    if os.path.exists(bm):
        return [bm]
    raise FileNotFoundError(f"no checkpoints (ensemble_*.pt / best_model.pt) in {MODEL_DIR}")


def build_views(use_tta):
    base = transforms.Compose([transforms.Resize((IMG_H, IMG_W)),
                               transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if not use_tta:
        return [base]
    views = [base]
    for deg, tx in [(0, 0.03), (0, -0.03), (2, 0.0), (-2, 0.0)]:
        views.append(transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.RandomAffine(degrees=abs(deg) or 0.01,
                                    translate=(abs(tx), 0.0) if tx else None, fill=255),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    return views


MODELS, IDX2CHAR, VIEWS = [], {}, []
LOCK = threading.Lock()


def load():
    global IDX2CHAR, VIEWS
    ckpts = find_ckpts()
    for p in ckpts:
        ck = torch.load(p, map_location=DEVICE)
        m = CaptchaNet(ck["num_chars"], ck["n_pos"]).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.eval()
        MODELS.append(m)
        IDX2CHAR = {int(k): v for k, v in ck["idx2char"].items()}
    VIEWS = build_views(USE_TTA)
    return [os.path.basename(p) for p in ckpts]


@torch.no_grad()
def predict_pil(img):
    img = img.convert("L")
    probs = None
    for v in VIEWS:
        x = v(img).unsqueeze(0).to(DEVICE)
        for m in MODELS:
            p = m(x).softmax(-1)
            probs = p if probs is None else probs + p
    probs = probs / (len(VIEWS) * len(MODELS))
    conf, idx = probs[0].max(-1)
    return "".join(IDX2CHAR[int(i)] for i in idx), float(conf.mean())


# ── HTTP server ────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"status": "ok", "models": len(MODELS), "device": str(DEVICE)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if body.get("image"):
                img = Image.open(io.BytesIO(base64.b64decode(body["image"])))
            elif body.get("path"):
                img = Image.open(body["path"])
            else:
                return self._send(400, {"error": "provide 'image' (base64) or 'path'"})
            with LOCK:
                text, conf = predict_pil(img)
            self._send(200, {"text": text, "conf": round(conf, 4)})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})


if __name__ == "__main__":
    names = load()
    print(f"OCR server ready: http://{HOST}:{PORT}  models={names} device={DEVICE} tta={USE_TTA}",
          flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
