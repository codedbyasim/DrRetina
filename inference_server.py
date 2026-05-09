#!/usr/bin/env python3
"""
DrRetina – AMD MI300X Inference Server
Runs on AMD GPU server; exposes REST API for HF Spaces Gradio UI.

Usage (on AMD server):
    pip install fastapi uvicorn python-multipart pillow
    python3 inference_server.py
"""

import os, io, base64, cv2, numpy as np
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTMAEModel
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

# ── Constants ─────────────────────────────────────────────────────
GRADES = {
    0: "No DR",  1: "Mild DR",  2: "Moderate DR",
    3: "Severe DR",  4: "Proliferative DR",
}
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ──────────────────────────────────────────────────────────
class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.backbone   = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        hidden          = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes),
        )
    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values, noise=None)
        return self.classifier(out.last_hidden_state[:, 0, :])


print(f"[Server] Loading model on {device}...")
model = DRClassifier().to(device)
ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()
print("[Server] Model ready ✅")

# ── Preprocessing ──────────────────────────────────────────────────
def circle_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img_bgr[y:y+h, x:x+w]

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

INFER_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = apply_clahe(circle_crop(bgr))
    bgr = cv2.resize(bgr, (224, 224))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), INFER_TF(Image.fromarray(rgb)).unsqueeze(0).to(device)

# ── GradCAM ────────────────────────────────────────────────────────
class ViTGradCAM:
    def __init__(self, m):
        self.m = m; self._f = self._g = None
        layer = m.backbone.encoder.layer[-1]
        layer.register_forward_hook(
            lambda *a: setattr(self, "_f", a[2][0] if isinstance(a[2], tuple) else a[2]))
        layer.register_full_backward_hook(
            lambda *a: setattr(self, "_g", a[2][0]))

    def generate(self, tensor, cls_idx):
        self.m.zero_grad()
        self.m(tensor)[0, cls_idx].backward()
        g = self._g[0, 1:, :]; f = self._f[0, 1:, :]
        cam = F.relu((g.mean(-1).unsqueeze(-1) * f).sum(-1))
        cam = cam.reshape(14, 14).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (224, 224))

def overlay_heatmap(pil224, cam):
    img  = np.array(pil224).astype(np.float32)
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.float32)
    return Image.fromarray((0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8))

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── FastAPI App ────────────────────────────────────────────────────
app = FastAPI(title="DrRetina Inference API")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model": "ViT-MAE DR Classifier"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents))
        pil224, tensor = preprocess(pil_img)

        gradcam = ViTGradCAM(model)
        with torch.set_grad_enabled(True):
            logits = model(tensor)
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        grade = int(probs.argmax())
        cam   = gradcam.generate(tensor.clone(), grade)
        cam_pil = overlay_heatmap(pil224, cam)

        return JSONResponse({
            "grade":      grade,
            "grade_name": GRADES[grade],
            "confidence": float(probs[grade]),
            "probs":      [float(p) for p in probs],
            "image_b64":  pil_to_b64(pil224),
            "cam_b64":    pil_to_b64(cam_pil),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[Server] Starting on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
