# 👁️ RetinAgent — AI-Powered Diabetic Retinopathy Detection

[![AMD MI300X](https://img.shields.io/badge/AMD-MI300X-ED1C24?logo=amd)](https://developer.amd.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kappa](https://img.shields.io/badge/Cohen's%20Kappa-0.9097-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-85.01%25-brightgreen)]()

> AMD Developer Hackathon 2026 · Track 3: Vision & Multimodal AI

RetinAgent is an end-to-end AI diagnostic system that accepts retinal fundus images and produces:
- **DR Grade classification** (0–4) using fine-tuned ViT-MAE
- **GradCAM visual explainability** heatmaps
- **Natural language diagnostic reports** (Qwen LLM)
- **Interactive Q&A** for clinical follow-up

---

## 🚀 Quick Start

```bash
git clone https://github.com/codedbyasim/RetinoAgent.git
cd RetinoAgent
pip install -r requirements.txt

# Run the Gradio demo
python app.py
```

Open `http://localhost:7860` in your browser.

---

## 🏋️ Training (AMD MI300X)

```bash
python train.py \
  --epochs 50 \
  --batch_size 128 \
  --data_dir ./aptos2019-blindness-detection
```

| Metric | Result | Target |
|--------|--------|--------|
| Cohen's Kappa | **0.9097** | > 0.85 ✅ |
| Test Accuracy | **85.01%** | > 80% ✅ |
| Training Time | **5.3 min** | < 3 hours ✅ |

---

## 🏗️ Architecture

```
Input Image
    │
    ▼
Preprocessing (Circle Crop → CLAHE → 224×224 → Normalize)
    │
    ▼
ViT-MAE Encoder (facebook/vit-mae-base, 12 blocks, 768-dim)
    │
    ├──► Classification Head → DR Grade 0-4
    │
    ├──► GradCAM Engine → Heatmap Overlay
    │
    └──► Qwen LLM Agent → Report + Q&A
```

---

## 📁 Project Structure

```
RetinoAgent/
├── train.py          # Training pipeline (FR-02, FR-03)
├── app.py            # Gradio demo (FR-04, FR-05, FR-06, FR-07)
├── requirements.txt  # Python dependencies
├── checkpoints/      # Saved model weights (not committed)
│   └── best_model.pth
└── LICENSE           # MIT License
```

---

## ⚙️ AMD ROCm Compatibility

All code runs natively on AMD GPUs via ROCm:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ROCm exposes AMD GPUs as 'cuda' — no code change needed
```

Tested on: **AMD Instinct MI300X** via AMD Developer Cloud

---

## 📋 SRS Requirements Coverage

| ID | Feature | Status |
|----|---------|--------|
| FR-01 | Image Upload & Validation | ✅ |
| FR-02 | Preprocessing Pipeline | ✅ |
| FR-03 | DR Grade Classification | ✅ |
| FR-04 | GradCAM Explainability | ✅ |
| FR-05 | NL Diagnostic Report | ✅ |
| FR-06 | Interactive Q&A Agent | ✅ |
| FR-07 | HF Spaces Deployment | ✅ |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
