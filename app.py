#!/usr/bin/env python3
"""
RetinAgent – Gradio Demo
FR-04: GradCAM | FR-05: NL Report | FR-06: Q&A Agent | FR-07: HF Deployment
AMD Developer Hackathon 2026
"""

import os, cv2, numpy as np
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTMAEModel
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
import gradio as gr

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
GRADES = {
    0: ("No DR",             "No visible signs of diabetic retinopathy."),
    1: ("Mild DR",           "Microaneurysms only present."),
    2: ("Moderate DR",       "More than microaneurysms but less than severe DR."),
    3: ("Severe DR",         "More than 20 intraretinal haemorrhages in each of 4 quadrants."),
    4: ("Proliferative DR",  "Neovascularisation or vitreous / pre-retinal haemorrhage."),
}
GRADE_EMOJI  = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🆘"}
URGENCY      = {
    0: "Routine follow-up in 12 months.",
    1: "Follow-up in 6 months.",
    2: "Ophthalmology referral within 3 months.",
    3: "Urgent referral within 1 month.",
    4: "Emergency referral — immediate risk of blindness.",
}
LESIONS = {
    0: "None expected.",
    1: "Microaneurysms (small red dots on the retina).",
    2: "Microaneurysms, hard exudates, retinal oedema.",
    3: "Extensive haemorrhages (>20/quadrant), venous beading, IRMA.",
    4: "Neovascularisation, vitreous haemorrhage, tractional detachment risk.",
}
TREATMENTS = {
    0: "No treatment needed. Maintain good glycaemic and BP control.",
    1: "No direct retinal treatment. Optimise HbA1c < 7%, BP < 130/80.",
    2: "Focal laser photocoagulation may be needed for macular oedema.",
    3: "Pan-retinal photocoagulation (PRP) laser; anti-VEGF may be considered.",
    4: "Anti-VEGF injections; vitreoretinal surgery if vitreous haemorrhage.",
}

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.backbone   = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        hidden          = self.backbone.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values, noise=None)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


_model = None
def get_model():
    global _model
    if _model is None:
        _model = DRClassifier().to(device)
        if os.path.exists(CHECKPOINT):
            ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            _model.load_state_dict(state, strict=False)
            print(f"[Model] Loaded from {CHECKPOINT}")
        else:
            print("[Model] WARNING: checkpoint not found — using random weights")
        _model.eval()
    return _model


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING  (FR-02)
# ─────────────────────────────────────────────────────────────────
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
    """Returns (224x224 PIL RGB, tensor [1,3,224,224])"""
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = apply_clahe(circle_crop(bgr))
    bgr = cv2.resize(bgr, (224, 224))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil224  = Image.fromarray(rgb)
    tensor  = INFER_TF(pil224).unsqueeze(0).to(device)
    return pil224, tensor


# ─────────────────────────────────────────────────────────────────
# GRADCAM  (FR-04)
# ─────────────────────────────────────────────────────────────────
class ViTGradCAM:
    """Gradient-weighted Class Activation Map for ViT encoder."""
    def __init__(self, model):
        self.model = model
        self._feats = self._grads = None
        layer = model.backbone.encoder.layer[-1]
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_feats", o[0] if isinstance(o, tuple) else o))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0]))

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        # Skip CLS token (index 0) → 196 patch tokens, 768-dim each
        g = self._grads[0, 1:, :]                    # (196, 768)
        f = self._feats[0, 1:, :]                    # (196, 768)
        w = g.mean(dim=-1)                           # (196,)
        cam = F.relu((w.unsqueeze(-1) * f).sum(-1))  # (196,)
        cam = cam.reshape(14, 14).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (224, 224))


def overlay_heatmap(pil224, cam_np):
    img = np.array(pil224).astype(np.float32)
    heat = (cm.jet(cam_np)[:, :, :3] * 255).astype(np.float32)
    blend = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blend)


# ─────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────
def predict(pil_img):
    model   = get_model()
    pil224, tensor = preprocess(pil_img)
    gradcam = ViTGradCAM(model)
    with torch.set_grad_enabled(True):
        logits = model(tensor)
    probs  = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    grade  = int(probs.argmax())
    cam    = gradcam.generate(tensor.clone(), grade)
    cam_pil = overlay_heatmap(pil224, cam)
    return grade, probs, pil224, cam_pil


# ─────────────────────────────────────────────────────────────────
# REPORT GENERATION  (FR-05)
# ─────────────────────────────────────────────────────────────────
def _qwen_report(grade, probs):
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        prob_txt = ", ".join(f"Grade {i}: {p*100:.1f}%" for i, p in enumerate(probs))
        prompt   = (
            f"You are a clinical ophthalmology AI. A diabetic retinopathy screening gave "
            f"Grade {grade} ({GRADES[grade][0]}) at {probs[grade]*100:.1f}% confidence. "
            f"All probabilities: {prob_txt}.\n"
            "Write a structured report: 1) Diagnosis Summary 2) Severity Assessment "
            "3) Likely Lesions 4) Clinical Recommendation 5) Follow-up Timeline. "
            "End with an AI disclaimer. Be clear and compassionate."
        )
        resp = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600, temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def generate_report(grade, probs):
    llm = _qwen_report(grade, probs)
    if llm:
        return llm

    name, desc = GRADES[grade]
    conf = probs[grade] * 100
    bar  = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
    prob_lines = "\n".join(
        f"  • Grade {i} – {GRADES[i][0]}: **{p*100:.1f}%**" for i, p in enumerate(probs))

    return f"""# {GRADE_EMOJI[grade]} RetinAgent Diagnostic Report

## 1. Diagnosis Summary
**Grade {grade} — {name}** | Confidence: {conf:.1f}% [{bar}]

> {desc}

## 2. Class Probabilities
{prob_lines}

## 3. Likely Lesions Present
{LESIONS[grade]}

## 4. Clinical Recommendation
{URGENCY[grade]}

## 5. Treatment Options
{TREATMENTS[grade]}

---
⚠️ *This report is AI-generated for screening purposes only and does not replace a qualified ophthalmologist.*
*Model: ViT-MAE | Dataset: APTOS 2019 | Test Kappa: 0.9097 | AMD MI300X*"""


# ─────────────────────────────────────────────────────────────────
# Q&A AGENT  (FR-06)
# ─────────────────────────────────────────────────────────────────
def _qwen_qa(question, grade, report):
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        system = (
            f"You are RetinAgent, a clinical AI assistant. The patient has Grade {grade} DR "
            f"({GRADES[grade][0]}). Report context: {report[:400]}. "
            "Answer clearly and compassionately. Always recommend consulting an ophthalmologist."
        )
        resp = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": question}],
            max_tokens=400, temperature=0.5,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def _template_qa(question, grade):
    q = question.lower()
    if any(w in q for w in ["what", "mean", "explain", "grade"]):
        return f"**Grade {grade} ({GRADES[grade][0]})**: {GRADES[grade][1]}\n\n**Lesions:** {LESIONS[grade]}"
    if any(w in q for w in ["urgent", "serious", "dangerous", "bad", "worry"]):
        return f"**Urgency:** {URGENCY[grade]}"
    if any(w in q for w in ["treat", "medicine", "therapy", "surgery", "laser"]):
        return f"**Treatment:** {TREATMENTS[grade]}"
    if any(w in q for w in ["lifestyle", "diet", "exercise", "prevent"]):
        return ("**Lifestyle recommendations:**\n"
                "• Control blood sugar (HbA1c < 7%)\n"
                "• Control blood pressure (< 130/80 mmHg)\n"
                "• Manage cholesterol levels\n"
                "• Quit smoking\n"
                "• Regular aerobic exercise (150 min/week)\n"
                "• Annual eye screening")
    if any(w in q for w in ["follow", "next", "appointment", "when"]):
        return f"**Follow-up:** {URGENCY[grade]}"
    return (f"I can answer questions about your Grade {grade} ({GRADES[grade][0]}) diagnosis.\n"
            f"Try asking: *What does this grade mean? What treatment is needed? "
            f"How urgent is this? What lifestyle changes help?*")


def chat_fn(message, history, grade_state, report_state):
    if grade_state is None:
        history.append((message, "⚠️ Please upload and analyse a retinal image first."))
        return history, history
    ans = _qwen_qa(message, grade_state, report_state) or _template_qa(message, grade_state)
    history.append((message, ans))
    return history, history


# ─────────────────────────────────────────────────────────────────
# GRADIO UI  (FR-07)
# ─────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', sans-serif; }
.grade-badge { font-size: 1.4rem; font-weight: 700; }
.report-box { background: #0f172a; color: #e2e8f0; border-radius: 12px; padding: 1rem; }
footer { display: none !important; }
"""

TITLE = """
<div style="text-align:center; padding: 1.5rem 0;">
  <h1 style="font-size:2.5rem; font-weight:800; margin:0;">
    👁️ RetinAgent
  </h1>
  <p style="font-size:1.1rem; color:#64748b; margin-top:0.4rem;">
    AI-Powered Diabetic Retinopathy Detection &amp; Diagnostic Agent<br>
    <b>AMD Instinct MI300X · ViT-MAE · APTOS 2019 · Kappa 0.9097</b>
  </p>
</div>
"""

def analyze(image):
    if image is None:
        return None, None, "⚠️ Please upload an image.", "", None

    try:
        pil    = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        grade, probs, pil224, cam_pil = predict(pil)
        report = generate_report(grade, probs)
        badge  = f"{GRADE_EMOJI[grade]} Grade {grade} — {GRADES[grade][0]}  ({probs[grade]*100:.1f}% confidence)"
        return pil224, cam_pil, badge, report, grade
    except Exception as e:
        return None, None, f"❌ Error: {e}", "", None


with gr.Blocks(css=CSS, title="RetinAgent") as demo:
    grade_state  = gr.State(None)
    report_state = gr.State("")
    chat_history = gr.State([])

    gr.HTML(TITLE)

    with gr.Tabs():

        # ── Tab 1: Diagnosis ───────────────────────────────────
        with gr.TabItem("🔬 Diagnosis"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Retinal Fundus Image",
                                         type="pil", height=300)
                    analyze_btn = gr.Button("🚀 Analyse", variant="primary", size="lg")

                with gr.Column(scale=1):
                    orig_out = gr.Image(label="Preprocessed Image (224×224)", height=220)
                    cam_out  = gr.Image(label="GradCAM Heatmap (FR-04)", height=220)

            grade_out  = gr.Markdown("Upload an image and click Analyse.", elem_classes="grade-badge")
            report_out = gr.Markdown(elem_classes="report-box")

            analyze_btn.click(
                fn=analyze,
                inputs=[img_input],
                outputs=[orig_out, cam_out, grade_out, report_out, grade_state],
            ).then(
                fn=lambda r: r,
                inputs=[report_out],
                outputs=[report_state],
            )

        # ── Tab 2: Q&A Agent ──────────────────────────────────
        with gr.TabItem("💬 Clinical Q&A (FR-06)"):
            gr.Markdown("### Ask follow-up questions about your diagnosis\n"
                        "*Analyse an image first, then ask anything about the result.*")

            chatbot  = gr.Chatbot(height=380, label="RetinAgent Chat")
            with gr.Row():
                msg_box  = gr.Textbox(placeholder="e.g. What does Grade 3 mean? How urgent is this?",
                                      scale=5, show_label=False)
                send_btn = gr.Button("Send", variant="primary", scale=1)

            examples = gr.Examples(
                examples=["What does this DR grade mean?",
                          "How urgent is my condition?",
                          "What treatment options are available?",
                          "What lifestyle changes should I make?",
                          "What lesions are likely present?"],
                inputs=msg_box,
            )

            def respond(msg, hist, g, r):
                if not msg.strip():
                    return hist, hist, ""
                new_hist, _ = chat_fn(msg, hist, g, r)
                return new_hist, new_hist, ""

            send_btn.click(respond,
                           [msg_box, chat_history, grade_state, report_state],
                           [chatbot, chat_history, msg_box])
            msg_box.submit(respond,
                           [msg_box, chat_history, grade_state, report_state],
                           [chatbot, chat_history, msg_box])

        # ── Tab 3: About ──────────────────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## RetinAgent — AMD Developer Hackathon 2026

| Component | Details |
|-----------|---------|
| **Track** | Track 3: Vision & Multimodal AI |
| **Backbone** | ViT-MAE (`facebook/vit-mae-base`) |
| **Dataset** | APTOS 2019 (3,662 retinal images) |
| **Hardware** | AMD Instinct MI300X via AMD Developer Cloud |
| **Framework** | PyTorch (ROCm 6.x) |
| **Test Kappa** | **0.9097** ✅ (Target: >0.85) |
| **Test Accuracy** | **85.01%** ✅ (Target: >80%) |

### DR Grading Scale
| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | 🟢 No DR | No visible lesions |
| 1 | 🟡 Mild | Microaneurysms only |
| 2 | 🟠 Moderate | Exudates, oedema |
| 3 | 🔴 Severe | Extensive haemorrhages |
| 4 | 🆘 Proliferative | Neovascularisation |

⚠️ *For research and educational purposes only. Not a substitute for clinical diagnosis.*
""")

    gr.HTML("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding:1rem 0;">
      RetinAgent · AMD Hackathon 2026 · ViT-MAE · ROCm · MIT License
    </div>""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
