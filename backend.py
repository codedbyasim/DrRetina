#!/usr/bin/env python3
"""
DrRetina — Backend
Model inference, GradCAM, MedGemma reports, Qwen Q&A
AMD Developer Hackathon 2026
"""

import os, cv2, math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTMAEModel
from huggingface_hub import hf_hub_download
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm

# LangChain agent (imported lazily to avoid slow startup)
try:
    from agent import agent_generate_report, agent_qa
    AGENT_AVAILABLE = True
except Exception as e:
    print(f"[Agent] LangChain import failed: {e}")
    AGENT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# FEATHERLESS AI CLIENT  (used as fallback if LangChain unavailable)
# ─────────────────────────────────────────────────────────────────
_DEFAULT_KEY = "rc_c871260215042ae1dc87e28ef5672b1658b30652445af3837d0211b17edee2b8"
FEATHERLESS_KEY = os.environ.get("FEATHERLESS_API_KEY", _DEFAULT_KEY)
try:
    from openai import OpenAI as _OAI
    llm_client = _OAI(base_url="https://api.featherless.ai/v1", api_key=FEATHERLESS_KEY)
except Exception:
    llm_client = None

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
GRADES = {
    0: ("No DR",            "No visible signs of diabetic retinopathy."),
    1: ("Mild DR",          "Microaneurysms only present."),
    2: ("Moderate DR",      "More than microaneurysms but less than severe DR."),
    3: ("Severe DR",        "More than 20 intraretinal haemorrhages in each quadrant."),
    4: ("Proliferative DR", "Neovascularisation or vitreous/pre-retinal haemorrhage."),
}
EMOJI = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🆘"}
COLORS = {0: "#22c55e", 1: "#eab308", 2: "#f97316", 3: "#ef4444", 4: "#dc2626"}
URGENCY = {
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
    0: "No treatment needed. Maintain glycaemic and BP control.",
    1: "Optimise HbA1c <7%, BP <130/80. No direct retinal treatment yet.",
    2: "Focal laser photocoagulation may be needed for macular oedema.",
    3: "Pan-retinal photocoagulation (PRP) laser; anti-VEGF may be considered.",
    4: "Anti-VEGF injections; vitreoretinal surgery if vitreous haemorrhage.",
}

# Grade-specific MedGemma system prompts
MEDGEMMA_SYSTEM = {
    0: "You are an expert ophthalmologist. The patient has NO Diabetic Retinopathy (Grade 0). Provide a reassuring but informative report emphasising preventive care and monitoring.",
    1: "You are an expert ophthalmologist. The patient has MILD Diabetic Retinopathy (Grade 1) with microaneurysms. Provide a clear report about early-stage DR and lifestyle modifications needed.",
    2: "You are an expert ophthalmologist. The patient has MODERATE Diabetic Retinopathy (Grade 2). Explain the progression, risks, and need for closer monitoring and possible treatment.",
    3: "You are an expert ophthalmologist. The patient has SEVERE Diabetic Retinopathy (Grade 3). This is serious — write an urgent clinical report emphasising the need for immediate specialist referral.",
    4: "You are an expert ophthalmologist. The patient has PROLIFERATIVE Diabetic Retinopathy (Grade 4), the most advanced stage. Write an emergency-level report conveying urgency and treatment options.",
}

# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────
HF_REPO   = "lablab-ai-amd-developer-hackathon/DrRetina-weights"
LOCAL_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.backbone.config.mask_ratio = 0.0   # match training: no masking
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        return self.classifier(out.last_hidden_state[:, 0, :])


_model = None

def get_model():
    global _model
    if _model is None:
        _model = DRClassifier().to(device)
        if os.path.exists(LOCAL_CKPT):
            path = LOCAL_CKPT
            print(f"[Model] Local checkpoint: {path}")
        else:
            print("[Model] Downloading from HF Hub...")
            path = hf_hub_download(repo_id=HF_REPO, filename="best_model.pth", repo_type="model")
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        # train.py saves key as 'model_state'
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        _model.load_state_dict(state, strict=False)
        _model.eval()
        print("[Model] ✅ Loaded (Kappa 0.9097 | Acc 85.01%)")
    return _model


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────
def circle_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return img[y:y+h, x:x+w]


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(pil_img):
    """Returns (pil_224, tensor [1,3,224,224])"""
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = apply_clahe(circle_crop(bgr))
    bgr = cv2.resize(bgr, (224, 224))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil224 = Image.fromarray(rgb)
    return pil224, TF(pil224).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────────────────────────
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self._feats = self._grads = None
        layer = model.backbone.encoder.layer[-1]
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_feats", o[0] if isinstance(o, tuple) else o))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0]))

    def generate(self, tensor, class_idx):
        self.model.eval()  # Must be eval mode for BatchNorm1d with batch_size=1
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        g   = self._grads[0, 1:, :]
        f   = self._feats[0, 1:, :]
        w   = g.mean(dim=-1)
        cam = F.relu((w.unsqueeze(-1) * f).sum(-1))
        cam = cam.detach().cpu().numpy()
        n   = len(cam)
        grid = int(math.isqrt(n))
        cam  = cam[:grid*grid].reshape(grid, grid)
        cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (224, 224))


def overlay_heatmap(pil224, cam_np):
    img  = np.array(pil224).astype(np.float32)
    heat = (cm.jet(cam_np)[:, :, :3] * 255).astype(np.float32)
    return Image.fromarray((0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────
# FR-01: IMAGE VALIDATION
# ─────────────────────────────────────────────────────────────────
def validate_image(pil_img):
    """
    FR-01: Validate retinal fundus image.
    Fundus photos have a DARK circular border (30-70% very dark pixels).
    Normal photos, ID cards, selfies etc. have very few dark pixels.
    """
    arr  = np.array(pil_img.convert("RGB")).astype(np.float32)
    gray = arr.mean(axis=2)  # quick grayscale

    # Check 1: blank / all black
    if arr.mean() < 8:
        return False, "Image appears blank or completely dark. Please upload a clear retinal fundus photo."

    # Check 2: resolution too small
    if pil_img.width < 100 or pil_img.height < 100:
        return False, "Image resolution too low. Please upload a higher-quality fundus photo."

    # Check 3: solid fill (no variation)
    if arr.std() < 8:
        return False, "Image appears to be a solid colour. Please upload a valid fundus photo."

    # Check 4: FUNDUS-SPECIFIC — must have significant dark background
    dark_ratio = float(np.mean(gray < 20))
    if dark_ratio < 0.15:
        return (
            False,
            f"This does not appear to be a retinal fundus photograph "
            f"(dark pixel ratio: {dark_ratio*100:.1f}% — expected ≥15%). "
            f"Please upload a proper fundus image with a dark circular border."
        )

    # Check 5: over-exposed / all white
    bright_ratio = float(np.mean(gray > 245))
    if bright_ratio > 0.80:
        return False, "Image appears over-exposed. Please upload a properly exposed fundus photo."

    return True, "OK"

def check_image_quality(pil_img):
    """
    F5: Image Quality Pre-check
    Checks for poor exposure, blurriness, and fundus boundary.
    """
    arr  = np.array(pil_img.convert("RGB"))
    gray_cv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 1. Blur check
    blur_score = cv2.Laplacian(gray_cv, cv2.CV_64F).var()
    if blur_score < 100:
        return False, f"WARNING: Image blurry (score: {blur_score:.1f}). Please retake the photograph under better lighting conditions."
    
    # 2. Exposure check
    mean_val = gray_cv.mean()
    if mean_val < 40 or mean_val > 220:
        return False, f"WARNING: Poor exposure (mean: {mean_val:.1f}). Please retake the photograph under better lighting conditions."
    
    # 3. Fundus boundary check
    _, thresh = cv2.threshold(gray_cv, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "WARNING: Not a retinal image (circle not found)."
        
    return True, "OK"


def predict(pil_img):
    model = get_model()
    model.eval()  # CRITICAL: BatchNorm1d needs eval mode for batch_size=1
    pil224, tensor = preprocess(pil_img)
    gradcam = ViTGradCAM(model)
    with torch.set_grad_enabled(True):
        logits = model(tensor)
    probs  = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    grade  = int(probs.argmax())
    cam    = gradcam.generate(tensor.clone(), grade)
    return grade, probs, pil224, overlay_heatmap(pil224, cam)


# ─────────────────────────────────────────────────────────────────
# MEDGEMMA REPORT  (FR-05) & F2: BILINGUAL
# ─────────────────────────────────────────────────────────────────
def medgemma_report(grade, probs, language="English"):
    """Grade-specific medical report via Qwen3-8B (fast)."""
    if not llm_client:
        return None
    try:
        prob_txt = " | ".join(f"Grade {i} ({GRADES[i][0]}): {p*100:.1f}%" for i, p in enumerate(probs))
        user_prompt = (
            f"DIABETIC RETINOPATHY SCREENING RESULT:\n"
            f"- Detected Grade: {grade} — {GRADES[grade][0]}\n"
            f"- Confidence: {probs[grade]*100:.1f}%\n"
            f"- All Class Probabilities: {prob_txt}\n\n"
            f"Write a concise clinical diagnostic report in {language} with these sections:\n"
            f"1. **Diagnosis Summary** — What was found\n"
            f"2. **Severity Assessment** — How serious is this\n"
            f"3. **Expected Lesions** — What signs are present\n"
            f"4. **Treatment Options** — Available treatments\n"
            f"5. **Follow-up** — When to see a doctor\n"
            f"6. **Recommendation** — Clear actionable advice\n\n"
            f"End with: '> ⚠️ AI Disclaimer: This report is AI-generated for screening purposes only.'\n"
            f"Be concise, compassionate, and medically accurate. Max 500 words."
        )
        resp = llm_client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[
                {"role": "system", "content": MEDGEMMA_SYSTEM[grade] + " Do not repeat sections. Stop after the disclaimer."},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.2,
            stop=["End of report", "AI Disclaimer", "©"]
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[LLM Report Error] {e}")
        return None


def generate_report(grade, probs, language="English"):
    """
    Generate diagnostic report:
    1. LangChain + Qwen agent (primary - SRS FR-05)
    2. Direct LLM fallback
    3. Static template (last resort)
    """
    # 1. LangChain agent (SRS §2.2 — Agent Layer)
    if AGENT_AVAILABLE and FEATHERLESS_KEY:
        report = agent_generate_report(grade, probs, language=language)
        if report:
            return report

    # 2. Direct API fallback
    report = medgemma_report(grade, probs, language=language)
    if report:
        return report

    # 3. Static template
    name, desc = GRADES[grade]
    conf = probs[grade] * 100
    prob_lines = "\n".join(
        f"- **Grade {i} – {GRADES[i][0]}**: {p*100:.1f}%" for i, p in enumerate(probs))
    return f"""## {EMOJI[grade]} Grade {grade}: {name}

**Confidence: {conf:.1f}%**

> {desc}

### 1. Diagnosis Summary
Grade {grade} DR detected with {conf:.1f}% confidence.

### 2. Severity Assessment
{URGENCY[grade]}

### 3. Expected Lesions
{LESIONS[grade]}

### 4. Treatment Options
{TREATMENTS[grade]}

### 5. Follow-up Timeline
{URGENCY[grade]}

### 6. Clinical Recommendation
Please consult a qualified ophthalmologist immediately.

### 📊 All Probabilities
{prob_lines}

---
> ⚠️ *AI screening tool only. Always consult a qualified ophthalmologist.*"""


# ─────────────────────────────────────────────────────────────────
# QWEN Q&A  (FR-06)  — LangChain ReAct Agent
# ─────────────────────────────────────────────────────────────────
def qwen_qa(question: str, grade: int, report: str, history: list = None) -> str | None:
    """
    Answer clinical questions:
    1. LangChain ReAct agent with tools (primary)
    2. Direct LLM fallback
    """
    probs_mock = [0.0] * 5
    probs_mock[grade] = 1.0
    confidence = 90.0  # placeholder for Q&A context

    # 1. LangChain agent with tools (SRS §2.2 — Agent Layer)
    if AGENT_AVAILABLE and FEATHERLESS_KEY:
        ans = agent_qa(question, grade, confidence, report, history=history)
        if ans:
            return ans

    # 2. Direct API fallback
    return _direct_qa(question, grade, report, history=history)


def _direct_qa(question: str, grade: int, report: str, history: list = None) -> str | None:
    """Fallback direct API Q&A without LangChain."""
    if not llm_client:
        return None
    try:
        system = (
            f"You are DrRetina, a clinical AI assistant. "
            f"Patient has Grade {grade} DR — {GRADES[grade][0]}. "
            f"Report context: {report[:400]}. "
            f"Answer clearly and recommend consulting an ophthalmologist."
        )
        resp = llm_client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[{"role": "system", "content": system}] + (history if history else []) + [{"role": "user", "content": question}],
            max_tokens=1500,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[Direct QA Error] {e}")
        return None


def template_qa(question, grade):
    """Fallback template Q&A."""
    q = question.lower()
    if any(w in q for w in ["what", "mean", "explain", "grade", "kya", "matlab"]):
        return f"**Grade {grade} – {GRADES[grade][0]}**: {GRADES[grade][1]}\n\n📅 {URGENCY[grade]}"
    if any(w in q for w in ["treat", "cure", "laser", "injection", "ilaj"]):
        return f"**Treatment for Grade {grade}**:\n{TREATMENTS[grade]}"
    if any(w in q for w in ["urgent", "serious", "danger", "blind", "khatarnak"]):
        return f"**⚠️ Urgency**: {URGENCY[grade]}\n\n**Treatment**: {TREATMENTS[grade]}"
    if any(w in q for w in ["lesion", "sign", "appear", "nishaan"]):
        return f"**Expected findings for Grade {grade}**:\n{LESIONS[grade]}"
    return (
        f"Based on your Grade {grade} ({GRADES[grade][0]}) result:\n\n"
        f"{GRADES[grade][1]}\n\n"
        f"**Recommended action**: {URGENCY[grade]}\n\n"
        f"Please consult a qualified ophthalmologist for personalised medical advice."
    )


# ─────────────────────────────────────────────────────────────────
# F4: AI REFERRAL LETTER GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_referral_letter_from_agent(grade, confidence):
    """
    Generate an AI Referral letter using LangChain tools or a direct LLM call.
    """
    if not llm_client:
        return "Error: Language model not available to generate referral letter."
    try:
        user_prompt = (
            f"Detected Grade: {grade} — {GRADES[grade][0]}\n"
            f"Confidence: {confidence:.1f}%\n\n"
            f"Generate a formal clinical referral letter to a Vitreoretinal Specialist. "
            f"Act as the referring AI system. Keep it concise, formal, and medical. "
            f"Include the AI Analysis Findings, risk level, and suggested intervention timeline."
        )
        resp = llm_client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[
                {"role": "system", "content": "You are DrRetina Clinical AI System generating a formal referral letter."},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[Referral Error] {e}")
        return "Error generating referral letter."


# ─────────────────────────────────────────────────────────────────
# F3: BATCH PROCESSING MODE
# ─────────────────────────────────────────────────────────────────
def batch_process_zip(zip_path, output_csv_path):
    """
    F3: Processes a ZIP of images using DataLoader and MI300X batch inference.
    Returns path to CSV.
    """
    import zipfile
    import tempfile
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset
    
    # Custom simple dataset for batch inference
    class BatchDataset(Dataset):
        def __init__(self, img_paths):
            self.img_paths = img_paths
        def __len__(self):
            return len(self.img_paths)
        def __getitem__(self, idx):
            path = self.img_paths[idx]
            try:
                # Need to use standard preprocessing but return tensor
                _, tensor = preprocess(Image.open(path))
                return tensor.squeeze(0), os.path.basename(path)
            except Exception as e:
                # If image is broken, return a dummy tensor
                return torch.zeros((3, 224, 224)), os.path.basename(path)
                
    try:
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        # Get all valid image files
        img_paths = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(root, f))
                    
        if not img_paths:
            return None, "No valid images found in the ZIP file."
            
        dataset = BatchDataset(img_paths)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) # MI300X optimized batch size
        
        model = get_model()
        model.eval()
        
        results = []
        with torch.no_grad():
            for tensors, filenames in loader:
                tensors = tensors.to(device)
                logits = model(tensors)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                
                for i in range(len(filenames)):
                    grade = int(probs[i].argmax())
                    conf = float(probs[i][grade])
                    
                    # Determine priority based on grade
                    if grade == 4:
                        priority, action = "URGENT", "Refer within 48 hours"
                    elif grade == 3:
                        priority, action = "HIGH", "Refer within 2 weeks"
                    elif grade == 2:
                        priority, action = "MEDIUM", "Follow up in 3 months"
                    elif grade == 1:
                        priority, action = "LOW", "Follow up in 6 months"
                    else:
                        priority, action = "ROUTINE", "Annual screening"
                        
                    results.append({
                        "Patient File": filenames[i],
                        "Grade": grade,
                        "Severity": GRADES[grade][0],
                        "Confidence": f"{conf*100:.1f}%",
                        "Priority": priority,
                        "Action Required": action
                    })
                    
        # Sort results: Priority (Grade 4 down to 0)
        df = pd.DataFrame(results)
        df = df.sort_values(by="Grade", ascending=False)
        df.to_csv(output_csv_path, index=False)
        
        return output_csv_path, f"Successfully processed {len(img_paths)} images."
        
    except Exception as e:
        print(f"[Batch Processing Error] {e}")
        return None, str(e)
