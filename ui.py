#!/usr/bin/env python3
"""
DrRetina — Gradio UI  (Light Theme, Clean Medical Design)
"""

import gradio as gr
from backend import (
    predict, generate_report, qwen_qa, template_qa,
    validate_image, check_image_quality, generate_referral_letter_from_agent, batch_process_zip,
    GRADES, EMOJI, COLORS, URGENCY,
)
import datetime
import os
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# CSS  —  Clean Light Medical Theme
# ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container * {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%) !important;
    background-attachment: fixed !important;
    color: #1a202c !important;
}

.gradio-container {
    background: transparent !important;
    max-width: 1250px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem 3rem !important;
}

footer, .built-with { display: none !important; }

/* ── Tab Navigation ── */
.tab-nav {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 6px !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04) !important;
}

.tab-nav button {
    background: transparent !important;
    color: #718096 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button:hover {
    background: rgba(255, 255, 255, 0.8) !important;
    color: #2d3748 !important;
    transform: translateY(-1px) !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(43,108,176,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Cards & Panels (Glassmorphism) ── */
.gr-panel, .gradio-group, .gr-box, fieldset {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05) !important;
    transition: transform 0.2s ease !important;
}

/* ── Labels ── */
label span, .gr-form label { color: #2d3748 !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.02em !important;}

/* ── Inputs ── */
textarea, input[type="text"], input[type="file"] {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #1a202c !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #3182ce !important;
    box-shadow: 0 0 0 3px rgba(49,130,206,0.15) !important;
    background: #ffffff !important;
    outline: none !important;
}

/* ── Primary Button ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    box-shadow: 0 4px 14px rgba(43,108,176,0.35), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
}

.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(43,108,176,0.45), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    background: linear-gradient(135deg, #3182ce 0%, #2a4365 100%) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%) !important;
    color: #2d3748 !important;
    border: 1px solid #cbd5e0 !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
}
.gr-button-secondary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    background: #ffffff !important;
}

/* ── Image ── */
.gr-image-preview, .image-container {
    background: rgba(255, 255, 255, 0.5) !important;
    border: 2px dashed #cbd5e0 !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
}
.gr-image-preview:hover, .image-container:hover {
    border-color: #3182ce !important;
    background: rgba(255, 255, 255, 0.8) !important;
}

/* ── Markdown ── */
.prose, .gr-markdown, .md {
    color: #2d3748 !important;
    line-height: 1.8 !important;
    font-size: 0.96rem !important;
}

.prose h1, .prose h2, .prose h3,
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #1a202c !important;
    font-weight: 800 !important;
    margin-top: 1.5rem !important;
    letter-spacing: -0.02em !important;
}

.prose strong, .gr-markdown strong { color: #1a202c !important; font-weight: 700 !important;}

.prose blockquote, .gr-markdown blockquote {
    border-left: 4px solid #3182ce !important;
    background: linear-gradient(90deg, rgba(235,248,255,0.8) 0%, rgba(235,248,255,0.2) 100%) !important;
    padding: 1rem 1.2rem !important;
    border-radius: 0 12px 12px 0 !important;
    color: #2c5282 !important;
    font-size: 0.92rem !important;
    font-style: italic !important;
}

/* ── Chatbot ── */
.chatbot-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

.chatbot {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    border-radius: 24px !important;
    padding: 1rem !important;
}

.message.user > div {
    background: #3182ce !important;
    color: #ffffff !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 12px 18px !important;
    font-size: 0.98rem !important;
    box-shadow: 0 4px 12px rgba(49,130,206,0.2) !important;
    border: none !important;
}

.message.bot > div, .message.assistant > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #2d3748 !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 12px 18px !important;
    font-size: 0.98rem !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
}

.chat-input-container {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 6px !important;
    margin-top: 1rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
    transition: all 0.3s ease !important;
}

.chat-input-container:focus-within {
    border-color: #3182ce !important;
    box-shadow: 0 10px 30px rgba(49,130,206,0.1) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3182ce; }
"""

# ─────────────────────────────────────────────────────────────────
# HTML COMPONENTS
# ─────────────────────────────────────────────────────────────────
HEADER_HTML = """
<div style="
    text-align: center;
    padding: 3rem 1.5rem 2.5rem;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 24px;
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 40px rgba(31, 38, 135, 0.08);
    position: relative;
    overflow: hidden;
">
  <!-- Decorative background elements -->
  <div style="position:absolute;top:-50px;left:-50px;width:150px;height:150px;background:rgba(66,153,225,0.15);border-radius:50%;filter:blur(40px);"></div>
  <div style="position:absolute;bottom:-50px;right:-50px;width:200px;height:200px;background:rgba(154,230,180,0.15);border-radius:50%;filter:blur(40px);"></div>

  <div style="
      position: relative;
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(235, 248, 255, 0.9); border: 1px solid #bee3f8;
      border-radius: 999px; padding: 6px 18px;
      margin-bottom: 1.5rem;
      font-size: 0.75rem; font-weight: 800; color: #2b6cb0;
      letter-spacing: 0.1em; text-transform: uppercase;
      box-shadow: 0 2px 8px rgba(43,108,176,0.1);
  ">
    Clinical Intelligence · Diagnostic Support System
  </div>

  <div style="position:relative; display:flex;align-items:center;justify-content:center;gap:18px;margin-bottom:0.8rem">
    <div style="
        width: 60px; height: 60px;
        background: linear-gradient(135deg, #3182ce, #2a4365);
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2rem;
        box-shadow: 0 8px 24px rgba(43,108,176,0.4), inset 0 2px 0 rgba(255,255,255,0.2);
    ">👁️</div>
    <h1 style="
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        letter-spacing: -2px;
        color: #1a202c;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.05);
    ">DrRetina</h1>
  </div>

  <p style="position:relative; color: #4a5568; font-size: 1.1rem; margin-bottom: 1.8rem; font-weight: 500">
    Clinical AI-Powered Diabetic Retinopathy Detection Agent
  </p>

  <div style="position:relative; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
    <span style="background:rgba(255,255,255,0.9);border:1px solid #e2e8f0;border-radius:10px;padding:6px 16px;
                 font-size:0.85rem;color:#4a5568;font-weight:700;box-shadow:0 2px 6px rgba(0,0,0,0.04)">
      🔥 Finetuned on AMD MI300X
    </span>
    <span style="background:rgba(255,255,255,0.9);border:1px solid #e2e8f0;border-radius:10px;padding:6px 16px;
                 font-size:0.85rem;color:#4a5568;font-weight:700;box-shadow:0 2px 6px rgba(0,0,0,0.04)">
      🧠 ViT-MAE & Qwen3-8B
    </span>
    <span style="background:linear-gradient(135deg, #f0fff4, #c6f6d5);border:1px solid #9ae6b4;border-radius:10px;padding:6px 16px;
                 font-size:0.85rem;color:#22543d;font-weight:800;box-shadow:0 2px 6px rgba(39,103,73,0.1)">
      ✅ Kappa 0.9097
    </span>
  </div>
</div>
"""

EMPTY_STATE_HTML = """
<div style="
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: #f7fafc;
    border: 2px dashed #e2e8f0;
    border-radius: 16px;
    margin: 1rem 0;
">
  <div style="font-size: 2.5rem; margin-bottom: 0.75rem">🔍</div>
  <p style="color: #718096; font-size: 0.95rem; margin: 0; line-height: 1.6">
    Upload a retinal fundus image and click<br>
    <strong style="color: #2b6cb0">Analyse Image</strong> to get your DR grade &amp; clinical report.
  </p>
</div>
"""

LOADING_HTML = """
<div style="
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: #ebf8ff;
    border: 1px solid #bee3f8;
    border-radius: 16px;
    margin: 1rem 0;
">
  <div style="font-size: 2rem; margin-bottom: 0.5rem">⏳</div>
  <p style="color: #2b6cb0; font-size: 0.95rem; font-weight: 600; margin: 0">
    Analysing image &amp; generating AI report...
  </p>
</div>
"""

LOADING_REPORT_HTML = "*⏳ AI report generating... please wait.*"

GRADE_BG = {0: "#f0fff4", 1: "#fffff0", 2: "#fff7ed", 3: "#fff5f5", 4: "#fff5f5"}
GRADE_BORDER = {0: "#9ae6b4", 1: "#f6e05e", 2: "#fbd38d", 3: "#fc8181", 4: "#fc8181"}
GRADE_TEXT = {0: "#22543d", 1: "#744210", 2: "#7b341e", 3: "#742a2a", 4: "#63171b"}

def make_grade_badge(grade, probs):
    color  = COLORS[grade]
    bg     = GRADE_BG[grade]
    border = GRADE_BORDER[grade]
    text   = GRADE_TEXT[grade]
    name   = GRADES[grade][0]
    conf   = probs[grade] * 100
    
    # F5: Confidence Tier
    if conf >= 88:
        conf_tier = "HIGH CONFIDENCE"
        conf_color = "#22c55e" # Green
    elif conf >= 72:
        conf_tier = "BORDERLINE"
        conf_color = "#eab308" # Yellow
    else:
        conf_tier = "LOW CONFIDENCE"
        conf_color = "#ef4444" # Red

    prob_pills = "".join(
        f"""<div style="
            background: {'#fff' if i != grade else bg};
            border: 1.5px solid {GRADE_BORDER[i] if i == grade else '#e2e8f0'};
            border-radius: 8px;
            padding: 5px 12px;
            font-size: 0.78rem;
            color: {GRADE_TEXT[i] if i == grade else '#4a5568'};
            font-weight: {'700' if i == grade else '500'};
            white-space: nowrap;
        "><span style='margin-right:4px'>{EMOJI[i]}</span>{GRADES[i][0]}: {p*100:.1f}%</div>"""
        for i, p in enumerate(probs)
    )

    return f"""
    <div style="
        background: {bg};
        border: 1.5px solid {border};
        border-radius: 20px;
        padding: 1.75rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    ">
      <div style="display:flex;align-items:flex-start;gap:1rem;flex-wrap:wrap">
        <div style="flex:1;min-width:200px">
          <div style="font-size:0.72rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem">
            DR Grade Detected
          </div>
          <div style="font-size:2rem;font-weight:800;color:{text};letter-spacing:-0.5px;margin-bottom:0.3rem">
            {EMOJI[grade]} Grade {grade} — {name}
          </div>
          <div style="color:{text};font-size:0.9rem;font-weight:500;opacity:0.8">
            {URGENCY[grade]}
          </div>
        </div>

        <div style="text-align:right;min-width:100px">
          <div style="font-size:0.72rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem">
            Confidence
          </div>
          <div style="font-size:2.2rem;font-weight:800;color:{text}">{conf:.1f}%</div>
          <div style="display:inline-block; margin-top:4px; padding:3px 8px; border-radius:4px; font-size:0.7rem; font-weight:bold; color:white; background-color:{conf_color};">
            {conf_tier}
          </div>
        </div>
      </div>

      <!-- Confidence bar -->
      <div style="height:8px;background:#e2e8f0;border-radius:999px;margin:1.25rem 0;overflow:hidden">
        <div style="
            height:100%;width:{conf:.1f}%;
            background:linear-gradient(90deg,{color},{color}bb);
            border-radius:999px;
        "></div>
      </div>

      <!-- Probabilities -->
      <div style="font-size:0.72rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem">
        All Class Probabilities
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        {prob_pills}
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────
# STEP 1: Fast inference (grade + heatmap, no LLM)
# ─────────────────────────────────────────────────────────────────
def fast_analyse(pil_img):
    """Returns grade + images immediately. No LLM call."""
    if pil_img is None:
        return None, None, EMPTY_STATE_HTML, LOADING_REPORT_HTML, None, None
        
    # F5: Image Quality Pre-check
    q_ok, q_msg = check_image_quality(pil_img)
    q_alert = ""
    if not q_ok:
        q_alert = f"<div style='background:#fff5f5;border:1px solid #fc8181;border-radius:8px;padding:8px;margin-bottom:10px;'><b style='color:#c53030;'>QC Alert:</b> <span style='color:#c53030;font-size:0.9rem'>{q_msg}</span></div>"

    # FR-01: Validate image
    ok, msg = validate_image(pil_img)
    if not ok:
        err = q_alert + f"""<div style='background:#fff5f5;border:1.5px solid #fc8181;border-radius:14px;
                   padding:1.25rem 1.5rem;margin:1rem 0'>
                   <div style='font-size:1.1rem;margin-bottom:0.3rem'>⚠️ Invalid Image</div>
                   <p style='color:#c53030;margin:0;font-size:0.9rem'>{msg}</p></div>"""
        return None, None, err, "", None, None
    try:
        grade, probs, pil224, cam_pil = predict(pil_img)
        max_prob = float(probs[grade])
        if max_prob < 0.50:
            ood_alert = f"<div style='background:#fff5f5;border:1px solid #fc8181;border-radius:8px;padding:8px;margin-bottom:10px;'><b style='color:#c53030;'>Low Confidence Alert:</b> <span style='color:#c53030;font-size:0.9rem'>Model confidence is only {max_prob*100:.1f}%. This image might be of poor quality, out-of-focus, or not a standard retinal fundus photo.</span></div>"
            q_alert = ood_alert + q_alert
        badge = q_alert + make_grade_badge(grade, probs)
            
        return pil224, cam_pil, badge, LOADING_REPORT_HTML, grade, probs.tolist()
    except Exception as e:
        import traceback; traceback.print_exc()
        err = f"""<div style='background:#fff5f5;border:1px solid #fc8181;border-radius:12px;
                   padding:1.25rem;margin:1rem 0'><p style='color:#c53030;font-weight:600;margin:0'>
                   ❌ Error: {e}</p></div>"""
        return None, None, err, "", None, None


# ─────────────────────────────────────────────────────────────────
# STEP 2: Slow LLM report
# ─────────────────────────────────────────────────────────────────
def get_report(grade, probs_list, language):
    """Called after fast_analyse via .then() — generates LLM report."""
    import gradio as gr
    if grade is None or probs_list is None:
        return "", "", gr.DownloadButton(visible=False)
    import numpy as np
    import tempfile
    import os
    import markdown
    from weasyprint import HTML
    probs  = np.array(probs_list)
    report = generate_report(grade, probs, language)
    
    html_body = markdown.markdown(report, extensions=['tables'])
    is_rtl = language in ["Urdu", "Arabic"]
    dir_attr = 'dir="rtl"' if is_rtl else 'dir="ltr"'
    text_align = 'right' if is_rtl else 'left'
    
    html_content = f"""<html>
    <head>
    <meta charset='utf-8'>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&family=Noto+Naskh+Arabic:wght@400;700&family=Noto+Sans+Devanagari:wght@400;700&display=swap');
    body {{ 
        font-family: 'Noto Sans', 'Noto Naskh Arabic', 'Noto Sans Devanagari', sans-serif; 
        line-height: 1.6; 
        padding: 2em; 
        text-align: {text_align};
    }} 
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }} 
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; }} 
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: {text_align}; }} 
    th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body {dir_attr}>
    <h1 style="text-align: {text_align}">DrRetina Clinical Report</h1>
    {html_body}
    </body>
    </html>"""
    
    tmp_path = os.path.join(tempfile.gettempdir(), "DrRetina_Clinical_Report.pdf")
    try:
        HTML(string=html_content).write_pdf(tmp_path)
    except Exception as e:
        # Fallback to TXT if pdfkit fails locally without wkhtmltopdf
        tmp_path = os.path.join(tempfile.gettempdir(), "DrRetina_Clinical_Report.txt")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("DrRetina Clinical Report\n==========================\n\n" + report)
            
    return gr.Markdown(value=report, rtl=is_rtl), report, gr.DownloadButton(value=tmp_path, visible=True)

def create_referral(grade, probs_list):
    if grade is None or probs_list is None:
        return "⚠️ Analyze an image first."
    conf = probs_list[grade] * 100
    letter = generate_referral_letter_from_agent(grade, conf)
    return letter


# ─────────────────────────────────────────────────────────────────
# CHAT FUNCTION
# ─────────────────────────────────────────────────────────────────
def chat_fn(message, history, g_state, r_state):
    if history is None:
        history = []
    if not message.strip():
        return history, history
    if g_state is None:
        history.append({
            "role": "assistant",
            "content": "⚠️ Please upload and analyse a retinal image first, then I can answer your questions."
        })
        return history, history
    ans = qwen_qa(message, g_state, r_state, history=history) or template_qa(message, g_state)
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": ans})
    return history, history


# ─────────────────────────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        css=CSS,
        title="DrRetina — AI Diabetic Retinopathy Detection",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Plus Jakarta Sans"),
        ),
    ) as demo:

        g_state     = gr.State(None)
        r_state     = gr.State("")
        probs_state = gr.State(None)

        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            # ━━━ Tab 1: Diagnosis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("🔬 Diagnosis"):
                with gr.Row(equal_height=False):

                    # Left: Upload + controls
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML("""<div style="font-size:0.85rem;font-weight:800;color:#2d3748;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.8rem;border-bottom:2px solid #e2e8f0;padding-bottom:4px;">Analysis Setup</div>""")
                        
                        with gr.Row():
                            lang_in = gr.Dropdown(["English", "Urdu", "Hindi", "Arabic", "Spanish", "French"], label="Report Language", value="English", scale=1)
                        
                        img_in = gr.Image(
                            type="pil",
                            label="Retinal Fundus Image",
                            height=280,
                            show_label=True,
                            show_download_button=False,
                        )
                        
                        btn = gr.Button("🔍 Analyse Image", variant="primary", size="lg")

                        gr.HTML("""
                        <div style="
                            margin-top:0.75rem;padding:0.9rem 1rem;
                            background:#f7fafc;border:1px solid #e2e8f0;
                            border-radius:10px;font-size:0.82rem;color:#718096;line-height:1.65;
                        ">
                          <strong style="color:#4a5568;display:block;margin-bottom:4px">📋 How to use</strong>
                          Upload a clear retinal fundus photo (JPG/PNG).
                          The AI detects the DR grade, highlights affected areas with GradCAM,
                          and generates a personalised clinical report.
                        </div>
                        """)

                    # Right: Outputs
                    with gr.Column(scale=1, min_width=300):
                        gr.HTML("""<div style="font-size:0.78rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem">Analysis Output</div>""")
                        with gr.Row():
                            img_pre = gr.Image(label="Enhanced View", height=145, show_download_button=False)
                            img_cam = gr.Image(label="Clinical Attention Map", height=145, show_download_button=False)
                        # GradCAM color legend
                        gr.HTML("""
                        <div style="display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap">
                          <span style="font-size:0.72rem;color:#718096;font-weight:600">GradCAM Legend:</span>
                          <div style="display:flex;align-items:center;gap:4px">
                            <div style="width:60px;height:10px;border-radius:4px;
                                        background:linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)"></div>
                            <span style="font-size:0.7rem;color:#718096">Low → High Attention</span>
                          </div>
                        </div>
                        """)

                # Grade result
                grade_html = gr.HTML(EMPTY_STATE_HTML)

                # Divider
                gr.HTML("""
                <div style="display:flex;align-items:center;gap:1rem;margin:1.5rem 0 0.75rem">
                  <div style="height:1px;flex:1;background:#e2e8f0"></div>
                  <span style="font-size:0.78rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap">
                    📋 AI Clinical Report
                  </span>
                  <div style="height:1px;flex:1;background:#e2e8f0"></div>
                </div>
                """)

                report_md = gr.Markdown(
                    value="*Analyse an image to generate a personalised AI clinical report.*",
                )
                download_btn = gr.DownloadButton("📥 Download Clinical Report", visible=False)
                
                btn.click(
                    fn=fast_analyse,
                    inputs=[img_in],
                    outputs=[img_pre, img_cam, grade_html, report_md, g_state, probs_state],
                    api_name=False,
                ).then(
                    fn=get_report,
                    inputs=[g_state, probs_state, lang_in],
                    outputs=[report_md, r_state, download_btn],
                    api_name=False,
                )

            # ━━━ Tab 2: Clinical Q&A ━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.TabItem("💬 Clinical Q&A"):
                with gr.Column(elem_classes="chatbot-container"):
                    gr.HTML("""
                    <div style="text-align: center; padding: 0.5rem 0 1rem">
                      <h2 style="color:#1a202c; font-size:1.6rem; font-weight:800; margin-bottom:0.3rem">DrRetina AI Assistant</h2>
                      <p style="color:#718096; font-size:0.95rem; max-width:600px; margin: 0 auto">
                        Ask questions about your screening results, treatment guidelines, or general eye health.
                        Powered by <strong>Qwen3-8B</strong>.
                      </p>
                    </div>
                    """)

                    chatbot = gr.Chatbot(
                        height=350,
                        type="messages",
                        show_label=False,
                        elem_classes="chatbot",
                        placeholder="<div style='text-align:center;color:#a0aec0;padding:2rem;font-size:1rem'>👋 Hello! I'm your clinical assistant.<br>Upload an image in the analysis tab to start a detailed discussion.</div>",
                    )

                    with gr.Row(elem_classes="chat-input-container"):
                        msg = gr.Textbox(
                            placeholder="Type your question here...",
                            scale=9,
                            show_label=False,
                            container=False,
                        )
                        send = gr.Button("↑", variant="primary", scale=1, min_width=50)

                    gr.HTML("""
                    <div style="margin-top:1.2rem; display:flex; justify-content: center; gap:10px; flex-wrap:wrap">
                      <span style="font-size:0.85rem; color:#718096; font-weight:600; margin-right:5px; align-self:center">Try asking:</span>
                      <button onclick="document.querySelector('textarea').value='Explain my DR grade in detail.'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="background:#f7fafc; border:1px solid #e2e8f0; border-radius:99px; padding:6px 15px; font-size:0.8rem; color:#4a5568; cursor:pointer; transition: all 0.2s">Grade explanation</button>
                      <button onclick="document.querySelector('textarea').value='What are the next steps for my treatment?'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="background:#f7fafc; border:1px solid #e2e8f0; border-radius:99px; padding:6px 15px; font-size:0.8rem; color:#4a5568; cursor:pointer; transition: all 0.2s">Next steps</button>
                      <button onclick="document.querySelector('textarea').value='How can I prevent further vision loss?'; document.querySelector('textarea').dispatchEvent(new Event('input'))" style="background:#f7fafc; border:1px solid #e2e8f0; border-radius:99px; padding:6px 15px; font-size:0.8rem; color:#4a5568; cursor:pointer; transition: all 0.2s">Prevention</button>
                    </div>
                    """)

                chat_hist = gr.State(None)

                send.click(
                    chat_fn,
                    inputs=[msg, chat_hist, g_state, r_state],
                    outputs=[chatbot, chat_hist],
                    api_name=False,
                ).then(lambda: "", outputs=msg)

                msg.submit(
                    chat_fn,
                    inputs=[msg, chat_hist, g_state, r_state],
                    outputs=[chatbot, chat_hist],
                    api_name=False,
                ).then(lambda: "", outputs=msg)

            # ━━━ Tab 3: Batch Processing (F3) ━━━━━━━━━━━━━━━━━
            with gr.TabItem("ℹ️ How it Works"):
                gr.Markdown("""
## 👁️ DrRetina: Advanced Clinical AI

DrRetina is a next-generation diagnostic assistant for Diabetic Retinopathy (DR) screening. It combines high-performance vision transformers with generative medical intelligence to provide clinicians with clear, actionable insights.

---

### 🛡️ Built-in Quality Assurance

Our system ensures clinical reliability through a sophisticated automated pipeline:
- **Smart Image Validation**: Automatically detects poor lighting, blur, or incorrect focus before analysis begins.
- **Retinal Enhancement**: Applies clinical-grade contrast enhancement to make subtle microaneurysms and haemorrhages more visible.
- **Explainable Diagnostics**: Generates a **Clinical Attention Map** that highlights the specific pathological areas identified by the AI.

---

### 🧠 Modern AI Architecture

DrRetina is built on cutting-edge infrastructure optimized for medical precision:
- **Vision Core**: A Vision Transformer (ViT-MAE) specialized in ophthalmic features.
- **Agentic Layer**: Powered by **Qwen3-8B**, providing structured clinical reporting and natural language Q&A.
- **High-Performance Hardware**: Fine-tuned on **AMD Instinct™ MI300X** for superior medical precision.

---

### 📋 Seamless Workflow

| Phase | Description |
|-------|-----------|
| **1. Analysis** | Instant grading from No DR to Proliferative DR with confidence metrics. |
| **2. Visualization** | Detailed attention mapping showing areas of clinical concern. |
| **3. Reporting** | Automated, multi-lingual clinical reports generated in seconds. |

---

### 📈 Grading Scale & Clinical Action

| Grade | Name | Recommended Action |
|-------|------|--------------------|
| 🟢 0 | No DR | Routine annual review. |
| 🟡 1 | Mild DR | 6-month follow-up and metabolic control. |
| 🟠 2 | Moderate DR | Specialist referral within 3 months. |
| 🔴 3 | Severe DR | Urgent referral within 1 month. |
| 🆘 4 | Proliferative | Emergency referral; immediate risk of vision loss. |

---

> ⚠️ **Clinical Note**: DrRetina is an AI screening tool designed to support, not replace, professional ophthalmic evaluation. Always consult a qualified medical professional for definitive diagnosis.
| ViT-MAE Encoder + Classification Head | PyTorch (ROCm), HuggingFace |
| **Explainability** | GradCAM Engine | pytorch-grad-cam |
| **Agent** | Report Generator + Q&A | Qwen3-8B (Featherless AI) |
| **Interface** | Gradio Web UI | Gradio 5.x |
| **Training Compute** | AMD Instinct MI300X | ROCm 6.x |
| **Deployment** | HF Spaces | AMD Hackathon Org |

---

### 🧠 Model Architecture

```
facebook/vit-mae-base
  └─ 12 Transformer Encoder Blocks (768-dim, 12 heads)
  └─ 224×224 input → 196 patches (16×16 each)
  └─ mask_ratio = 0.0 (all patches used for inference)

Classification Head:
  Linear(768, 256) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(256, 5)
```

**Training:** Fine-tuned on APTOS 2019 (3,662 images) on AMD Instinct MI300X

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW + Weight Decay |
| Backbone LR | 2e-5 |
| Head LR | 1e-3 |
| Scheduler | Cosine Decay + 5-epoch warmup |
| Batch Size | 128 |
| Epochs | 30 |
| Loss | Focal Loss (γ=2) + Label Smoothing |

---

### 📊 Performance Results

| Metric | Target | **Achieved** |
|--------|--------|-------------|
| Cohen’s Kappa | > 0.85 | **0.9097** ✅ |
| Accuracy | > 80% | **85.01%** ✅ |
| Inference Latency | < 5s | **~2–3s** ✅ |
""")
                gr.Gallery(
                    value=[
                        "graphs/acc_plot.png",
                        "graphs/kappa_plot.png",
                        "graphs/loss_plot.png",
                        "graphs/lr_plot.png"
                    ],
                    columns=2,
                    show_label=False,
                    elem_id="graphs-gallery"
                )
                gr.Markdown("""
---

### 📈 DR Grading Scale (ICDR Classification)

| Grade | Name | Expected Lesions | Action |
|-------|------|-----------------|--------|
| 🟢 0 | No DR | None | 12-month routine review |
| 🟡 1 | Mild DR | Microaneurysms only | 6-month follow-up |
| 🟠 2 | Moderate DR | Microaneurysms, exudates, oedema | Referral within 3 months |
| 🔴 3 | Severe DR | >20 haemorrhages/quadrant, IRMA | Urgent referral 1 month |
| 🆘 4 | Proliferative DR | Neovascularisation, vitreous haemorrhage | Emergency referral |

---


                """)

    return demo
