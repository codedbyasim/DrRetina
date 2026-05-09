#!/usr/bin/env python3
"""
DrRetina — LangChain Agentic Layer  (SRS §2.2, FR-05, FR-06)

Uses LangChain with Qwen3-8B via Featherless AI (OpenAI-compatible endpoint).
The agent has access to clinical tools to:
  - Explain DR grades
  - Provide treatment recommendations
  - Answer follow-up clinical questions
  - Generate structured diagnostic reports
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# ─────────────────────────────────────────────────────────────────
# FEATHERLESS AI — Qwen via OpenAI-compatible endpoint
# ─────────────────────────────────────────────────────────────────
_DEFAULT_KEY = "rc_c871260215042ae1dc87e28ef5672b1658b30652445af3837d0211b17edee2b8"
FEATHERLESS_KEY = os.environ.get("FEATHERLESS_API_KEY", _DEFAULT_KEY)

def get_llm(temperature: float = 0.3, max_tokens: int = 800, stop_tokens: list = None):
    """Returns LangChain ChatOpenAI configured for Qwen3-8B via Featherless."""
    if not FEATHERLESS_KEY:
        return None
    
    kwargs = {}
    if stop_tokens:
        kwargs["stop"] = stop_tokens
        
    return ChatOpenAI(
        model="Qwen/Qwen3-8B",
        openai_api_key=FEATHERLESS_KEY,
        openai_api_base="https://api.featherless.ai/v1",
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=kwargs
    )


# ─────────────────────────────────────────────────────────────────
# DR KNOWLEDGE BASE (tools use this)
# ─────────────────────────────────────────────────────────────────
DR_GRADES = {
    0: {
        "name": "No Diabetic Retinopathy",
        "description": "No visible signs of diabetic retinopathy detected.",
        "lesions": "None expected.",
        "urgency": "Routine follow-up in 12 months.",
        "treatment": "No retinal treatment needed. Maintain HbA1c <7%, BP <130/80, regular exercise.",
        "lifestyle": "Control blood sugar, blood pressure, and cholesterol. Annual eye screening.",
        "severity": "None",
    },
    1: {
        "name": "Mild Diabetic Retinopathy",
        "description": "Early stage with microaneurysms only — small bulges in blood vessels.",
        "lesions": "Microaneurysms (tiny red dots on the retina).",
        "urgency": "Follow-up in 6 months.",
        "treatment": "Optimise HbA1c <7%, BP <130/80. No direct retinal treatment yet.",
        "lifestyle": "Strict diabetes management, smoking cessation, dietary changes.",
        "severity": "Mild",
    },
    2: {
        "name": "Moderate Diabetic Retinopathy",
        "description": "More extensive retinal changes with multiple lesion types.",
        "lesions": "Microaneurysms, hard exudates (lipid deposits), retinal haemorrhages, macular oedema possible.",
        "urgency": "Ophthalmology referral within 3 months.",
        "treatment": "Focal laser photocoagulation for macular oedema; anti-VEGF injections if oedema present.",
        "lifestyle": "Urgent diabetes optimisation; blood pressure control critical.",
        "severity": "Moderate",
    },
    3: {
        "name": "Severe Diabetic Retinopathy",
        "description": "Advanced non-proliferative DR with significant retinal ischaemia.",
        "lesions": "More than 20 intraretinal haemorrhages per quadrant, venous beading, IRMA (intraretinal microvascular abnormalities).",
        "urgency": "Urgent ophthalmology referral within 1 month.",
        "treatment": "Pan-retinal photocoagulation (PRP) laser; anti-VEGF injections; close monitoring.",
        "lifestyle": "Immediate hospitalisation risk if untreated; strict metabolic control essential.",
        "severity": "Severe",
    },
    4: {
        "name": "Proliferative Diabetic Retinopathy",
        "description": "Most advanced stage with new abnormal blood vessel growth (neovascularisation).",
        "lesions": "Neovascularisation of disc/retina, vitreous haemorrhage, tractional retinal detachment risk.",
        "urgency": "Emergency referral — immediate risk of blindness.",
        "treatment": "Anti-VEGF injections (bevacizumab/ranibizumab); PRP laser; vitreoretinal surgery if haemorrhage.",
        "lifestyle": "Emergency condition — do not delay. Same-day or next-day ophthalmologist visit required.",
        "severity": "Critical/Emergency",
    },
}


# ─────────────────────────────────────────────────────────────────
# LANGCHAIN TOOLS  (SRS: Agent Layer Tools)
# ─────────────────────────────────────────────────────────────────
@tool
def get_grade_info(grade: int) -> str:
    """Get detailed clinical information about a specific DR grade (0-4).
    Use this when the user asks what their grade means."""
    if grade not in DR_GRADES:
        return "Invalid grade. DR grades range from 0 (No DR) to 4 (Proliferative DR)."
    g = DR_GRADES[grade]
    return (
        f"**Grade {grade} — {g['name']}**\n"
        f"Description: {g['description']}\n"
        f"Severity: {g['severity']}\n"
        f"Expected Lesions: {g['lesions']}\n"
        f"Urgency: {g['urgency']}\n"
        f"Treatment: {g['treatment']}"
    )


@tool
def get_treatment_options(grade: int) -> str:
    """Get treatment options and recommendations for a specific DR grade (0-4).
    Use this when the user asks about treatment, what to do next, or how to manage their condition."""
    if grade not in DR_GRADES:
        return "Invalid grade. Please specify a grade between 0 and 4."
    g = DR_GRADES[grade]
    return (
        f"**Treatment for Grade {grade} ({g['name']}):**\n"
        f"• Medical Treatment: {g['treatment']}\n"
        f"• Urgency: {g['urgency']}\n"
        f"• Lifestyle: {g['lifestyle']}"
    )


@tool
def get_urgency_level(grade: int) -> str:
    """Get the urgency level and recommended follow-up timeline for a DR grade (0-4).
    Use this when asked how serious/urgent the condition is."""
    if grade not in DR_GRADES:
        return "Invalid grade."
    g = DR_GRADES[grade]
    return (
        f"**Urgency for Grade {grade} ({g['name']}):**\n"
        f"Severity: {g['severity']}\n"
        f"Action Required: {g['urgency']}\n"
        f"Details: {g['description']}"
    )


@tool
def get_lifestyle_advice(grade: int) -> str:
    """Get lifestyle and diabetes management advice for a specific DR grade (0-4).
    Use this when asked about lifestyle changes, diet, exercise, or diabetes management."""
    if grade not in DR_GRADES:
        return "Invalid grade."
    g = DR_GRADES[grade]
    return (
        f"**Lifestyle Advice for Grade {grade} ({g['name']}):**\n"
        f"• {g['lifestyle']}\n"
        f"General recommendations:\n"
        f"• Keep HbA1c below 7%\n"
        f"• Maintain blood pressure below 130/80 mmHg\n"
        f"• Quit smoking immediately\n"
        f"• Regular aerobic exercise (30 min/day)\n"
        f"• Low-glycaemic diet\n"
        f"• Annual dilated eye examination"
    )


@tool
def compare_grades(grade_a: int, grade_b: int) -> str:
    """Compare two DR grades to explain the difference in severity.
    Use when user asks about progression or wants to understand how serious their grade is relative to others."""
    if grade_a not in DR_GRADES or grade_b not in DR_GRADES:
        return "Invalid grades. Please specify grades between 0 and 4."
    a = DR_GRADES[grade_a]
    b = DR_GRADES[grade_b]
    return (
        f"**Comparison: Grade {grade_a} vs Grade {grade_b}**\n"
        f"Grade {grade_a} ({a['name']}): {a['severity']} severity — {a['urgency']}\n"
        f"Grade {grade_b} ({b['name']}): {b['severity']} severity — {b['urgency']}"
    )


# ─────────────────────────────────────────────────────────────────
# F4: REFERRAL LETTER TOOLS
# ─────────────────────────────────────────────────────────────────
@tool
def analyze_severity(grade: int, confidence: float) -> str:
    """Analyze the clinical severity of the findings based on grade and confidence."""
    if grade not in DR_GRADES:
        return "Invalid grade."
    g = DR_GRADES[grade]
    return f"Severity Analysis: Grade {grade} ({g['name']}). Confidence is {confidence:.1f}%. Expected lesions: {g['lesions']}. Risk level is {g['severity']}."

@tool
def recommend_treatment(grade: int) -> str:
    """Provide evidence-based treatment protocol for a given DR grade."""
    if grade not in DR_GRADES:
        return "Invalid grade."
    return f"Recommended Intervention: {DR_GRADES[grade]['treatment']}"

@tool
def calculate_urgency(grade: int, progression_rate: str = "Unknown") -> str:
    """Calculate the referral timeline and urgency based on grade."""
    if grade not in DR_GRADES:
        return "Invalid grade."
    return f"Suggested Timeline: {DR_GRADES[grade]['urgency']}. Progression: {progression_rate}."

@tool
def generate_referral_letter(patient_name: str, findings: str, urgency: str) -> str:
    """Generate a formatted clinical referral letter."""
    import datetime
    today = datetime.datetime.now().strftime("%B %d, %Y")
    return f"""Date: {today}
From: DrRetina Clinical AI System
To: Vitreoretinal Specialist

RE: Referral — {patient_name}

I am referring this patient for evaluation of Diabetic Retinopathy.

AI Analysis Findings:
{findings}

{urgency}

Generated by DrRetina v1.0 | AMD MI300X | Kappa: 0.9097
"""

TOOLS = [get_grade_info, get_treatment_options, get_urgency_level,
         get_lifestyle_advice, compare_grades,
         analyze_severity, recommend_treatment, calculate_urgency, generate_referral_letter]


# ─────────────────────────────────────────────────────────────────
# AGENT BUILDER LOGIC REPLACED BY create_agent IN agent_qa
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# AGENT REPORT GENERATION  (FR-05)
# ─────────────────────────────────────────────────────────────────
def agent_generate_report(grade: int, probs, language: str = "English") -> Optional[str]:
    """Generate diagnostic report using LangChain + Qwen."""
    llm = get_llm(temperature=0.3, max_tokens=1500, stop_tokens=["End of report", "AI Disclaimer", "©"])
    if not llm:
        return None
    try:
        grade_info = DR_GRADES[grade]
        prob_txt = " | ".join(
            f"Grade {i} ({DR_GRADES[i]['name'][:8]}): {p*100:.1f}%"
            for i, p in enumerate(probs)
        )
        messages = [
            SystemMessage(content=(
                f"You are an expert ophthalmologist generating a clinical report in {language}. "
                f"Patient has Grade {grade} DR ({grade_info['name']}). "
                f"Severity: {grade_info['severity']}. "
                f"Instructions: Be compassionate, professional, and clinically accurate. "
                f"IMPORTANT: Do not repeat any sections. Stop immediately after the AI disclaimer. "
                f"Use simple, non-technical terms if the language is not English."
            )),
            HumanMessage(content=(
                f"Generate a structured clinical diagnostic report for this DR screening result in {language}:\n\n"
                f"**Detected Grade:** {grade} — {grade_info['name']}\n"
                f"**Confidence:** {probs[grade]*100:.1f}%\n"
                f"**All Probabilities:** {prob_txt}\n\n"
                f"The report MUST include these sections ONLY (translated to {language}):\n"
                f"## 1. Diagnosis Summary\n"
                f"## 2. Severity Assessment\n"
                f"## 3. Expected Lesions\n"
                f"## 4. Treatment Options\n"
                f"## 5. Follow-up Timeline\n"
                f"## 6. Clinical Recommendation\n\n"
                f"Finish the report with: 'End of report.' followed by a brief AI disclaimer. "
                f"Do not write more than 400 words. Do not repeat sections."
            )),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"[Agent Report Error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# AGENT Q&A  (FR-06)
# ─────────────────────────────────────────────────────────────────
def agent_qa(question: str, grade: int, confidence: float, report: str, history: list = None) -> Optional[str]:
    """Answer clinical questions quickly without slow tool roundtrips."""
    llm = get_llm(temperature=0.6, max_tokens=1500)
    if not llm:
        return None
        
    g_info = DR_GRADES[grade]
    sys_msg = f"""You are DrRetina, a clinical AI assistant specializing in Diabetic Retinopathy (DR).

Patient's current condition:
- DR Grade: {grade} — {g_info['name']}
- Severity: {g_info['severity']}
- Expected Lesions: {g_info['lesions']}
- Urgency: {g_info['urgency']}
- Recommended Treatment: {g_info['treatment']}
- Lifestyle Advice: {g_info['lifestyle']}
- Confidence: {confidence:.1f}%

IMPORTANT INSTRUCTIONS:
1. Use the clinical context above to answer the user's questions accurately.
2. Be compassionate, clear, and professional.
3. Always recommend consulting a qualified ophthalmologist.
4. MULTILINGUAL SUPPORT: You MUST reply in the exact same language that the user asks the question in (e.g., if they ask in Urdu, reply in fluent Urdu; if Hindi, reply in Hindi)."""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        msg = [SystemMessage(content=sys_msg)]
        if history:
            for h in history:
                if h["role"] == "user": msg.append(HumanMessage(content=h["content"]))
                else: msg.append(AIMessage(content=h["content"]))
        msg.append(HumanMessage(content=question))
        return llm.invoke(msg).content
    except Exception as e:
        print(f"[Agent QA Error] {e}")
        return None
