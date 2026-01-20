import joblib
import json
import gradio as gr
import numpy as np
import os

# -----------------------------
# Load models
# -----------------------------
BASE_DIR = "saved_models"

human_ai_model = joblib.load(os.path.join(BASE_DIR, "human_ai_model.joblib"))
attrib_model = joblib.load(os.path.join(BASE_DIR, "attrib_model.joblib"))

# Load config (optional but recommended)
config_path = os.path.join(BASE_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        CONFIG = json.load(f)
else:
    CONFIG = {
        "min_words": 30,
        "human_confidence_threshold": 0.5,
        "attrib_confidence_gap": 0.15
    }

# -----------------------------
# Helper functions
# -----------------------------

def classify_with_confidence(text, model, top_k=3):
    scores = model.decision_function([text])[0]
    classes = model.classes_
    
    ranked = sorted(
        zip(classes, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]


def safe_attribution(text):
    ranked = classify_with_confidence(text, attrib_model)
    
    best_model, best_score = ranked[0]
    second_model, second_score = ranked[1]
    
    confidence_gap = best_score - second_score
    
    return {
        "predicted_model": best_model,
        "confidence_gap": round(confidence_gap, 3),
        "top_candidates": [
            (m, round(s, 3)) for m, s in ranked
        ]
    }


def human_ai_decision(text):
    score = human_ai_model.decision_function([text])[0]
    label = human_ai_model.predict([text])[0]
    
    return {
        "label": label,
        "confidence": round(abs(score), 3),
        "raw_score": round(score, 3)
    }


def full_classify(text):
    text = text.strip()
    word_count = len(text.split())
    
    # -------- Guardrails --------
    if word_count < CONFIG["min_words"]:
        return {
            "final_label": "uncertain",
            "reason": f"Text too short ({word_count} words)"
        }
    
    # -------- Stage 1: Human vs AI --------
    hvai = human_ai_decision(text)
    
    if hvai["label"] == "human" and hvai["confidence"] >= CONFIG["human_confidence_threshold"]:
        return {
            "final_label": "human",
            "details": hvai
        }
    
    # -------- Stage 2: Attribution --------
    attrib = safe_attribution(text)
    
    if attrib["predicted_model"] == "human_story":
        return {
            "final_label": "human",
            "details": attrib
        }
    
    if attrib["confidence_gap"] < CONFIG["attrib_confidence_gap"]:
        return {
            "final_label": "ai_uncertain",
            "details": attrib
        }
    
    return {
        "final_label": "ai",
        "predicted_model": attrib["predicted_model"],
        "details": attrib
    }

# -----------------------------
# Gradio Interface
# -----------------------------

# -----------------------------
# Gradio UI (Improved)
# -----------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AI Text Attribution Demo

        This tool analyzes a piece of text to determine:

        **1.** Whether it is likely **human-written or AI-generated**  
        **2.** If AI-generated, which **language model style it most closely resembles**

        Results are **probabilistic**. Some outputs are marked **uncertain** to avoid overclaiming.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=12,
                label="Input Text",
                placeholder="Paste text here (minimum ~30 words recommended)..."
            )
            analyze_btn = gr.Button("Analyze Text")

        with gr.Column(scale=1):
            headline = gr.Markdown("### Result will appear here")
            explanation = gr.Markdown("")

    gr.Markdown("### 🔍 Detailed Model Output")
    raw_output = gr.JSON(label="Raw Output")

    def ui_wrapper(text):
        result = full_classify(text)

        # ---------- Headline & explanation ----------
        if result["final_label"] == "human":
            headline_md = "## ✅ Likely Human-Written"
            explanation_md = (
                "The text shows high stylistic variability and does not strongly match "
                "known AI generation patterns."
            )

        elif result["final_label"] == "ai":
            model = result["predicted_model"]
            gap = result["details"]["confidence_gap"]

            headline_md = "## 🤖 Likely AI-Generated"
            explanation_md = (
                f"The text most closely resembles **{model}**.\n\n"
                f"Confidence gap: **{gap}** — higher values indicate stronger attribution confidence."
            )

        elif result["final_label"] == "ai_uncertain":
            headline_md = "## ⚠️ AI-Generated (Uncertain Attribution)"
            explanation_md = (
                "The text appears AI-generated, but its style is similar to multiple models.\n\n"
                "A confident attribution cannot be made."
            )

        else:
            headline_md = "## ❓ Uncertain"
            explanation_md = result.get(
                "reason",
                "The system could not make a reliable determination."
            )

        return headline_md, explanation_md, result

    analyze_btn.click(
        fn=ui_wrapper,
        inputs=text_input,
        outputs=[headline, explanation, raw_output]
    )


if __name__ == "__main__":
    demo.launch()

