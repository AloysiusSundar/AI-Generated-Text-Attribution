# AI Text Attribution Demo

A lightweight NLP system for **AI text detection and attribution**.  
The tool determines whether a given text is **human-written or AI-generated**, and if AI-generated, estimates the **most likely source language model**, while explicitly handling uncertainty.

The system is designed to be **prompt-safe**, **interpretable**, and **conservative** in its claims.

---

## 🔗 Live Demo

👉 **Hugging Face Space**  
https://huggingface.co/spaces/AloysiusJoy/TextDetector

---

## 🧠 Method Overview

This project follows a **two-stage classification pipeline**:

### 1️⃣ Human vs AI Detection  
A binary classifier determines whether the input text is human-written or AI-generated using:
- **TF-IDF n-gram features (1–3 grams)**
- **Linear Support Vector Machine (SVM)**

This stage acts as a safety gate to prevent forced attribution on human text.

---

### 2️⃣ Model Attribution
If the text is classified as AI-generated, a multi-class classifier estimates which model’s **writing style** it most closely resembles.

- Uses the same **TF-IDF + Linear SVM** approach  
- Applies **confidence-gap thresholds** to explicitly mark ambiguous cases as *uncertain* rather than overclaiming

---

## 🧪 Output Categories

The system returns one of the following outcomes:

- **✅ Likely Human-Written**
- **🤖 Likely AI-Generated (with model attribution)**
- **⚠️ AI-Generated (Uncertain Attribution)**
- **❓ Uncertain** 

All results are **probabilistic** and represent *stylistic similarity*, not definitive authorship.

---
