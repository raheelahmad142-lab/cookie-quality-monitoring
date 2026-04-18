# 🍪 Cookie Quality Monitoring System

A real-time AI-powered visual quality monitoring system that detects cookie defects and scores tray quality automatically.

## 🚀 Live Demo
👉 https://huggingface.co/spaces/raheelahmad142/cookie-quality-monitoring

## 📌 Project Overview
This project simulates a real-world production quality monitoring system for a cookie manufacturer. A computer vision model inspects each tray, classifies every cookie, and outputs a quality score with detailed issue reporting.

Inspired by a real Computer Vision Engineer job posting requiring defect detection, visual inspection, and product grading systems.

## ✅ What It Does
- Detects each cookie as **Good**, **Broken**, or **Burned**
- Analyzes **color and brightness** per cookie using HSV color space
- Computes a **tray quality score (0-100)** based on defects and count
- Flags issues automatically with detailed report
- Runs inference in **~11ms per image**

## 🧠 How Scoring Works
| Issue | Penalty |
|---|---|
| Wrong cookie count | -10 per missing/extra |
| Burned cookie | -15 per cookie |
| Broken cookie | -10 per cookie |

Score >= 80 → ✅ PASS
Score 50-79 → ⚠️ WARNING
Score < 50 → ❌ FAIL

## 🛠️ Tech Stack
- **YOLO11** — object detection
- **OpenCV** — color and brightness analysis
- **Gradio** — web demo interface
- **Python** — core logic
- **Google Colab** — training environment
- **Hugging Face Spaces** — deployment

## 📊 Model Performance
- **mAP50:** 99.5%
- **Precision:** 99.9%
- **Recall:** 100%
- **Dataset:** 1076 images (Good / Broken / Burned)

## 🗂️ Project Structure
cookie-quality-monitoring/
├── app.py              # Gradio demo + full pipeline
├── best.pt             # Trained YOLO11 model weights
├── requirements.txt    # Dependencies
└── README.md
