# 🏦 AI Credit Risk & Portfolio Intelligence System

> **End-to-end ML pipeline for credit risk assessment + interactive business dashboard.** Predicts probability of default with AUC 0.80, visualises portfolio risk in real time, and generates board-level executive summaries via Gemini LLM.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?style=flat-square&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi)
![Gemini](https://img.shields.io/badge/Gemini-LLM-8e44ad?style=flat-square)
![Koyeb](https://img.shields.io/badge/Backend-Koyeb-121212?style=flat-square)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-black?style=flat-square&logo=vercel)

---

## 🌐 Live Demo

🚀 **[ai-credit-risk-intelligence.vercel.app](https://ai-credit-risk-intelligence.vercel.app)**

---

## 🎯 What it does

An end-to-end credit risk intelligence system that trains ML models, analyses a loan portfolio, and presents insights through an interactive dashboard.

1. **Trains ML models** to predict Probability of Default (PD) on the German Credit dataset
2. **Segments the portfolio** into low / medium / high risk tiers using feature importance analysis
3. **Runs 1,000-scenario Monte Carlo simulation** to compute Expected Loss and Value-at-Risk
4. **Auto-generates executive summaries** using Gemini LLM for board-level reporting
5. **Visualises everything** in a clean React dashboard with live charts

---

## 📊 Model Results

| Model | AUC Score |
|-------|-----------|
| Logistic Regression | 0.76 |
| **Random Forest** | **0.80** |

---

## ✨ Features

- 📊 **Portfolio Risk Breakdown** — pie chart of low / medium / high risk borrowers
- 💰 **Expected Loss by Tier** — average EL per risk segment using PD × LGD × EAD
- 📉 **Monte Carlo Loss Distribution** — 1,000 simulated scenarios, full loss curve
- 🎯 **Value-at-Risk Cards** — 95% and 99% VaR thresholds at a glance
- 🔍 **Feature Importance** — top 10 risk drivers from Random Forest
- 🤖 **AI Executive Summary** — Gemini LLM generates a board-ready risk report on demand
- 📱 **Fully Responsive** — works on desktop and mobile

---

## 🚀 Deployment

### Backend → Koyeb (free)

1. Go to [koyeb.com](https://koyeb.com) → sign up free
2. New App → GitHub → select this repo
3. Set **root directory** to `/` (uses Procfile at root)
4. Add environment variable: `GEMINI_API_KEY=your_key`
5. Deploy → copy your live URL e.g. `https://your-app.koyeb.app`

### Frontend → Vercel (free)

1. Go to [vercel.com](https://vercel.com) → import this repo
2. Set **root directory** to `frontend/`
3. Add environment variable: `REACT_APP_API_URL=https://your-app.koyeb.app`
4. Deploy

---

## 💻 Local Development

### Backend
```bash
pip install -r requirements.txt
cd api
uvicorn index:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
echo "REACT_APP_API_URL=http://localhost:8000" > .env
npm start
```

---

## 🏗️ Project Structure

```
AI-Credit-Risk-Intelligence/
├── api/
│   ├── index.py              # FastAPI server
│   ├── model.py              # ML pipeline (Random Forest, Monte Carlo, VaR)
│   └── german_credit.csv     # Dataset
├── frontend/
│   └── src/
│       ├── App.jsx           # Dashboard UI
│       └── App.css           # Styles
├── Procfile                  # Koyeb start command
├── requirements.txt          # Python dependencies
└── vercel.json               # Vercel frontend config
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Pipeline | Python, scikit-learn, pandas, NumPy |
| Simulation | Monte Carlo, SciPy, NumPy |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18, Recharts |
| LLM | Google Gemini API |
| Backend Host | Koyeb |
| Frontend Host | Vercel |

---

## 📚 Key Concepts

| Term | Definition |
|------|-----------|
| **PD** | Probability of Default — likelihood a borrower won't repay |
| **LGD** | Loss Given Default — % of exposure lost if default occurs |
| **EAD** | Exposure at Default — total outstanding value at risk |
| **VaR** | Value-at-Risk — maximum expected loss at a confidence level |
| **Monte Carlo** | Running 1,000 random scenarios to model portfolio loss uncertainty |

---

## 📄 License

MIT — feel free to use and build on this project.
