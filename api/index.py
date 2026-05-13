import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import (
    get_portfolio_breakdown,
    get_expected_loss,
    get_var,
    get_loss_distribution,
    get_feature_importance,
    get_model_performance,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Credit Risk Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-credit-risk-intelligence.vercel.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/portfolio")
def portfolio():
    return get_portfolio_breakdown()

@app.get("/api/expected-loss")
def expected_loss():
    return get_expected_loss()

@app.get("/api/var")
def var():
    return get_var()

@app.get("/api/loss-distribution")
def loss_distribution():
    return get_loss_distribution()

@app.get("/api/feature-importance")
def feature_importance():
    return get_feature_importance()

@app.get("/api/performance")
def model_performance():
    return get_model_performance()

@app.get("/api/summary")
def executive_summary():
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

        var_data       = get_var()
        portfolio_data = get_portfolio_breakdown()
        fi_data        = get_feature_importance()
        perf_data      = get_model_performance()

        top_features = ", ".join([f["feature"] for f in fi_data[:3]])
        breakdown    = portfolio_data["riskBreakdown"]
        low_pct      = next(t["percentage"] for t in breakdown if t["tier"] == "Low Risk")
        med_pct      = next(t["percentage"] for t in breakdown if t["tier"] == "Medium Risk")
        high_pct     = next(t["percentage"] for t in breakdown if t["tier"] == "High Risk")

        # Concise prompt for faster Gemini response
        prompt = f"""You are a senior credit risk analyst. Write a concise executive summary (3 short paragraphs, plain text, no bullets) for a board audience:

Portfolio: {portfolio_data['total']} borrowers — {low_pct}% Low, {med_pct}% Medium, {high_pct}% High Risk
Expected Loss: ${var_data['expectedLoss']} per borrower
VaR 95%: ${var_data['var95']} | VaR 99%: ${var_data['var99']}
Top Risk Drivers: {top_features}
Model AUC: {perf_data['bestAuc']} (Random Forest)

Cover: portfolio health, risk concentration, VaR interpretation, 1 recommendation."""

        model = GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 300}
        )
        return {"summary": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
