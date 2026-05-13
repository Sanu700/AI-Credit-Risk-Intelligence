import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel

from model import (
    get_portfolio_breakdown,
    get_expected_loss,
    get_var,
    get_loss_distribution,
    get_feature_importance,
    get_model_performance,
)

# ─────────────────────────────────────────────────────────────
# Gemini Setup
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Intelligence API",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-credit-risk-intelligence.vercel.app",
        "https://*.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Root Route
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Credit Risk Intelligence API Running"
    }

# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }

# ─────────────────────────────────────────────────────────────
# Portfolio Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def portfolio():
    try:
        return get_portfolio_breakdown()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio Error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────
# Expected Loss Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/expected-loss")
def expected_loss():
    try:
        return get_expected_loss()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Expected Loss Error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────
# VaR Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/var")
def var():
    try:
        return get_var()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"VaR Error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────
# Loss Distribution Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/loss-distribution")
def loss_distribution():
    try:
        return get_loss_distribution()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Loss Distribution Error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────
# Feature Importance Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/feature-importance")
def feature_importance():
    try:
        return get_feature_importance()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature Importance Error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────
# Performance Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/performance")
def model_performance():
    try:
        return get_model_performance()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Performance Error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────
# Executive Summary Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/summary")
def executive_summary():

    try:

        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY not set"
            )

        # Fetch analytics data
        var_data = get_var()
        portfolio_data = get_portfolio_breakdown()
        fi_data = get_feature_importance()
        perf_data = get_model_performance()

        # Extract feature names safely
        top_features = ", ".join(
            [f["feature"] for f in fi_data[:3]]
        )

        breakdown = portfolio_data["riskBreakdown"]

        low_pct = next(
            t["percentage"]
            for t in breakdown
            if t["tier"] == "Low Risk"
        )

        med_pct = next(
            t["percentage"]
            for t in breakdown
            if t["tier"] == "Medium Risk"
        )

        high_pct = next(
            t["percentage"]
            for t in breakdown
            if t["tier"] == "High Risk"
        )

        # Gemini Prompt
        prompt = f"""
You are a senior credit risk analyst.

Write a concise executive summary in 3 short paragraphs for a banking board audience.

Portfolio Summary:
- Total Borrowers: {portfolio_data['total']}
- Low Risk: {low_pct}%
- Medium Risk: {med_pct}%
- High Risk: {high_pct}%

Risk Metrics:
- Expected Loss: ${var_data['expectedLoss']} per borrower
- VaR 95%: ${var_data['var95']}
- VaR 99%: ${var_data['var99']}

Top Risk Drivers:
{top_features}

Model Performance:
- Random Forest AUC: {perf_data['bestAuc']}

Discuss:
1. Portfolio health
2. Risk concentration
3. VaR interpretation
4. One recommendation

Keep response professional and concise.
"""

        # Create Gemini Model
        model = GenerativeModel("gemini-1.5-flash")

        # Generate Response
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 300,
                "temperature": 0.4,
            }
        )

        summary_text = response.text.strip()

        return {
            "summary": summary_text
        }

    except HTTPException as http_error:
        raise http_error

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summary Generation Error: {str(e)}"
        )
