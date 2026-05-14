import sys
import os
import requests

sys.path.insert(0, os.path.dirname(__file__))

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

# ─────────────────────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    allow_origins=["*"],
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
            return {
                "summary": "Gemini API key not configured."
            }

        var_data = get_var()

        portfolio_data = get_portfolio_breakdown()

        fi_data = get_feature_importance()

        perf_data = get_model_performance()

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

        prompt = f"""
Generate a professional executive summary for a credit risk dashboard.

Portfolio:
- Total Borrowers: {portfolio_data['total']}
- Low Risk: {low_pct}%
- Medium Risk: {med_pct}%
- High Risk: {high_pct}%

Risk Metrics:
- Expected Loss: ${var_data['expectedLoss']}
- VaR95: ${var_data['var95']}
- VaR99: ${var_data['var99']}

Top Risk Drivers:
{top_features}

Model:
- Best Model: {perf_data['bestModel']}
- AUC: {perf_data['bestAuc']}

Write 3 concise professional paragraphs.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        data = response.json()

        summary = data["candidates"][0]["content"]["parts"][0]["text"]

        return {
            "summary": summary
        }

    except Exception as e:

        print("SUMMARY ERROR:", str(e))

        return {
            "summary": f"Summary generation failed: {str(e)}"
        }
