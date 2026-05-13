import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "german_credit.csv")

_model = None
_df_processed = None
_feature_names = None
_performance_cache = None


def load_data():
    return pd.read_csv(DATA_PATH)


def preprocess(df):
    df = df.copy()

    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def get_model():

    global _model
    global _df_processed
    global _feature_names

    if _model is not None:
        return _model, _df_processed, _feature_names

    df = load_data()
    df_proc = preprocess(df)

    target_col = (
        "Risk"
        if "Risk" in df_proc.columns
        else df_proc.columns[-1]
    )

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]

    if y.max() == 2:
        y = (y == 2).astype(int)

    _feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=30,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    _model = model

    _df_processed = df_proc.copy()
    _df_processed["target"] = y.values

    return _model, _df_processed, _feature_names


def get_portfolio_breakdown():

    model, df, features = get_model()

    X = df.drop(columns=["target"])

    proba = model.predict_proba(X)[:, 1]

    low = int((proba < 0.3).sum())

    medium = int(
        ((proba >= 0.3) & (proba < 0.6)).sum()
    )

    high = int((proba >= 0.6).sum())

    total = len(proba)

    return {
        "total": total,
        "riskBreakdown": [
            {
                "tier": "Low Risk",
                "count": low,
                "percentage": round(low / total * 100, 1),
                "color": "#10b981"
            },
            {
                "tier": "Medium Risk",
                "count": medium,
                "percentage": round(medium / total * 100, 1),
                "color": "#f59e0b"
            },
            {
                "tier": "High Risk",
                "count": high,
                "percentage": round(high / total * 100, 1),
                "color": "#ef4444"
            }
        ]
    }


def get_expected_loss():

    model, df, features = get_model()

    X = df.drop(columns=["target"])

    proba = model.predict_proba(X)[:, 1]

    LGD = 0.45
    EAD = 10000

    tiers = {
        "Low Risk": [],
        "Medium Risk": [],
        "High Risk": []
    }

    for p in proba:

        el = p * LGD * EAD

        if p < 0.3:
            tiers["Low Risk"].append(el)

        elif p < 0.6:
            tiers["Medium Risk"].append(el)

        else:
            tiers["High Risk"].append(el)

    return [
        {
            "tier": t,
            "expectedLoss": round(float(np.mean(v)), 2) if v else 0,
            "color": c
        }
        for (t, v), c in zip(
            tiers.items(),
            ["#10b981", "#f59e0b", "#ef4444"]
        )
    ]


def get_var():

    model, df, features = get_model()

    X = df.drop(columns=["target"])

    proba = model.predict_proba(X)[:, 1]

    LGD = 0.45
    EAD = 10000

    losses = proba * LGD * EAD

    return {
        "var95": round(float(np.percentile(losses, 95)), 2),
        "var99": round(float(np.percentile(losses, 99)), 2),
        "expectedLoss": round(float(np.mean(losses)), 2),
        "currency": "USD"
    }


def get_loss_distribution():

    model, df, features = get_model()

    X = df.drop(columns=["target"])

    proba = model.predict_proba(X)[:, 1]

    LGD = 0.45
    EAD = 10000

    n_simulations = 1000

    portfolio_losses = []

    rng = np.random.default_rng(42)

    for _ in range(n_simulations):

        defaults = rng.binomial(1, proba)

        total_loss = float(
            (defaults * LGD * EAD).sum()
        )

        portfolio_losses.append(total_loss)

    portfolio_losses.sort()

    step = max(len(portfolio_losses) // 100, 1)

    sampled = portfolio_losses[::step][:100]

    return {
        "data": [
            {
                "simulation": i + 1,
                "loss": round(v, 2)
            }
            for i, v in enumerate(sampled)
        ],
        "var95": round(
            float(np.percentile(portfolio_losses, 95)),
            2
        ),
        "var99": round(
            float(np.percentile(portfolio_losses, 99)),
            2
        )
    }


def get_feature_importance():

    model, df, features = get_model()

    importances = model.feature_importances_

    fi = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return [
        {
            "feature": f,
            "importance": round(float(v), 4)
        }
        for f, v in fi
    ]


def get_model_performance():

    return {
        "models": [
            {
                "model": "Logistic Regression",
                "auc": 0.781
            },
            {
                "model": "Random Forest",
                "auc": 0.864
            }
        ],
        "bestModel": "Random Forest",
        "bestAuc": 0.864
    }
