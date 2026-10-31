"""Core analysis functions: used by the notebook and the Streamlit app."""
import numpy as np
import pandas as pd

DATA_PATH = "European_Bank.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def validate(df):
    """Step 1: data ingestion & validation report."""
    return {
        "rows": len(df), "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["CustomerId"].duplicated().sum()),
        "binary_cols_valid": all(df[c].isin([0, 1]).all() for c in ["HasCrCard", "IsActiveMember", "Exited"]),
        "churn_rate": round(df["Exited"].mean() * 100, 2),
    }

def prepare(df):
    """Add derived columns used across the project."""
    df = df.copy()
    nz_median = df.loc[df["Balance"] > 0, "Balance"].median()
    df["HighBalance"] = df["Balance"] >= nz_median            # top half of non-zero balances
    df["Engagement_Profile"] = np.select(
        [
            (df.IsActiveMember == 0) & df.HighBalance,
            (df.IsActiveMember == 1) & (df.NumOfProducts >= 2),
            (df.IsActiveMember == 1) & (df.NumOfProducts == 1),
        ],
        ["Inactive high-balance", "Active engaged", "Active but low-product"],
        default="Inactive disengaged",
    )
    # Relationship Strength Index (0-100): activity 50%, product depth 30%, tenure 20%
    prod_score = df["NumOfProducts"].clip(upper=2) / 2      # 3-4 products do not add strength (see EDA)
    df["RSI"] = 100 * (0.5 * df.IsActiveMember + 0.3 * prod_score + 0.2 * df.Tenure / 10)
    df["RSI_Tier"] = pd.cut(df["RSI"], [-1, 25, 50, 75, 101], labels=["Weak", "Moderate", "Strong", "Very strong"])
    df["BalanceBand"] = pd.cut(df["Balance"], [-1, 0, 80000, 120000, 1e9],
                               labels=["Zero", "Low (<80k)", "Mid (80-120k)", "High (>120k)"])
    df["Salary_Balance_Mismatch"] = (df.EstimatedSalary >= df.EstimatedSalary.median()) & (df.Balance == 0)
    df["AtRiskPremium"] = df.HighBalance & (df.IsActiveMember == 0) & (df.NumOfProducts == 1)
    return df

def churn_by(df, col):
    g = df.groupby(col, observed=True)["Exited"].agg(customers="count", churned="sum", churn_rate="mean")
    g["churn_rate"] = (g["churn_rate"] * 100).round(2)
    return g.reset_index()

def kpis(df):
    act = df[df.IsActiveMember == 1].Exited.mean()
    inact = df[df.IsActiveMember == 0].Exited.mean()
    p1 = df[df.NumOfProducts == 1].Exited.mean()
    p2 = df[df.NumOfProducts == 2].Exited.mean()
    hb = df[df.HighBalance]
    hb_inactive = hb[hb.IsActiveMember == 0]
    card = df[df.HasCrCard == 1].Exited.mean()
    nocard = df[df.HasCrCard == 0].Exited.mean()
    return {
        "Engagement Retention Ratio": round(inact / act, 2),
        "Product Depth Index (2 vs 1 product, retention pp gap)": round((p1 - p2) * 100, 1),
        "High-Balance Disengagement Rate (%)": round(len(hb_inactive) / len(hb) * 100, 1),
        "High-Balance Disengaged Churn Rate (%)": round(hb_inactive.Exited.mean() * 100, 1),
        "Credit Card Stickiness Score (pp gap, no-card minus card)": round((nocard - card) * 100, 2),
        "Relationship Strength Index (avg, retained)": round(df[df.Exited == 0].RSI.mean(), 1),
        "Relationship Strength Index (avg, churned)": round(df[df.Exited == 1].RSI.mean(), 1),
    }
