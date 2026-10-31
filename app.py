import pandas as pd
import plotly.express as px
import streamlit as st
import analysis as a

st.set_page_config(page_title="Retention Analytics", page_icon="🏦", layout="wide")

@st.cache_data
def get_data():
    return a.prepare(a.load_data())

df_all = get_data()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
eng = st.sidebar.multiselect("Engagement status", ["Active", "Inactive"], default=["Active", "Inactive"])
prod = st.sidebar.slider("Number of products", 1, 4, (1, 4))
bal = st.sidebar.slider("Balance range", 0, int(df_all.Balance.max()) + 1, (0, int(df_all.Balance.max()) + 1), step=5000)
sal = st.sidebar.slider("Estimated salary range", 0, int(df_all.EstimatedSalary.max()) + 1, (0, int(df_all.EstimatedSalary.max()) + 1), step=5000)
geo = st.sidebar.multiselect("Geography", sorted(df_all.Geography.unique()), default=sorted(df_all.Geography.unique()))

df = df_all[
    df_all.IsActiveMember.map({1: "Active", 0: "Inactive"}).isin(eng)
    & df_all.NumOfProducts.between(*prod)
    & df_all.Balance.between(*bal)
    & df_all.EstimatedSalary.between(*sal)
    & df_all.Geography.isin(geo)
]

st.title("🏦 Customer Engagement & Product Utilization Analytics")
st.caption("Retention strategy dashboard · European bank customer data")
if df.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ---------- KPI cards ----------
k = a.kpis(df) if df.IsActiveMember.nunique() == 2 and df.NumOfProducts.nunique() >= 2 and df.HasCrCard.nunique() == 2 else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Churn rate", f"{df.Exited.mean()*100:.1f}%")
c3.metric("Engagement Retention Ratio", k["Engagement Retention Ratio"] if k else "n/a",
          help="Churn of inactive ÷ churn of active customers")
c4.metric("At-risk premium customers", f"{int(df.AtRiskPremium.sum()):,}",
          help="High balance + inactive + single product")

tab1, tab2, tab3, tab4 = st.tabs(["Engagement vs churn", "Product utilization", "High-value disengaged", "Retention strength"])

with tab1:
    l, r = st.columns(2)
    d = a.churn_by(df.assign(Status=df.IsActiveMember.map({1: "Active", 0: "Inactive"})), "Status")
    l.plotly_chart(px.bar(d, x="Status", y="churn_rate", text="churn_rate", title="Churn rate by activity status",
                          labels={"churn_rate": "Churn %"}), width='stretch')
    d = a.churn_by(df, "Engagement_Profile").sort_values("churn_rate")
    r.plotly_chart(px.bar(d, x="churn_rate", y="Engagement_Profile", orientation="h", text="churn_rate",
                          title="Churn rate by engagement profile", labels={"churn_rate": "Churn %"}), width='stretch')
    st.dataframe(d, width='stretch', hide_index=True)

with tab2:
    l, r = st.columns(2)
    d = a.churn_by(df, "NumOfProducts")
    l.plotly_chart(px.bar(d, x="NumOfProducts", y="churn_rate", text="churn_rate",
                          title="Churn rate by number of products", labels={"churn_rate": "Churn %"}), width='stretch')
    r.plotly_chart(px.bar(d, x="NumOfProducts", y="customers", title="Customers by number of products"), width='stretch')
    heat = df.pivot_table(index="IsActiveMember", columns="NumOfProducts", values="Exited", aggfunc="mean").mul(100).round(1)
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                              title="Churn % — activity × product count",
                              labels=dict(color="Churn %")), width='stretch')
    cc = a.churn_by(df.assign(CreditCard=df.HasCrCard.map({1: "Has card", 0: "No card"})), "CreditCard")
    st.subheader("Credit card ownership")
    st.dataframe(cc, width='stretch', hide_index=True)

with tab3:
    st.markdown("**High-value disengaged customers**: balance in the top half of funded accounts, inactive, single product.")
    risk = df[df.AtRiskPremium]
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers flagged", f"{len(risk):,}")
    m2.metric("Churn rate of this group", f"{risk.Exited.mean()*100:.1f}%" if len(risk) else "n/a")
    m3.metric("Balance at stake (still active customers)", f"€{risk[risk.Exited == 0].Balance.sum():,.0f}")
    d = a.churn_by(df, "BalanceBand")
    st.plotly_chart(px.bar(d, x="BalanceBand", y="churn_rate", text="churn_rate", title="Churn by balance band",
                           labels={"churn_rate": "Churn %"}), width='stretch')
    mm = df[df.Salary_Balance_Mismatch]
    st.caption(f"Salary–balance mismatch (above-median salary but zero balance): {len(mm):,} customers, "
               f"churn {mm.Exited.mean()*100:.1f}%" if len(mm) else "No salary–balance mismatches in selection.")
    st.subheader("Retention watch-list (not yet churned)")
    wl = risk[risk.Exited == 0][["CustomerId", "Surname", "Geography", "Age", "Tenure", "Balance", "EstimatedSalary", "RSI"]]
    st.dataframe(wl.sort_values("Balance", ascending=False), width='stretch', hide_index=True)
    st.download_button("Download watch-list (CSV)", wl.to_csv(index=False), "at_risk_premium.csv", "text/csv")

with tab4:
    st.markdown("**Relationship Strength Index (RSI)** = 50% activity + 30% product depth (capped at 2 products) + 20% tenure, scaled 0–100.")
    l, r = st.columns(2)
    d = a.churn_by(df, "RSI_Tier")
    l.plotly_chart(px.bar(d, x="RSI_Tier", y="churn_rate", text="churn_rate", title="Churn by RSI tier",
                          labels={"churn_rate": "Churn %"}), width='stretch')
    r.plotly_chart(px.histogram(df.assign(Outcome=df.Exited.map({1: "Churned", 0: "Retained"})), x="RSI", color="Outcome",
                                barmode="overlay", opacity=0.6, title="RSI distribution"), width='stretch')
    t = a.churn_by(df, "Tenure")
    st.plotly_chart(px.line(t, x="Tenure", y="churn_rate", markers=True, title="Churn by tenure (years)",
                            labels={"churn_rate": "Churn %"}), width='stretch')
    if k:
        st.subheader("KPI summary")
        st.dataframe(pd.DataFrame(k.items(), columns=["KPI", "Value"]), width='stretch', hide_index=True)
