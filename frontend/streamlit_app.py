"""
ForecastOps — Streamlit Dashboard

Tabs:
  1. Forecast — interactive forecast chart with confidence bands
  2. Model Registry — comparison table of all registered versions
  3. Drift Monitor — PSI heatmap per feature
  4. Retrain — trigger retraining manually
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ForecastOps",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ ForecastOps — Energy Demand Forecasting")
st.caption("End-to-end MLOps pipeline · Prophet vs LightGBM · MLflow Model Registry")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🏆 Model Registry", "🔍 Drift Monitor", "🔁 Retrain"])


# ─── Tab 1: Forecast ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Generate Forecast")
    hours = st.slider("Forecast horizon (hours)", min_value=1, max_value=168, value=24, step=1)
    if st.button("Run Forecast", type="primary"):
        with st.spinner("Fetching forecast…"):
            try:
                resp = requests.get(f"{API_URL}/forecast", params={"hours": hours}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                df = pd.DataFrame(data["forecast"])
                df["date"] = pd.to_datetime(df["date"])
                model_info = data.get("model", {})

                col1, col2, col3 = st.columns(3)
                col1.metric("Model", model_info.get("model_type", "—").upper())
                col2.metric("RMSE (test)", f"{model_info.get('rmse', 0):.3f}")
                col3.metric("MAPE (test)", f"{model_info.get('mape', 0):.2f}%")

                # Forecast chart
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["yhat"],
                    name="Forecast", line=dict(color="#2563eb", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=pd.concat([df["date"], df["date"].iloc[::-1]]),
                    y=pd.concat([df["yhat_upper"], df["yhat_lower"].iloc[::-1]]),
                    fill="toself", fillcolor="rgba(37,99,235,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% PI",
                ))
                fig.update_layout(
                    title=f"{hours}-hour Energy Demand Forecast",
                    xaxis_title="Time",
                    yaxis_title="Demand (kWh)",
                    hovermode="x unified",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Forecast failed: {e}")


# ─── Tab 2: Model Registry ────────────────────────────────────────────────────
with tab2:
    st.subheader("Registered Model Versions")
    if st.button("Refresh"):
        pass
    try:
        resp = requests.get(f"{API_URL}/model/versions", timeout=10)
        resp.raise_for_status()
        versions = resp.json().get("versions", [])
        if versions:
            df_v = pd.DataFrame(versions)
            # Highlight Production row
            def highlight_prod(row):
                return ["background-color: #dcfce7" if row["stage"] == "Production" else "" for _ in row]
            st.dataframe(
                df_v.style.apply(highlight_prod, axis=1),
                use_container_width=True,
            )
        else:
            st.info("No registered models yet. Run training first.")
    except Exception as e:
        st.error(f"Could not fetch model versions: {e}")


# ─── Tab 3: Drift Monitor ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Data Drift — PSI Check")
    st.caption("Compares last 4-week window against the preceding reference period.")
    if st.button("Run Drift Check", type="primary"):
        with st.spinner("Checking drift…"):
            try:
                resp = requests.get(f"{API_URL}/drift", timeout=30)
                resp.raise_for_status()
                report = resp.json()

                target_psi = report.get("target_psi", 0)
                level = report.get("target_level", "ok")
                color = {"ok": "green", "warn": "orange", "alert": "red"}.get(level, "gray")
                st.markdown(f"**Target PSI:** `{target_psi:.4f}` — :{color}[{level.upper()}]")

                if report.get("retraining_recommended"):
                    st.warning("⚠️ Retraining recommended — significant drift detected.")
                else:
                    st.success("✅ No significant drift detected.")

                # Feature PSI table
                features = report.get("features", {})
                if features:
                    feat_df = pd.DataFrame([
                        {"feature": k, "psi": v["psi"], "level": v["level"]}
                        for k, v in features.items()
                    ]).sort_values("psi", ascending=False)
                    st.dataframe(feat_df, use_container_width=True)
            except Exception as e:
                st.error(f"Drift check failed: {e}")


# ─── Tab 4: Retrain ───────────────────────────────────────────────────────────
with tab4:
    st.subheader("Manual Retraining")
    st.info("Triggers a full pipeline run: ingest → feature engineer → train both models → compare → promote winner.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Trigger Retraining Now", type="primary"):
            with st.spinner("Retraining… this may take a few minutes."):
                try:
                    resp = requests.post(f"{API_URL}/retrain", timeout=300)
                    resp.raise_for_status()
                    result = resp.json().get("result", {})
                    winner = result.get("winner", {})
                    st.success(
                        f"✅ Done! Winner: **{winner.get('model_name', '?').upper()}** "
                        f"— RMSE={winner.get('rmse', 0):.4f}"
                    )
                    st.json(winner)
                except Exception as e:
                    st.error(f"Retraining failed: {e}")

    with col_b:
        st.markdown("**Scheduler Status**")
        try:
            resp = requests.get(f"{API_URL}/scheduler/status", timeout=5)
            resp.raise_for_status()
            status = resp.json()
            st.json(status)
        except Exception as e:
            st.error(f"Could not fetch scheduler status: {e}")
