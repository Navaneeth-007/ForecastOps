"""ForecastOps operator dashboard; API is the single source of model state."""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="ForecastOps | Energy Intelligence", page_icon="⚡", layout="wide")


def api(path, *, method="get", **kwargs):
    response = requests.request(method, f"{API_URL}{path}", **kwargs)
    if not response.ok:
        try:
            message = response.json().get("detail", response.reason)
        except ValueError:
            message = response.reason
        raise RuntimeError(f"{response.status_code}: {message}")
    return response.json()


st.title("⚡ ForecastOps")
st.caption("ENERGY INTELLIGENCE / MODEL OPERATIONS")
st.markdown("Forecast hourly demand, inspect model decisions, and monitor distribution changes.")
with st.sidebar:
    st.header("Control room")
    st.caption("Prophet · LightGBM · MLflow · FastAPI")
    try:
        info = api("/model/info", timeout=10)
        st.success(f"Champion: {info['model_type'].upper()} · v{info['version']}")
        st.caption(f"Run {info['run_id'][:12]}")
    except (requests.RequestException, RuntimeError) as exc:
        st.warning(f"Model unavailable. Run training first. {exc}")
    st.divider()
    st.caption(
        "Default demo uses seeded synthetic data. Forecast dates follow the latest observation, not today's clock."
    )
    st.caption("LightGBM and Prophet are forecasting models. No LLM or paid inference API is used.")
    if not DEMO_MODE:
        st.caption("Interactive API docs are available on the API host at /docs.")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Model registry", "Drift monitor", "Retraining"])
with tab1:
    st.subheader("Demand outlook")
    hours = st.select_slider("Forecast horizon", options=[1, 6, 12, 24, 48, 72, 168], value=24)
    if st.button("Generate forecast", type="primary"):
        with st.spinner("Generating forecast…"):
            try:
                st.session_state.forecast = api("/forecast", params={"hours": hours}, timeout=120)
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))
    data = st.session_state.get("forecast")
    if data:
        df = pd.DataFrame(data["forecast"])
        df["date"] = pd.to_datetime(df["date"])
        info = data["model"]
        a, b, c = st.columns(3)
        a.metric("Holdout RMSE", f"{info['rmse']:.2f} kWh")
        b.metric("Average forecast", f"{df.yhat.mean():.1f} kWh")
        c.metric("Peak forecast", f"{df.yhat.max():.1f} kWh")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date, y=df.yhat_upper, line={"width": 0}, showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=df.date,
                y=df.yhat_lower,
                fill="tonexty",
                fillcolor="rgba(14,165,233,0.15)",
                line={"width": 0},
                name="Uncertainty band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.date, y=df.yhat, name="Demand forecast", line={"color": "#0ea5e9", "width": 3}
            )
        )
        fig.update_layout(
            height=430,
            hovermode="x unified",
            xaxis_title="Time (UTC)",
            yaxis_title="Hourly demand (kWh)",
            margin={"t": 20},
            legend={"orientation": "h", "x": 0, "y": 1.12},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{data['horizon_hours']}-hour forecast · model version {info['version']} · "
            "LightGBM bands are heuristic; Prophet bands are model-based estimates."
        )
        st.download_button(
            "Download forecast CSV", df.to_csv(index=False), "forecast.csv", "text/csv"
        )
        with st.expander("Inspect predictions"):
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Choose a horizon and generate a forecast to explore demand and uncertainty.")

with tab2:
    st.subheader("Versioned model history")
    st.caption(
        "The champion alias routes new requests to a versioned artifact. Previous versions remain available for rollback."
    )
    if st.button("Load model history"):
        try:
            versions = api("/model/versions", timeout=15)["versions"]
            st.session_state.versions = versions
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))
    versions = st.session_state.get("versions", [])
    if versions:
        st.dataframe(pd.DataFrame(versions), use_container_width=True, hide_index=True)
    else:
        st.info("Load history after the first training run.")

with tab3:
    st.subheader("Distribution health")
    st.caption(
        "PSI compares the last four weeks with the preceding four weeks. Seasonality can cause shifts without model degradation."
    )
    if st.button("Check distributions", type="primary"):
        try:
            st.session_state.drift = api("/drift", timeout=30)
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))
    report = st.session_state.get("drift")
    if report:
        st.metric("Demand PSI", f"{report['target_psi']:.4f}")
        if report["retraining_recommended"]:
            st.warning(
                "Significant distribution shift. Investigate before deciding whether to retrain."
            )
        else:
            st.success("No significant distribution shift detected.")
        st.dataframe(
            pd.DataFrame([{"feature": key, **value} for key, value in report["features"].items()]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("PSI < 0.10: stable · 0.10–0.25: monitor · ≥ 0.25: investigate")

with tab4:
    st.subheader("Train, compare, promote")
    st.info(
        "Trains both candidates and promotes the lower-RMSE candidate in this run. This operation may take several minutes."
    )
    if DEMO_MODE:
        st.caption(
            "This public demo is read-only. Models train during deployment; run the local Docker stack to explore authenticated retraining."
        )
    else:
        secret = st.text_input("Retraining API key", type="password")
        if st.button("Start training", type="primary", disabled=not secret):
            with st.spinner("Training models and recording experiments…"):
                try:
                    result = api(
                        "/retrain", method="post", headers={"X-API-Key": secret}, timeout=900
                    )
                    st.success("Training complete. Champion updated.")
                    st.json(result["result"]["winner"])
                except (requests.RequestException, RuntimeError) as exc:
                    st.error(str(exc))
    if st.button("Refresh training status"):
        try:
            st.json(api("/scheduler/status", timeout=10))
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))
