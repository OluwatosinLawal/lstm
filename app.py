# =============================================================================
#  ALERZO LSTM DEMAND FORECASTING — STREAMLIT APP
#  File : app.py
#  Run  : streamlit run app.py
#  Requires: pip install streamlit
# =============================================================================

import os
import pickle
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Alerzo Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a clean look ───────────────────────────────────
st.markdown(
    """
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #1565C0;
    }
    .metric-label { font-size: 13px; color: #555; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: bold; color: #1565C0; }
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


@st.cache_resource
def load_model_and_scaler():
    """
    Load the saved LSTM model and MinMaxScaler from disk.
    @st.cache_resource means this only runs once — not on every page refresh.
    """
    try:
        from tensorflow.keras.models import load_model

        model = load_model(os.path.join("model", "lstm_demand_forecast.keras"))
        with open(os.path.join("model", "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


def load_and_clean_csv(uploaded_file):
    """
    Reads an uploaded CSV file and applies the same cleaning steps
    used in the training pipeline.
    """
    df = pd.read_csv(uploaded_file, low_memory=False)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from text values
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Remove commas from numeric columns
    for col in ["final_amount", "unitPrice", "orderTotal", "quantitySold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

    # Parse dates with dayfirst=True (dd/mm/yyyy format)
    if "orderDate" in df.columns:
        df["orderDate"] = pd.to_datetime(
            df["orderDate"], dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["orderDate"])

    # Standardise salesCategory
    if "salesCategory" in df.columns:
        df["salesCategory"] = df["salesCategory"].str.strip().str.title()

    return df


def build_daily_series(df):
    """
    Aggregates transaction rows into a clean daily total sales series.
    Removes bottom 1% outlier days, matching the training pipeline.
    """
    daily = (
        df.groupby("orderDate")["final_amount"]
        .sum()
        .reset_index()
        .rename(columns={"orderDate": "date", "final_amount": "total_sales"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Fill full calendar (no gaps)
    full_cal = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_cal)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Remove zero and bottom 1% days
    daily = daily[daily["total_sales"] > 0].reset_index(drop=True)
    low_thresh = daily["total_sales"].quantile(0.01)
    daily = daily[daily["total_sales"] > low_thresh].reset_index(drop=True)

    return daily


def make_sequences(data, look_back=60):
    """Converts a 1-D normalised array into (X, y) supervised pairs."""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def compute_smape(y_true, y_pred):
    """Symmetric MAPE — robust to near-zero actual values."""
    return float(
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        * 100
    )


def forecast_future(model, scaler, last_60_raw, n_days=30):
    """
    Generates n_days of future forecasts using rolling one-step prediction.
    Each prediction is fed back as input for the next step.
    """
    last_60_scaled = scaler.transform(last_60_raw.reshape(-1, 1)).flatten()
    window = list(last_60_scaled)
    preds_scaled = []

    for _ in range(n_days):
        x = np.array(window[-60:]).reshape(1, 60, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(pred)
        window.append(pred)

    preds_naira = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    # Clip negative forecasts to 0 (sales cannot be negative)
    preds_naira = np.maximum(preds_naira, 0)
    return preds_naira


def load_comparison_results():
    """
    Loads comparison results from outputs/model_comparison.csv if it exists.
    Parses formatted strings (₦ and %) back into numbers for plotting.
    """
    csv_path = os.path.join("outputs", "model_comparison.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        # Create a numeric version for plotting
        plot_df = df.copy()

        def clean_val(val):
            if isinstance(val, str):
                return float(val.replace("₦", "").replace("%", "").replace(",", ""))
            return val

        for col in ["RMSE", "MAE", "MAPE", "sMAPE"]:
            if col in plot_df.columns:
                plot_df[col] = plot_df[col].apply(clean_val)

        return df, plot_df
    except Exception:
        return None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x60?text=Alerzo+LSTM", use_column_width=True
    )
    st.markdown("## Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "📂 Upload & Forecast",
            "🔮 Future Predictions",
            "📊 Model Performance",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Model info**")

    model, scaler, load_err = load_model_and_scaler()
    if model is not None:
        st.success("Model loaded ✓")
        st.caption("LSTM • 64 units • 60-day window")
        st.caption("TensorFlow 2.21 | 16,961 params")
    else:
        st.error("Model not found")
        st.caption(f"Error: {load_err}")
        st.caption("Make sure model/ folder is in the same directory as app.py")

    st.markdown("---")
    st.caption("Alerzo B2B Demand Forecasting")
    st.caption("Postgraduate Diploma Research Project")


# =============================================================================
# PAGE: HOME
# =============================================================================

if page == "🏠 Home":
    st.title("Alerzo Product Demand Forecasting System")
    st.markdown("##### LSTM-Based Sales Forecasting | Jan 2023 – Nov 2025")
    st.markdown("---")

    # Try to get latest sMAPE from results
    results_data = load_comparison_results()
    test_smape = "68.77%"
    if results_data:
        df_res, _ = results_data
        lstm_row = df_res[df_res["Model"] == "Vanilla LSTM"]
        if not lstm_row.empty:
            test_smape = lstm_row["sMAPE"].values[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Model Type</div>
            <div class="metric-value">Vanilla LSTM</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Training Period</div>
            <div class="metric-value">Jan 2023–Nov 2025</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Look-back Window</div>
            <div class="metric-value">60 Days</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Test sMAPE</div>
            <div class="metric-value">{test_smape}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### What this app does")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📂 Upload & Forecast")
        st.write(
            "Upload your cleaned CSV sales data and the model will generate forecasts against the actual values, showing RMSE, MAE, and sMAPE on your data."
        )
    with c2:
        st.markdown("#### 🔮 Future Predictions")
        st.write(
            "Upload your most recent sales data and generate forward-looking forecasts for the next 7, 14, or 30 days using the trained LSTM model."
        )
    with c3:
        st.markdown("#### 📊 Model Performance")
        st.write(
            "View the final comparative results of the LSTM against ARIMA and Prophet baselines on the original test set."
        )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Make sure your CSV file has an **orderDate** column (dd/mm/yyyy format) and a **final_amount** column
    2. Go to **Upload & Forecast** to evaluate the model on your data
    3. Go to **Future Predictions** to forecast upcoming sales
    """)

    if model is None:
        st.markdown(
            """
        <div class="warn-box">
        Model files not found. Make sure the <b>model/</b> folder containing
        <b>lstm_demand_forecast.keras</b> and <b>scaler.pkl</b> is in the
        same directory as app.py before using the forecast pages.
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE: UPLOAD & FORECAST
# =============================================================================

elif page == "📂 Upload & Forecast":
    st.title("Upload Data & Evaluate Forecast")
    st.markdown(
        "Upload one or more cleaned CSV files to run the LSTM model and see how well it forecasts against actual sales."
    )

    if model is None:
        st.error(
            "Model not loaded. Check that model/lstm_demand_forecast.keras and model/scaler.pkl exist."
        )
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload cleaned CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Files must have orderDate (dd/mm/yyyy) and final_amount columns",
    )

    look_back = 60  # Must match training

    if uploaded_files:
        with st.spinner("Loading and cleaning data…"):
            frames = []
            for f in uploaded_files:
                df = load_and_clean_csv(f)
                frames.append(df)
                st.success(f"✔ {f.name} — {len(df):,} rows loaded")

            master = pd.concat(frames, ignore_index=True)
            daily = build_daily_series(master)

        st.markdown(f"**Total trading days after cleaning:** {len(daily):,}")
        st.markdown(
            f"**Date range:** {daily['date'].min().date()} → {daily['date'].max().date()}"
        )
        st.markdown(f"**Mean daily sales:** ₦{daily['total_sales'].mean():,.0f}")

        if len(daily) < look_back + 10:
            st.error(
                f"Not enough data. Need at least {look_back + 10} trading days, got {len(daily)}."
            )
            st.stop()

        # ── Normalise and create sequences ──────────────────────
        sales = daily["total_sales"].values.reshape(-1, 1)
        scaled = scaler.transform(sales)
        X_all, y_all = make_sequences(scaled, look_back)
        X_all = X_all.reshape(*X_all.shape, 1)

        with st.spinner("Generating predictions…"):
            y_pred_scaled = model.predict(X_all, verbose=0)

        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_actual = scaler.inverse_transform(y_all.reshape(-1, 1)).flatten()
        dates = daily["date"].values[look_back:]

        # ── Metrics ─────────────────────────────────────────────
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        smape = compute_smape(y_actual, y_pred)
        thresh = np.mean(y_actual) * 0.01
        mask = y_actual > thresh
        mape = np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100

        st.markdown("---")
        st.markdown("### Forecast Performance on Uploaded Data")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"₦{rmse:,.0f}")
        m2.metric("MAE", f"₦{mae:,.0f}")
        m3.metric(
            "MAPE",
            f"{mape:.1f}%",
            help="Filtered: excludes days below 1% of mean sales",
        )
        m4.metric(
            "sMAPE", f"{smape:.1f}%", help="Symmetric MAPE — robust to near-zero days"
        )

        # ── Plot ────────────────────────────────────────────────
        st.markdown("### Forecast vs Actual Sales")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            dates,
            y_actual,
            color="#1565C0",
            linewidth=1.0,
            label="Actual Sales",
            alpha=0.85,
        )
        ax.plot(
            dates,
            y_pred,
            color="#E53935",
            linewidth=1.2,
            linestyle="--",
            label="LSTM Forecast",
        )
        ax.fill_between(
            dates,
            np.minimum(y_actual, y_pred),
            np.maximum(y_actual, y_pred),
            alpha=0.10,
            color="#E53935",
            label="Error Band",
        )
        ax.set_title(
            "LSTM Forecast vs Actual Daily Sales", fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("Total Sales (₦)")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Download results ────────────────────────────────────
        st.markdown("### Download Forecast Results")
        results_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dates).strftime("%d/%m/%Y"),
                "actual_sales_ngn": np.round(y_actual, 2),
                "predicted_sales_ngn": np.round(y_pred, 2),
                "error_ngn": np.round(y_actual - y_pred, 2),
            }
        )
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download forecast as CSV",
            data=csv_bytes,
            file_name="lstm_forecast_results.csv",
            mime="text/csv",
        )


# =============================================================================
# PAGE: FUTURE PREDICTIONS
# =============================================================================

elif page == "🔮 Future Predictions":
    st.title("Future Sales Forecast")
    st.markdown(
        "Upload your most recent sales data to generate forward-looking predictions."
    )

    if model is None:
        st.error("Model not loaded. Check that model/ folder exists with both files.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload your most recent cleaned CSV",
            type=["csv"],
            help="Must contain at least 60 trading days of data",
        )
    with col2:
        n_days = st.selectbox(
            "Forecast horizon",
            options=[7, 14, 30],
            index=1,
            help="Number of future days to forecast",
        )

    if uploaded:
        with st.spinner("Preparing data…"):
            df = load_and_clean_csv(uploaded)
            daily = build_daily_series(df)

        st.success(
            f"✔ {len(daily):,} trading days loaded | "
            f"{daily['date'].min().date()} → {daily['date'].max().date()}"
        )

        if len(daily) < 60:
            st.error("Need at least 60 trading days of data to generate forecasts.")
            st.stop()

        # Use last 60 days as the seed window
        last_60 = daily["total_sales"].values[-60:]
        last_date = daily["date"].iloc[-1]

        with st.spinner(f"Generating {n_days}-day forecast…"):
            future_preds = forecast_future(model, scaler, last_60, n_days)

        # Build future date index (skip Sundays — Alerzo typically does not trade)
        future_dates = []
        current = last_date
        while len(future_dates) < n_days:
            current = current + pd.Timedelta(days=1)
            if current.weekday() != 6:  # 6 = Sunday
                future_dates.append(current)

        future_df = pd.DataFrame(
            {
                "date": future_dates,
                "forecast_ngn": np.round(future_preds[: len(future_dates)], 2),
            }
        )

        st.markdown("---")
        st.markdown(f"### {n_days}-Day Sales Forecast")
        st.markdown(
            f"Forecast period: **{future_dates[0].date()}** → **{future_dates[-1].date()}**"
        )

        # ── Summary metrics ──────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Total forecast (period)", f"₦{future_preds.sum():,.0f}")
        c2.metric("Average daily forecast", f"₦{future_preds.mean():,.0f}")
        c3.metric("Peak day forecast", f"₦{future_preds.max():,.0f}")

        # ── Plot: history + forecast ──────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))

        # Show last 90 days of history for context
        hist = daily.tail(90)
        ax.plot(
            hist["date"],
            hist["total_sales"],
            color="#1565C0",
            linewidth=1.0,
            label="Historical Sales",
            alpha=0.8,
        )

        # Forecast line
        ax.plot(
            future_df["date"],
            future_df["forecast_ngn"],
            color="#E53935",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=4,
            label=f"{n_days}-Day LSTM Forecast",
        )

        # Shaded forecast region
        ax.axvspan(future_dates[0], future_dates[-1], alpha=0.06, color="#E53935")

        # Vertical divider between history and forecast
        ax.axvline(
            last_date,
            color="gray",
            linewidth=1.0,
            linestyle=":",
            label="Forecast start",
        )

        ax.set_title(
            f"LSTM {n_days}-Day Demand Forecast — Alerzo",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylabel("Total Sales (₦)")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Table of forecast values ──────────────────────────────
        st.markdown("### Daily Forecast Breakdown")
        display_df = future_df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%A, %d %b %Y")
        display_df["forecast_ngn"] = display_df["forecast_ngn"].apply(
            lambda v: f"₦{v:,.0f}"
        )
        display_df.columns = ["Date", "Forecast (₦)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────
        csv_bytes = (
            future_df.assign(date=future_df["date"].dt.strftime("%d/%m/%Y"))
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            label="⬇ Download forecast as CSV",
            data=csv_bytes,
            file_name=f"alerzo_forecast_{n_days}day.csv",
            mime="text/csv",
        )


# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================

elif page == "📊 Model Performance":
    st.title("Model Performance — Test Set Results")
    st.markdown("Final evaluation results from the held-out test set (Jun–Nov 2025).")

    # ── Try to load dynamic results ──────────────────────────────
    results_data = load_comparison_results()

    if results_data:
        res_df, plot_df = results_data

        # Table
        st.markdown("### Comparative Results: LSTM vs Baselines")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Charts
        st.markdown("### Visual Comparison")
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        colours = ["#1565C0", "#2E7D32", "#E53935", "#FBC02D"]
        metrics_meta = [
            ("RMSE", "RMSE (₦)"),
            ("MAE", "MAE (₦)"),
            ("MAPE", "MAPE (%)"),
            ("sMAPE", "sMAPE (%)"),
        ]

        models = plot_df["Model"].tolist()

        for ax, (metric, label) in zip(axes, metrics_meta):
            if metric in plot_df.columns:
                vals = plot_df[metric].values
                bars = ax.bar(
                    models,
                    vals,
                    color=colours[: len(models)],
                    edgecolor="white",
                    width=0.5,
                )
                ax.set_title(label, fontsize=11, fontweight="bold")
                ax.set_ylabel(label, fontsize=9)
                for bar in bars:
                    h = bar.get_height()
                    txt = f"₦{h:,.0f}" if "₦" in label else f"{h:.1f}%"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h * 1.02,
                        txt,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
                ax.grid(True, alpha=0.3, axis="y")
                ax.tick_params(axis="x", labelsize=8)

        plt.suptitle(
            "Model Comparison: LSTM vs Baseline Models", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Update description dynamically if possible
        lstm_row = res_df[res_df["Model"] == "Vanilla LSTM"]
        if not lstm_row.empty:
            l_rmse = lstm_row["RMSE"].values[0]
            l_mae = lstm_row["MAE"].values[0]
            l_smape = lstm_row["sMAPE"].values[0]
            st.info(
                f"**Current Run Summary (Vanilla LSTM):** RMSE: {l_rmse} | MAE: {l_mae} | sMAPE: {l_smape}"
            )

    else:
        # Fallback to hardcoded if CSV missing
        st.warning("Comparison CSV not found. Showing baseline estimates.")
        results = {
            "Model": ["Vanilla LSTM", "Prophet", "ARIMA"],
            "RMSE (₦)": ["66,358,732", "167,442,799", "207,790,264"],
            "MAE (₦)": ["53,692,800", "144,600,650", "194,490,642"],
            "MAPE (%)": ["424.40", "1,262.95", "1,525.71"],
            "sMAPE (%)": ["68.77", "98.04", "109.89"],
        }
        pd_df = pd.DataFrame(results)
        st.dataframe(pd_df, use_container_width=True, hide_index=True)

    # ── Interpretation ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Interpretation")

    if results_data:
        res_df, plot_df = results_data
        lstm = (
            plot_df[plot_df["Model"] == "Vanilla LSTM"].iloc[0]
            if "Vanilla LSTM" in plot_df["Model"].values
            else None
        )
        arima = (
            plot_df[plot_df["Model"] == "ARIMA"].iloc[0]
            if "ARIMA" in plot_df["Model"].values
            else None
        )
        prophet = (
            plot_df[plot_df["Model"] == "Prophet"].iloc[0]
            if "Prophet" in plot_df["Model"].values
            else None
        )

        if lstm is not None and arima is not None and prophet is not None:
            rmse_ratio = arima["RMSE"] / lstm["RMSE"]
            mae_ratio_arima = arima["MAE"] / lstm["MAE"]
            mae_ratio_prophet = prophet["MAE"] / lstm["MAE"]

            st.markdown(f"""
            - **RMSE**: The LSTM achieves **₦{lstm["RMSE"]:,.0f}** compared to ₦{prophet["RMSE"]:,.0f} (Prophet) and ₦{arima["RMSE"]:,.0f} (ARIMA) — approximately **{rmse_ratio:.1f}× lower error** than ARIMA.
            - **MAE**: The LSTM's average daily error of **₦{lstm["MAE"]:,.0f}** is **{mae_ratio_arima:.1f}× lower** than ARIMA and **{mae_ratio_prophet:.1f}× lower** than Prophet.
            - **sMAPE**: At **{lstm["sMAPE"]:.2f}%**, the LSTM is meaningfully more accurate than Prophet ({prophet["sMAPE"]:.2f}%) and ARIMA ({arima["sMAPE"]:.2f}%) on a scale-independent, symmetrically normalised basis.
            - **MAPE** values are elevated for all models due to near-zero actual sales on low-volume trading days. sMAPE is the more reliable percentage metric for this dataset.
            - The LSTM **outperforms both classical baselines on every metric**, confirming the effectiveness of the deep learning approach for B2B e-commerce demand forecasting.
            """)
        else:
            st.markdown(
                "Results for one or more models (LSTM, ARIMA, Prophet) are missing from the comparison file."
            )
    else:
        st.markdown("""
        - **RMSE**: The LSTM achieves ₦66.4M compared to ₦167.4M (Prophet) and ₦207.8M (ARIMA) — approximately **3× lower error** than ARIMA.
        - **MAE**: The LSTM's average daily error of ₦53.7M is **2.7× lower** than ARIMA and **2.6× lower** than Prophet.
        - **sMAPE**: At 68.77%, the LSTM is meaningfully more accurate than Prophet (98.04%) and ARIMA (109.89%) on a scale-independent, symmetrically normalised basis.
        - **MAPE** values are elevated for all models due to near-zero actual sales on low-volume trading days. sMAPE is the more reliable percentage metric for this dataset.
        - The LSTM **outperforms both classical baselines on every metric**, confirming the effectiveness of the deep learning approach for B2B e-commerce demand forecasting.
        """)

    st.markdown("### Training Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Architecture | Vanilla LSTM |
        | LSTM units | 64 |
        | Dropout rate | 0.2 |
        | Look-back window | 60 days |
        | Total parameters | 16,961 |
        """)
    with col2:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Optimiser | Adam |
        | Loss function | MSE |
        | Epochs trained | 22 (early stopped) |
        | Best epoch | 2 |
        | Best val loss | 0.02645 |
        """)


# =============================================================================
# PAGE: ABOUT
# =============================================================================

elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown("""
    ### Design and Development of an LSTM-Based Product Demand Forecasting System

    This application is the practical implementation component of a postgraduate diploma
    research project. It demonstrates a trained vanilla LSTM neural network for forecasting
    daily product demand from historical B2B e-commerce sales data.

    ---

    ### Data Source
    - **Company**: Alerzo Limited (Nigerian B2B e-commerce)
    - **Period**: January 2023 – November 2025
    - **Volume**: 2,381,940 transaction records across 5 files
    - **Trading days used**: 982 (after cleaning and outlier removal)

    ---

    ### Model
    - Single-layer LSTM with 64 units
    - 60-day look-back window
    - Dropout regularisation (rate = 0.2)
    - Trained with Adam optimiser and MSE loss
    - Early stopping with patience = 20

    ---

    ### Technologies
    - Python 3.11.9
    - TensorFlow 2.21 / Keras
    - pandas, NumPy, scikit-learn
    - Streamlit (this app)
    - Matplotlib

    ---

    ### How to run locally
    ```
    pip install streamlit tensorflow pandas numpy scikit-learn matplotlib
    streamlit run app.py
    ```
    """)
