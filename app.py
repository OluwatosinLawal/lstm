# =============================================================================
#  AN LSTM DEMAND FORECASTING — STREAMLIT APP
#  File : app.py
#  Run  : streamlit run app.py
#
#  FOLDER STRUCTURE REQUIRED:
#  lstm-forecasting/
#  ├── app.py
#  ├── model/
#  │   ├── lstm_demand_forecast.keras
#  │   └── scaler.pkl
#  └── outputs/
#      └── model_comparison.csv
# =============================================================================

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(
    page_title="LSTM Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATHS — always relative to this file so it works locally and on cloud
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "lstm_demand_forecast.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")

# =============================================================================
# HELPERS
# =============================================================================

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        return None, None, (
            f"File not found: {MODEL_PATH}\n"
            "Run lstm_forecasting.py first, then make sure the model/ folder "
            "is in the same directory as app.py."
        )
    if not os.path.exists(SCALER_PATH):
        return None, None, f"File not found: {SCALER_PATH}. Run lstm_forecasting.py."
    try:
        from tensorflow.keras.models import load_model
        model  = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


def load_comparison_csv():
    """
    Reads outputs/model_comparison.csv saved by lstm_forecasting.py.
    Returns (display_df, numeric_df) or (None, None).
    """
    if not os.path.exists(CSV_PATH):
        return None, None
    try:
        df = pd.read_csv(CSV_PATH)

        def to_float(val):
            if isinstance(val, str):
                return float(val.replace("₦","").replace("%","").replace(",","").strip())
            return float(val)

        num = df.copy()
        for col in ["RMSE", "MAE", "MAPE", "sMAPE"]:
            if col in num.columns:
                num[col] = num[col].apply(to_float)
        return df, num
    except Exception as e:
        st.warning(f"Could not parse model_comparison.csv: {e}")
        return None, None


def clean_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    for col in ["final_amount", "unitPrice", "orderTotal", "quantitySold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    if "orderDate" in df.columns:
        df["orderDate"] = pd.to_datetime(df["orderDate"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["orderDate"])
    if "salesCategory" in df.columns:
        df["salesCategory"] = df["salesCategory"].str.strip().str.title()
    return df


def build_daily(df):
    daily = (
        df.groupby("orderDate")["final_amount"].sum().reset_index()
        .rename(columns={"orderDate": "date", "final_amount": "total_sales"})
        .sort_values("date").reset_index(drop=True)
    )
    cal = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date").reindex(cal).fillna(0)
        .reset_index().rename(columns={"index": "date"})
    )
    daily = daily[daily["total_sales"] > 0].reset_index(drop=True)
    thresh = daily["total_sales"].quantile(0.01)
    return daily[daily["total_sales"] > thresh].reset_index(drop=True)


def make_sequences(data, lb=60):
    X, y = [], []
    for i in range(lb, len(data)):
        X.append(data[i-lb:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def smape(y_true, y_pred):
    return float(np.mean(
        2*np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)+1e-8)
    )*100)


def future_forecast(model, scaler, last_60, n):
    s = scaler.transform(last_60.reshape(-1,1)).flatten()
    w = list(s)
    out = []
    for _ in range(n):
        x = np.array(w[-60:]).reshape(1,60,1)
        p = model.predict(x, verbose=0)[0,0]
        out.append(p); w.append(p)
    return np.maximum(
        scaler.inverse_transform(np.array(out).reshape(-1,1)).flatten(), 0
    )


# =============================================================================
# LOAD ONCE
# =============================================================================
model, scaler, load_err = load_model_and_scaler()
disp_df, num_df = load_comparison_csv()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", [
        "🏠 Home", "📂 Upload & Forecast",
        "🔮 Future Predictions", "📊 Model Performance", "ℹ️ About"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Model status**")
    if model is not None:
        st.success("Model loaded ✓")
        st.caption("LSTM · 64 units · 60-day window")
    else:
        st.error("Model not found")
        st.caption("Run lstm_forecasting.py")
        st.caption("then place model/ next to app.py")

    st.markdown("---")
    if num_df is not None:
        row = num_df[num_df["Model"] == "Vanilla LSTM"]
        if not row.empty:
            st.markdown("**Latest results (live)**")
            st.caption(f"RMSE : ₦{row['RMSE'].values[0]:,.0f}")
            st.caption(f"MAE  : ₦{row['MAE'].values[0]:,.0f}")
            st.caption(f"sMAPE: {row['sMAPE'].values[0]:.2f}%")
    else:
        st.caption("Run lstm_forecasting.py")
        st.caption("to see live metrics here")

    st.markdown("---")
    st.caption("Nigerian B2B Demand Forecasting")
    st.caption("Postgraduate Diploma Research")

# =============================================================================
# PAGE: HOME
# =============================================================================
if page == "🏠 Home":
    st.title("LSTM Product Demand Forecasting System")
    st.markdown("##### LSTM-Based Daily Sales Forecasting | Jan 2023 – Nov 2025")
    st.markdown("---")

    # sMAPE pulled live from CSV — updates automatically after each training run
    if num_df is not None:
        row = num_df[num_df["Model"] == "Vanilla LSTM"]
        smape_val = f"{row['sMAPE'].values[0]:.2f}%" if not row.empty else "—"
    else:
        smape_val = "Run model first"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Model Type", "Training Period", "Look-back Window", "Test sMAPE"],
        ["Vanilla LSTM", "Jan 2023–Nov 2025", "60 Days", smape_val]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 📂 Upload & Forecast")
        st.write("Upload cleaned CSV files. The model evaluates against actual sales and shows RMSE, MAE, and sMAPE.")
    with b:
        st.markdown("#### 🔮 Future Predictions")
        st.write("Upload recent data and generate 7, 14, or 30-day forward forecasts.")
    with c:
        st.markdown("#### 📊 Model Performance")
        st.write("LSTM vs ARIMA vs Prophet comparison, updated automatically from the latest training run.")

    st.markdown("---")
    st.markdown("### Setup checklist")
    m_ok = "✅" if model   is not None else "❌"
    c_ok = "✅" if disp_df is not None else "⚠️"
    st.markdown(f"""
{m_ok} **model/lstm_demand_forecast.keras** — {"found" if model is not None else "not found — run lstm_forecasting.py"}
{m_ok} **model/scaler.pkl** — {"found" if model is not None else "not found — run lstm_forecasting.py"}
{c_ok} **outputs/model_comparison.csv** — {"found — metrics are live and will update after each run" if disp_df is not None else "not found — run lstm_forecasting.py to generate it"}
    """)

    if model is None:
        st.error(f"**Model error:** {load_err}")
        st.info("""
**How to fix:**
1. Open VS Code terminal in your project folder
2. Activate your venv: `venv\\Scripts\\activate`
3. Run: `python lstm_forecasting.py`
4. Wait for training to complete (~2 minutes)
5. Run: `streamlit run app.py`
        """)

# =============================================================================
# PAGE: UPLOAD & FORECAST
# =============================================================================
elif page == "📂 Upload & Forecast":
    st.title("Upload Data & Evaluate Forecast")
    if model is None:
        st.error("Model not loaded. Run lstm_forecasting.py first.")
        st.stop()

    files = st.file_uploader("Upload cleaned CSV file(s)", type=["csv"],
                              accept_multiple_files=True,
                              help="Must have orderDate (dd/mm/yyyy) and final_amount columns")
    if files:
        with st.spinner("Cleaning data…"):
            frames = []
            for f in files:
                d = clean_csv(f)
                frames.append(d)
                st.success(f"✔ {f.name} — {len(d):,} rows")
            master = pd.concat(frames, ignore_index=True)
            daily  = build_daily(master)

        st.markdown(f"**Trading days:** {len(daily):,} &nbsp;|&nbsp; "
                    f"**Range:** {daily['date'].min().date()} → {daily['date'].max().date()} &nbsp;|&nbsp; "
                    f"**Mean daily sales:** ₦{daily['total_sales'].mean():,.0f}")

        if len(daily) < 70:
            st.error("Need at least 70 trading days. Upload more data.")
            st.stop()

        sales  = daily["total_sales"].values.reshape(-1, 1)
        scaled = scaler.transform(sales)
        X, y   = make_sequences(scaled)
        X      = X.reshape(*X.shape, 1)

        with st.spinner("Generating predictions…"):
            yp_s = model.predict(X, verbose=0)

        yp  = scaler.inverse_transform(yp_s).flatten()
        ya  = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        dts = daily["date"].values[60:]

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse_v  = np.sqrt(mean_squared_error(ya, yp))
        mae_v   = mean_absolute_error(ya, yp)
        smape_v = smape(ya, yp)
        mask    = ya > np.mean(ya) * 0.01
        mape_v  = np.mean(np.abs((ya[mask]-yp[mask])/ya[mask]))*100

        st.markdown("---")
        st.markdown("### Performance on Uploaded Data")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE",  f"₦{rmse_v:,.0f}")
        m2.metric("MAE",   f"₦{mae_v:,.0f}")
        m3.metric("MAPE",  f"{mape_v:.1f}%",  help="Excludes near-zero days")
        m4.metric("sMAPE", f"{smape_v:.1f}%", help="Symmetric — robust to near-zero days")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dts, ya, color="#1565C0", linewidth=1.0, label="Actual Sales", alpha=0.85)
        ax.plot(dts, yp, color="#E53935", linewidth=1.2, linestyle="--", label="LSTM Forecast")
        ax.fill_between(dts, np.minimum(ya,yp), np.maximum(ya,yp),
                        alpha=0.10, color="#E53935", label="Error Band")
        ax.set_title("LSTM Forecast vs Actual Daily Sales", fontsize=13, fontweight="bold")
        ax.set_ylabel("Total Sales (₦)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        out = pd.DataFrame({
            "date":                pd.to_datetime(dts).strftime("%d/%m/%Y"),
            "actual_sales_ngn":    np.round(ya, 2),
            "predicted_sales_ngn": np.round(yp, 2),
            "error_ngn":           np.round(ya - yp, 2),
        })
        st.download_button("⬇ Download CSV", out.to_csv(index=False).encode(),
                           "forecast_results.csv", "text/csv")

# =============================================================================
# PAGE: FUTURE PREDICTIONS
# =============================================================================
elif page == "🔮 Future Predictions":
    st.title("Future Sales Forecast")
    if model is None:
        st.error("Model not loaded. Run lstm_forecasting.py first.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        up = st.file_uploader("Upload most recent cleaned CSV", type=["csv"],
                              help="Needs at least 60 trading days")
    with col2:
        n = st.selectbox("Forecast horizon", [7, 14, 30], index=1)

    if up:
        with st.spinner("Preparing…"):
            df    = clean_csv(up)
            daily = build_daily(df)

        st.success(f"✔ {len(daily):,} trading days | "
                   f"{daily['date'].min().date()} → {daily['date'].max().date()}")

        if len(daily) < 60:
            st.error("Need at least 60 trading days."); st.stop()

        last60    = daily["total_sales"].values[-60:]
        last_date = daily["date"].iloc[-1]

        with st.spinner(f"Forecasting {n} days…"):
            preds = future_forecast(model, scaler, last60, n)

        fut = []
        cur = last_date
        while len(fut) < n:
            cur += pd.Timedelta(days=1)
            if cur.weekday() != 6:
                fut.append(cur)

        fdf = pd.DataFrame({"date": fut, "forecast_ngn": np.round(preds[:len(fut)], 2)})

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total",   f"₦{preds.sum():,.0f}")
        c2.metric("Average", f"₦{preds.mean():,.0f}")
        c3.metric("Peak",    f"₦{preds.max():,.0f}")

        fig, ax = plt.subplots(figsize=(14, 5))
        hist = daily.tail(90)
        ax.plot(hist["date"], hist["total_sales"], color="#1565C0",
                linewidth=1.0, label="Historical", alpha=0.8)
        ax.plot(fdf["date"], fdf["forecast_ngn"], color="#E53935",
                linewidth=2.0, linestyle="--", marker="o", markersize=4,
                label=f"{n}-Day Forecast")
        ax.axvspan(fut[0], fut[-1], alpha=0.06, color="#E53935")
        ax.axvline(last_date, color="gray", linewidth=1.0, linestyle=":",
                   label="Forecast start")
        ax.set_title(f"LSTM {n}-Day Demand Forecast", fontsize=13, fontweight="bold")
        ax.set_ylabel("Total Sales (₦)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        show = fdf.copy()
        show["date"]         = show["date"].dt.strftime("%A, %d %b %Y")
        show["forecast_ngn"] = show["forecast_ngn"].apply(lambda v: f"₦{v:,.0f}")
        show.columns = ["Date", "Forecast (₦)"]
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.download_button("⬇ Download CSV",
            fdf.assign(date=fdf["date"].dt.strftime("%d/%m/%Y"))
               .to_csv(index=False).encode(),
            f"forecast_{n}day.csv", "text/csv")

# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================
elif page == "📊 Model Performance":
    st.title("Model Performance — Test Set Results")
    st.markdown("Updates automatically each time you re-run `lstm_forecasting.py`.")

    if disp_df is not None and num_df is not None:
        st.success("Live results loaded from `outputs/model_comparison.csv`")
        st.markdown("### Comparative Results")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        st.markdown("### Visual Comparison")
        colours = ["#1565C0", "#2E7D32", "#E53935"]
        mlist   = num_df["Model"].tolist()
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        for ax, (metric, label) in zip(axes, [
            ("RMSE","RMSE (₦)"),("MAE","MAE (₦)"),
            ("MAPE","MAPE (%)"),("sMAPE","sMAPE (%)")
        ]):
            if metric in num_df.columns:
                vals = num_df[metric].values
                bars = ax.bar(mlist, vals, color=colours[:len(mlist)],
                              edgecolor="white", width=0.5)
                ax.set_title(label, fontsize=11, fontweight="bold")
                ax.set_ylabel(label, fontsize=9)
                for bar in bars:
                    h = bar.get_height()
                    txt = f"₦{h:,.0f}" if "₦" in label else f"{h:.1f}%"
                    ax.text(bar.get_x()+bar.get_width()/2, h*1.02,
                            txt, ha="center", va="bottom", fontsize=7)
                ax.grid(True, alpha=0.3, axis="y")
                ax.tick_params(axis="x", labelsize=8)
        plt.suptitle("Model Comparison: LSTM vs Baselines",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        lstm    = num_df[num_df["Model"]=="Vanilla LSTM"].iloc[0] if "Vanilla LSTM" in num_df["Model"].values else None
        arima   = num_df[num_df["Model"]=="ARIMA"].iloc[0]        if "ARIMA"        in num_df["Model"].values else None
        prophet = num_df[num_df["Model"]=="Prophet"].iloc[0]      if "Prophet"      in num_df["Model"].values else None

        st.markdown("---")
        st.markdown("### Interpretation")
        if all(x is not None for x in [lstm, arima, prophet]):
            st.markdown(f"""
- **RMSE**: LSTM ₦{lstm['RMSE']:,.0f} vs Prophet ₦{prophet['RMSE']:,.0f} vs ARIMA ₦{arima['RMSE']:,.0f} — LSTM is **{arima['RMSE']/lstm['RMSE']:.1f}× more accurate** than ARIMA.
- **MAE**: LSTM ₦{lstm['MAE']:,.0f} average error is **{arima['MAE']/lstm['MAE']:.1f}× lower** than ARIMA and **{prophet['MAE']/lstm['MAE']:.1f}× lower** than Prophet.
- **sMAPE**: LSTM {lstm['sMAPE']:.2f}% vs Prophet {prophet['sMAPE']:.2f}% vs ARIMA {arima['sMAPE']:.2f}%.
- MAPE values are elevated due to near-zero actual sales on low-volume days. sMAPE is the primary percentage metric for this dataset.
            """)
    else:
        st.warning("outputs/model_comparison.csv not found — showing placeholder values.")
        st.info("Run `python lstm_forecasting.py` to generate live results.")
        st.dataframe(pd.DataFrame({
            "Model":     ["Vanilla LSTM","Prophet","ARIMA"],
            "RMSE (₦)":  ["66,358,732","167,442,799","207,790,264"],
            "MAE (₦)":   ["53,692,800","144,600,650","194,490,642"],
            "MAPE (%)":  ["424.40","1,262.95","1,525.71"],
            "sMAPE (%)": ["68.77","98.04","109.89"],
        }), use_container_width=True, hide_index=True)

    st.markdown("### Training Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
| Parameter | Value |
|---|---|
| Architecture | Vanilla LSTM |
| LSTM units | 64 |
| Dropout rate | 0.2 |
| Look-back window | 60 days |
| Total parameters | 16,961 |
        """)
    with c2:
        st.markdown("""
| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Loss function | MSE |
| Max epochs | 100 |
| Early stopping patience | 20 |
| LR reduction patience | 8 |
        """)

# =============================================================================
# PAGE: ABOUT
# =============================================================================
elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown("""
### Design and Development of an LSTM-Based Product Demand Forecasting System
Postgraduate Diploma Research — Nigerian B2B E-Commerce

---
### Data
- **Period**: January 2023 – November 2025
- **Raw records**: ~2.66 million transactions across 5 files
- **Trading days used**: 982 (after cleaning and outlier removal)
- **Date format in source files**: dd/mm/yyyy

### Model
- Vanilla LSTM · 64 units · 60-day look-back · Dropout 0.2
- Adam optimiser · MSE loss · Early stopping (patience 20)

### Technologies
Python 3.11.9 · TensorFlow 2.21 · pandas · NumPy · scikit-learn · Streamlit · Matplotlib · pmdarima · Prophet

### Required folder structure
```
lstm-forecasting/
├── app.py
├── lstm_forecasting.py
├── model/
│   ├── lstm_demand_forecast.keras
│   └── scaler.pkl
└── outputs/
    └── model_comparison.csv
```

### Run locally
```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib
streamlit run app.py
```
    """)
