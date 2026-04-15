# =============================================================================
#  LSTM DEMAND FORECASTING — STREAMLIT APP  (Full Version)
#  Run : streamlit run app.py
#
#  Folder structure required:
#  project/
#  ├── app.py
#  ├── model/
#  │   ├── lstm_demand_forecast.keras
#  │   └── scaler.pkl
#  └── outputs/
#      ├── model_comparison.csv
#      ├── 01_daily_sales_timeseries.png
#      ├── 02_training_loss.png
#      ├── 03_predictions_vs_actual.png
#      └── 04_model_comparison.png
# =============================================================================

import os, pickle, warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1b3a2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #4caf82;
    }
    .metric-label { font-size: 13px; color: #a8d5b5; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: bold; color: #c8f7dc; }

    .explain-box {
        background: #1a3d2b;
        border-left: 5px solid #4caf82;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #d4f5c0 !important;
        font-size: 14px;
        line-height: 1.7;
    }
    .explain-box strong {
        color: #b8ffcc !important;
    }
    .explain-box em {
        color: #a8f0b8 !important;
        font-style: italic;
    }

    .step-box {
        background: #1e2a3a;
        border-left: 4px solid #7F77DD;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0 12px 0;
        color: #c8d8f0 !important;
        font-size: 13px;
    }

    .warn-box {
        background: #3a2e10;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #f5dfa0 !important;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "lstm_demand_forecast.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")

REQUIRED_COLS = {
    "orderDate":     "Date the order was created (format: DD/MM/YYYY)",
    "final_amount":  "Sales value of the order line (unit price × quantity). Numbers only — no commas.",
    "displayTitle":  "Product name (optional — used for reference only)",
    "productId":     "Unique product identifier / SKU (optional)",
    "quantitySold":  "Number of units sold (optional)",
    "unitPrice":     "Price per unit in NGN (optional)",
    "orderTotal":    "Total order value including all items (optional)",
    "salesCategory": "Label: 'Regular Sales' or 'Promo Sales' (optional)",
    "SubOrderNumber":"Unique transaction identifier (optional)",
}
MANDATORY_COLS = ["orderDate", "final_amount"]


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found: {MODEL_PATH}"
    if not os.path.exists(SCALER_PATH):
        return None, None, f"Scaler file not found: {SCALER_PATH}"
    try:
        from tensorflow.keras.models import load_model
        model  = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


def load_comparison_csv():
    if not os.path.exists(CSV_PATH):
        return None, None
    try:
        df = pd.read_csv(CSV_PATH)
        def to_float(v):
            if isinstance(v, str):
                return float(v.replace("₦","").replace("%","").replace(",","").strip())
            return float(v)
        num = df.copy()
        for col in ["RMSE","MAE","MAPE","sMAPE"]:
            if col in num.columns:
                num[col] = num[col].apply(to_float)
        return df, num
    except Exception as e:
        st.warning(f"Could not parse model_comparison.csv: {e}")
        return None, None


def show_column_guide():
    """Renders the required/optional column reference table."""
    st.markdown("#### Column reference")
    rows = ""
    for col, desc in REQUIRED_COLS.items():
        req = "**Required**" if col in MANDATORY_COLS else "Optional"
        rows += f"| `{col}` | {req} | {desc} |\n"
    st.markdown(
        "| Column name | Status | Description |\n"
        "|---|---|---|\n" + rows
    )
    st.markdown(
        "> **Date format:** `orderDate` must be DD/MM/YYYY (e.g. `05/01/2023`).  \n"
        "> **Numeric format:** `final_amount` must be plain numbers — remove currency symbols and commas before uploading.  \n"
        "> Column names are **case-sensitive** and must match exactly."
    )


def map_columns(df):
    """
    Let the user map their CSV columns to the required names
    when the names don't match exactly.
    Returns a renamed dataframe or None if mandatory cols are missing.
    """
    csv_cols = list(df.columns)
    st.markdown("#### Map your columns")
    st.caption(
        "Your file's column names are listed below. "
        "Use the dropdowns to tell the app which column in your file "
        "corresponds to each required field."
    )
    mapping = {}
    c1, c2 = st.columns(2)
    for i, (req_col, desc) in enumerate(
        [("orderDate", "Date column (DD/MM/YYYY)"),
         ("final_amount", "Sales amount column (₦)")]
    ):
        col_widget = c1 if i % 2 == 0 else c2
        with col_widget:
            # Pre-select the matching column if it already exists
            default_idx = csv_cols.index(req_col) if req_col in csv_cols else 0
            chosen = st.selectbox(
                f"{req_col} — {desc}",
                options=csv_cols,
                index=default_idx,
                key=f"map_{req_col}"
            )
            mapping[req_col] = chosen
    return mapping


def apply_mapping_and_clean(df, mapping):
    """Rename columns per mapping then apply cleaning."""
    rename = {v: k for k, v in mapping.items() if v != k}
    if rename:
        df = df.rename(columns=rename)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    for col in ["final_amount","unitPrice","orderTotal","quantitySold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",","",regex=False), errors="coerce"
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
        .rename(columns={"orderDate":"date","final_amount":"total_sales"})
        .sort_values("date").reset_index(drop=True)
    )
    cal = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date").reindex(cal).fillna(0)
        .reset_index().rename(columns={"index":"date"})
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


def calc_smape(y_true, y_pred):
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


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plot_daily_series(daily):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(daily["date"], daily["total_sales"],
                 color="#1565C0", linewidth=0.7, alpha=0.8)
    axes[0].set_title("Daily Total Sales", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Total Sales (₦)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3)

    roll30 = daily["total_sales"].rolling(window=30, min_periods=1).mean()
    axes[1].plot(daily["date"], daily["total_sales"],
                 color="#90CAF9", linewidth=0.5, alpha=0.6, label="Daily Sales")
    axes[1].plot(daily["date"], roll30,
                 color="#E53935", linewidth=1.8, label="30-Day Rolling Average")
    axes[1].set_title("Daily Sales with 30-Day Rolling Average", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Total Sales (₦)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_training_history(history_dict):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history_dict["loss"],     color="#1565C0", linewidth=1.5, label="Training Loss (MSE)")
    ax.plot(history_dict["val_loss"], color="#E53935", linewidth=1.5, label="Validation Loss (MSE)")
    ax.set_title("Training vs Validation Loss per Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def explain_metrics(rmse, mae, smape_v, mape_v, mean_sales, n_total, n_mape):
    rmse_pct = rmse / mean_sales * 100
    mae_pct  = mae  / mean_sales * 100
    st.markdown(f"""
<div class="explain-box">
<strong>What these numbers mean</strong><br><br>
<strong>RMSE — ₦{rmse:,.0f}</strong><br>
On average, the model's predictions deviated from actual sales by about ₦{rmse:,.0f} per day,
with larger errors penalised more heavily. This is {rmse_pct:.1f}% of the mean daily sales
of ₦{mean_sales:,.0f}, indicating the typical magnitude of the largest prediction errors.<br><br>

<strong>MAE — ₦{mae:,.0f}</strong><br>
The average absolute daily prediction error is ₦{mae:,.0f} ({mae_pct:.1f}% of mean daily sales).
Unlike RMSE, MAE treats all errors equally regardless of size, giving a cleaner picture of
typical day-to-day accuracy.<br><br>

<strong>sMAPE — {smape_v:.2f}%</strong><br>
The Symmetric Mean Absolute Percentage Error (sMAPE) is the primary percentage metric here.
A sMAPE of {smape_v:.2f}% means that on average across all {n_total} test days,
the model's daily prediction deviated from the actual value by approximately
{smape_v/2:.1f}% of the combined magnitude of both. sMAPE is used instead of standard
MAPE because it handles near-zero actual values without producing extreme outlier percentages.<br><br>

<strong>MAPE — {mape_v:.2f}%</strong><br>
Standard MAPE was evaluated on {n_mape} of {n_total} days (days where actual sales exceeded
1% of the mean). The remaining {n_total - n_mape} days with very low actual sales were excluded
because dividing by a near-zero denominator produces extreme percentage values that inflate
the average unfairly. The MAPE figure should be read alongside sMAPE for a complete picture.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD ONCE
# ══════════════════════════════════════════════════════════════════
model, scaler, load_err = load_model_and_scaler()
disp_df, num_df = load_comparison_csv()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "📂 Upload & Forecast",
        "🔮 Future Predictions",
        "📊 Training Results",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Model status**")
    if model is not None:
        st.success("Model loaded ✓")
        st.caption("Vanilla LSTM · 64 units · 60-day window")
        st.caption("16,961 trainable parameters")
    else:
        st.error("Model not found")
        st.caption("Run lstm_forecasting.py first")

    st.markdown("---")
    if num_df is not None:
        row = num_df[num_df["Model"] == "Vanilla LSTM"]
        if not row.empty:
            st.markdown("**Live test-set results**")
            st.caption(f"RMSE  : ₦{row['RMSE'].values[0]:,.0f}")
            st.caption(f"MAE   : ₦{row['MAE'].values[0]:,.0f}")
            st.caption(f"sMAPE : {row['sMAPE'].values[0]:.2f}%")
    else:
        st.caption("Run lstm_forecasting.py")
        st.caption("to see live results here")

    st.markdown("---")
    st.caption("Nigerian B2B Demand Forecasting")
    st.caption("Postgraduate Diploma Research")


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("LSTM Product Demand Forecasting System")
    st.markdown("##### Vanilla LSTM · Nigerian B2B E-Commerce · Jan 2023 – Nov 2025")
    st.markdown("---")

    smape_val = "—"
    if num_df is not None:
        row = num_df[num_df["Model"] == "Vanilla LSTM"]
        if not row.empty:
            smape_val = f"{row['sMAPE'].values[0]:.2f}%"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Model", "Training period", "Look-back window", "Test sMAPE"],
        ["Vanilla LSTM", "Jan 2023 – Nov 2025", "60 days", smape_val]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### What this app does")
    a, b, c, d = st.columns(4)
    with a:
        st.markdown("**📂 Upload & Forecast**")
        st.write("Upload your sales CSV, map columns, and the model evaluates forecasts against your actual sales with full graphs and explanations.")
    with b:
        st.markdown("**🔮 Future Predictions**")
        st.write("Upload recent data to generate 7, 14, or 30-day forward demand forecasts with a dated breakdown table.")
    with c:
        st.markdown("**📊 Training Results**")
        st.write("View all training graphs — daily sales series, training loss curves, test forecasts, and metric comparisons — from the original model training run.")
    with d:
        st.markdown("**ℹ️ About**")
        st.write("Project background, model architecture, and data source details.")

    st.markdown("---")
    st.markdown("### Setup checklist")
    m_ok = "✅" if model   is not None else "❌"
    c_ok = "✅" if disp_df is not None else "⚠️"
    g1 = os.path.exists(os.path.join(OUT_DIR, "01_daily_sales_timeseries.png"))
    g2 = os.path.exists(os.path.join(OUT_DIR, "02_training_loss.png"))
    g3 = os.path.exists(os.path.join(OUT_DIR, "03_predictions_vs_actual.png"))
    g4 = os.path.exists(os.path.join(OUT_DIR, "04_model_comparison.png"))
    st.markdown(f"""
{m_ok} `model/lstm_demand_forecast.keras` — {"found" if model is not None else "not found — run lstm_forecasting.py"}
{m_ok} `model/scaler.pkl` — {"found" if model is not None else "not found — run lstm_forecasting.py"}
{c_ok} `outputs/model_comparison.csv` — {"found — live metrics active" if disp_df is not None else "not found — run lstm_forecasting.py"}
{"✅" if g1 else "⚠️"} `outputs/01_daily_sales_timeseries.png` — {"found" if g1 else "not found — run lstm_forecasting.py"}
{"✅" if g2 else "⚠️"} `outputs/02_training_loss.png` — {"found" if g2 else "not found — run lstm_forecasting.py"}
{"✅" if g3 else "⚠️"} `outputs/03_predictions_vs_actual.png` — {"found" if g3 else "not found — run lstm_forecasting.py"}
{"✅" if g4 else "⚠️"} `outputs/04_model_comparison.png` — {"found" if g4 else "not found — run lstm_forecasting.py"}
    """)

    if model is None:
        st.error(f"**Model error:** {load_err}")
        st.info("""
**How to fix this:**
1. Open a VS Code terminal in your project folder
2. Activate the virtual environment: `venv\\Scripts\\activate`
3. Run: `python lstm_forecasting.py`
4. Wait for training to finish (approximately 2 minutes)
5. Return here and refresh the page
        """)


# ══════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & FORECAST
# ══════════════════════════════════════════════════════════════════
elif page == "📂 Upload & Forecast":
    st.title("Upload Data & Evaluate Forecast")
    st.markdown(
        "Upload one or more sales CSV files. The app will walk you through "
        "column mapping, data preparation, model inference, and result interpretation."
    )

    if model is None:
        st.error("Model not loaded. Run `python lstm_forecasting.py` first.")
        st.stop()

    # ── Step 1: Column guide ──────────────────────────────────────
    with st.expander("📋  Step 1 — Column requirements (read before uploading)", expanded=True):
        st.markdown(
            "Your CSV file must contain at minimum the two **Required** columns below. "
            "All other columns are optional. Column names must match exactly — "
            "they are case-sensitive."
        )
        show_column_guide()
        st.markdown("**Example of the minimum required CSV structure:**")
        st.code(
            "orderDate,final_amount\n"
            "01/01/2023,125000\n"
            "02/01/2023,340500\n"
            "03/01/2023,89000",
            language="text"
        )

    st.markdown("---")

    # ── Step 2: Upload ────────────────────────────────────────────
    st.markdown("### Step 2 — Upload your CSV file(s)")
    files = st.file_uploader(
        "Upload one or more cleaned CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Multiple files are merged automatically (e.g. separate year files)",
    )

    if not files:
        st.info("Waiting for file upload…")
        st.stop()

    # Read raw files
    raw_frames = []
    for f in files:
        raw_frames.append(pd.read_csv(f, low_memory=False))
        raw_frames[-1].columns = raw_frames[-1].columns.str.strip()
    raw_df = pd.concat(raw_frames, ignore_index=True)

    st.markdown(
        f"**{len(files)} file(s) uploaded** — {len(raw_df):,} total rows, "
        f"{len(raw_df.columns)} columns detected: `{'`, `'.join(raw_df.columns.tolist())}`"
    )

    # ── Step 3: Column mapping ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3 — Map your columns")

    # Check if mandatory cols already present
    already_ok = all(c in raw_df.columns for c in MANDATORY_COLS)
    if already_ok:
        st.success(
            "Both required columns (`orderDate` and `final_amount`) were found automatically. "
            "You can adjust the mapping below if needed."
        )
    else:
        st.markdown(
            '<div class="warn-box">One or more required columns were not found automatically. '
            "Use the dropdowns below to select which column in your file corresponds to each "
            "required field.</div>",
            unsafe_allow_html=True,
        )

    mapping = map_columns(raw_df)

    # Validate
    missing = [k for k, v in mapping.items() if v not in raw_df.columns]
    if missing:
        st.error(f"Mapped columns not found in file: {missing}. Check your mapping.")
        st.stop()

    # ── Step 4: Run pipeline ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 4 — Run the forecasting pipeline")
    run_btn = st.button("▶  Run forecast", type="primary", use_container_width=True)

    if not run_btn:
        st.stop()

    # ── Progress ──────────────────────────────────────────────────
    prog = st.progress(0, text="Starting…")

    prog.progress(10, text="Step 1 / 6 — Applying column mapping and cleaning data…")
    st.markdown('<div class="step-box">Renaming columns, stripping whitespace, parsing dates (DD/MM/YYYY), converting numeric strings…</div>', unsafe_allow_html=True)
    df_clean = apply_mapping_and_clean(raw_df.copy(), mapping)

    missing_mandatory = [c for c in MANDATORY_COLS if c not in df_clean.columns]
    if missing_mandatory:
        st.error(f"Mandatory columns still missing after mapping: {missing_mandatory}")
        st.stop()
    if len(df_clean) == 0:
        st.error("No valid rows remain after cleaning. Check that orderDate is in DD/MM/YYYY format.")
        st.stop()

    st.success(f"✔ Cleaning complete — {len(df_clean):,} valid rows retained")

    prog.progress(25, text="Step 2 / 6 — Aggregating to daily total sales…")
    st.markdown('<div class="step-box">Grouping all transaction rows by date and summing final_amount to produce one total sales figure per day. Removing zero-sales and bottom 1% outlier days…</div>', unsafe_allow_html=True)
    daily = build_daily(df_clean)

    if len(daily) < 70:
        st.error(f"Only {len(daily)} trading days after cleaning. Need at least 70.")
        st.stop()

    mean_sales = float(daily["total_sales"].mean())
    st.success(
        f"✔ {len(daily):,} trading days | "
        f"{daily['date'].min().date()} → {daily['date'].max().date()} | "
        f"Mean daily sales: ₦{mean_sales:,.0f}"
    )

    # Graph 1: Daily series
    st.markdown("**Daily Sales Time Series**")
    fig1 = plot_daily_series(daily)
    st.pyplot(fig1); plt.close()
    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> The top panel shows raw daily total sales. "
        "The bottom panel overlays a 30-day rolling average (red line) to reveal the "
        "underlying trend while smoothing out day-to-day volatility. "
        "Wide swings between high and low days are normal for B2B e-commerce where "
        "bulk orders from large retailers create irregular spikes."
        "</div>",
        unsafe_allow_html=True,
    )

    prog.progress(45, text="Step 3 / 6 — Normalising and creating sequences…")
    st.markdown('<div class="step-box">Applying Min-Max scaling (fitted on training portion only) to compress all values to [0, 1]. Building 60-day sliding window sequences as model inputs…</div>', unsafe_allow_html=True)

    sales  = daily["total_sales"].values.reshape(-1, 1)
    n      = len(sales)
    t_end  = int(n * 0.70)
    v_end  = int(n * 0.85)
    scaled = scaler.transform(sales)
    X_all, y_all = make_sequences(scaled)
    adj_t = t_end - 60
    adj_v = v_end - 60
    X_tr, y_tr = X_all[:adj_t], y_all[:adj_t]
    X_va, y_va = X_all[adj_t:adj_v], y_all[adj_t:adj_v]
    X_te, y_te = X_all[adj_v:], y_all[adj_v:]
    X_te = X_te.reshape(*X_te.shape, 1)

    st.success(
        f"✔ Sequences ready — Training: {len(X_tr)} | Validation: {len(X_va)} | Test: {len(X_te)}"
    )

    prog.progress(60, text="Step 4 / 6 — Running model inference on test set…")
    st.markdown('<div class="step-box">Feeding the test sequences through the trained LSTM model to generate one-step-ahead sales predictions…</div>', unsafe_allow_html=True)

    yp_s    = model.predict(X_te.reshape(len(X_te), 60, 1), verbose=0)
    y_pred  = scaler.inverse_transform(yp_s).flatten()
    y_actual= scaler.inverse_transform(y_te.reshape(-1,1)).flatten()
    dates   = daily["date"].values[v_end + 60 - (n - len(y_te)):]
    # safe date alignment
    dates   = daily["date"].values[len(daily) - len(y_pred):]

    st.success(f"✔ {len(y_pred)} predictions generated")

    prog.progress(75, text="Step 5 / 6 — Computing evaluation metrics…")
    st.markdown('<div class="step-box">Calculating RMSE, MAE, MAPE, and sMAPE on the inverse-transformed (₦) predictions…</div>', unsafe_allow_html=True)

    rmse_v  = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae_v   = float(mean_absolute_error(y_actual, y_pred))
    smape_v = calc_smape(y_actual, y_pred)
    mask    = y_actual > mean_sales * 0.01
    mape_v  = float(np.mean(np.abs((y_actual[mask]-y_pred[mask])/y_actual[mask]))*100)
    n_mape  = int(mask.sum())

    st.markdown("---")
    st.markdown("### Results")
    st.markdown("#### Evaluation metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",  f"₦{rmse_v:,.0f}")
    m2.metric("MAE",   f"₦{mae_v:,.0f}")
    m3.metric("sMAPE", f"{smape_v:.2f}%")
    m4.metric("MAPE",  f"{mape_v:.2f}%", help=f"Evaluated on {n_mape}/{len(y_pred)} days")

    explain_metrics(rmse_v, mae_v, smape_v, mape_v, mean_sales, len(y_pred), n_mape)

    prog.progress(90, text="Step 6 / 6 — Generating forecast plots…")

    # Graph 2: Forecast vs actual
    st.markdown("#### Forecast vs Actual Sales")
    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_actual, color="#1565C0", linewidth=1.0, label="Actual Sales", alpha=0.85)
    ax.plot(dates, y_pred,   color="#E53935", linewidth=1.2, linestyle="--", label="LSTM Forecast")
    ax.fill_between(dates,
                    np.minimum(y_actual, y_pred),
                    np.maximum(y_actual, y_pred),
                    alpha=0.10, color="#E53935", label="Error Band")
    ax.set_title("LSTM Forecast vs Actual Daily Sales", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Sales (₦)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> The blue line shows the actual recorded daily sales. "
        "The red dashed line is the LSTM model's prediction for each day. "
        "The shaded pink region is the error band — the gap between what the model predicted "
        "and what actually happened. A narrower band means better accuracy. "
        "Notice that the model captures the general trend direction well but smooths out "
        "the sharp single-day spikes — this is expected for a model that only sees past sales "
        "and has no information about upcoming promotions or bulk orders."
        "</div>",
        unsafe_allow_html=True,
    )
    plt.close()

    # Graph 3: Error distribution
    st.markdown("#### Daily Prediction Error Distribution")
    errors = y_actual - y_pred
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))
    axes3[0].plot(dates, errors, color="#534AB7", linewidth=0.8, alpha=0.8)
    axes3[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes3[0].fill_between(dates, 0, errors,
                           where=errors > 0, alpha=0.3, color="#E53935", label="Under-forecast")
    axes3[0].fill_between(dates, 0, errors,
                           where=errors < 0, alpha=0.3, color="#1565C0", label="Over-forecast")
    axes3[0].set_title("Prediction Error Over Time  (Actual − Predicted)", fontsize=11, fontweight="bold")
    axes3[0].set_ylabel("Error (₦)")
    axes3[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes3[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes3[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes3[0].legend(); axes3[0].grid(True, alpha=0.3)

    axes3[1].hist(errors, bins=30, color="#7F77DD", edgecolor="white", alpha=0.85)
    axes3[1].axvline(0, color="black", linewidth=1.0, linestyle="--", label="Zero error")
    axes3[1].axvline(float(np.mean(errors)), color="#E53935", linewidth=1.2,
                     linestyle="-", label=f"Mean error: ₦{np.mean(errors):,.0f}")
    axes3[1].set_title("Distribution of Prediction Errors", fontsize=11, fontweight="bold")
    axes3[1].set_xlabel("Error (₦)"); axes3[1].set_ylabel("Number of days")
    axes3[1].legend(); axes3[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading these charts:</strong> The left panel shows how the prediction error "
        "(actual minus predicted) varies day by day. Red regions are days where the model "
        "<em>under-forecast</em> (actual was higher than predicted); blue regions are days where "
        "it <em>over-forecast</em>. The right panel is a histogram — if it is roughly centred "
        "on zero, the model has no systematic bias. A histogram skewed to the right means "
        "the model tends to under-predict; skewed to the left means it tends to over-predict."
        "</div>",
        unsafe_allow_html=True,
    )
    plt.close()

    prog.progress(100, text="Complete!")
    st.success("Pipeline complete.")

    # Download
    out_df = pd.DataFrame({
        "date":                pd.to_datetime(dates).strftime("%d/%m/%Y"),
        "actual_sales_ngn":    np.round(y_actual, 2),
        "predicted_sales_ngn": np.round(y_pred,   2),
        "error_ngn":           np.round(errors,    2),
    })
    st.download_button(
        "⬇  Download forecast results as CSV",
        out_df.to_csv(index=False).encode(),
        "forecast_results.csv", "text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: FUTURE PREDICTIONS
# ══════════════════════════════════════════════════════════════════
elif page == "🔮 Future Predictions":
    st.title("Future Sales Forecast")
    st.markdown(
        "Upload your most recent sales data. The model uses the last 60 trading days "
        "as input to generate forward-looking predictions."
    )

    if model is None:
        st.error("Model not loaded. Run lstm_forecasting.py first.")
        st.stop()

    with st.expander("📋  Column requirements", expanded=False):
        show_column_guide()

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Upload your most recent sales CSV", type=["csv"],
                              help="Must contain at least 60 trading days")
    with c2:
        n = st.selectbox("Forecast horizon (days)", [7, 14, 30], index=1)

    if not up:
        st.info("Waiting for file upload…")
        st.stop()

    raw = pd.read_csv(up, low_memory=False)
    raw.columns = raw.columns.str.strip()

    st.markdown("#### Map your columns")
    mapping = map_columns(raw)
    run_btn = st.button("▶  Generate forecast", type="primary", use_container_width=True)
    if not run_btn:
        st.stop()

    prog = st.progress(0, text="Preparing data…")
    prog.progress(20, text="Cleaning and aggregating…")
    df_c  = apply_mapping_and_clean(raw.copy(), mapping)
    daily = build_daily(df_c)

    if len(daily) < 60:
        st.error(f"Need at least 60 trading days. Found {len(daily)}.")
        st.stop()

    st.success(
        f"✔ {len(daily):,} trading days | "
        f"{daily['date'].min().date()} → {daily['date'].max().date()}"
    )

    prog.progress(50, text="Running forecast…")
    last60    = daily["total_sales"].values[-60:]
    last_date = daily["date"].iloc[-1]
    preds     = future_forecast(model, scaler, last60, n)

    fut = []
    cur = last_date
    while len(fut) < n:
        cur += pd.Timedelta(days=1)
        fut.append(cur)

    fdf = pd.DataFrame({"date": fut, "forecast_ngn": np.round(preds[:len(fut)], 2)})

    prog.progress(80, text="Building charts…")

    st.markdown("---")
    st.markdown("### Forecast results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total forecast", f"₦{preds.sum():,.0f}")
    c2.metric("Daily average",  f"₦{preds.mean():,.0f}")
    c3.metric("Peak day",       f"₦{preds.max():,.0f}")

    # Chart: history + forecast
    fig, ax = plt.subplots(figsize=(14, 5))
    hist = daily.tail(90)
    ax.plot(hist["date"], hist["total_sales"],
            color="#1565C0", linewidth=1.0, label="Historical Sales", alpha=0.8)
    ax.plot(fdf["date"], fdf["forecast_ngn"],
            color="#E53935", linewidth=2.0, linestyle="--",
            marker="o", markersize=4, label=f"{n}-Day Forecast")
    ax.axvspan(fut[0], fut[-1], alpha=0.06, color="#E53935")
    ax.axvline(last_date, color="gray", linewidth=1.0, linestyle=":",
               label="Forecast start")
    ax.set_title(f"LSTM {n}-Day Demand Forecast", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Sales (₦)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> The blue line shows the last 90 days of actual "
        "historical sales. The vertical dotted grey line marks where history ends and the "
        "forecast begins. The red dashed line and dots show the model's predicted daily sales "
        "for each of the next {n} days. The model generates these by repeatedly predicting "
        "one day ahead, feeding each prediction back as input for the next step. "
        "Note that uncertainty grows with each step further into the future."
        "</div>".format(n=n),
        unsafe_allow_html=True,
    )

    # Forecast bar chart
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(
        [d.strftime("%d %b") for d in fut],
        fdf["forecast_ngn"],
        color="#7F77DD", edgecolor="white"
    )
    ax2.set_title(f"{n}-Day Daily Forecast Breakdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Forecast Sales (₦)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

    # Table
    show = fdf.copy()
    show["date"]         = show["date"].dt.strftime("%A, %d %b %Y")
    show["forecast_ngn"] = show["forecast_ngn"].apply(lambda v: f"₦{v:,.0f}")
    show.columns = ["Date", "Forecast (₦)"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    prog.progress(100, text="Done!")
    st.download_button(
        "⬇  Download forecast CSV",
        fdf.assign(date=fdf["date"].dt.strftime("%d/%m/%Y")).to_csv(index=False).encode(),
        f"forecast_{n}day.csv", "text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS (all 4 graphs from original training run)
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Training Results":
    st.title("Training Results")
    st.markdown(
        "All graphs and metrics from the original model training run on the full "
        "2023–2025 dataset. Graphs are loaded from the `outputs/` folder and update "
        "automatically whenever `lstm_forecasting.py` is re-run."
    )

    # ── Graph 1: Daily sales ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### Graph 1 — Daily total sales time series")
    p = os.path.join(OUT_DIR, "01_daily_sales_timeseries.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True)
        st.markdown(
            '<div class="explain-box">'
            "<strong>What this shows:</strong> The top panel plots raw daily total sales for the "
            "entire study period (Jan 2023 – Nov 2025). The bottom panel overlays a 30-day "
            "rolling average (red line) to expose the underlying demand trend. "
            "Sales grew from early 2023, peaked around late 2024, and declined toward "
            "November 2025. This declining tail formed the test period and represents the "
            "most challenging forecasting scenario for the model. "
            "The high day-to-day volatility throughout the series reflects the irregular "
            "bulk ordering patterns typical of B2B e-commerce platforms."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Graph not found. Run `python lstm_forecasting.py` to generate it.")

    # ── Graph 2: Training loss ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Graph 2 — LSTM training vs validation loss")
    p = os.path.join(OUT_DIR, "02_training_loss.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True)
        st.markdown(
            '<div class="explain-box">'
            "<strong>What this shows:</strong> Each point on the blue line is the model's "
            "training loss (Mean Squared Error on normalised values) at the end of that epoch. "
            "The red line is the validation loss — how well the model performed on data it "
            "did not train on. "
            "Both losses drop steeply in the first few epochs and then level off, "
            "running close together without the validation loss rising away from training loss. "
            "This pattern confirms <em>clean convergence with no overfitting</em>: the model "
            "learned real patterns from the training data rather than memorising it. "
            "Early stopping fired after the validation loss showed no further improvement, "
            "and the best weights were automatically restored."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Graph not found. Run `python lstm_forecasting.py` to generate it.")

    # ── Graph 3: Forecast vs actual ───────────────────────────────
    st.markdown("---")
    st.markdown("### Graph 3 — LSTM forecast vs actual sales (test period)")
    p = os.path.join(OUT_DIR, "03_predictions_vs_actual.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True)
        st.markdown(
            '<div class="explain-box">'
            "<strong>What this shows:</strong> The blue line is the actual daily total sales "
            "recorded during the held-out test period (approximately Jun – Nov 2025). "
            "The red dashed line is the LSTM model's prediction for each of those days, "
            "generated without the model ever having seen that period during training. "
            "The shaded pink region is the error band between forecast and actual. "
            "<br><br>"
            "The model correctly captures the general <em>downward trend</em> of the test "
            "period. However, it does not replicate the sharp individual-day spikes — "
            "this is expected behaviour for a univariate model that has no access to "
            "external information such as promotions, restocking events, or seasonal campaigns. "
            "For strategic planning purposes (weekly or monthly demand estimates), "
            "the trend-level accuracy is the most operationally relevant output."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Graph not found. Run `python lstm_forecasting.py` to generate it.")

    # ── Graph 4: Model comparison ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Graph 4 — Model performance metrics")
    p = os.path.join(OUT_DIR, "04_model_comparison.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True)

        if num_df is not None:
            lstm_row = num_df[num_df["Model"] == "Vanilla LSTM"]
            if not lstm_row.empty:
                l = lstm_row.iloc[0]
                st.markdown(
                    '<div class="explain-box">'
                    "<strong>What this shows:</strong> Bar charts comparing the four evaluation "
                    "metrics across the models evaluated. Lower values on all metrics indicate "
                    "better performance."
                    "<br><br>"
                    f"<strong>RMSE (₦{l['RMSE']:,.0f}):</strong> The root mean squared error on the "
                    "original Naira scale. Larger errors are penalised disproportionately, making "
                    "RMSE sensitive to the biggest prediction misses."
                    "<br><br>"
                    f"<strong>MAE (₦{l['MAE']:,.0f}):</strong> The average absolute daily error. "
                    "Every day's error is weighted equally, giving a robust measure of "
                    "typical accuracy."
                    "<br><br>"
                    f"<strong>MAPE ({l['MAPE']:.2f}%):</strong> The mean absolute percentage error. "
                    "Elevated values here are caused by near-zero actual sales on low-volume "
                    "days where percentage errors become very large. This metric should be "
                    "interpreted alongside sMAPE."
                    "<br><br>"
                    f"<strong>sMAPE ({l['sMAPE']:.2f}%):</strong> The symmetric MAPE — the primary "
                    "percentage metric for this project. It bounds the error between 0 and 200% "
                    "and is not distorted by near-zero actual values, making it the most "
                    "reliable single-figure summary of forecast accuracy."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Run lstm_forecasting.py to load live metric values.")
    else:
        st.warning("Graph not found. Run `python lstm_forecasting.py` to generate it.")

    # ── Live metrics table ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Live metrics table")
    if disp_df is not None:
        st.success("Loaded from `outputs/model_comparison.csv` — updates automatically on re-run.")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
    else:
        st.warning("model_comparison.csv not found. Run lstm_forecasting.py to generate it.")

    # ── Training configuration ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Model training configuration")
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
| Input shape | (60, 1) |
        """)
    with c2:
        st.markdown("""
| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Loss function | MSE |
| Max epochs | 100 |
| Early stopping patience | 20 |
| Min delta | 0.0001 |
| LR reduction patience | 8 |
        """)


# ══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown("""
### Design and Development of an LSTM-Based Product Demand Forecasting System
**Postgraduate Diploma Research** — Nigerian B2B E-Commerce

---
### Research objective
Design, implement, and evaluate a vanilla Long Short-Term Memory (LSTM) neural network
for forecasting daily product demand from historical transactional sales data, applied to
a Nigerian B2B e-commerce platform, and deploy the model as an interactive web application.

---
### Dataset
| Property | Detail |
|---|---|
| Company | Nigerian B2B e-commerce (anonymous) |
| Period | January 2023 – November 2025 |
| Raw transactions | 2,381,940 rows across 5 CSV files |
| Trading days used | 982 (after cleaning and outlier removal) |
| Date format in source | DD/MM/YYYY |

---
### Model architecture
| Layer | Type | Detail |
|---|---|---|
| 1 | LSTM | 64 units · return_sequences=False |
| 2 | Dropout | Rate = 0.2 |
| 3 | Dense | 1 neuron · linear activation |
| — | Total params | 16,961 (66.25 KB) |

**Training:** Adam optimiser · MSE loss · Early stopping (patience 20) · LR reduction on plateau

---
### Technologies
| Tool | Version / Role |
|---|---|
| Python | 3.11.9 |
| TensorFlow / Keras | 2.21.0 |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | MinMaxScaler |
| Matplotlib | All charts |
| Streamlit | This web application |

---
### Required folder structure
```
project/
├── app.py
├── lstm_forecasting.py
├── model/
│   ├── lstm_demand_forecast.keras
│   └── scaler.pkl
└── outputs/
    ├── model_comparison.csv
    ├── 01_daily_sales_timeseries.png
    ├── 02_training_loss.png
    ├── 03_predictions_vs_actual.png
    └── 04_model_comparison.png
```

### Run locally
```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib
streamlit run app.py
```
    """)
