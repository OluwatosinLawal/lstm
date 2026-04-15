# =============================================================================
#  LSTM DEMAND FORECASTING — STREAMLIT APP  (Full Version v4)
#  Run : streamlit run app.py
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
.explain-box strong { color: #b8ffcc !important; }
.explain-box em { color: #a8f0b8 !important; font-style: italic; }

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

.acc-excellent { background:#1a3d2b; color:#b8ffcc; padding:4px 10px; border-radius:4px; font-weight:bold; }
.acc-good      { background:#1a2e3d; color:#a8d5f5; padding:4px 10px; border-radius:4px; font-weight:bold; }
.acc-fair      { background:#3a3010; color:#f5e0a0; padding:4px 10px; border-radius:4px; font-weight:bold; }
.acc-poor      { background:#3d1a1a; color:#f5b0b0; padding:4px 10px; border-radius:4px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "lstm_demand_forecast.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")

LOOK_BACK = 60

# Accuracy scale from supervisor
def accuracy_label(smape_val):
    if smape_val < 10:
        return "Excellent / Highly Accurate", "acc-excellent"
    elif smape_val < 25:
        return "Good", "acc-good"
    elif smape_val < 50:
        return "Reasonable / Fair", "acc-fair"
    else:
        return "Inaccurate", "acc-poor"


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
        model = load_model(MODEL_PATH)
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
        return None, None


def read_and_merge_files(uploaded_files):
    """
    Reads multiple CSV files and merges them into one DataFrame.
    Handles files with the same columns regardless of order.
    """
    frames = []
    for f in uploaded_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    return merged


def clean_dataframe(df, date_col, amount_col, qty_col=None,
                    product_col=None, product_id_col=None):
    """
    Clean the dataframe: strip whitespace, parse dates, convert numerics.
    Returns cleaned df.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Numeric columns
    for col in [amount_col, qty_col, "unitPrice", "orderTotal"]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",","",regex=False), errors="coerce"
            )

    # Date parsing (dayfirst for DD/MM/YYYY)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    return df


def get_product_categories(df, product_id_col):
    """
    Extract product categories from productId prefix e.g. NGA-FDI, NGA-BEV.
    Returns a dict: {category_code: full_label}
    """
    if product_id_col not in df.columns:
        return {}
    prefixes = df[product_id_col].dropna().str.extract(r"NGA-([A-Z]+)-")[0].dropna().unique()
    label_map = {
        "FDI": "Food Items (FDI)",
        "BEV": "Beverages (BEV)",
        "HME": "Home & Household (HME)",
        "PRF": "Personal & Cooking (PRF)",
        "PHA": "Pharmacy (PHA)",
        "AGR": "Agriculture (AGR)",
        "OTH": "Other",
    }
    return {p: label_map.get(p, f"Category {p}") for p in sorted(prefixes)}


def filter_by_category(df, product_id_col, category_code):
    if category_code == "ALL":
        return df
    mask = df[product_id_col].str.contains(f"NGA-{category_code}-", na=False)
    return df[mask]


def aggregate_series(df, date_col, value_col, freq):
    """
    Aggregate to daily ('D') or monthly ('MS') frequency.
    Returns a clean (date, total_value) DataFrame.
    """
    series = (
        df.groupby(date_col)[value_col].sum()
        .reset_index()
        .rename(columns={date_col: "date", value_col: "total"})
        .sort_values("date")
    )
    # Resample to chosen frequency
    series = (
        series.set_index("date")
        .resample(freq)["total"].sum()
        .reset_index()
    )
    series = series[series["total"] > 0].reset_index(drop=True)
    # Remove bottom 1% outliers
    thresh = series["total"].quantile(0.01)
    series = series[series["total"] > thresh].reset_index(drop=True)
    return series


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


def calc_mape(y_true, y_pred, mean_val):
    mask = y_true > mean_val * 0.01
    if mask.sum() == 0:
        return np.nan, 0
    m = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    return m, int(mask.sum())


def rolling_forecast(model, scaler, seed_window, n_steps):
    """
    Generates n_steps future predictions using rolling one-step-ahead inference.
    seed_window: 1D array of raw (un-normalised) values, length >= LOOK_BACK.
    """
    s = scaler.transform(seed_window[-LOOK_BACK:].reshape(-1, 1)).flatten()
    window = list(s)
    out = []
    for _ in range(n_steps):
        x = np.array(window[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
        p = model.predict(x, verbose=0)[0, 0]
        out.append(p)
        window.append(p)
    raw = scaler.inverse_transform(np.array(out).reshape(-1, 1)).flatten()
    return np.maximum(raw, 0)


def show_column_guide():
    st.markdown("""
| Column name | Status | Description |
|---|---|---|
| `orderDate` | **Required** | Order date — format DD/MM/YYYY (e.g. `05/01/2023`) |
| `final_amount` | **Required*** | Sales value per order line (₦). Plain numbers only — no commas or symbols. |
| `quantitySold` | **Required*** | Units sold per order line. Required if forecasting quantity. |
| `displayTitle` | Recommended | Product name — used for product-level filtering |
| `productId` | Recommended | SKU / product ID — used to extract categories (e.g. NGA-FDI-...) |
| `salesCategory` | Optional | Label: Regular Sales or Promo Sales |
| `orderTotal` | Optional | Total order value including all items |

> \* At least one of `final_amount` or `quantitySold` must be present depending on your forecast target.  
> Column names are **case-sensitive** and must match exactly.  
> Multiple files with the same columns are merged automatically.
""")


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
        "✅ Forecast vs Actual",
        "📊 Training Results",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Model status**")
    if model is not None:
        st.success("Model loaded ✓")
        st.caption("Vanilla LSTM · 64 units · 60-day window")
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
        ["Vanilla LSTM", "Jan 2023 – Nov 2025", "60 periods", smape_val]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Accuracy scale reference")
    st.markdown("""
| MAPE | sMAPE | Rating |
|---|---|---|
| < 10% | < 10% | Excellent / Highly Accurate |
| 10% – 20% | 10% – 25% | Good |
| 20% – 50% | 25% – 50% | Reasonable / Fair |
| > 50% | > 50% | Inaccurate |
""")

    st.markdown("---")
    st.markdown("### Pages")
    a, b, c, d, e = st.columns(5)
    with a:
        st.markdown("**📂 Upload & Forecast**")
        st.write("Evaluate the model against actual historical data. Supports product category filtering, daily or monthly aggregation, and sales amount or quantity targets.")
    with b:
        st.markdown("**🔮 Future Predictions**")
        st.write("Generate forward forecasts for a custom date range or number of periods. Select product category, aggregation level, and forecast target.")
    with c:
        st.markdown("**✅ Forecast vs Actual**")
        st.write("Upload a forecast file and actual data file. The app compares them side by side with MAPE, sMAPE, and an accuracy rating per product/period.")
    with d:
        st.markdown("**📊 Training Results**")
        st.write("All four training graphs from the original model run with plain-language explanations.")
    with e:
        st.markdown("**ℹ️ About**")
        st.write("Project background, model architecture, and data format guide.")

    if model is None:
        st.error(f"Model not loaded: {load_err}")
        st.info("Run `python lstm_forecasting.py` then refresh this page.")


# ══════════════════════════════════════════════════════════════════
# SHARED UPLOAD + CONFIG WIDGET (used by Upload & Forecast and Future Predictions)
# ══════════════════════════════════════════════════════════════════
def upload_and_configure(page_key):
    """
    Renders: multi-file uploader, column mapper, aggregation level,
    forecast target, and product/category filter.
    Returns: (daily_series, config_dict) or (None, None) if not ready.
    """
    with st.expander("📋  Column requirements", expanded=False):
        show_column_guide()

    st.markdown("### Step 1 — Upload CSV file(s)")
    st.markdown(
        "You can upload **multiple CSV files** at once (e.g. separate year files). "
        "They will be merged automatically provided they have the same columns."
    )
    files = st.file_uploader(
        "Upload one or more sales CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key=f"upload_{page_key}",
        help="Multiple files are stacked row-by-row after upload"
    )
    if not files:
        st.info("Waiting for file upload…")
        return None, None

    raw = read_and_merge_files(files)
    if raw is None or len(raw) == 0:
        st.error("No data found in uploaded files.")
        return None, None

    st.success(f"✔ {len(files)} file(s) merged — {len(raw):,} total rows | Columns: `{'`, `'.join(raw.columns.tolist())}`")

    # ── Column mapping ────────────────────────────────────────────
    st.markdown("### Step 2 — Map your columns")
    csv_cols = list(raw.columns)

    def pick(label, hint_names, key):
        default = next((c for c in hint_names if c in csv_cols), csv_cols[0])
        idx = csv_cols.index(default)
        return st.selectbox(label, csv_cols, index=idx, key=f"{page_key}_{key}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        date_col = pick("Date column", ["orderDate"], "date")
    with c2:
        amt_col  = pick("Sales amount column", ["final_amount"], "amt")
    with c3:
        qty_col  = pick("Quantity column", ["quantitySold"], "qty")
    with c4:
        prod_col = pick("Product name column", ["displayTitle"], "prod")

    pid_col = pick("Product ID column (for categories)", ["productId"], "pid")

    # ── Forecast configuration ────────────────────────────────────
    st.markdown("### Step 3 — Forecast configuration")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        freq_label = st.selectbox(
            "Aggregation level",
            ["Daily", "Monthly"],
            key=f"{page_key}_freq",
            help="Monthly is more stable and aligns with business planning cycles."
        )
        freq = "D" if freq_label == "Daily" else "MS"

    with fc2:
        target_label = st.selectbox(
            "Forecast target",
            ["Sales Amount (₦)", "Sales Quantity (units)"],
            key=f"{page_key}_target"
        )
        value_col = amt_col if "Amount" in target_label else qty_col
        unit      = "₦" if "Amount" in target_label else "units"

    with fc3:
        # Category filter
        cats = get_product_categories(raw, pid_col)
        cat_options = {"All Products": "ALL"} | cats
        cat_label = st.selectbox(
            "Product category",
            list(cat_options.keys()),
            key=f"{page_key}_cat",
            help="Filter by product category extracted from productId prefix (e.g. NGA-FDI = Food Items)."
        )
        cat_code = cat_options[cat_label]

    # ── Build series ──────────────────────────────────────────────
    if st.button("▶  Load & process data", key=f"{page_key}_load", type="primary"):
        with st.spinner("Cleaning and aggregating data…"):
            df_clean = clean_dataframe(raw, date_col, value_col, qty_col, prod_col, pid_col)

            if cat_code != "ALL":
                df_clean = filter_by_category(df_clean, pid_col, cat_code)
                if len(df_clean) == 0:
                    st.error(f"No rows found for category '{cat_label}'. Try a different filter.")
                    return None, None

            series = aggregate_series(df_clean, date_col, value_col, freq)

            if len(series) < LOOK_BACK + 5:
                st.error(
                    f"Only {len(series)} periods after aggregation. "
                    f"Need at least {LOOK_BACK + 5} to run the model. "
                    "Try switching to Daily aggregation or uploading more data."
                )
                return None, None

        st.session_state[f"{page_key}_series"]  = series
        st.session_state[f"{page_key}_unit"]    = unit
        st.session_state[f"{page_key}_freq"]    = freq
        st.session_state[f"{page_key}_flabel"]  = freq_label
        st.session_state[f"{page_key}_cat"]     = cat_label
        st.session_state[f"{page_key}_target"]  = target_label
        st.session_state[f"{page_key}_df"]      = df_clean
        st.session_state[f"{page_key}_vcol"]    = value_col
        st.success(
            f"✔ {len(series)} {freq_label.lower()} periods | "
            f"{series['date'].min().date()} → {series['date'].max().date()} | "
            f"Mean: {unit}{'₦' if unit=='₦' else ''}{series['total'].mean():,.0f}"
        )

    if f"{page_key}_series" not in st.session_state:
        return None, None

    series = st.session_state[f"{page_key}_series"]
    cfg = {
        "unit":    st.session_state[f"{page_key}_unit"],
        "freq":    st.session_state[f"{page_key}_freq"],
        "flabel":  st.session_state[f"{page_key}_flabel"],
        "cat":     st.session_state[f"{page_key}_cat"],
        "target":  st.session_state[f"{page_key}_target"],
        "df":      st.session_state[f"{page_key}_df"],
        "vcol":    st.session_state[f"{page_key}_vcol"],
    }
    return series, cfg


# ══════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & FORECAST
# ══════════════════════════════════════════════════════════════════
elif page == "📂 Upload & Forecast":
    st.title("Upload Data & Evaluate Forecast")
    st.markdown(
        "Upload historical sales data and evaluate how well the LSTM model forecasts it. "
        "Supports filtering by product category, daily or monthly aggregation, "
        "and sales amount or quantity as the forecast target."
    )

    if model is None:
        st.error("Model not loaded. Run `python lstm_forecasting.py` first.")
        st.stop()

    series, cfg = upload_and_configure("uf")
    if series is None:
        st.stop()

    unit   = cfg["unit"]
    flabel = cfg["flabel"]

    st.markdown("---")
    st.markdown("### Step 4 — Run evaluation")

    n_total = len(series)
    t_end   = int(n_total * 0.70)
    v_end   = int(n_total * 0.85)

    # Normalise
    sales = series["total"].values.reshape(-1, 1)
    scaler_local = __import__("sklearn.preprocessing", fromlist=["MinMaxScaler"]).MinMaxScaler()
    scaler_local.fit(sales[:t_end])
    scaled = scaler_local.transform(sales)

    X_all, y_all = make_sequences(scaled, LOOK_BACK)
    adj_t = t_end - LOOK_BACK
    adj_v = v_end - LOOK_BACK
    X_te  = X_all[adj_v:].reshape(-1, LOOK_BACK, 1)
    y_te  = y_all[adj_v:]

    with st.spinner("Running model inference…"):
        yp_s   = model.predict(X_te, verbose=0)
    y_pred   = scaler_local.inverse_transform(yp_s).flatten()
    y_actual = scaler_local.inverse_transform(y_te.reshape(-1,1)).flatten()
    dates    = series["date"].values[len(series)-len(y_pred):]

    mean_val = float(series["total"].mean())
    rmse_v   = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae_v    = float(mean_absolute_error(y_actual, y_pred))
    smape_v  = calc_smape(y_actual, y_pred)
    mape_v, n_mape = calc_mape(y_actual, y_pred, mean_val)
    acc_label, acc_cls = accuracy_label(smape_v)

    st.markdown("### Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",   f"{unit}{rmse_v:,.0f}")
    m2.metric("MAE",    f"{unit}{mae_v:,.0f}")
    m3.metric("sMAPE",  f"{smape_v:.2f}%")
    m4.metric("MAPE",   f"{mape_v:.2f}%" if not np.isnan(mape_v) else "N/A")
    m5.metric("Rating", acc_label)

    st.markdown(
        f'<div class="explain-box">'
        f"<strong>Accuracy rating: <span class='{acc_cls}'>{acc_label}</span></strong><br><br>"
        f"<strong>RMSE ({unit}{rmse_v:,.0f}):</strong> Average prediction error penalising large misses more heavily. "
        f"This is {rmse_v/mean_val*100:.1f}% of the mean {flabel.lower()} {cfg['target'].lower()}.<br><br>"
        f"<strong>MAE ({unit}{mae_v:,.0f}):</strong> Average absolute error per {flabel.lower()} period ({mae_v/mean_val*100:.1f}% of mean).<br><br>"
        f"<strong>sMAPE ({smape_v:.2f}%):</strong> Primary percentage metric. Bounded 0–200%, robust to near-zero values. "
        f"Rated <em>{acc_label}</em> on the accuracy scale.<br><br>"
        f"<strong>MAPE ({mape_v:.2f}%):</strong> Evaluated on {n_mape}/{len(y_pred)} periods "
        f"(excluded near-zero periods to avoid division errors)."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Plot 1: full series
    st.markdown("#### Full series — daily/monthly sales")
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(series["date"], series["total"], color="#1565C0", linewidth=0.8, alpha=0.8)
    roll = series["total"].rolling(window=6 if cfg["freq"]=="MS" else 30, min_periods=1).mean()
    ax1.plot(series["date"], roll, color="#E53935", linewidth=1.5, label="Rolling average")
    ax1.set_title(f"{flabel} {cfg['target']} — {cfg['cat']}", fontsize=12, fontweight="bold")
    ax1.set_ylabel(f"{cfg['target']} ({unit})")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig1); plt.close()

    # Plot 2: forecast vs actual
    st.markdown("#### Forecast vs Actual (test period)")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(pd.to_datetime(dates), y_actual, color="#1565C0", linewidth=1.0, label="Actual")
    ax2.plot(pd.to_datetime(dates), y_pred,   color="#E53935", linewidth=1.2,
             linestyle="--", label="LSTM Forecast")
    ax2.fill_between(pd.to_datetime(dates),
                     np.minimum(y_actual, y_pred), np.maximum(y_actual, y_pred),
                     alpha=0.12, color="#E53935", label="Error Band")
    ax2.set_title("LSTM Forecast vs Actual", fontsize=12, fontweight="bold")
    ax2.set_ylabel(f"{cfg['target']} ({unit})")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading the forecast chart:</strong> The blue line is actual recorded values. "
        "The red dashed line is the model prediction. A narrower pink error band means more "
        "accurate predictions. The model captures the general trend but smooths out "
        "individual-period spikes — expected for a univariate model without promotional signals."
        "</div>", unsafe_allow_html=True
    )

    # Plot 3: error distribution
    st.markdown("#### Prediction error analysis")
    errors = y_actual - y_pred
    fig3, ax3 = plt.subplots(1, 2, figsize=(14, 4))
    ax3[0].bar(range(len(errors)), errors,
               color=["#E53935" if e > 0 else "#1565C0" for e in errors], alpha=0.7)
    ax3[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3[0].set_title("Error per Period (Actual − Predicted)", fontsize=11, fontweight="bold")
    ax3[0].set_ylabel(f"Error ({unit})")
    ax3[0].set_xlabel(f"{flabel} period index")
    ax3[0].grid(True, alpha=0.3)

    ax3[1].hist(errors, bins=25, color="#7F77DD", edgecolor="white", alpha=0.85)
    ax3[1].axvline(0, color="black", linewidth=1.0, linestyle="--", label="Zero error")
    ax3[1].axvline(float(np.mean(errors)), color="#E53935", linewidth=1.2,
                   label=f"Mean: {unit}{np.mean(errors):,.0f}")
    ax3[1].set_title("Error Distribution", fontsize=11, fontweight="bold")
    ax3[1].set_xlabel(f"Error ({unit})"); ax3[1].set_ylabel("Periods")
    ax3[1].legend(); ax3[1].grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Download
    out_df = pd.DataFrame({
        "period":     pd.to_datetime(dates).strftime("%d/%m/%Y" if cfg["freq"]=="D" else "%b %Y"),
        "actual":     np.round(y_actual, 2),
        "predicted":  np.round(y_pred,   2),
        "error":      np.round(errors,    2),
        "smape_pct":  [round(calc_smape(np.array([a]), np.array([p])), 2)
                       for a, p in zip(y_actual, y_pred)],
    })
    out_df.columns = [f"period", f"actual_{unit}", f"predicted_{unit}",
                      f"error_{unit}", "smape_pct"]
    st.download_button(
        "⬇  Download forecast results CSV",
        out_df.to_csv(index=False).encode(),
        "forecast_results.csv", "text/csv", use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: FUTURE PREDICTIONS
# ══════════════════════════════════════════════════════════════════
elif page == "🔮 Future Predictions":
    st.title("Future Sales Forecast")
    st.markdown(
        "Upload recent sales data and generate forward-looking predictions. "
        "Select a specific date range or number of periods, and choose "
        "whether to forecast by product category, daily or monthly, "
        "and for sales amount or quantity."
    )

    if model is None:
        st.error("Model not loaded. Run `python lstm_forecasting.py` first.")
        st.stop()

    series, cfg = upload_and_configure("fp")
    if series is None:
        st.stop()

    unit   = cfg["unit"]
    flabel = cfg["flabel"]
    freq   = cfg["freq"]

    st.markdown("---")
    st.markdown("### Step 4 — Forecast horizon")

    hz1, hz2 = st.columns(2)
    with hz1:
        horizon_type = st.radio(
            "Specify forecast period by:",
            ["Number of periods", "Date range"],
            horizontal=True, key="fp_htype"
        )
    with hz2:
        last_date = pd.Timestamp(series["date"].iloc[-1])
        if horizon_type == "Number of periods":
            n_periods = st.number_input(
                f"Number of {flabel.lower()} periods to forecast",
                min_value=1, max_value=365 if freq=="D" else 36,
                value=30 if freq=="D" else 6, key="fp_n"
            )
            if freq == "D":
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), periods=n_periods, freq="D"
                )
            else:
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1), periods=n_periods, freq="MS"
                )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                start_date = st.date_input(
                    "Forecast start date",
                    value=(last_date + pd.Timedelta(days=1)).date(), key="fp_start"
                )
            with col_b:
                end_date = st.date_input(
                    "Forecast end date",
                    value=(last_date + pd.Timedelta(days=30)).date(), key="fp_end"
                )
            future_dates = pd.date_range(
                start=start_date, end=end_date,
                freq="D" if freq=="D" else "MS"
            )
            n_periods = len(future_dates)

    if n_periods == 0:
        st.warning("No forecast periods selected. Adjust your date range.")
        st.stop()

    if st.button("▶  Generate forecast", type="primary", key="fp_run", use_container_width=True):
        seed = series["total"].values

        # Fit local scaler on full series (we don't have a train/test split here)
        from sklearn.preprocessing import MinMaxScaler as MMS
        sc = MMS(); sc.fit(seed.reshape(-1,1))

        with st.spinner(f"Forecasting {n_periods} {flabel.lower()} periods…"):
            preds = rolling_forecast(model, sc, seed, n_periods)

        fdf = pd.DataFrame({"date": future_dates[:len(preds)], "forecast": np.round(preds[:len(future_dates)], 2)})

        st.markdown("---")
        st.markdown("### Forecast results")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Total ({flabel.lower()})", f"{unit}{preds.sum():,.0f}")
        c2.metric("Average per period",         f"{unit}{preds.mean():,.0f}")
        c3.metric("Peak period",                f"{unit}{preds.max():,.0f}")

        # Chart: history + forecast
        fig, ax = plt.subplots(figsize=(14, 5))
        hist_show = series.tail(12 if freq=="MS" else 90)
        ax.plot(hist_show["date"], hist_show["total"],
                color="#1565C0", linewidth=1.0, label="Historical", alpha=0.8)
        ax.plot(fdf["date"], fdf["forecast"],
                color="#E53935", linewidth=2.0, linestyle="--",
                marker="o", markersize=4, label=f"{n_periods}-period Forecast")
        if len(future_dates) > 0:
            ax.axvspan(future_dates[0], future_dates[-1], alpha=0.06, color="#E53935")
        ax.axvline(last_date, color="gray", linewidth=1.0, linestyle=":", label="Forecast start")
        ax.set_title(
            f"LSTM {n_periods}-{flabel} Forecast — {cfg['cat']} — {cfg['target']}",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylabel(f"{cfg['target']} ({unit})")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Bar breakdown
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.bar(
            fdf["date"].dt.strftime("%b %Y" if freq=="MS" else "%d %b"),
            fdf["forecast"], color="#7F77DD", edgecolor="white"
        )
        ax2.set_title(f"Forecast Breakdown per {flabel} Period", fontsize=11, fontweight="bold")
        ax2.set_ylabel(f"{cfg['target']} ({unit})")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown(
            '<div class="explain-box">'
            "<strong>How to read this forecast:</strong> The blue line shows recent actual sales. "
            "The dotted grey line marks where history ends. "
            "The red dashed line and the bar chart show predicted values for each future period. "
            "<em>Uncertainty increases the further ahead the forecast extends</em> — "
            "treat later periods as directional estimates rather than precise targets."
            "</div>", unsafe_allow_html=True
        )

        # Table
        show = fdf.copy()
        fmt = "%b %Y" if freq=="MS" else "%A, %d %b %Y"
        show["date"]     = show["date"].dt.strftime(fmt)
        show["forecast"] = show["forecast"].apply(lambda v: f"{unit}{v:,.0f}")
        show.columns     = ["Period", f"Forecast ({unit})"]
        st.dataframe(show, use_container_width=True, hide_index=True)

        # Download raw forecast for use in Forecast vs Actual page
        dl = fdf.copy()
        dl["date"] = dl["date"].dt.strftime("%d/%m/%Y")
        dl.columns = ["date", f"forecast_{unit}"]
        st.download_button(
            "⬇  Download forecast CSV  (use this file in Forecast vs Actual)",
            dl.to_csv(index=False).encode(),
            f"forecast_{n_periods}{freq}.csv", "text/csv",
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════
# PAGE: FORECAST VS ACTUAL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "✅ Forecast vs Actual":
    st.title("Forecast vs Actual Comparison")
    st.markdown(
        "Upload the forecast file (generated on the Future Predictions page) "
        "and a file containing the actual sales data for the same period. "
        "The app aligns them by date, calculates accuracy metrics, "
        "and rates the forecast using the accuracy scale."
    )

    st.markdown("---")
    st.markdown("### Accuracy scale reference")
    st.markdown("""
| MAPE | sMAPE | Rating |
|---|---|---|
| < 10% | < 10% | Excellent / Highly Accurate |
| 10% – 20% | 10% – 25% | Good |
| 20% – 50% | 25% – 50% | Reasonable / Fair |
| > 50% | > 50% | Inaccurate |
""")

    st.markdown("---")
    st.markdown("### Step 1 — Upload your files")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Forecast file** (from Future Predictions page)")
        fc_file = st.file_uploader(
            "Upload forecast CSV", type=["csv"], key="fva_fc",
            help="Must have a date column and a forecast column"
        )
    with col2:
        st.markdown("**Actual sales file(s)**")
        act_files = st.file_uploader(
            "Upload actual data CSV(s)", type=["csv"],
            accept_multiple_files=True, key="fva_act",
            help="Same format as your main sales data. Multiple files are merged."
        )

    if not fc_file or not act_files:
        st.info("Upload both files to continue.")
        st.stop()

    # ── Configuration ─────────────────────────────────────────────
    st.markdown("### Step 2 — Configure columns")
    fc_df = pd.read_csv(fc_file)
    fc_df.columns = fc_df.columns.str.strip()

    act_raw = read_and_merge_files(act_files)
    act_raw.columns = act_raw.columns.str.strip()

    fc_cols  = list(fc_df.columns)
    act_cols = list(act_raw.columns)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fc_date_col = st.selectbox("Forecast: date column", fc_cols,
                                   index=fc_cols.index("date") if "date" in fc_cols else 0,
                                   key="fva_fcdate")
    with c2:
        fc_val_col  = st.selectbox("Forecast: value column", fc_cols,
                                   index=1 if len(fc_cols) > 1 else 0, key="fva_fcval")
    with c3:
        act_date_col = st.selectbox("Actual: date column", act_cols,
                                    index=act_cols.index("orderDate") if "orderDate" in act_cols else 0,
                                    key="fva_actdate")
    with c4:
        act_val_col  = st.selectbox("Actual: value column", act_cols,
                                    index=act_cols.index("final_amount") if "final_amount" in act_cols else 0,
                                    key="fva_actval")

    c5, c6, c7 = st.columns(3)
    with c5:
        freq_fva = st.selectbox("Aggregation", ["Daily", "Monthly"], key="fva_freq")
        freq_code = "D" if freq_fva == "Daily" else "MS"
        unit_fva  = "₦"
    with c6:
        # Optional product filter
        pid_col_act = st.selectbox(
            "Product ID column in actual (for grouping)",
            ["None"] + act_cols, key="fva_pid"
        )
    with c7:
        prod_col_act = st.selectbox(
            "Product name column in actual",
            ["None"] + act_cols, key="fva_pname"
        )

    run_fva = st.button("▶  Compare forecast vs actual", type="primary",
                        key="fva_run", use_container_width=True)
    if not run_fva:
        st.stop()

    # ── Process ───────────────────────────────────────────────────
    with st.spinner("Processing…"):
        # Parse forecast file
        fc_df[fc_date_col] = pd.to_datetime(fc_df[fc_date_col], dayfirst=True, errors="coerce")
        fc_df = fc_df.dropna(subset=[fc_date_col])
        fc_df[fc_val_col]  = pd.to_numeric(
            fc_df[fc_val_col].astype(str).str.replace(",","",regex=False).str.replace(unit_fva,"",regex=False),
            errors="coerce"
        )
        # Resample forecast to chosen freq
        fc_series = (
            fc_df.set_index(fc_date_col)[fc_val_col]
            .resample(freq_code).sum()
            .reset_index()
            .rename(columns={fc_date_col:"date", fc_val_col:"forecast"})
        )

        # Parse actual file
        act_df = clean_dataframe(act_raw, act_date_col, act_val_col)
        act_agg = (
            act_df.set_index(act_date_col)[act_val_col]
            .resample(freq_code).sum()
            .reset_index()
            .rename(columns={act_date_col:"date", act_val_col:"actual"})
        )

        # Merge on date
        merged = pd.merge(fc_series, act_agg, on="date", how="inner")
        merged = merged[(merged["forecast"] > 0) | (merged["actual"] > 0)]

        if len(merged) == 0:
            st.error(
                "No overlapping dates found between forecast and actual. "
                "Check that the date ranges match."
            )
            st.stop()

        # Per-period metrics
        merged["error"]    = merged["actual"] - merged["forecast"]
        merged["abs_err"]  = merged["error"].abs()
        merged["smape_pct"]= merged.apply(
            lambda r: round(
                200 * abs(r["actual"]-r["forecast"]) /
                (abs(r["actual"])+abs(r["forecast"])+1e-8), 2
            ), axis=1
        )
        merged["mape_pct"] = merged.apply(
            lambda r: round(abs(r["actual"]-r["forecast"])/r["actual"]*100, 2)
            if r["actual"] > 0 else np.nan, axis=1
        )
        merged["rating"] = merged["smape_pct"].apply(lambda v: accuracy_label(v)[0])

        # Format date for display
        date_fmt = "%b %Y" if freq_code=="MS" else "%d/%m/%Y"
        merged["period"] = merged["date"].dt.strftime(date_fmt)

    # ── Overall metrics ───────────────────────────────────────────
    y_a = merged["actual"].values
    y_f = merged["forecast"].values
    ov_rmse  = float(np.sqrt(mean_squared_error(y_a, y_f)))
    ov_mae   = float(mean_absolute_error(y_a, y_f))
    ov_smape = calc_smape(y_a, y_f)
    ov_mape, nm = calc_mape(y_a, y_f, float(np.mean(y_a)))
    ov_label, ov_cls = accuracy_label(ov_smape)

    st.markdown("---")
    st.markdown("### Overall accuracy")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",   f"{unit_fva}{ov_rmse:,.0f}")
    m2.metric("MAE",    f"{unit_fva}{ov_mae:,.0f}")
    m3.metric("sMAPE",  f"{ov_smape:.2f}%")
    m4.metric("MAPE",   f"{ov_mape:.2f}%" if not np.isnan(ov_mape) else "N/A")
    m5.metric("Rating", ov_label)

    st.markdown(
        f'<div class="explain-box">'
        f"<strong>Overall accuracy: <span class='{ov_cls}'>{ov_label}</span></strong><br><br>"
        f"Across {len(merged)} overlapping periods, the forecast had a sMAPE of {ov_smape:.2f}% "
        f"and a MAE of {unit_fva}{ov_mae:,.0f} per period.<br>"
        f"MAPE evaluated on {nm}/{len(merged)} periods (excluded zero-actual periods)."
        f"</div>", unsafe_allow_html=True
    )

    # ── Chart ─────────────────────────────────────────────────────
    st.markdown("### Forecast vs Actual chart")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged["date"], merged["actual"],   color="#1565C0", linewidth=1.2, label="Actual", marker="o", markersize=4)
    ax.plot(merged["date"], merged["forecast"], color="#E53935", linewidth=1.2, linestyle="--", label="Forecast", marker="s", markersize=4)
    ax.fill_between(merged["date"],
                    np.minimum(merged["actual"], merged["forecast"]),
                    np.maximum(merged["actual"], merged["forecast"]),
                    alpha=0.12, color="#E53935", label="Error Band")
    ax.set_title("Forecast vs Actual Sales", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Value ({unit_fva})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Detailed table ────────────────────────────────────────────
    st.markdown("### Period-by-period breakdown")
    display = merged[[
        "period", "forecast", "actual", "error", "smape_pct", "mape_pct", "rating"
    ]].copy()
    display.columns = [
        "Period",
        f"Forecast ({unit_fva})",
        f"Actual ({unit_fva})",
        f"Error ({unit_fva})",
        "sMAPE (%)",
        "MAPE (%)",
        "Rating"
    ]
    # Format numbers
    for col in [f"Forecast ({unit_fva})", f"Actual ({unit_fva})", f"Error ({unit_fva})"]:
        display[col] = display[col].apply(lambda v: f"{v:,.0f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────
    dl = merged[["period","forecast","actual","error","smape_pct","mape_pct","rating"]].copy()
    dl.columns = ["period", f"forecast_{unit_fva}", f"actual_{unit_fva}",
                  f"error_{unit_fva}", "smape_pct", "mape_pct", "rating"]
    st.download_button(
        "⬇  Download comparison as CSV",
        dl.to_csv(index=False).encode(),
        "forecast_vs_actual.csv", "text/csv",
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Training Results":
    st.title("Training Results")
    st.markdown(
        "All graphs from the original model training run. "
        "They update automatically when `lstm_forecasting.py` is re-run."
    )

    graphs = [
        ("01_daily_sales_timeseries.png", "Daily total sales time series",
         "The top panel shows raw daily total sales (Jan 2023–Nov 2025). "
         "The bottom panel adds a 30-day rolling average (red) to reveal the underlying trend. "
         "Sales grew from early 2023, peaked around late 2024, and declined toward Nov 2025. "
         "This declining tail formed the test period — the hardest scenario for the model. "
         "High day-to-day volatility reflects irregular bulk ordering patterns typical of B2B platforms."),
        ("02_training_loss.png", "LSTM training vs validation loss",
         "The blue line is training loss (MSE on normalised values) per epoch. "
         "The red line is validation loss. Both drop steeply in the first few epochs then "
         "level off running close together with no upward divergence. "
         "This confirms clean convergence with no overfitting. "
         "Early stopping fired after validation loss showed no further improvement, "
         "and best weights were automatically restored."),
        ("03_predictions_vs_actual.png", "LSTM forecast vs actual (test period)",
         "Blue = actual daily sales in the held-out test period (Jun–Nov 2025). "
         "Red dashed = LSTM predictions for those same days, generated without the model "
         "ever seeing this period during training. "
         "The model captures the general downward trend correctly but smooths out individual spikes — "
         "expected for a univariate model with no access to promotional or restocking signals. "
         "For strategic planning the trend-level accuracy is the most operationally relevant output."),
        ("04_model_comparison.png", "Model performance metrics",
         "Bar charts comparing RMSE, MAE, MAPE, and sMAPE across the models evaluated. "
         "Lower bars on all metrics = better performance. "
         "sMAPE is the primary percentage metric here because it is bounded 0–200% and not "
         "distorted by near-zero actual values on low-volume days, which inflate conventional MAPE."),
    ]

    for fname, title, explanation in graphs:
        p = os.path.join(OUT_DIR, fname)
        st.markdown(f"---\n### {title}")
        if os.path.exists(p):
            st.image(p, use_column_width=True)
            st.markdown(
                f'<div class="explain-box"><strong>What this shows:</strong> {explanation}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning(f"`{fname}` not found — run `python lstm_forecasting.py` to generate it.")

    st.markdown("---")
    st.markdown("### Live metrics table")
    if disp_df is not None:
        st.success("Loaded from `outputs/model_comparison.csv` — updates on re-run.")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Run `python lstm_forecasting.py` to generate the comparison CSV.")

    st.markdown("---")
    st.markdown("### Model configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
| Parameter | Value |
|---|---|
| Architecture | Vanilla LSTM |
| LSTM units | 64 |
| Dropout rate | 0.2 |
| Look-back window | 60 periods |
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


# ══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown("""
### Design and Development of an LSTM-Based Product Demand Forecasting System
**Postgraduate Diploma Research** — Nigerian B2B E-Commerce (Anonymous)

---
### Research objective
Design, implement, and evaluate a vanilla LSTM neural network for forecasting
daily product demand from historical transactional sales data, and deploy the
model as an interactive web application.

---
### Dataset
| Property | Detail |
|---|---|
| Company | Nigerian B2B e-commerce (anonymous) |
| Period | January 2023 – November 2025 |
| Raw transactions | ~2.38 million rows across 5 CSV files |
| Trading days used | 982 (after cleaning and outlier removal) |
| Date format | DD/MM/YYYY |
| Product categories | Food (FDI), Beverages (BEV), Home (HME), Personal/Cooking (PRF), others |

---
### Model architecture
| Layer | Type | Detail |
|---|---|---|
| 1 | LSTM | 64 units, return_sequences=False |
| 2 | Dropout | Rate = 0.2 |
| 3 | Dense | 1 neuron, linear activation |
| — | Total | 16,961 trainable parameters |

---
### Accuracy scale
| MAPE | sMAPE | Rating |
|---|---|---|
| < 10% | < 10% | Excellent / Highly Accurate |
| 10%–20% | 10%–25% | Good |
| 20%–50% | 25%–50% | Reasonable / Fair |
| > 50% | > 50% | Inaccurate |

---
### Technologies
Python 3.11.9 · TensorFlow 2.21 · pandas · NumPy · scikit-learn · Matplotlib · Streamlit

---
### Minimum CSV format required
```
orderDate,final_amount
05/01/2023,125000
06/01/2023,340500
```
For quantity forecasting, also include a `quantitySold` column.
For category filtering, include `productId` (format: `NGA-FDI-PST-000025`).

---
### Run locally
```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib
streamlit run app.py
```
    """)
