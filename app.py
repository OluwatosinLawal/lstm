# =============================================================================
#  LSTM DEMAND FORECASTING — STREAMLIT APP  (v5)
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
from sklearn.preprocessing import MinMaxScaler

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
.explain-box em     { color: #a8f0b8 !important; font-style: italic; }

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

.acc-excellent { background:#1a3d2b; color:#b8ffcc; padding:4px 10px;
                 border-radius:4px; font-weight:bold; display:inline-block; }
.acc-good      { background:#1a2e3d; color:#a8d5f5; padding:4px 10px;
                 border-radius:4px; font-weight:bold; display:inline-block; }
.acc-fair      { background:#3a3010; color:#f5e0a0; padding:4px 10px;
                 border-radius:4px; font-weight:bold; display:inline-block; }
.acc-poor      { background:#3d1a1a; color:#f5b0b0; padding:4px 10px;
                 border-radius:4px; font-weight:bold; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "lstm_demand_forecast.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
LOOK_BACK   = 60

# ── Accuracy helpers ──────────────────────────────────────────────
def accuracy_label(smape_val):
    if smape_val < 10:   return "Excellent / Highly Accurate", "acc-excellent"
    elif smape_val < 25: return "Good", "acc-good"
    elif smape_val < 50: return "Reasonable / Fair", "acc-fair"
    else:                return "Inaccurate", "acc-poor"

def acc_scale_table():
    """Renders the accuracy scale table — only call this after forecast results."""
    st.markdown("""
**Accuracy scale reference**

| MAPE | sMAPE | Rating |
|---|---|---|
| < 10% | < 10% | Excellent / Highly Accurate |
| 10% – 20% | 10% – 25% | Good |
| 20% – 50% | 25% – 50% | Reasonable / Fair |
| > 50% | > 50% | Inaccurate |
""")

# ══════════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Not found: {MODEL_PATH}"
    if not os.path.exists(SCALER_PATH):
        return None, None, f"Not found: {SCALER_PATH}"
    try:
        from tensorflow.keras.models import load_model
        m = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            s = pickle.load(f)
        return m, s, None
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
    except:
        return None, None


def read_and_merge(files):
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d.columns = d.columns.str.strip()
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def clean_df(df, date_col, val_col, qty_col=None):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.apply(lambda c: c.str.strip() if c.dtype=="object" else c)
    for col in [val_col, qty_col, "unitPrice", "orderTotal"]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",","",regex=False), errors="coerce"
            )
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    return df


def get_categories(df, pid_col):
    if pid_col not in df.columns:
        return {}
    label_map = {
        "FDI": "Food Items (FDI)",
        "BEV": "Beverages (BEV)",
        "HME": "Home & Household (HME)",
        "PRF": "Personal & Cooking (PRF)",
        "PHA": "Pharmacy (PHA)",
        "AGR": "Agriculture (AGR)",
    }
    codes = df[pid_col].dropna().str.extract(r"NGA-([A-Z]+)-")[0].dropna().unique()
    return {label_map.get(c, f"Category {c}"): c for c in sorted(codes)}


def aggregate(df, date_col, val_col, freq, product_filter=None,
              prod_col=None, pid_col=None, cat_code=None, products=None):
    """
    Aggregate df to chosen frequency after optional category/product filtering.
    Returns a (date, total) series DataFrame.
    """
    d = df.copy()

    # Filter by category
    if cat_code and cat_code != "ALL" and pid_col and pid_col in d.columns:
        d = d[d[pid_col].str.contains(f"NGA-{cat_code}-", na=False)]

    # Filter by specific products
    if products and prod_col and prod_col in d.columns:
        d = d[d[prod_col].isin(products)]

    if len(d) == 0:
        return None

    series = (
        d.groupby(date_col)[val_col].sum()
        .reset_index()
        .rename(columns={date_col:"date", val_col:"total"})
        .sort_values("date")
    )
    series = (
        series.set_index("date")
        .resample(freq)["total"].sum()
        .reset_index()
    )
    series = series[series["total"] > 0].reset_index(drop=True)
    thresh = series["total"].quantile(0.01)
    series = series[series["total"] > thresh].reset_index(drop=True)
    return series


def make_sequences(data, lb=60):
    X, y = [], []
    for i in range(lb, len(data)):
        X.append(data[i-lb:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def calc_smape(yt, yp):
    return float(np.mean(2*np.abs(yt-yp)/(np.abs(yt)+np.abs(yp)+1e-8))*100)


def calc_mape(yt, yp, mean_v):
    mask = yt > mean_v * 0.01
    if mask.sum() == 0:
        return np.nan, 0
    return float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100), int(mask.sum())


def rolling_forecast(model, sc, seed, n):
    s = sc.transform(seed[-LOOK_BACK:].reshape(-1,1)).flatten()
    w = list(s)
    out = []
    for _ in range(n):
        x = np.array(w[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
        p = model.predict(x, verbose=0)[0,0]
        out.append(p); w.append(p)
    return np.maximum(sc.inverse_transform(np.array(out).reshape(-1,1)).flatten(), 0)


def show_col_guide():
    st.markdown("""
| Column | Status | Description |
|---|---|---|
| `orderDate` | **Required** | Date — format DD/MM/YYYY |
| `final_amount` | **Required*** | Sales value per order line (₦) — plain numbers only |
| `quantitySold` | **Required*** | Units sold — required if forecasting quantity |
| `displayTitle` | Recommended | Product name — used for product-level filtering |
| `productId` | Recommended | SKU — used to extract categories (e.g. NGA-FDI-...) |
| `salesCategory` | Optional | Regular Sales or Promo Sales |

> \* At least one of `final_amount` or `quantitySold` is required.  
> Column names are **case-sensitive**.  
> Multiple files with the same columns are merged automatically.
""")


# ══════════════════════════════════════════════════════════════════
# SHARED UPLOAD + CONFIGURE  (separate key per page to avoid state clash)
# ══════════════════════════════════════════════════════════════════

def upload_and_configure(pk):
    """
    pk = page key string, e.g. "uf" or "fp".
    Returns (series DataFrame, config dict) once user clicks Load,
    or (None, None) if not ready yet.
    """
    with st.expander("📋  Column requirements", expanded=False):
        show_col_guide()

    st.markdown("### Step 1 — Upload CSV file(s)")
    st.caption(
        "You can upload **multiple files** at once (e.g. one per year). "
        "They are merged automatically as long as they share the same column names."
    )
    files = st.file_uploader(
        "Upload one or more sales CSV files",
        type=["csv"], accept_multiple_files=True, key=f"{pk}_files"
    )
    if not files:
        st.info("Waiting for file upload…")
        return None, None

    raw = read_and_merge(files)
    if raw is None or len(raw) == 0:
        st.error("No data found.")
        return None, None

    st.success(
        f"✔ {len(files)} file(s) merged — {len(raw):,} rows | "
        f"Columns: `{'`, `'.join(raw.columns.tolist())}`"
    )

    # ── Step 2: Column mapping ────────────────────────────────────
    st.markdown("### Step 2 — Map your columns")
    cols = list(raw.columns)

    def pick(label, hints, key):
        d = next((c for c in hints if c in cols), cols[0])
        return st.selectbox(label, cols, index=cols.index(d), key=f"{pk}_{key}")

    c1, c2, c3, c4 = st.columns(4)
    with c1: date_col = pick("Date column",          ["orderDate"],     "date")
    with c2: amt_col  = pick("Sales amount column",  ["final_amount"],  "amt")
    with c3: qty_col  = pick("Quantity column",       ["quantitySold"],  "qty")
    with c4: prod_col = pick("Product name column",  ["displayTitle"],  "prod")
    pid_col = pick("Product ID column (for categories)", ["productId"], "pid")

    # ── Step 3: Forecast configuration ───────────────────────────
    st.markdown("### Step 3 — Forecast configuration")

    fc1, fc2 = st.columns(2)
    with fc1:
        freq_label = st.selectbox(
            "Aggregation level",
            ["Daily", "Monthly"],
            key=f"{pk}_agg",
            help="Monthly is more stable and aligns with business planning cycles."
        )
        freq = "D" if freq_label == "Daily" else "MS"

    with fc2:
        target_label = st.selectbox(
            "Forecast target",
            ["Sales Amount (₦)", "Sales Quantity (units)"],
            key=f"{pk}_tgt"
        )
        val_col = amt_col if "Amount" in target_label else qty_col
        unit    = "₦" if "Amount" in target_label else "units"

    # Rolling average window for chart (visual only, does not affect model)
    if freq == "D":
        roll_win = st.select_slider(
            "Chart rolling average window (days — display only, does not affect model)",
            options=[7, 15, 30], value=30, key=f"{pk}_roll"
        )
    else:
        roll_win = 3  # 3-month rolling for monthly

    # ── Product / category filter ─────────────────────────────────
    st.markdown("#### Product filter")
    cats = get_categories(raw, pid_col)
    cat_name_to_code = {"All Products": "ALL"} | {v: k for k, v in cats.items()}
    # Flip: display label → code
    cat_display      = {"All Products": "ALL"} | {v: k for k, v in cats.items()}

    filter_mode = st.radio(
        "Filter by:",
        ["All Products", "Product Category", "Specific Products (up to 10)"],
        horizontal=True, key=f"{pk}_fmode"
    )

    cat_code      = "ALL"
    chosen_prods  = None

    if filter_mode == "Product Category":
        cat_labels = list(cats.values())
        if cat_labels:
            chosen_cat_label = st.selectbox(
                "Select category", cat_labels, key=f"{pk}_cat"
            )
            # cats is {display_label: code}
            cat_code = cats[chosen_cat_label]
        else:
            st.warning("No categories detected — ensure productId column is correct.")

    elif filter_mode == "Specific Products (up to 10)":
        if prod_col in raw.columns:
            all_prods = sorted(raw[prod_col].dropna().unique().tolist())
            chosen_prods = st.multiselect(
                "Select up to 10 products",
                options=all_prods,
                max_selections=10,
                key=f"{pk}_prods",
                help="The forecast will sum the selected products together."
            )
            if not chosen_prods:
                st.info("Select at least one product to continue.")
                return None, None
        else:
            st.warning(f"Column `{prod_col}` not found.")

    # ── Load button ───────────────────────────────────────────────
    if st.button("▶  Load & process data", key=f"{pk}_load", type="primary"):
        with st.spinner("Cleaning and aggregating…"):
            df_c = clean_df(raw, date_col, val_col, qty_col)
            series = aggregate(
                df_c, date_col, val_col, freq,
                prod_col=prod_col, pid_col=pid_col,
                cat_code=cat_code, products=chosen_prods
            )

        if series is None or len(series) == 0:
            st.error("No data after filtering. Try a different product or category.")
            return None, None

        if len(series) < LOOK_BACK + 5:
            st.error(
                f"Only {len(series)} periods after filtering. Need at least {LOOK_BACK+5}. "
                "Try Monthly aggregation, All Products, or upload more data."
            )
            return None, None

        # Store in session with unique per-page keys
        st.session_state[f"{pk}_s_series"]   = series
        st.session_state[f"{pk}_s_unit"]     = unit
        st.session_state[f"{pk}_s_freq"]     = freq
        st.session_state[f"{pk}_s_flabel"]   = freq_label
        st.session_state[f"{pk}_s_cat"]      = filter_mode if filter_mode != "Product Category" else chosen_cat_label if cat_labels else "All"
        st.session_state[f"{pk}_s_target"]   = target_label
        st.session_state[f"{pk}_s_roll"]     = roll_win
        st.session_state[f"{pk}_s_prods"]    = chosen_prods

        st.success(
            f"✔ {len(series)} {freq_label.lower()} periods | "
            f"{series['date'].min().date()} → {series['date'].max().date()} | "
            f"Mean: {'₦' if unit=='₦' else ''}{series['total'].mean():,.0f} {unit if unit!='₦' else ''}"
        )

    if f"{pk}_s_series" not in st.session_state:
        return None, None

    cfg = {
        "series":  st.session_state[f"{pk}_s_series"],
        "unit":    st.session_state[f"{pk}_s_unit"],
        "freq":    st.session_state[f"{pk}_s_freq"],
        "flabel":  st.session_state[f"{pk}_s_flabel"],
        "cat":     st.session_state[f"{pk}_s_cat"],
        "target":  st.session_state[f"{pk}_s_target"],
        "roll":    st.session_state[f"{pk}_s_roll"],
        "prods":   st.session_state[f"{pk}_s_prods"],
    }
    return cfg["series"], cfg


# ══════════════════════════════════════════════════════════════════
# LOAD MODEL ONCE
# ══════════════════════════════════════════════════════════════════
model, base_scaler, load_err = load_model_and_scaler()
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
        st.caption("Vanilla LSTM · 64 units · 60-period window")
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
    for col, lbl, val in zip(
        [c1,c2,c3,c4],
        ["Model","Training period","Look-back window","Test sMAPE"],
        ["Vanilla LSTM","Jan 2023–Nov 2025","60 periods", smape_val]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Pages")
    a, b, c, d, e = st.columns(5)
    with a:
        st.markdown("**📂 Upload & Forecast**")
        st.write("Evaluate the model against historical data. Filter by category or specific products, choose daily/monthly, and select sales amount or quantity.")
    with b:
        st.markdown("**🔮 Future Predictions**")
        st.write("Generate forward forecasts for a custom date range or number of periods, with the same filtering and aggregation options.")
    with c:
        st.markdown("**✅ Forecast vs Actual**")
        st.write("Upload a forecast file and actual data. The app compares them with MAPE, sMAPE, and an accuracy rating per period.")
    with d:
        st.markdown("**📊 Training Results**")
        st.write("All four original training graphs with plain-language explanations.")
    with e:
        st.markdown("**ℹ️ About**")
        st.write("Project background, column guide, and model details.")

    if model is None:
        st.error(f"Model not loaded: {load_err}")
        st.info("Run `python lstm_forecasting.py` then refresh.")


# ══════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & FORECAST
# ══════════════════════════════════════════════════════════════════
if page == "📂 Upload & Forecast":
    st.title("Upload Data & Evaluate Forecast")
    st.markdown(
        "Upload historical sales data and evaluate how well the LSTM model forecasts it. "
        "Filter by product category or select up to 10 specific products."
    )

    if model is None:
        st.error("Model not loaded. Run `python lstm_forecasting.py` first.")
        st.stop()

    series, cfg = upload_and_configure("uf")
    if series is None:
        st.stop()

    unit   = cfg["unit"]
    freq   = cfg["freq"]
    flabel = cfg["flabel"]
    roll   = cfg["roll"]

    st.markdown("---")
    st.markdown("### Step 4 — Run evaluation")

    n    = len(series)
    t_e  = int(n * 0.70)
    v_e  = int(n * 0.85)

    sales  = series["total"].values.reshape(-1,1)
    sc     = MinMaxScaler(); sc.fit(sales[:t_e])
    scaled = sc.transform(sales)

    X_all, y_all = make_sequences(scaled, LOOK_BACK)
    adj_t = t_e - LOOK_BACK
    adj_v = v_e - LOOK_BACK
    X_te  = X_all[adj_v:].reshape(-1, LOOK_BACK, 1)
    y_te  = y_all[adj_v:]

    with st.spinner("Running model…"):
        yp_s    = model.predict(X_te, verbose=0)
    y_pred   = sc.inverse_transform(yp_s).flatten()
    y_actual = sc.inverse_transform(y_te.reshape(-1,1)).flatten()
    dates    = series["date"].values[n - len(y_pred):]

    mean_v   = float(series["total"].mean())
    rmse_v   = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae_v    = float(mean_absolute_error(y_actual, y_pred))
    sm_v     = calc_smape(y_actual, y_pred)
    mp_v, nm = calc_mape(y_actual, y_pred, mean_v)
    lbl, cls = accuracy_label(sm_v)

    st.markdown("### Results")

    # ── Metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",   f"{'₦' if unit=='₦' else ''}{rmse_v:,.0f}{'' if unit=='₦' else ' '+unit}")
    m2.metric("MAE",    f"{'₦' if unit=='₦' else ''}{mae_v:,.0f}{'' if unit=='₦' else ' '+unit}")
    m3.metric("sMAPE",  f"{sm_v:.2f}%")
    m4.metric("MAPE",   f"{mp_v:.2f}%" if not np.isnan(mp_v) else "N/A")
    m5.metric("Rating", lbl)

    st.markdown(
        f'<div class="explain-box">'
        f"<strong>Accuracy rating: <span class='{cls}'>{lbl}</span></strong><br><br>"
        f"<strong>RMSE</strong> measures error with larger mistakes penalised more. "
        f"At {'₦' if unit=='₦' else ''}{rmse_v:,.0f}, this is {rmse_v/mean_v*100:.1f}% of the mean {flabel.lower()} value.<br><br>"
        f"<strong>MAE</strong> is the average absolute error per period — "
        f"{'₦' if unit=='₦' else ''}{mae_v:,.0f} ({mae_v/mean_v*100:.1f}% of mean).<br><br>"
        f"<strong>sMAPE ({sm_v:.2f}%)</strong> is the primary percentage metric. Rated "
        f"<em>{lbl}</em> on the scale below.<br><br>"
        f"<strong>MAPE ({mp_v:.2f}%)</strong> evaluated on {nm}/{len(y_pred)} periods "
        f"(near-zero periods excluded to prevent division errors)."
        f"</div>",
        unsafe_allow_html=True
    )
    acc_scale_table()

    # ── Graph 1: full series ──────────────────────────────────────
    st.markdown("#### Full series — historical data")
    fig1, ax1 = plt.subplots(figsize=(14,4))
    ax1.plot(series["date"], series["total"], color="#1565C0", linewidth=0.8, alpha=0.8, label="Sales")
    roll_series = series["total"].rolling(window=roll, min_periods=1).mean()
    ax1.plot(series["date"], roll_series, color="#E53935", linewidth=1.5,
             label=f"{roll}-period rolling avg")
    ax1.set_title(f"{flabel} {cfg['target']} — {cfg['cat']}", fontsize=12, fontweight="bold")
    ax1.set_ylabel(f"{cfg['target']} ({unit})")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig1); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> The blue line is the raw daily or monthly total. "
        "The red line is the rolling average which smooths out short-term spikes to reveal the "
        "underlying demand trend. Wide swings are typical of B2B bulk ordering."
        "</div>", unsafe_allow_html=True
    )

    # ── Graph 2: forecast vs actual ───────────────────────────────
    st.markdown("#### Forecast vs Actual (test period)")
    fig2, ax2 = plt.subplots(figsize=(14,5))
    ax2.plot(pd.to_datetime(dates), y_actual, color="#1565C0", linewidth=1.0, label="Actual")
    ax2.plot(pd.to_datetime(dates), y_pred,   color="#E53935", linewidth=1.2,
             linestyle="--", label="LSTM Forecast")
    ax2.fill_between(pd.to_datetime(dates),
                     np.minimum(y_actual,y_pred), np.maximum(y_actual,y_pred),
                     alpha=0.12, color="#E53935", label="Error Band")
    ax2.set_title("LSTM Forecast vs Actual", fontsize=12, fontweight="bold")
    ax2.set_ylabel(f"{cfg['target']} ({unit})")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> Blue = actual recorded values. "
        "Red dashed = model predictions. Narrower pink error band = more accurate. "
        "The model captures the overall trend but smooths individual-period spikes — "
        "expected when no promotional or external signals are in the input."
        "</div>", unsafe_allow_html=True
    )

    # ── Graph 3: error distribution ───────────────────────────────
    st.markdown("#### Error analysis")
    errors = y_actual - y_pred
    fig3, ax3 = plt.subplots(1, 2, figsize=(14,4))
    ax3[0].bar(range(len(errors)), errors,
               color=["#E53935" if e>0 else "#1565C0" for e in errors], alpha=0.7)
    ax3[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3[0].set_title("Error per Period (Actual − Predicted)", fontsize=11, fontweight="bold")
    ax3[0].set_ylabel(f"Error ({unit})"); ax3[0].grid(True, alpha=0.3)

    ax3[1].hist(errors, bins=25, color="#7F77DD", edgecolor="white", alpha=0.85)
    ax3[1].axvline(0, color="black", linewidth=1.0, linestyle="--", label="Zero error")
    ax3[1].axvline(float(np.mean(errors)), color="#E53935", linewidth=1.2,
                   label=f"Mean error")
    ax3[1].set_title("Error Distribution", fontsize=11, fontweight="bold")
    ax3[1].set_xlabel(f"Error ({unit})"); ax3[1].set_ylabel("Periods")
    ax3[1].legend(); ax3[1].grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading the error charts:</strong> "
        "Red bars = the model under-predicted (actual was higher). "
        "Blue bars = the model over-predicted. "
        "The histogram shows whether errors are centred on zero (no systematic bias) "
        "or skewed in one direction (consistent over or under-forecasting)."
        "</div>", unsafe_allow_html=True
    )

    # Download
    out = pd.DataFrame({
        "period":    pd.to_datetime(dates).strftime("%d/%m/%Y" if freq=="D" else "%b %Y"),
        "actual":    np.round(y_actual,2),
        "predicted": np.round(y_pred,2),
        "error":     np.round(errors,2),
        "smape_pct": [round(calc_smape(np.array([a]),np.array([p])),2)
                      for a,p in zip(y_actual,y_pred)],
    })
    st.download_button("⬇  Download forecast results CSV",
        out.to_csv(index=False).encode(),
        "forecast_results.csv","text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: FUTURE PREDICTIONS
# ══════════════════════════════════════════════════════════════════
if page == "🔮 Future Predictions":
    st.title("Future Sales Forecast")
    st.markdown(
        "Upload recent sales data to generate forward-looking predictions. "
        "Select a date range or number of periods, and filter by category or specific products."
    )

    if model is None:
        st.error("Model not loaded. Run `python lstm_forecasting.py` first.")
        st.stop()

    series, cfg = upload_and_configure("fp")
    if series is None:
        st.stop()

    unit   = cfg["unit"]
    freq   = cfg["freq"]
    flabel = cfg["flabel"]
    roll   = cfg["roll"]

    st.markdown("---")
    st.markdown("### Step 4 — Forecast horizon")

    hz1, hz2 = st.columns(2)
    with hz1:
        h_type = st.radio("Specify forecast period by:",
                          ["Number of periods", "Date range"],
                          horizontal=True, key="fp_htype")
    with hz2:
        last_date = pd.Timestamp(series["date"].iloc[-1])
        if h_type == "Number of periods":
            n_periods = st.number_input(
                f"Number of {flabel.lower()} periods",
                min_value=1, max_value=365 if freq=="D" else 36,
                value=30 if freq=="D" else 6, key="fp_nper"
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
            ca, cb = st.columns(2)
            with ca:
                s_date = st.date_input("Start date",
                    value=(last_date + pd.Timedelta(days=1)).date(), key="fp_sdate")
            with cb:
                e_date = st.date_input("End date",
                    value=(last_date + pd.Timedelta(days=30)).date(), key="fp_edate")
            future_dates = pd.date_range(
                start=s_date, end=e_date, freq="D" if freq=="D" else "MS"
            )
            n_periods = len(future_dates)

    if n_periods == 0:
        st.warning("No periods selected. Adjust your date range.")
        st.stop()

    if not st.button("▶  Generate forecast", type="primary",
                     key="fp_run", use_container_width=True):
        st.stop()

    # Fit local scaler on the full series (no train/test split for future)
    seed = series["total"].values
    sc   = MinMaxScaler(); sc.fit(seed.reshape(-1,1))

    with st.spinner(f"Forecasting {n_periods} {flabel.lower()} period(s)…"):
        preds = rolling_forecast(model, sc, seed, n_periods)

    fdf = pd.DataFrame({
        "date":     future_dates[:len(preds)],
        "forecast": np.round(preds[:len(future_dates)],2)
    })

    st.markdown("---")
    st.markdown("### Forecast results")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total ({flabel.lower()})", f"{'₦' if unit=='₦' else ''}{preds.sum():,.0f}")
    c2.metric("Average per period",         f"{'₦' if unit=='₦' else ''}{preds.mean():,.0f}")
    c3.metric("Peak period",                f"{'₦' if unit=='₦' else ''}{preds.max():,.0f}")

    # Chart: history + forecast
    fig, ax = plt.subplots(figsize=(14,5))
    hist = series.tail(12 if freq=="MS" else 90)
    ax.plot(hist["date"], hist["total"],
            color="#1565C0", linewidth=1.0, label="Historical", alpha=0.8)
    roll_h = hist["total"].rolling(window=roll, min_periods=1).mean()
    ax.plot(hist["date"], roll_h,
            color="#90CAF9", linewidth=1.2, linestyle=":",
            label=f"{roll}-period rolling avg")
    ax.plot(fdf["date"], fdf["forecast"],
            color="#E53935", linewidth=2.0, linestyle="--",
            marker="o", markersize=4, label=f"Forecast ({n_periods} periods)")
    if len(future_dates) > 0:
        ax.axvspan(future_dates[0], future_dates[-1], alpha=0.06, color="#E53935")
    ax.axvline(last_date, color="gray", linewidth=1.0, linestyle=":", label="Forecast start")
    ax.set_title(
        f"LSTM {n_periods}-period Forecast — {cfg['cat']} — {cfg['target']}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(f"{cfg['target']} ({unit})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Bar breakdown
    fig2, ax2 = plt.subplots(figsize=(max(12, n_periods//2), 4))
    fmt = "%b %Y" if freq=="MS" else "%d %b"
    ax2.bar(fdf["date"].dt.strftime(fmt), fdf["forecast"],
            color="#7F77DD", edgecolor="white")
    ax2.set_title(f"Forecast Breakdown — {flabel} Periods", fontsize=11, fontweight="bold")
    ax2.set_ylabel(f"{cfg['target']} ({unit})")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>How to read this forecast:</strong> "
        "Blue = recent actual sales. Grey dotted = rolling average trend. "
        "The dotted grey vertical line marks where history ends. "
        "Red dashed + bar chart = predicted values for each future period. "
        "<em>Uncertainty increases further into the future</em> — "
        "treat later periods as directional estimates, not precise targets."
        "</div>", unsafe_allow_html=True
    )

    # Table
    show = fdf.copy()
    show["date"]     = show["date"].dt.strftime("%b %Y" if freq=="MS" else "%A, %d %b %Y")
    show["forecast"] = show["forecast"].apply(
        lambda v: f"{'₦' if unit=='₦' else ''}{v:,.0f}{'' if unit=='₦' else ' '+unit}"
    )
    show.columns = ["Period", f"Forecast ({unit})"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    dl = fdf.copy()
    dl["date"] = dl["date"].dt.strftime("%d/%m/%Y")
    dl.columns = ["date", f"forecast_{unit}"]
    st.download_button(
        "⬇  Download forecast CSV  (use on Forecast vs Actual page)",
        dl.to_csv(index=False).encode(),
        f"forecast_{n_periods}{freq}.csv","text/csv", use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: FORECAST vs ACTUAL
# ══════════════════════════════════════════════════════════════════
if page == "✅ Forecast vs Actual":
    st.title("Forecast vs Actual Comparison")
    st.markdown(
        "Upload the forecast file (from Future Predictions) and the actual sales data "
        "for the same period. The app aligns by date, computes accuracy metrics, "
        "and rates each period using the accuracy scale."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Forecast file** (downloaded from Future Predictions page)")
        fc_file = st.file_uploader("Upload forecast CSV", type=["csv"], key="fva_fc")
    with col2:
        st.markdown("**Actual sales file(s)**")
        act_files = st.file_uploader("Upload actual data CSV(s)", type=["csv"],
                                     accept_multiple_files=True, key="fva_act")

    if not fc_file or not act_files:
        st.info("Upload both files to continue.")
        st.stop()

    st.markdown("### Step 2 — Configure")
    fc_df    = pd.read_csv(fc_file); fc_df.columns = fc_df.columns.str.strip()
    act_raw  = read_and_merge(act_files)
    fc_cols  = list(fc_df.columns)
    act_cols = list(act_raw.columns)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fc_date = st.selectbox("Forecast: date column", fc_cols,
            index=fc_cols.index("date") if "date" in fc_cols else 0, key="fva_fcdate")
    with c2:
        fc_val  = st.selectbox("Forecast: value column", fc_cols,
            index=1 if len(fc_cols)>1 else 0, key="fva_fcval")
    with c3:
        act_date = st.selectbox("Actual: date column", act_cols,
            index=act_cols.index("orderDate") if "orderDate" in act_cols else 0,
            key="fva_actdate")
    with c4:
        act_val  = st.selectbox("Actual: value column", act_cols,
            index=act_cols.index("final_amount") if "final_amount" in act_cols else 0,
            key="fva_actval")

    c5, c6 = st.columns(2)
    with c5:
        fva_freq  = st.selectbox("Aggregation", ["Daily","Monthly"], key="fva_freq")
        fva_fcode = "D" if fva_freq=="Daily" else "MS"
    with c6:
        unit_fva  = st.selectbox("Unit label", ["₦","units"], key="fva_unit")

    if not st.button("▶  Compare forecast vs actual", type="primary",
                     key="fva_run", use_container_width=True):
        st.stop()

    with st.spinner("Processing…"):
        # Forecast series
        fc_df[fc_date] = pd.to_datetime(fc_df[fc_date], dayfirst=True, errors="coerce")
        fc_df[fc_val]  = pd.to_numeric(
            fc_df[fc_val].astype(str).str.replace(",","",regex=False)
                         .str.replace("₦","",regex=False), errors="coerce"
        )
        fc_s = (
            fc_df.dropna(subset=[fc_date])
            .set_index(fc_date)[fc_val]
            .resample(fva_fcode).sum()
            .reset_index()
            .rename(columns={fc_date:"date", fc_val:"forecast"})
        )

        # Actual series
        act_c = clean_df(act_raw, act_date, act_val)
        act_s = (
            act_c.set_index(act_date)[act_val]
            .resample(fva_fcode).sum()
            .reset_index()
            .rename(columns={act_date:"date", act_val:"actual"})
        )

        merged = pd.merge(fc_s, act_s, on="date", how="inner")
        merged = merged[(merged["forecast"]>0)|(merged["actual"]>0)]

    if len(merged) == 0:
        st.error("No overlapping dates. Check that date ranges match between files.")
        st.stop()

    # Per-period metrics
    merged["error"]     = merged["actual"] - merged["forecast"]
    merged["smape_pct"] = merged.apply(
        lambda r: round(200*abs(r["actual"]-r["forecast"])/(abs(r["actual"])+abs(r["forecast"])+1e-8),2), axis=1
    )
    merged["mape_pct"]  = merged.apply(
        lambda r: round(abs(r["actual"]-r["forecast"])/r["actual"]*100,2)
        if r["actual"]>0 else np.nan, axis=1
    )
    merged["rating"] = merged["smape_pct"].apply(lambda v: accuracy_label(v)[0])

    date_fmt = "%b %Y" if fva_fcode=="MS" else "%d/%m/%Y"
    merged["period"] = merged["date"].dt.strftime(date_fmt)

    # Overall
    ya, yf = merged["actual"].values, merged["forecast"].values
    ov_rmse  = float(np.sqrt(mean_squared_error(ya,yf)))
    ov_mae   = float(mean_absolute_error(ya,yf))
    ov_sm    = calc_smape(ya,yf)
    ov_mp,nm = calc_mape(ya,yf,float(np.mean(ya)))
    ov_lbl, ov_cls = accuracy_label(ov_sm)

    st.markdown("---")
    st.markdown("### Overall accuracy")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("RMSE",   f"{unit_fva}{ov_rmse:,.0f}")
    m2.metric("MAE",    f"{unit_fva}{ov_mae:,.0f}")
    m3.metric("sMAPE",  f"{ov_sm:.2f}%")
    m4.metric("MAPE",   f"{ov_mp:.2f}%" if not np.isnan(ov_mp) else "N/A")
    m5.metric("Rating", ov_lbl)

    st.markdown(
        f'<div class="explain-box">'
        f"<strong>Overall accuracy: <span class='{ov_cls}'>{ov_lbl}</span></strong><br>"
        f"Across {len(merged)} overlapping periods — sMAPE {ov_sm:.2f}%, "
        f"MAE {unit_fva}{ov_mae:,.0f} per period. "
        f"MAPE evaluated on {nm}/{len(merged)} periods (zero-actual excluded)."
        f"</div>", unsafe_allow_html=True
    )
    acc_scale_table()

    # Chart
    st.markdown("### Forecast vs Actual chart")
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(merged["date"], merged["actual"],   color="#1565C0", linewidth=1.2,
            label="Actual", marker="o", markersize=4)
    ax.plot(merged["date"], merged["forecast"], color="#E53935", linewidth=1.2,
            linestyle="--", label="Forecast", marker="s", markersize=4)
    ax.fill_between(merged["date"],
                    np.minimum(merged["actual"],merged["forecast"]),
                    np.maximum(merged["actual"],merged["forecast"]),
                    alpha=0.12, color="#E53935", label="Error Band")
    ax.set_title("Forecast vs Actual", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Value ({unit_fva})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Detailed table
    st.markdown("### Period-by-period breakdown")
    disp = merged[["period","forecast","actual","error","smape_pct","mape_pct","rating"]].copy()
    disp.columns = ["Period",f"Forecast ({unit_fva})",f"Actual ({unit_fva})",
                    f"Error ({unit_fva})","sMAPE (%)","MAPE (%)","Rating"]
    for c in [f"Forecast ({unit_fva})",f"Actual ({unit_fva})",f"Error ({unit_fva})"]:
        disp[c] = disp[c].apply(lambda v: f"{v:,.0f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    dl = merged[["period","forecast","actual","error","smape_pct","mape_pct","rating"]].copy()
    dl.columns = ["period",f"forecast_{unit_fva}",f"actual_{unit_fva}",
                  f"error_{unit_fva}","smape_pct","mape_pct","rating"]
    st.download_button("⬇  Download comparison as CSV",
        dl.to_csv(index=False).encode(),
        "forecast_vs_actual.csv","text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════
if page == "📊 Training Results":
    st.title("Training Results")
    st.markdown("All graphs from the original model training run. Updated automatically when `lstm_forecasting.py` is re-run.")

    graphs = [
        ("01_daily_sales_timeseries.png","Daily total sales time series",
         "The top panel shows raw daily total sales (Jan 2023–Nov 2025). "
         "The bottom panel adds a 30-day rolling average (red) to reveal the underlying trend. "
         "Sales grew from early 2023, peaked around late 2024, and declined toward Nov 2025. "
         "This declining tail formed the test period — the hardest forecasting scenario. "
         "High volatility reflects irregular bulk ordering typical of B2B e-commerce."),
        ("02_training_loss.png","LSTM training vs validation loss",
         "The blue line is training loss (MSE) per epoch. The red line is validation loss. "
         "Both drop steeply in the first few epochs then level off running close together "
         "with no upward divergence — confirming clean convergence with no overfitting. "
         "Early stopping fired after validation loss showed no further improvement "
         "and best weights were automatically restored."),
        ("03_predictions_vs_actual.png","LSTM forecast vs actual (test period)",
         "Blue = actual daily sales in the held-out test period (Jun–Nov 2025). "
         "Red dashed = LSTM predictions generated without the model ever seeing this period. "
         "The model captures the general downward trend correctly but smooths individual spikes — "
         "expected for a univariate model with no promotional signals in the input. "
         "For strategic planning, trend-level accuracy is the most operationally relevant output."),
        ("04_model_comparison.png","Model performance metrics",
         "Bar charts comparing RMSE, MAE, MAPE, and sMAPE across the models evaluated. "
         "Lower = better. sMAPE is the primary percentage metric — bounded 0–200% and not "
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
            st.warning(f"`{fname}` not found — run `python lstm_forecasting.py`.")

    st.markdown("---")
    st.markdown("### Live metrics table")
    if disp_df is not None:
        st.success("Loaded from `outputs/model_comparison.csv` — updates on re-run.")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Run `python lstm_forecasting.py` to generate the comparison CSV.")

    st.markdown("---")
    st.markdown("### Model configuration")
    c1,c2 = st.columns(2)
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
if page == "ℹ️ About":
    st.title("About This Project")
    st.markdown("""
### Design and Development of an LSTM-Based Product Demand Forecasting System
**Postgraduate Diploma Research** — Nigerian B2B E-Commerce (Anonymous)

---
### Minimum CSV format
```
orderDate,final_amount
05/01/2023,125000
06/01/2023,340500
```
For quantity forecasting, add `quantitySold`.  
For category/product filtering, add `productId` (format: `NGA-FDI-PST-000025`) and `displayTitle`.

---
### Product filtering options
| Mode | How it works |
|---|---|
| All Products | Aggregates total sales across every product |
| Product Category | Filters by category prefix in productId (FDI=Food, BEV=Beverages, HME=Home, PRF=Personal/Cooking) |
| Specific Products (up to 10) | You select individual products by name; their sales are summed |

> Note: The LSTM model was trained on aggregate daily total sales. Category and product filters
> are applied before aggregation to produce a filtered series — the model then forecasts
> the total of that filtered group. The rolling window (7/15/30 days) only affects
> the display chart and does not change the model's predictions.

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
### Run locally
```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib
streamlit run app.py
```
    """)
