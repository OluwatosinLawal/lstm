# =============================================================================
#  LSTM DEMAND FORECASTING — STREAMLIT APP  (v6)
#  Run : streamlit run app.py
#
#  Fixes in this version:
#  - Sunday exclusion from daily series and future forecast dates
#  - Monthly look-back scales to 12 periods (not 60)
#  - Optional columns can be set to "— None —"
#  - UTF-8 → latin-1 → cp1252 encoding fallback for CSV uploads
#  - Forecast vs Actual: product/category filter applied to actual data
#  - Per-product breakdown table in Forecast vs Actual
#  - dynamic look-back passed through rolling_forecast
# =============================================================================

import os, pickle, warnings
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
LOOK_BACK   = 60   # default (daily); monthly uses 12 — see get_lb()

NONE_LABEL  = "— None —"   # sentinel for optional column dropdowns


# ══════════════════════════════════════════════════════════════════
# ACCURACY HELPERS
# ══════════════════════════════════════════════════════════════════

def accuracy_label(smape_val):
    if smape_val < 10:   return "Excellent / Highly Accurate", "acc-excellent"
    elif smape_val < 25: return "Good", "acc-good"
    elif smape_val < 50: return "Reasonable / Fair", "acc-fair"
    else:                return "Inaccurate", "acc-poor"


def acc_scale_table():
    """Renders accuracy scale — called only inside results sections."""
    st.markdown("""
**Accuracy scale reference**

| MAPE | sMAPE | Rating |
|---|---|---|
| < 10% | < 10% | Excellent / Highly Accurate |
| 10% – 20% | 10% – 25% | Good |
| 20% – 50% | 25% – 50% | Reasonable / Fair |
| > 50% | > 50% | Inaccurate |
""")


def get_lb(freq):
    """Return appropriate look-back window for the chosen frequency."""
    return 12 if freq == "MS" else LOOK_BACK


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


def read_csv_safe(f):
    """Read a single CSV with UTF-8 → latin-1 → cp1252 fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            f.seek(0)
            d = pd.read_csv(f, low_memory=False, encoding=enc)
            d.columns = d.columns.str.strip()
            return d
        except (UnicodeDecodeError, Exception):
            continue
    f.seek(0)
    d = pd.read_csv(f, low_memory=False, encoding="latin-1", errors="replace")
    d.columns = d.columns.str.strip()
    return d


def read_and_merge(files):
    """Merge multiple uploaded CSV files into one DataFrame."""
    frames = [read_csv_safe(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else None


def clean_df(df, date_col, val_col, qty_col=None):
    """
    Strip whitespace, convert numeric columns, parse dates.
    qty_col may be None if user chose — None —.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Only apply str operations to object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda c: c.str.strip())

    numeric_targets = [val_col]
    if qty_col:
        numeric_targets.append(qty_col)
    for col in numeric_targets + ["unitPrice", "orderTotal"]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    return df


def get_categories(df, pid_col):
    """Extract product categories from productId prefix (NGA-XXX-)."""
    if not pid_col or pid_col not in df.columns:
        return {}
    label_map = {
        "FDI": "Food Items (FDI)",
        "BEV": "Beverages (BEV)",
        "HME": "Home & Household (HME)",
        "PRF": "Personal & Cooking (PRF)",
        "PHA": "Pharmacy (PHA)",
        "AGR": "Agriculture (AGR)",
    }
    extracted = df[pid_col].dropna().astype(str).str.extract(r"NGA-([A-Z]+)-")[0]
    codes = extracted.dropna().unique()
    return {label_map.get(c, f"Category {c}"): c for c in sorted(codes)}


def aggregate(df, date_col, val_col, freq,
              prod_col=None, pid_col=None, cat_code=None, products=None):
    """
    Filter then aggregate to (date, total) series at chosen frequency.
    Sundays are excluded from daily aggregation.
    Returns a DataFrame with columns [date, total], or None if empty.
    """
    d = df.copy()

    # Category filter
    if cat_code and cat_code != "ALL" and pid_col and pid_col in d.columns:
        d = d[d[pid_col].astype(str).str.contains(f"NGA-{cat_code}-", na=False)]

    # Specific product filter
    if products and prod_col and prod_col in d.columns:
        d = d[d[prod_col].isin(products)]

    if len(d) == 0:
        return None

    series = (
        d.groupby(date_col)[val_col].sum()
        .reset_index()
        .rename(columns={date_col: "date", val_col: "total"})
        .sort_values("date")
    )
    series = (
        series.set_index("date")
        .resample(freq)["total"].sum()
        .reset_index()
    )
    series = series[series["total"] > 0].reset_index(drop=True)

    # Remove bottom 1% outlier periods — daily only.
    # Monthly totals are never near-zero anomalies so no removal needed.
    if freq == "D":
        thresh = series["total"].quantile(0.01)
        series = series[series["total"] > thresh].reset_index(drop=True)
        # Exclude Sundays from daily series
        series = series[series["date"].dt.weekday != 6].reset_index(drop=True)

    return series if len(series) > 0 else None


def breakdown_by_group(df, date_col, val_col, group_col, freq):
    """
    Returns a pivoted breakdown DataFrame: rows = period, cols = each product/category.
    Used for per-product/category breakdown tables.
    """
    if not group_col or group_col not in df.columns:
        return None
    df = df.copy()
    df["_period"] = df[date_col].dt.to_period(
        "M" if freq == "MS" else "D"
    ).dt.to_timestamp()
    gb = (
        df.groupby(["_period", group_col])[val_col]
        .sum().reset_index()
        .rename(columns={"_period": "period", group_col: "group", val_col: "total"})
    )
    gb = gb[gb["total"] > 0]
    fmt = "%b %Y" if freq == "MS" else "%d/%m/%Y"
    gb["period"] = gb["period"].dt.strftime(fmt)
    pivot = gb.pivot_table(index="period", columns="group",
                           values="total", aggfunc="sum", fill_value=0)
    pivot = pivot.reset_index()
    return pivot


def make_sequences(data, lb):
    X, y = [], []
    for i in range(lb, len(data)):
        X.append(data[i-lb:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def calc_smape(yt, yp):
    return float(np.mean(2*np.abs(yt-yp) / (np.abs(yt)+np.abs(yp)+1e-8)) * 100)


def calc_mape(yt, yp, mean_v):
    mask = yt > mean_v * 0.01
    if mask.sum() == 0:
        return np.nan, 0
    return float(np.mean(np.abs((yt[mask]-yp[mask]) / yt[mask])) * 100), int(mask.sum())


def rolling_forecast(model, sc, seed, n, lb=LOOK_BACK):
    """
    Generate n one-step-ahead predictions using rolling inference.
    lb: look-back window (60 for daily, 12 for monthly).
    """
    use_seed = seed[-lb:] if len(seed) >= lb else seed
    s = sc.transform(use_seed.reshape(-1, 1)).flatten()
    w = list(s)
    out = []
    for _ in range(n):
        x = np.array(w[-lb:]).reshape(1, lb, 1)
        p = model.predict(x, verbose=0)[0, 0]
        out.append(p)
        w.append(p)
    return np.maximum(
        sc.inverse_transform(np.array(out).reshape(-1, 1)).flatten(), 0
    )


def make_future_dates(last_date, n_periods, freq):
    """
    Generate n_periods future dates starting after last_date.
    For daily freq, Sundays are excluded so the count reflects trading days only.
    """
    if freq == "D":
        dates = []
        current = pd.Timestamp(last_date)
        while len(dates) < n_periods:
            current += pd.Timedelta(days=1)
            if current.weekday() != 6:   # skip Sunday
                dates.append(current)
        return pd.DatetimeIndex(dates)
    else:
        return pd.date_range(
            start=pd.Timestamp(last_date) + pd.DateOffset(months=1),
            periods=n_periods, freq="MS"
        )


def make_future_dates_range(start_date, end_date, freq):
    """
    Generate dates between start and end for the given frequency.
    Sundays are excluded for daily frequency.
    """
    if freq == "D":
        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DatetimeIndex([d for d in all_dates if d.weekday() != 6])
    else:
        return pd.date_range(start=start_date, end=end_date, freq="MS")


def show_col_guide():
    st.markdown("""
| Column | Status | Description |
|---|---|---|
| `orderDate` | **Required** | Date — format DD/MM/YYYY |
| `final_amount` | **Required*** | Sales value per order line (₦) — plain numbers, no commas |
| `quantitySold` | Optional | Units sold — select if forecasting quantity |
| `displayTitle` | Optional | Product name — needed for product-level filtering |
| `productId` | Optional | SKU code — needed for category filtering (e.g. NGA-FDI-...) |
| `salesCategory` | Optional | Regular Sales or Promo Sales |

> * `final_amount` or `quantitySold` — at least one required depending on forecast target.  
> Column names are **case-sensitive**. Multiple files with identical columns are merged automatically.  
> If your dataset does not have optional columns, select **— None —** in those dropdowns.
""")


# ══════════════════════════════════════════════════════════════════
# SHARED UPLOAD + CONFIGURE
# ══════════════════════════════════════════════════════════════════

def upload_and_configure(pk):
    """
    Shared upload/config widget. pk = page key ("uf" or "fp").
    Returns (series, cfg) after user clicks Load, else (None, None).
    """
    with st.expander("📋  Column requirements", expanded=False):
        show_col_guide()

    st.markdown("### Step 1 — Upload CSV file(s)")
    st.caption(
        "Upload **one or more CSV files** — they are merged automatically "
        "as long as all files share the same column names."
    )
    files = st.file_uploader(
        "Upload sales CSV file(s)",
        type=["csv"], accept_multiple_files=True, key=f"{pk}_files"
    )
    if not files:
        st.info("Waiting for file upload…")
        return None, None

    raw = read_and_merge(files)
    if raw is None or len(raw) == 0:
        st.error("No data found in uploaded files.")
        return None, None

    st.success(
        f"✔ {len(files)} file(s) merged — {len(raw):,} rows | "
        f"Columns: `{'`, `'.join(raw.columns.tolist())}`"
    )

    # ── Step 2: Column mapping ─────────────────────────────────────
    st.markdown("### Step 2 — Map your columns")
    cols      = list(raw.columns)
    opt_cols  = [NONE_LABEL] + cols   # options list that includes None for optional fields

    def pick_required(label, hints, key):
        """Selectbox with no None option — for required columns."""
        default = next((c for c in hints if c in cols), cols[0])
        return st.selectbox(label, cols, index=cols.index(default), key=f"{pk}_{key}")

    def pick_optional(label, hints, key):
        """Selectbox with — None — option — for optional columns."""
        default = next((c for c in hints if c in cols), NONE_LABEL)
        idx = opt_cols.index(default)
        chosen = st.selectbox(label, opt_cols, index=idx, key=f"{pk}_{key}")
        return None if chosen == NONE_LABEL else chosen

    c1, c2, c3, c4 = st.columns(4)
    with c1: date_col = pick_required("Date column",                  ["orderDate"],    "date")
    with c2: amt_col  = pick_required("Sales amount column",          ["final_amount"], "amt")
    with c3: qty_col  = pick_optional("Quantity column (optional)",   ["quantitySold"], "qty")
    with c4: prod_col = pick_optional("Product name (optional)",      ["displayTitle"], "prod")
    pid_col = pick_optional("Product ID column (optional — for categories)", ["productId"], "pid")

    # ── Step 3: Forecast configuration ────────────────────────────
    st.markdown("### Step 3 — Forecast configuration")

    fc1, fc2 = st.columns(2)
    with fc1:
        freq_label = st.selectbox(
            "Aggregation level",
            ["Daily", "Monthly"],
            key=f"{pk}_agg",
            help="Monthly is more stable and better suited to planning cycles."
        )
        freq = "D" if freq_label == "Daily" else "MS"

    with fc2:
        tgt_options = ["Sales Amount (₦)"]
        if qty_col:
            tgt_options.append("Sales Quantity (units)")
        target_label = st.selectbox(
            "Forecast target", tgt_options, key=f"{pk}_tgt"
        )
        val_col = amt_col if "Amount" in target_label else qty_col
        unit    = "₦"     if "Amount" in target_label else "units"

    # Rolling average window (display only)
    if freq == "D":
        roll_win = st.select_slider(
            "Chart rolling average window — display only, does not affect the model",
            options=[7, 15, 30], value=30, key=f"{pk}_roll"
        )
    else:
        roll_win = 3

    # ── Product / category filter ──────────────────────────────────
    st.markdown("#### Product filter")
    cats = get_categories(raw, pid_col) if pid_col else {}

    filter_mode = st.radio(
        "Filter by:",
        ["All Products", "Product Category", "Specific Products (up to 10)"],
        horizontal=True, key=f"{pk}_fmode"
    )

    cat_code     = "ALL"
    chosen_prods = None
    cat_label_display = "All Products"

    if filter_mode == "Product Category":
        if cats:
            chosen_cat_label   = st.selectbox("Select category", list(cats.keys()), key=f"{pk}_cat")
            cat_code           = cats[chosen_cat_label]
            cat_label_display  = chosen_cat_label
        else:
            st.warning(
                "No product categories detected. "
                "Ensure the Product ID column is set and follows the NGA-XXX- format."
            )

    elif filter_mode == "Specific Products (up to 10)":
        if prod_col and prod_col in raw.columns:
            all_prods    = sorted(raw[prod_col].dropna().unique().tolist())
            chosen_prods = st.multiselect(
                "Select up to 10 products",
                options=all_prods,
                max_selections=10,
                key=f"{pk}_prods",
                help="Their sales will be summed to form a single forecast series."
            )
            if not chosen_prods:
                st.info("Select at least one product to continue.")
                return None, None
            cat_label_display = f"{len(chosen_prods)} product(s)"
        else:
            st.warning(
                "Product name column is set to None or not found. "
                "Map the Product name column in Step 2 to use this filter."
            )

    # ── Load button ────────────────────────────────────────────────
    if st.button("▶  Load & process data", key=f"{pk}_load", type="primary"):
        with st.spinner("Cleaning and aggregating…"):
            df_c   = clean_df(raw, date_col, val_col, qty_col)
            series = aggregate(
                df_c, date_col, val_col, freq,
                prod_col=prod_col, pid_col=pid_col,
                cat_code=cat_code, products=chosen_prods
            )

        if series is None or len(series) == 0:
            st.error("No data after filtering and aggregation. Try a broader filter or upload more data.")
            return None, None

        lb          = get_lb(freq)
        # Monthly: need lb + 1 (13 months minimum for one full look-back + 1 target).
        # Daily:   need lb + 5 to have a meaningful test split.
        min_periods = lb + 1 if freq == "MS" else lb + 5

        if len(series) < min_periods:
            unit_word = "months" if freq == "MS" else "trading days"
            st.error(
                f"Only {len(series)} {unit_word} after filtering. "
                f"Need at least {min_periods}. "
                f"{'Upload at least 13 months of data.' if freq == 'MS' else 'Upload more data, switch to a broader product filter, or try Monthly aggregation.'}"
            )
            return None, None

        # Store results in session state with unique per-page keys
        st.session_state[f"{pk}_s_series"]  = series
        st.session_state[f"{pk}_s_unit"]    = unit
        st.session_state[f"{pk}_s_freq"]    = freq
        st.session_state[f"{pk}_s_flabel"]  = freq_label
        st.session_state[f"{pk}_s_cat"]     = cat_label_display
        st.session_state[f"{pk}_s_target"]  = target_label
        st.session_state[f"{pk}_s_roll"]    = roll_win
        st.session_state[f"{pk}_s_prods"]   = chosen_prods
        st.session_state[f"{pk}_s_df"]      = df_c
        st.session_state[f"{pk}_s_vcol"]    = val_col
        st.session_state[f"{pk}_s_pcol"]    = prod_col
        st.session_state[f"{pk}_s_pidcol"]  = pid_col
        st.session_state[f"{pk}_s_catcode"] = cat_code

        mean_disp = series["total"].mean()
        st.success(
            f"✔ {len(series)} {freq_label.lower()} periods | "
            f"{series['date'].min().date()} → {series['date'].max().date()} | "
            f"Mean: {'₦' if unit=='₦' else ''}{mean_disp:,.0f}"
            f"{'' if unit=='₦' else ' '+unit}"
        )

    if f"{pk}_s_series" not in st.session_state:
        return None, None

    cfg = {k.replace(f"{pk}_s_",""): v
           for k, v in st.session_state.items() if k.startswith(f"{pk}_s_")}
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
        st.caption("Vanilla LSTM · 64 units · 60-day / 12-month look-back")
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
        [c1, c2, c3, c4],
        ["Model", "Training period", "Look-back window", "Test sMAPE"],
        ["Vanilla LSTM", "Jan 2023 – Nov 2025", "60 days / 12 months", smape_val]
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
        st.write("Evaluate the model on historical data. Filter by category or specific products, choose daily/monthly, sales amount or quantity.")
    with b:
        st.markdown("**🔮 Future Predictions**")
        st.write("Generate forward forecasts for a custom date range or number of periods. Sundays automatically excluded from daily forecasts.")
    with c:
        st.markdown("**✅ Forecast vs Actual**")
        st.write("Upload a forecast and actual data. Compare side by side with per-period MAPE, sMAPE, and accuracy rating. Download as CSV.")
    with d:
        st.markdown("**📊 Training Results**")
        st.write("All four original training graphs with explanations.")
    with e:
        st.markdown("**ℹ️ About**")
        st.write("Column guide, model details, and project background.")

    if model is None:
        st.error(f"Model not loaded: {load_err}")
        st.info("Run `python lstm_forecasting.py` then refresh.")


# ══════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & FORECAST
# ══════════════════════════════════════════════════════════════════
if page == "📂 Upload & Forecast":
    st.title("Upload Data & Evaluate Forecast")
    st.markdown(
        "Upload historical sales data to evaluate how accurately the LSTM model "
        "forecasts it. Sundays are automatically excluded from daily series."
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
    lb     = get_lb(freq)

    st.markdown("---")
    st.markdown("### Step 4 — Run evaluation")

    n   = len(series)
    t_e = int(n * 0.70)
    v_e = int(n * 0.85)

    sales  = series["total"].values.reshape(-1, 1)
    sc     = MinMaxScaler()
    sc.fit(sales[:t_e])
    scaled = sc.transform(sales)

    X_all, y_all = make_sequences(scaled, lb)
    adj_t = t_e - lb
    adj_v = v_e - lb
    X_te  = X_all[adj_v:].reshape(-1, lb, 1)
    y_te  = y_all[adj_v:]

    with st.spinner("Running model inference…"):
        yp_s    = model.predict(X_te, verbose=0)
    y_pred   = sc.inverse_transform(yp_s).flatten()
    y_actual = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()
    dates    = series["date"].values[n - len(y_pred):]

    mean_v   = float(series["total"].mean())
    rmse_v   = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae_v    = float(mean_absolute_error(y_actual, y_pred))
    sm_v     = calc_smape(y_actual, y_pred)
    mp_v, nm = calc_mape(y_actual, y_pred, mean_v)
    lbl, cls = accuracy_label(sm_v)

    pfx = "₦" if unit == "₦" else ""
    sfx = "" if unit == "₦" else f" {unit}"

    st.markdown("### Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",   f"{pfx}{rmse_v:,.0f}{sfx}")
    m2.metric("MAE",    f"{pfx}{mae_v:,.0f}{sfx}")
    m3.metric("sMAPE",  f"{sm_v:.2f}%")
    m4.metric("MAPE",   f"{mp_v:.2f}%" if not np.isnan(mp_v) else "N/A")
    m5.metric("Rating", lbl)

    st.markdown(
        f'<div class="explain-box">'
        f"<strong>Accuracy rating: <span class='{cls}'>{lbl}</span></strong><br><br>"
        f"<strong>RMSE</strong> penalises large errors more heavily — "
        f"{pfx}{rmse_v:,.0f}{sfx} is {rmse_v/mean_v*100:.1f}% of the mean {flabel.lower()} value.<br><br>"
        f"<strong>MAE</strong> is the average absolute error per period — "
        f"{pfx}{mae_v:,.0f}{sfx} ({mae_v/mean_v*100:.1f}% of mean).<br><br>"
        f"<strong>sMAPE ({sm_v:.2f}%)</strong> is the primary percentage metric. "
        f"Rated <em>{lbl}</em> on the scale below.<br><br>"
        f"<strong>MAPE ({mp_v:.2f}%)</strong> evaluated on {nm}/{len(y_pred)} periods "
        f"(near-zero periods excluded to prevent division-by-zero errors)."
        f"</div>",
        unsafe_allow_html=True
    )
    acc_scale_table()

    # Graph 1: full series
    st.markdown("#### Full series — historical data")
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(series["date"], series["total"],
             color="#1565C0", linewidth=0.8, alpha=0.8, label="Sales")
    roll_s = series["total"].rolling(window=roll, min_periods=1).mean()
    ax1.plot(series["date"], roll_s, color="#E53935", linewidth=1.5,
             label=f"{roll}-period rolling avg")
    ax1.set_title(f"{flabel} {cfg['target']} — {cfg['cat']}", fontsize=12, fontweight="bold")
    ax1.set_ylabel(f"{cfg['target']} ({unit})")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig1); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> The blue line is the raw sales total per period. "
        "The red line is the rolling average — it smooths short-term spikes to expose the "
        "underlying demand trend. Wide swings are typical of B2B bulk ordering patterns."
        "</div>", unsafe_allow_html=True
    )

    # Graph 2: forecast vs actual
    st.markdown("#### Forecast vs Actual (test period)")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(pd.to_datetime(dates), y_actual,
             color="#1565C0", linewidth=1.0, label="Actual")
    ax2.plot(pd.to_datetime(dates), y_pred,
             color="#E53935", linewidth=1.2, linestyle="--", label="LSTM Forecast")
    ax2.fill_between(pd.to_datetime(dates),
                     np.minimum(y_actual, y_pred), np.maximum(y_actual, y_pred),
                     alpha=0.12, color="#E53935", label="Error Band")
    ax2.set_title("LSTM Forecast vs Actual", fontsize=12, fontweight="bold")
    ax2.set_ylabel(f"{cfg['target']} ({unit})")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading this chart:</strong> Blue = actual recorded values. "
        "Red dashed = model predictions. A narrower pink error band means more accurate predictions. "
        "The model captures the overall demand trend but smooths individual-period spikes — "
        "expected for a model that has no access to promotional or external demand signals."
        "</div>", unsafe_allow_html=True
    )

    # Graph 3: error analysis
    st.markdown("#### Error analysis")
    errors = y_actual - y_pred
    fig3, ax3 = plt.subplots(1, 2, figsize=(14, 4))
    ax3[0].bar(range(len(errors)), errors,
               color=["#E53935" if e > 0 else "#1565C0" for e in errors], alpha=0.7)
    ax3[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3[0].set_title("Error per Period (Actual − Predicted)", fontsize=11, fontweight="bold")
    ax3[0].set_ylabel(f"Error ({unit})"); ax3[0].grid(True, alpha=0.3)

    ax3[1].hist(errors, bins=25, color="#7F77DD", edgecolor="white", alpha=0.85)
    ax3[1].axvline(0, color="black", linewidth=1.0, linestyle="--", label="Zero error")
    ax3[1].axvline(float(np.mean(errors)), color="#E53935", linewidth=1.2,
                   label="Mean error")
    ax3[1].set_title("Error Distribution", fontsize=11, fontweight="bold")
    ax3[1].set_xlabel(f"Error ({unit})"); ax3[1].set_ylabel("Periods")
    ax3[1].legend(); ax3[1].grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown(
        '<div class="explain-box">'
        "<strong>Reading the error charts:</strong> "
        "Red bars = the model under-predicted (actual was higher than forecast). "
        "Blue bars = the model over-predicted. "
        "The histogram shows whether errors are centred on zero (no bias) "
        "or consistently skewed in one direction."
        "</div>", unsafe_allow_html=True
    )

    # Download
    out = pd.DataFrame({
        "period":    pd.to_datetime(dates).strftime("%d/%m/%Y" if freq=="D" else "%b %Y"),
        "actual":    np.round(y_actual, 2),
        "predicted": np.round(y_pred,   2),
        "error":     np.round(errors,   2),
        "smape_pct": [round(calc_smape(np.array([a]), np.array([p])), 2)
                      for a, p in zip(y_actual, y_pred)],
    })
    st.download_button(
        "⬇  Download forecast results CSV",
        out.to_csv(index=False).encode(),
        "forecast_results.csv", "text/csv", use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: FUTURE PREDICTIONS
# ══════════════════════════════════════════════════════════════════
if page == "🔮 Future Predictions":
    st.title("Future Sales Forecast")
    st.markdown(
        "Generate forward-looking predictions from recent sales data. "
        "Sundays are automatically excluded from daily forecasts."
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
    lb     = get_lb(freq)

    st.markdown("---")
    st.markdown("### Step 4 — Forecast horizon")

    hz1, hz2 = st.columns(2)
    with hz1:
        h_type = st.radio(
            "Specify forecast period by:",
            ["Number of periods", "Date range"],
            horizontal=True, key="fp_htype"
        )
    with hz2:
        last_date = pd.Timestamp(series["date"].iloc[-1])

        if h_type == "Number of periods":
            n_periods = st.number_input(
                f"Number of {flabel.lower()} periods to forecast",
                min_value=1,
                max_value=365 if freq == "D" else 36,
                value=30 if freq == "D" else 6,
                key="fp_nper"
            )
            future_dates = make_future_dates(last_date, int(n_periods), freq)

        else:
            ca, cb = st.columns(2)
            with ca:
                s_date = st.date_input(
                    "Forecast start date",
                    value=(last_date + pd.Timedelta(days=1)).date(),
                    key="fp_sdate"
                )
            with cb:
                e_date = st.date_input(
                    "Forecast end date",
                    value=(last_date + pd.Timedelta(days=30)).date(),
                    key="fp_edate"
                )
            future_dates = make_future_dates_range(s_date, e_date, freq)
            n_periods    = len(future_dates)

    if n_periods == 0:
        st.warning("No forecast periods generated. Adjust your date range.")
        st.stop()

    if freq == "D":
        st.caption(
            f"ℹ️ {n_periods} trading day(s) selected — Sundays excluded automatically."
        )

    if not st.button("▶  Generate forecast", type="primary",
                     key="fp_run", use_container_width=True):
        st.stop()

    seed = series["total"].values
    sc   = MinMaxScaler()
    sc.fit(seed.reshape(-1, 1))

    with st.spinner(f"Forecasting {n_periods} {flabel.lower()} period(s)…"):
        preds = rolling_forecast(model, sc, seed, int(n_periods), lb=lb)

    fdf = pd.DataFrame({
        "date":     future_dates[:len(preds)],
        "forecast": np.round(preds[:len(future_dates)], 2)
    })

    pfx = "₦" if unit == "₦" else ""
    sfx = "" if unit == "₦" else f" {unit}"

    st.markdown("---")
    st.markdown("### Forecast results")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total ({flabel.lower()})", f"{pfx}{preds.sum():,.0f}{sfx}")
    c2.metric("Average per period",        f"{pfx}{preds.mean():,.0f}{sfx}")
    c3.metric("Peak period",               f"{pfx}{preds.max():,.0f}{sfx}")

    # Chart: history + forecast
    fig, ax = plt.subplots(figsize=(14, 5))
    hist   = series.tail(12 if freq == "MS" else 90)
    roll_h = hist["total"].rolling(window=roll, min_periods=1).mean()
    ax.plot(hist["date"], hist["total"],
            color="#1565C0", linewidth=1.0, label="Historical", alpha=0.8)
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

    # Bar chart
    bar_w  = max(12, n_periods // 2)
    fig2, ax2 = plt.subplots(figsize=(bar_w, 4))
    fmt = "%b %Y" if freq == "MS" else "%d %b"
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
        "The vertical dotted line marks where history ends. "
        "Red dashed + bars = predicted values. "
        "<em>Uncertainty grows further into the future</em> — "
        "later periods are directional estimates, not precise targets."
        "</div>", unsafe_allow_html=True
    )

    # Table
    show = fdf.copy()
    show["date"]     = show["date"].dt.strftime("%b %Y" if freq == "MS" else "%A, %d %b %Y")
    show["forecast"] = show["forecast"].apply(lambda v: f"{pfx}{v:,.0f}{sfx}")
    show.columns     = ["Period", f"Forecast ({unit})"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Download
    dl = fdf.copy()
    dl["date"] = dl["date"].dt.strftime("%d/%m/%Y")
    dl.columns = ["date", f"forecast_{unit}"]
    st.download_button(
        "⬇  Download forecast CSV  (use this on the Forecast vs Actual page)",
        dl.to_csv(index=False).encode(),
        f"forecast_{n_periods}{freq}.csv", "text/csv",
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: FORECAST vs ACTUAL
# ══════════════════════════════════════════════════════════════════
if page == "✅ Forecast vs Actual":
    st.title("Forecast vs Actual Comparison")
    st.markdown(
        "Upload the forecast file (generated on the Future Predictions page) "
        "and the actual sales data for the same period. "
        "The app aligns by date, computes accuracy metrics per period, "
        "rates each using the accuracy scale, and shows a per-product breakdown."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Forecast file** (from the Future Predictions download)")
        fc_file = st.file_uploader("Upload forecast CSV", type=["csv"], key="fva_fc")
    with col2:
        st.markdown("**Actual sales file(s)**")
        act_files = st.file_uploader(
            "Upload actual sales CSV(s)", type=["csv"],
            accept_multiple_files=True, key="fva_act"
        )

    if not fc_file or not act_files:
        st.info("Upload both files to continue.")
        st.stop()

    # ── Read files ────────────────────────────────────────────────
    fc_df   = read_csv_safe(fc_file)
    act_raw = read_and_merge(act_files)
    fc_cols  = list(fc_df.columns)
    act_cols = list(act_raw.columns)
    act_opt  = [NONE_LABEL] + act_cols

    st.markdown("### Step 2 — Column configuration")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        fc_date = st.selectbox("Forecast: date column", fc_cols,
            index=fc_cols.index("date") if "date" in fc_cols else 0, key="fva_fcdate")
    with r1c2:
        fc_val  = st.selectbox("Forecast: value column", fc_cols,
            index=1 if len(fc_cols) > 1 else 0, key="fva_fcval")
    with r1c3:
        act_date = st.selectbox("Actual: date column", act_cols,
            index=act_cols.index("orderDate") if "orderDate" in act_cols else 0,
            key="fva_actdate")
    with r1c4:
        act_val  = st.selectbox("Actual: value column", act_cols,
            index=act_cols.index("final_amount") if "final_amount" in act_cols else 0,
            key="fva_actval")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        fva_freq  = st.selectbox("Aggregation", ["Daily", "Monthly"], key="fva_freq")
        fva_fcode = "D" if fva_freq == "Daily" else "MS"
    with r2c2:
        unit_fva = st.selectbox("Unit label", ["₦", "units"], key="fva_unit")
    with r2c3:
        pid_col_fva  = st.selectbox("Product ID column (for category filter)",
                                    act_opt, key="fva_pid")
        pid_col_fva  = None if pid_col_fva == NONE_LABEL else pid_col_fva
    prod_col_fva = st.selectbox("Product name column (for breakdown & filter)",
                                act_opt, key="fva_pname")
    prod_col_fva = None if prod_col_fva == NONE_LABEL else prod_col_fva

    # ── Product/category filter for actual ───────────────────────
    st.markdown("#### Filter actual data to match your forecast selection")
    fva_filter = st.radio(
        "Filter actual by:",
        ["All Products", "Product Category", "Specific Products"],
        horizontal=True, key="fva_fmode"
    )
    fva_cat_code  = None
    fva_prod_list = None

    if fva_filter == "Product Category" and pid_col_fva:
        cats_fva = get_categories(act_raw, pid_col_fva)
        if cats_fva:
            chosen_fva = st.selectbox("Category", list(cats_fva.keys()), key="fva_cat_sel")
            fva_cat_code = cats_fva[chosen_fva]
        else:
            st.warning("No categories detected in actual data.")

    elif fva_filter == "Specific Products" and prod_col_fva:
        all_p_fva    = sorted(act_raw[prod_col_fva].dropna().unique().tolist())
        fva_prod_list = st.multiselect(
            "Select products to include in actual",
            all_p_fva, max_selections=10, key="fva_prod_sel"
        )

    if not st.button("▶  Compare forecast vs actual", type="primary",
                     key="fva_run", use_container_width=True):
        st.stop()

    # ── Process ───────────────────────────────────────────────────
    with st.spinner("Processing…"):
        # Forecast series
        fc_df[fc_date] = pd.to_datetime(fc_df[fc_date], dayfirst=True, errors="coerce")
        fc_df[fc_val]  = pd.to_numeric(
            fc_df[fc_val].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₦", "", regex=False),
            errors="coerce"
        )
        fc_s = (
            fc_df.dropna(subset=[fc_date])
            .set_index(fc_date)[fc_val]
            .resample(fva_fcode).sum()
            .reset_index()
            .rename(columns={fc_date: "date", fc_val: "forecast"})
        )
        # Remove Sundays from daily forecast series
        if fva_fcode == "D":
            fc_s = fc_s[fc_s["date"].dt.weekday != 6].reset_index(drop=True)

        # Actual series — apply filter
        act_c = clean_df(act_raw, act_date, act_val)

        if fva_filter == "Product Category" and fva_cat_code and pid_col_fva:
            act_c = act_c[
                act_c[pid_col_fva].astype(str)
                .str.contains(f"NGA-{fva_cat_code}-", na=False)
            ]
        elif fva_filter == "Specific Products" and fva_prod_list and prod_col_fva:
            act_c = act_c[act_c[prod_col_fva].isin(fva_prod_list)]

        act_s = (
            act_c.set_index(act_date)[act_val]
            .resample(fva_fcode).sum()
            .reset_index()
            .rename(columns={act_date: "date", act_val: "actual"})
        )
        # Remove Sundays from daily actual series
        if fva_fcode == "D":
            act_s = act_s[act_s["date"].dt.weekday != 6].reset_index(drop=True)

        merged = pd.merge(fc_s, act_s, on="date", how="inner")
        merged = merged[(merged["forecast"] > 0) | (merged["actual"] > 0)]

    if len(merged) == 0:
        st.error(
            "No overlapping dates found between forecast and actual. "
            "Check that the date ranges match and the same aggregation level is used."
        )
        st.stop()

    # Per-period metrics
    merged["error"]     = merged["actual"] - merged["forecast"]
    merged["smape_pct"] = merged.apply(
        lambda r: round(
            200 * abs(r["actual"] - r["forecast"]) /
            (abs(r["actual"]) + abs(r["forecast"]) + 1e-8), 2
        ), axis=1
    )
    merged["mape_pct"] = merged.apply(
        lambda r: round(abs(r["actual"] - r["forecast"]) / r["actual"] * 100, 2)
        if r["actual"] > 0 else np.nan, axis=1
    )
    merged["rating"] = merged["smape_pct"].apply(lambda v: accuracy_label(v)[0])
    date_fmt = "%b %Y" if fva_fcode == "MS" else "%d/%m/%Y"
    merged["period"] = merged["date"].dt.strftime(date_fmt)

    # Overall metrics
    ya, yf   = merged["actual"].values, merged["forecast"].values
    ov_rmse  = float(np.sqrt(mean_squared_error(ya, yf)))
    ov_mae   = float(mean_absolute_error(ya, yf))
    ov_sm    = calc_smape(ya, yf)
    ov_mp, nm = calc_mape(ya, yf, float(np.mean(ya)))
    ov_lbl, ov_cls = accuracy_label(ov_sm)

    st.markdown("---")
    st.markdown("### Overall accuracy")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",   f"{unit_fva}{ov_rmse:,.0f}")
    m2.metric("MAE",    f"{unit_fva}{ov_mae:,.0f}")
    m3.metric("sMAPE",  f"{ov_sm:.2f}%")
    m4.metric("MAPE",   f"{ov_mp:.2f}%" if not np.isnan(ov_mp) else "N/A")
    m5.metric("Rating", ov_lbl)

    st.markdown(
        f'<div class="explain-box">'
        f"<strong>Overall accuracy: <span class='{ov_cls}'>{ov_lbl}</span></strong><br>"
        f"Across {len(merged)} overlapping periods — "
        f"sMAPE {ov_sm:.2f}%, MAE {unit_fva}{ov_mae:,.0f} per period. "
        f"MAPE evaluated on {nm}/{len(merged)} periods (zero-actual periods excluded)."
        f"</div>", unsafe_allow_html=True
    )
    acc_scale_table()

    # Chart
    st.markdown("### Forecast vs Actual chart")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged["date"], merged["actual"],
            color="#1565C0", linewidth=1.2, label="Actual", marker="o", markersize=4)
    ax.plot(merged["date"], merged["forecast"],
            color="#E53935", linewidth=1.2, linestyle="--",
            label="Forecast", marker="s", markersize=4)
    ax.fill_between(merged["date"],
                    np.minimum(merged["actual"], merged["forecast"]),
                    np.maximum(merged["actual"], merged["forecast"]),
                    alpha=0.12, color="#E53935", label="Error Band")
    ax.set_title("Forecast vs Actual", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Value ({unit_fva})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Period-by-period table
    st.markdown("### Period-by-period breakdown")
    disp = merged[["period","forecast","actual","error","smape_pct","mape_pct","rating"]].copy()
    disp.columns = [
        "Period", f"Forecast ({unit_fva})", f"Actual ({unit_fva})",
        f"Error ({unit_fva})", "sMAPE (%)", "MAPE (%)", "Rating"
    ]
    for c in [f"Forecast ({unit_fva})", f"Actual ({unit_fva})", f"Error ({unit_fva})"]:
        disp[c] = disp[c].apply(lambda v: f"{v:,.0f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Per-product breakdown
    if prod_col_fva and prod_col_fva in act_c.columns:
        st.markdown("### Per-product breakdown")
        st.caption(
            "Total actual sales/quantity per product per period "
            "(based on the filtered actual data)."
        )
        pivot = breakdown_by_group(act_c, act_date, act_val, prod_col_fva, fva_fcode)
        if pivot is not None and len(pivot) > 0:
            # Format numeric columns
            num_cols = [c for c in pivot.columns if c != "period"]
            for c in num_cols:
                pivot[c] = pivot[c].apply(lambda v: f"{v:,.0f}")
            st.dataframe(pivot, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇  Download product breakdown CSV",
                pivot.to_csv(index=False).encode(),
                "product_breakdown.csv", "text/csv"
            )
        else:
            st.info("No product-level data to display for this filter selection.")

    # Download full comparison
    dl = merged[["period","forecast","actual","error","smape_pct","mape_pct","rating"]].copy()
    dl.columns = [
        "period", f"forecast_{unit_fva}", f"actual_{unit_fva}",
        f"error_{unit_fva}", "smape_pct", "mape_pct", "rating"
    ]
    st.download_button(
        "⬇  Download full comparison as CSV",
        dl.to_csv(index=False).encode(),
        "forecast_vs_actual.csv", "text/csv", use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════
if page == "📊 Training Results":
    st.title("Training Results")
    st.markdown(
        "All graphs from the original model training run on the full 2023–2025 dataset. "
        "Updated automatically when `lstm_forecasting.py` is re-run."
    )

    graphs = [
        ("01_daily_sales_timeseries.png", "Daily total sales time series",
         "The top panel shows raw daily total sales (Jan 2023–Nov 2025). "
         "The bottom panel adds a 30-day rolling average (red) to expose the underlying trend. "
         "Sales grew from early 2023, peaked around late 2024, and declined toward Nov 2025. "
         "This declining tail formed the test period — the hardest forecasting scenario. "
         "High day-to-day volatility reflects the irregular bulk ordering patterns of B2B platforms."),
        ("02_training_loss.png", "LSTM training vs validation loss",
         "The blue line is training loss (MSE on normalised values) per epoch. "
         "The red line is validation loss. Both drop steeply in the first few epochs, "
         "then level off running close together with no upward divergence — "
         "confirming clean convergence with no overfitting. "
         "Early stopping fired at epoch 24 and best weights were restored from epoch 4."),
        ("03_predictions_vs_actual.png", "LSTM forecast vs actual (test period)",
         "Blue = actual daily sales in the held-out test period (Jun–Nov 2025). "
         "Red dashed = LSTM predictions generated without the model ever seeing this period. "
         "The model captures the general downward trend correctly but smooths individual spikes — "
         "expected for a univariate model with no access to promotional or external signals. "
         "Trend-level accuracy is the most operationally relevant output for strategic planning."),
        ("04_model_comparison.png", "Model performance metrics",
         "Bar charts comparing RMSE, MAE, MAPE, and sMAPE across all models evaluated. "
         "Lower = better on all metrics. "
         "sMAPE is the primary percentage metric — it is bounded 0–200% and not "
         "distorted by near-zero actual values on low-volume days, which inflate MAPE."),
    ]

    for fname, title, explanation in graphs:
        p = os.path.join(OUT_DIR, fname)
        st.markdown(f"---\n### {title}")
        if os.path.exists(p):
            st.image(p, use_column_width=True)
            st.markdown(
                f'<div class="explain-box">'
                f"<strong>What this shows:</strong> {explanation}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning(f"`{fname}` not found — run `python lstm_forecasting.py` to generate it.")

    st.markdown("---")
    st.markdown("### Live metrics table")
    if disp_df is not None:
        st.success("Loaded from `outputs/model_comparison.csv` — updates automatically on re-run.")
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
| Look-back (daily) | 60 periods |
| Look-back (monthly) | 12 periods |
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
| Sunday trading | Excluded |
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
### Minimum required CSV format
```
orderDate,final_amount
05/01/2023,125000
06/01/2023,340500
```
For quantity forecasting, add `quantitySold`.  
For category filtering, add `productId` (format: `NGA-FDI-PST-000025`).  
For product-level filtering and breakdown, add `displayTitle`.  
If a column does not exist in your file, select **— None —** in that dropdown.

---
### Product filtering options
| Mode | How it works |
|---|---|
| All Products | Aggregates total sales across all products |
| Product Category | Filters by the prefix in productId (FDI = Food, BEV = Beverages, HME = Home, PRF = Personal/Cooking) |
| Specific Products (up to 10) | Select individual product names; their sales are summed into one series |

---
### Sunday exclusion
The company does not trade on Sundays. Daily aggregation automatically removes Sunday rows
from both the historical series and the future forecast dates. Monthly aggregation is
unaffected because Sunday sales are zero and the monthly sum absorbs them without distortion.

---
### Look-back windows
| Aggregation | Look-back used |
|---|---|
| Daily | 60 trading days |
| Monthly | 12 months |

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
