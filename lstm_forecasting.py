# =============================================================================
#  LSTM-BASED PRODUCT DEMAND FORECASTING SYSTEM
#  Project : Design and Development of an LSTM-Based Product Demand
#            Forecasting System from Historical Sales Data
#  Company : Alerzo Limited (Nigerian B2B E-Commerce)
#  Period  : January 1, 2023 – November 22, 2025
#  Author  : Oluwatosin Oluwabukola Lawal
#  Tool    : Python 3.11.9 | VS Code | TensorFlow / Keras
# =============================================================================

# =============================================================================
# SECTION 1 ── IMPORT LIBRARIES
# Every library used in this project is imported here at the top.
# =============================================================================

import os  # For building file paths and creating folders
import pickle  # For saving/loading Python objects (e.g. the scaler)
import warnings  # To suppress harmless warning messages

warnings.filterwarnings("ignore")  # Keep terminal output clean

import matplotlib.dates as mdates  # Formats date axes on charts

# --- Plotting ---
import matplotlib.pyplot as plt  # Core plotting library
import numpy as np  # Numerical operations (arrays, math)
import pandas as pd  # Data loading, manipulation, aggregation
import seaborn as sns  # Higher-level statistical plots

# --- TensorFlow / Keras (Deep Learning) ---
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Scikit-learn (preprocessing + evaluation metrics) ---
from sklearn.preprocessing import MinMaxScaler  # Scales data to [0, 1]

#   LSTM    – the recurrent layer that learns sequential patterns
#   Dense   – a standard fully-connected layer (used for the output)
#   Dropout – randomly zeros a fraction of outputs to reduce overfitting
from tensorflow.keras.callbacks import (
    EarlyStopping,  # Stops training when validation loss stops improving
    ModelCheckpoint,  # Saves the best model weights to disk automatically
    ReduceLROnPlateau,  # Lowers the learning rate when training stalls
)
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential  # Linear stack of layers

# --- Baseline Models ---
# Install: pip install pmdarima
try:
    import pmdarima as pm  # Automatic ARIMA parameter selection

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠  pmdarima not found. Run:  pip install pmdarima")

# Install: pip install prophet
try:
    from prophet import Prophet  # Meta's Prophet time-series model

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠  prophet not found. Run:  pip install prophet")

# Fix random seeds so results are reproducible every run
np.random.seed(42)
tf.random.set_seed(42)

print("✅ Libraries loaded successfully.")
print(f"   TensorFlow  : {tf.__version__}")
print(f"   pandas      : {pd.__version__}")
print(f"   numpy       : {np.__version__}")


# =============================================================================
# SECTION 2 ── CONFIGURATION
# Change ONLY this section to match your folder structure or tune the model.
# =============================================================================

# ── Folder where your 5 cleaned CSV files live ──
# Example: If your files are in C:\Users\YourName\Documents\alerzo\data\cleaned
# then set DATA_DIR = r"C:\Users\YourName\Documents\alerzo\data\cleaned"
DATA_DIR = "./data/cleaned"

# ── Exact CSV filenames (adjust if yours differ) ──
DATA_FILES = {
    "2023_1": os.path.join(DATA_DIR, "cleaned_2023_1.csv"),
    "2023_2": os.path.join(DATA_DIR, "cleaned_2023_2.csv"),
    "2023_3": os.path.join(DATA_DIR, "cleaned_2023_3.csv"),
    "2024": os.path.join(DATA_DIR, "cleaned_2024.csv"),
    "2025": os.path.join(DATA_DIR, "cleaned_2025.csv"),
}

# ── Output & model folders (created automatically if absent) ──
OUTPUT_DIR = "./outputs"  # Plots, CSVs, and results go here
MODEL_DIR = "./model"  # Saved Keras model and scaler go here
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── LSTM Hyperparameters ──
LOOK_BACK = 60  # Number of past days fed as input to the model
LSTM_UNITS = 64  # Number of memory units inside the LSTM layer
DROPOUT_RATE = 0.2  # Fraction of neurons turned off during training (regularisation)
BATCH_SIZE = 32  # Samples processed per weight-update step
MAX_EPOCHS = 100  # Hard ceiling on training iterations
PATIENCE = 10  # Early stopping: halt if no improvement for this many epochs

# ── Data Split Ratios ──
TRAIN_RATIO = 0.70  # 70 % of the timeline used for training
VAL_RATIO = 0.15  # 15 % used for validation (tuning callbacks)
TEST_RATIO = 0.15  # 15 % withheld as the final, unseen test set

print("✅ Configuration ready.")


# =============================================================================
# SECTION 3 ── LOAD AND MERGE THE FIVE CSV FILES
# Each file is read separately, then combined into one master DataFrame.
# =============================================================================
print("\n📂 Loading data files…")

frames = []  # Accumulate each file's DataFrame here

for label, path in DATA_FILES.items():
    if not os.path.exists(path):
        print(f"  ⚠  File not found: {path}  ← check DATA_DIR and filenames")
        continue

    # read_csv automatically reads the file; parse_dates converts orderDate
    # from text/integer to a proper Python datetime object
    df = pd.read_csv(path, low_memory=False)

    # Strip leading/trailing spaces from ALL column names across all files
    df = pd.read_csv(path, low_memory=False)

    # Strip spaces from all column names
    df.columns = df.columns.str.strip()

    # Strip spaces from all text values
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Remove commas from numeric columns so "15,500" becomes 15500
    for col in ["final_amount", "unitPrice", "orderTotal", "quantitySold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

    # Parse dates properly
    # dayfirst=True tells pandas your dates are dd/mm/yyyy format
    df["orderDate"] = pd.to_datetime(df["orderDate"], dayfirst=True, errors="coerce")

    df["_source"] = label

    # Strip spaces from text columns too, in case values have padding
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Force orderDate to proper datetime — errors="coerce" turns bad values to NaT
    # dayfirst=True tells pandas your dates are dd/mm/yyyy format
    df["orderDate"] = pd.to_datetime(df["orderDate"], dayfirst=True, errors="coerce")

    df["_source"] = label
    frames.append(df)
    print(f"  ✔  {label:<8s}  →  {len(df):>10,} rows loaded")

# pd.concat stacks all DataFrames on top of each other (axis=0)
# ignore_index=True resets the row index so it runs 0, 1, 2, … continuously
master = pd.concat(frames, ignore_index=True)

print(f"\n✅ Total rows after merge  : {len(master):,}")
print(f"   Columns                : {list(master.columns)}")
print(
    f"   Date range             : {master['orderDate'].min().date()} → {master['orderDate'].max().date()}"
)


# =============================================================================
# SECTION 4 ── EXPLORATORY DATA ANALYSIS (EDA)
# Quick sanity-checks on the merged dataset before modelling.
# =============================================================================
print("\n📊 Exploratory Data Analysis…")

# Descriptive statistics for numeric columns
print("\n── Descriptive statistics ──")
print(
    master[["quantitySold", "unitPrice", "final_amount", "orderTotal"]]
    .describe()
    .round(2)
)

# Count of nulls per column (should be very low after cleaning)
print("\n── Missing values per column ──")
print(master.isnull().sum())

# Distribution of the categorical sales label
print("\n── Sales category distribution ──")
print(master["salesCategory"].value_counts())


# =============================================================================
# SECTION 5 ── TEMPORAL AGGREGATION
# Collapse the product-level transaction rows into one total-sales figure per day.
# This turns millions of rows into a single daily time series.
# =============================================================================
print("\n📅 Aggregating to daily total sales…")

# Ensure the date column is datetime (in case it wasn't parsed)
# Drop any rows where orderDate could not be parsed (NaT values)
master["orderDate"] = pd.to_datetime(master["orderDate"], errors="coerce")
master = master.dropna(subset=["orderDate"])

# Standardise salesCategory capitalisation across all files
master["salesCategory"] = master["salesCategory"].str.strip().str.title()

# groupby("orderDate") groups all rows that share the same date,
# then ["final_amount"].sum() totals the sales for each group
daily = (
    master.groupby("orderDate")["final_amount"]
    .sum()
    .reset_index()  # Turns the index back into a column
    .rename(columns={"orderDate": "date", "final_amount": "total_sales"})
    .sort_values("date")  # Critical: must be in chronological order
    .reset_index(drop=True)
)

print(f"✅ Daily series : {len(daily)} days")
print(f"   Min sales    : ₦{daily['total_sales'].min():>15,.2f}")
print(f"   Max sales    : ₦{daily['total_sales'].max():>15,.2f}")
print(f"   Mean sales   : ₦{daily['total_sales'].mean():>15,.2f}")


# =============================================================================
# SECTION 6 ── FILL MISSING CALENDAR DATES
# Business days without any orders leave gaps. We fill them with 0 to keep
# the series continuous. LSTMs require equally-spaced time steps.
# =============================================================================
print("\n📅 Ensuring a gapless daily calendar…")

# date_range creates every single calendar day between start and end
full_calendar = pd.date_range(
    start=daily["date"].min(), end=daily["date"].max(), freq="D"
)

# reindex replaces the date index with the complete calendar;
# any day that had no sales gets NaN, then fillna(0) replaces NaN with 0
daily = (
    daily.set_index("date")
    .reindex(full_calendar)  # Inserts missing dates as NaN rows
    .fillna(0)  # Zero sales on days with no orders
    .reset_index()
    .rename(columns={"index": "date"})
)

# Remove zero-sale days caused by weekends/holidays or bad rows
# so MAPE doesn't divide by zero and the date index stays aligned
daily = daily[daily["total_sales"] > 0].reset_index(drop=True)

# Also remove extreme outlier low days (bottom 1% of sales)
# These are likely partial trading days or data errors
low_threshold = daily["total_sales"].quantile(0.01)
daily = daily[daily["total_sales"] > low_threshold].reset_index(drop=True)
print(f"   After removing bottom 1% outliers: {len(daily)} days")

print(f"✅ Complete calendar series : {len(daily)} days (zeros removed)")


# =============================================================================
# SECTION 7 ── VISUALISE THE RAW TIME SERIES
# Always plot the data first – it reveals trends, outliers, and seasonality.
# =============================================================================
print("\n📈 Plotting raw daily sales time series…")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ── Top panel: raw daily sales ──
axes[0].plot(
    daily["date"], daily["total_sales"], color="#1565C0", linewidth=0.7, alpha=0.8
)
axes[0].set_title(
    "Alerzo – Daily Total Sales (Jan 2023 – Nov 2025)", fontsize=13, fontweight="bold"
)
axes[0].set_ylabel("Total Sales (₦)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[0].grid(True, alpha=0.3)

# ── Bottom panel: daily sales + 30-day rolling average ──
roll30 = daily["total_sales"].rolling(window=30, min_periods=1).mean()
axes[1].plot(
    daily["date"],
    daily["total_sales"],
    color="#90CAF9",
    linewidth=0.5,
    alpha=0.6,
    label="Daily Sales",
)
axes[1].plot(
    daily["date"],
    roll30,
    color="#E53935",
    linewidth=1.8,
    label="30-Day Rolling Average",
)
axes[1].set_title(
    "Daily Sales with 30-Day Rolling Average", fontsize=13, fontweight="bold"
)
axes[1].set_ylabel("Total Sales (₦)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_daily_sales_timeseries.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print("✅ Plot saved → outputs/01_daily_sales_timeseries.png")


# =============================================================================
# SECTION 8 ── NORMALISE THE DATA
# MinMaxScaler maps every value to the range [0, 1].
# IMPORTANT: fit only on training data to avoid leaking future info.
# =============================================================================
print("\n🔧 Normalising sales data…")

# Extract the values as a 2-D column array (required by scikit-learn)
# Shape: (n_days, 1)
sales = daily["total_sales"].values.reshape(-1, 1)

# ── Compute split indices ──
n = len(sales)
train_end = int(n * TRAIN_RATIO)  # E.g. day 0 … 728
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))  # E.g. day 729 … 885
# Test set runs from val_end to the end of the series

print(f"   Total days   : {n}")
print(f"   Training     : days 0 – {train_end}   ({train_end} days)")
print(f"   Validation   : days {train_end} – {val_end}   ({val_end - train_end} days)")
print(f"   Test         : days {val_end} – {n}   ({n - val_end} days)")

# Fit the scaler ONLY on training rows (learn the min and max from training data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(sales[:train_end])

# Transform the ENTIRE array using the training-set scale parameters
scaled = scaler.transform(sales)
# All values now lie in [0, 1]

print(
    f"✅ Scaling complete. Value range after scaling: [{scaled.min():.4f}, {scaled.max():.4f}]"
)


# =============================================================================
# SECTION 9 ── CREATE SUPERVISED SEQUENCES (Sliding Window)
# We convert the 1-D time series into (input window, target) pairs that the
# LSTM can learn from.
#
# Example with LOOK_BACK=3:
#   Day index : 0   1   2   3   4   5
#   Sales     : 10  20  30  40  50  60
#   →  X[0] = [10, 20, 30],  y[0] = 40
#   →  X[1] = [20, 30, 40],  y[1] = 50  … and so on
# =============================================================================
print(f"\n🔗 Creating sequences (look-back window = {LOOK_BACK} days)…")


def make_sequences(data, look_back):
    """
    Converts a 1-D time series array into (X, y) pairs for supervised learning.

    Parameters
    ----------
    data      : numpy array of shape (n, 1) – the normalised time series
    look_back : int – how many past days to use as the model input

    Returns
    -------
    X : numpy array of shape (n - look_back, look_back) – input windows
    y : numpy array of shape (n - look_back,)            – next-day targets
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])  # Window of LOOK_BACK past values
        y.append(data[i, 0])  # The value the model should predict
    return np.array(X), np.array(y)


X_all, y_all = make_sequences(scaled, LOOK_BACK)
# X_all shape : (n - LOOK_BACK, LOOK_BACK)
# y_all shape : (n - LOOK_BACK,)

# ── Adjust split boundaries for the sequence offset ──
# The first LOOK_BACK days are consumed to build the first input window,
# so all split indices need to be shifted accordingly.
adj_train = train_end - LOOK_BACK
adj_val = val_end - LOOK_BACK

X_train, y_train = X_all[:adj_train], y_all[:adj_train]
X_val, y_val = X_all[adj_train:adj_val], y_all[adj_train:adj_val]
X_test, y_test = X_all[adj_val:], y_all[adj_val:]

# ── Reshape inputs to 3-D: (samples, time_steps, features) ──
# Keras LSTM expects exactly this shape. "features" = 1 (univariate)
X_train = X_train.reshape(*X_train.shape, 1)
X_val = X_val.reshape(*X_val.shape, 1)
X_test = X_test.reshape(*X_test.shape, 1)

print(f"✅ Sequences ready:")
print(f"   X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"   X_val   : {X_val.shape}     y_val   : {y_val.shape}")
print(f"   X_test  : {X_test.shape}    y_test  : {y_test.shape}")


# =============================================================================
# SECTION 10 ── BUILD THE VANILLA LSTM MODEL
# A simple, single-layer LSTM followed by Dropout and a linear output neuron.
# =============================================================================
print("\n🏗️  Building the Vanilla LSTM model…")

model = Sequential(
    [
        # ── Layer 1: LSTM ──────────────────────────────────────────────────────
        # input_shape = (LOOK_BACK, 1): tell Keras what shape each sample has.
        # LSTM_UNITS = 64: the number of memory cells (hidden state dimensions).
        # return_sequences=False: only output the hidden state from the LAST
        #   time step (we're doing one-step-ahead prediction, not sequence-to-sequence).
        LSTM(LSTM_UNITS, input_shape=(LOOK_BACK, 1), return_sequences=False),
        # ── Layer 2: Dropout ───────────────────────────────────────────────────
        # During each training step, 20 % of LSTM output neurons are randomly
        # zeroed. This forces the model to not rely on any single neuron,
        # improving generalisation (reduces overfitting).
        Dropout(DROPOUT_RATE),
        # ── Layer 3: Dense output neuron ───────────────────────────────────────
        # A single neuron with a linear activation: it outputs one number,
        # the predicted (normalised) sales for the next day.
        Dense(1),  # No activation argument = default linear activation
    ]
)

# ── Compile ────────────────────────────────────────────────────────────────
# optimizer="adam" : Adaptive Moment Estimation – adjusts learning rate per
#   parameter automatically. Works well for RNNs.
# loss="mse"       : Mean Squared Error – the value we minimise during training.
# metrics=["mae"]  : Also track Mean Absolute Error so we can read it each epoch.
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.summary()  # Prints a table of layers, output shapes, and parameter counts


# =============================================================================
# SECTION 11 ── TRAINING CALLBACKS
# Callbacks are functions Keras calls automatically at the end of each epoch.
# =============================================================================
print("\n⚙️  Configuring training callbacks…")

# ── Early Stopping ──
# Stops training when validation loss (val_loss) has not improved for
# PATIENCE (10) epochs. restore_best_weights=True means the model's
# weights are rolled back to the best epoch found, not the final epoch.
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,  # was PATIENCE — give model more time to improve
    min_delta=0.0001,  # only count as improvement if loss drops by this much
    restore_best_weights=True,
    verbose=1,
)

# ── Learning Rate Scheduler ──
# If val_loss doesn't improve for 5 consecutive epochs, the learning rate
# is multiplied by factor=0.5, helping the optimiser escape flat areas.
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,  # New LR = current LR × 0.5
    patience=8,
    min_lr=1e-6,  # Never go below this learning rate
    verbose=1,
)

# ── Model Checkpoint ──
# Saves model weights to disk every time a new best val_loss is achieved.
# This guarantees we don't lose the best model if training overshoots.
best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.keras")
checkpoint = ModelCheckpoint(
    filepath=best_model_path, monitor="val_loss", save_best_only=True, verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]


# =============================================================================
# SECTION 12 ── TRAIN THE LSTM MODEL
# This is the main training loop. Keras handles the forward pass,
# loss calculation, backpropagation, and weight updates automatically.
# =============================================================================
print("\n🚀 Training the LSTM model (this may take a few minutes)…\n")

history = model.fit(
    X_train,
    y_train,  # Training inputs and targets
    validation_data=(X_val, y_val),  # Used only to monitor val_loss
    epochs=MAX_EPOCHS,  # Maximum number of passes through training data
    batch_size=BATCH_SIZE,  # Samples per gradient update
    callbacks=callbacks,  # Callbacks defined above
    verbose=1,  # Print one line per epoch
)

actual_epochs = len(history.history["loss"])
print(
    f"\n✅ Training finished after {actual_epochs} epochs "
    f"(max was {MAX_EPOCHS}, early stopped: {actual_epochs < MAX_EPOCHS})."
)


# =============================================================================
# SECTION 13 ── PLOT TRAINING / VALIDATION LOSS
# This plot shows whether the model is overfitting (train loss << val loss)
# or underfitting (both losses still decreasing at termination).
# =============================================================================
print("\n📉 Plotting training history…")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    history.history["loss"], color="#1565C0", linewidth=1.5, label="Training Loss (MSE)"
)
ax.plot(
    history.history["val_loss"],
    color="#E53935",
    linewidth=1.5,
    label="Validation Loss (MSE)",
)
ax.set_title(
    "LSTM Training vs Validation Loss per Epoch", fontsize=13, fontweight="bold"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Squared Error (MSE)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_training_loss.png"), dpi=150, bbox_inches="tight"
)
plt.show()
print("✅ Loss plot saved → outputs/02_training_loss.png")


# =============================================================================
# SECTION 14 ── GENERATE TEST SET PREDICTIONS
# The trained model predicts on unseen test data. Because the model outputs
# normalised values (in [0,1]), we use the scaler to convert them back to ₦.
# =============================================================================
print("\n🔮 Generating predictions on the test set…")

# model.predict returns a 2-D array of shape (n_test, 1)
y_pred_scaled = model.predict(X_test, verbose=0)

# inverse_transform converts the [0,1] predictions back to original ₦ values
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

# Also inverse-transform the actual test targets for fair comparison
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

print(f"✅ {len(y_pred)} test-set predictions generated.")


# =============================================================================
# SECTION 15 ── EVALUATION METRICS
# We compute RMSE, MAE, and MAPE on the original-scale (₦) values.
# sMAPE — more robust than MAPE when actuals approach zero
# =============================================================================
print("\n📏 Computing evaluation metrics…")


def evaluate(y_true, y_hat, name="Model"):
    """Computes and prints RMSE, MAE, and MAPE for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mae = mean_absolute_error(y_true, y_hat)

    # Only include days where actual sales exceed 1% of the mean
    # This removes extreme low-sale outlier days that destroy MAPE
    threshold = np.mean(y_true) * 0.01
    mask = y_true > threshold
    mape = np.mean(np.abs((y_true[mask] - y_hat[mask]) / y_true[mask])) * 100

    # sMAPE — symmetric, handles near-zero actuals without exclusion
    smape = (
        np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat) + 1e-8))
        * 100
    )

    print(f"\n{'═' * 44}")
    print(f"  {name} — Performance on Test Set")
    print(f"{'═' * 44}")
    print(f"  RMSE  :  ₦{rmse:>15,.2f}")
    print(f"  MAE   :  ₦{mae:>15,.2f}")
    print(f"  MAPE  :   {mape:>14.2f} %")
    print(f"  sMAPE :   {smape:>14.2f} %")
    print(f"  Days evaluated (MAPE): {mask.sum()} of {len(y_true)}")
    print(f"{'═' * 44}")

    return {"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape, "sMAPE": smape}


lstm_result = evaluate(y_actual, y_pred, name="Vanilla LSTM")


# =============================================================================
# SECTION 16 ── PLOT PREDICTIONS VS ACTUAL
# A visual overlay of the model's test-set forecast on the true sales values.
# =============================================================================
print("\n📈 Plotting test-set predictions vs actual sales…")


# Align test dates precisely to the sequence index
test_dates = daily["date"].values[len(daily) - len(y_test) :]

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test_dates, y_actual, color="#1565C0", linewidth=1.3, label="Actual Sales")
ax.plot(
    test_dates,
    y_pred,
    color="#E53935",
    linewidth=1.3,
    linestyle="--",
    label="LSTM Forecast",
)
ax.fill_between(
    test_dates,
    np.minimum(y_actual, y_pred),
    np.maximum(y_actual, y_pred),
    alpha=0.12,
    color="#E53935",
    label="Error Band",
)
ax.set_title(
    "LSTM Demand Forecast vs Actual Daily Sales — Test Period",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Date")
ax.set_ylabel("Total Sales (₦)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "03_predictions_vs_actual.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print("✅ Forecast plot saved → outputs/03_predictions_vs_actual.png")


# =============================================================================
# SECTION 17 ── BASELINE: ARIMA
# ARIMA is trained on the same training window and asked to forecast the
# same number of days as the LSTM test set.
# =============================================================================
print("\n📊 Training ARIMA baseline model…")

# Raw (un-normalised) training and test sales
train_raw = daily["total_sales"].values[:train_end]
test_raw = daily["total_sales"].values[val_end:]

arima_result = None

if ARIMA_AVAILABLE:
    try:
        print("  Searching for optimal ARIMA (p, d, q) via auto_arima…")
        # auto_arima tries many (p,d,q) combinations and picks the one
        # with the lowest AIC (Akaike Information Criterion)
        arima_model = pm.auto_arima(
            train_raw,
            seasonal=False,  # Non-seasonal; set True + m=7 for weekly seasonality
            stepwise=True,  # Stepwise search is faster than exhaustive
            information_criterion="aic",
            max_p=5,
            max_q=5,  # Upper bounds for the search space
            error_action="ignore",
            suppress_warnings=True,
        )
        print(f"  ✅ ARIMA order selected: {arima_model.order}")

        # Forecast exactly as many steps forward as the test set has days
        n_test = len(test_raw)
        arima_fc = arima_model.predict(n_periods=n_test)
        arima_fc = np.maximum(arima_fc, 0)  # Sales cannot be negative

        arima_result = evaluate(test_raw, arima_fc, name="ARIMA")

    except Exception as err:
        print(f"  ⚠  ARIMA failed: {err}")
else:
    print("  Skipping ARIMA – pmdarima not installed.")


# =============================================================================
# SECTION 18 ── BASELINE: PROPHET
# Prophet requires a DataFrame with columns 'ds' (dates) and 'y' (values).
# =============================================================================
print("\n📊 Training Prophet baseline model…")

prophet_result = None

if PROPHET_AVAILABLE:
    try:
        # Build the training DataFrame in the format Prophet expects
        prophet_train = pd.DataFrame(
            {
                "ds": daily["date"].values[:train_end],
                "y": daily["total_sales"].values[:train_end],
            }
        )

        # Initialise Prophet with all three seasonality components enabled
        prophet_model = Prophet(
            daily_seasonality=True,  # Day-of-week patterns
            weekly_seasonality=True,  # Weekly cycles
            yearly_seasonality=True,  # Annual patterns
        )
        prophet_model.fit(prophet_train)

        # Create a future DataFrame covering exactly the test period dates
        future = pd.DataFrame({"ds": daily["date"].values[val_end:]})
        forecast = prophet_model.predict(future)

        prophet_fc = np.maximum(forecast["yhat"].values, 0)  # Clip negatives to 0

        prophet_result = evaluate(test_raw, prophet_fc, name="Prophet")

    except Exception as err:
        print(f"  ⚠  Prophet failed: {err}")
else:
    print("  Skipping Prophet – prophet not installed.")


# =============================================================================
# SECTION 19 ── COMPARATIVE RESULTS TABLE
# Collect all model metrics into a DataFrame, print, and save as CSV.
# =============================================================================
print("\n📋 Comparative model results…")

all_results = [lstm_result]
if arima_result:
    all_results.append(arima_result)
if prophet_result:
    all_results.append(prophet_result)

results_df = pd.DataFrame(all_results).set_index("Model")

# Pretty-print with currency formatting
display_df = results_df.copy()
display_df["RMSE"] = display_df["RMSE"].apply(lambda v: f"₦{v:,.2f}")
display_df["MAE"] = display_df["MAE"].apply(lambda v: f"₦{v:,.2f}")
display_df["MAPE"] = display_df["MAPE"].apply(lambda v: f"{v:.2f}%")
display_df["sMAPE"] = display_df["sMAPE"].apply(lambda v: f"{v:.2f}%")

print("\n" + "═" * 55)
print("         COMPARATIVE MODEL PERFORMANCE SUMMARY")
print("═" * 55)
print(display_df.to_string())
print("═" * 55)

# Save to CSV
results_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
display_df.to_csv(results_path)
print(f"\n✅ Results table saved → {results_path}")


# =============================================================================
# SECTION 20 ── COMPARATIVE BAR CHART
# A grouped bar chart to visually compare all three models on all three metrics.
# =============================================================================
print("\n📊 Plotting comparative bar chart…")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
colours = ["#1565C0", "#E53935", "#2E7D32"]
metrics = [
    ("RMSE", "RMSE (₦)"),
    ("MAE", "MAE (₦)"),
    ("MAPE", "MAPE (%)"),
    ("sMAPE", "sMAPE (%)"),
]


for ax, (metric, label) in zip(axes, metrics):
    vals = results_df[metric].values
    names = results_df.index.tolist()
    bars = ax.bar(
        names, vals, color=colours[: len(names)], edgecolor="white", width=0.5
    )
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylabel(label)
    for bar in bars:
        h = bar.get_height()
        # Add value label above each bar for quick reading
        # Format Naira metrics differently from percentage metrics
        if metric in ("RMSE", "MAE"):
            label_text = f"₦{h:,.0f}"
        else:
            label_text = f"{h:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=9)

plt.suptitle(
    "Model Comparison: LSTM vs Baseline Forecasting Models",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "04_model_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.show()
print("✅ Comparison chart saved → outputs/04_model_comparison.png")


# =============================================================================
# SECTION 21 ── SAVE THE TRAINED MODEL AND SCALER
# Saving allows you to load the model later and make predictions without
# retraining. The scaler is also saved because you need it to convert
# future predictions back to ₦ values.
# =============================================================================
print("\n💾 Saving model and scaler…")

# Save the Keras model in the native .keras format
final_model_path = os.path.join(MODEL_DIR, "lstm_demand_forecast.keras")
model.save(final_model_path)
print(f"✅ Model saved   → {final_model_path}")

# Save the MinMaxScaler using pickle (standard Python serialisation)
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved  → {scaler_path}")


# =============================================================================
# SECTION 22 ── HOW TO LOAD AND USE THE MODEL LATER
# (Informational – not executed, but shows you how to reload and forecast)
# =============================================================================

"""
──────────────────────────────────────────────────────────────
HOW TO LOAD THE SAVED MODEL AND MAKE NEW PREDICTIONS
──────────────────────────────────────────────────────────────

from tensorflow.keras.models import load_model
import pickle, numpy as np

# 1. Load the saved model
model = load_model("model/lstm_demand_forecast.keras")

# 2. Load the saved scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 3. Prepare your latest 60-day window (already normalised)
#    last_60 should be a numpy array of shape (60,) with normalised values
last_60_normalised = ...   # your last 60 days of normalised sales

# 4. Reshape to (1, 60, 1) — batch of 1, 60 time steps, 1 feature
X_new = last_60_normalised.reshape(1, 60, 1)

# 5. Predict
pred_normalised = model.predict(X_new)

# 6. Convert back to ₦
pred_naira = scaler.inverse_transform(pred_normalised)
print(f"Tomorrow's predicted sales: ₦{pred_naira[0, 0]:,.2f}")
──────────────────────────────────────────────────────────────
"""

# =============================================================================
# ── DONE ──
# =============================================================================
print("\n" + "═" * 55)
print("  🎉  LSTM Demand Forecasting Pipeline — COMPLETE")
print("═" * 55)
print(f"  Outputs  :  {os.path.abspath(OUTPUT_DIR)}")
print(f"  Model    :  {os.path.abspath(MODEL_DIR)}")
print("═" * 55)
