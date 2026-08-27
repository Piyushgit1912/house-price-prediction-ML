"""
train_model.py
---------------
Main training script for the House Price Prediction project.

Steps performed:
  1. Generate dataset (if not present)
  2. Load & preprocess data
  3. Train three ML models (Linear Regression, KNN, SVR)
  4. Evaluate and compare models
  5. Save models and scaler to disk
  6. Generate all visualisation plots

Run this script BEFORE running app.py.

Usage:
    python train_model.py

Author  : ML Practical Project
Purpose : Training pipeline for House Price Prediction
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")   # Suppress sklearn convergence warnings

# Fix Unicode / emoji output on Windows (CP1252 → UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # Non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Local Imports
# ─────────────────────────────────────────────────────────────────────────────
from model import HousePriceModel


# ═════════════════════════════════════════════════════════════════════════════
# STEP 0 ─ Constants / Configuration
# ═════════════════════════════════════════════════════════════════════════════
DATASET_PATH = "house_price.csv"   # Path to the CSV dataset
MODEL_DIR    = "models"            # Directory to save model .pkl files
PLOTS_DIR    = "plots"             # Directory to save visualisation images
TEST_SIZE    = 0.20                # 20% data used for testing
RANDOM_STATE = 42                  # Seed for reproducibility


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Dataset Generation (if CSV not present)
# ═════════════════════════════════════════════════════════════════════════════
def generate_dataset(path: str) -> None:
    """
    Generate a realistic synthetic house price CSV dataset.

    Parameters
    ----------
    path : str
        File path where the CSV will be saved.
    """
    np.random.seed(RANDOM_STATE)
    N = 800   # Total number of records

    area       = np.random.randint(500, 5001, size=N)
    bedrooms   = np.random.randint(1, 7, size=N)
    bathrooms  = np.clip(np.random.randint(1, 5, size=N), 1, bedrooms)
    house_age  = np.random.randint(0, 51, size=N)
    distance   = np.random.randint(1, 51, size=N)

    # Realistic price formula (same as generate_dataset.py)
    noise = np.random.normal(0, 15000, size=N)
    price = (
        area       * 120
        + bedrooms * 25000
        + bathrooms * 18000
        - house_age * 3000
        - distance  * 5000
        + 50000
        + noise
    )
    price = np.clip(price, 50000, None).astype(int)

    df = pd.DataFrame({
        "Area"              : area,
        "Bedrooms"          : bedrooms,
        "Bathrooms"         : bathrooms,
        "House_Age"         : house_age,
        "Distance_From_City": distance,
        "Price"             : price
    })
    df.to_csv(path, index=False)
    print(f"  ✅  Dataset generated → {path} ({len(df)} records)")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ Data Loading & Preprocessing
# ═════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(path: str):
    """
    Load the dataset, clean it, and split into train / test sets.

    Preprocessing steps:
      - Check and handle null values
      - Remove duplicate rows
      - Feature selection (X) and target extraction (y)
      - Train-Test Split (80 / 20)
      - StandardScaler (important for KNN and SVR)

    Parameters
    ----------
    path : str
        Path to the CSV dataset file.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Scaled feature matrices.
    y_train, y_test : pd.Series
        Target price values.
    scaler : StandardScaler
        Fitted scaler (needed to scale new user inputs during prediction).
    df : pd.DataFrame
        Original cleaned DataFrame (used for visualisations).
    """
    print("\n📂  Loading dataset ...")
    df = pd.read_csv(path)
    print(f"    Shape: {df.shape}")
    print(f"\n    First 5 rows:\n{df.head()}\n")

    # ── 2a. Check for null values ───────────────────────────────────────────
    print("🔍  Checking for null values ...")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("    No null values found. ✅")
    else:
        print(f"    Null values found:\n{null_counts}")
        df.dropna(inplace=True)   # Drop rows with null values
        print("    Null rows dropped. ✅")

    # ── 2b. Remove duplicate rows ───────────────────────────────────────────
    print("\n🔍  Checking for duplicate rows ...")
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"    Removed {before - after} duplicate(s). Remaining: {after} rows. ✅")

    # ── 2c. Feature Selection ───────────────────────────────────────────────
    # X = input features, y = target variable (Price)
    feature_cols = ["Area", "Bedrooms", "Bathrooms", "House_Age", "Distance_From_City"]
    X = df[feature_cols]   # Feature matrix
    y = df["Price"]         # Target column

    print(f"\n✅  Features selected : {feature_cols}")
    print(f"    Target variable   : Price")

    # ── 2d. Train-Test Split (80% train, 20% test) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"\n✂️   Train-Test Split:")
    print(f"    Training samples  : {len(X_train)}")
    print(f"    Testing  samples  : {len(X_test)}")

    # ── 2e. Feature Scaling (StandardScaler) ────────────────────────────────
    # StandardScaler standardises features to mean=0, std=1.
    # This is REQUIRED for distance-based models like KNN and SVR.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit on train, transform train
    X_test_scaled  = scaler.transform(X_test)        # Only transform test (no fitting)

    # ── 2f. Target Scaling (for SVR) ────────────────────────────────────────
    # SVR is sensitive to the scale of the output variable too.
    # We scale y using a separate scaler so SVR can learn effectively.
    # We'll de-scale predictions back to rupees for reporting metrics.
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled  = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    print("\n⚖️   Feature Scaling applied (StandardScaler). ✅")
    print("    Target (Price) also scaled for SVR. ✅")

    return X_train_scaled, X_test_scaled, y_train, y_test, y_train_scaled, y_test_scaled, scaler, y_scaler, df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Visualisations
# ═════════════════════════════════════════════════════════════════════════════
def generate_visualisations(df: pd.DataFrame, results: dict, y_test, scaler, hpm: HousePriceModel) -> None:
    """
    Generate and save the following plots to the 'plots/' directory:
      1. Correlation Heatmap
      2. Feature Distributions
      3. Actual vs Predicted (all 3 models)
      4. Prediction Comparison Bar Chart
      5. Linear Regression Line (Area vs Price)

    Parameters
    ----------
    df      : pd.DataFrame    – Original cleaned dataset
    results : dict            – Model evaluation metrics from HousePriceModel
    y_test  : pd.Series       – True test labels
    scaler  : StandardScaler  – Fitted scaler (not used here directly)
    hpm     : HousePriceModel – Trained model object
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Colour palette used across all plots
    COLORS = {
        "Linear Regression": "#4C9BE8",
        "KNN Regressor"    : "#F4845F",
        "SVR (RBF)"        : "#5CB85C"
    }

    # ── Plot 1: Correlation Heatmap ─────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))   # Show only lower triangle
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"  📊  Saved: {PLOTS_DIR}/01_correlation_heatmap.png")

    # ── Plot 2: Feature Distribution Histograms ─────────────────────────────
    feature_cols = ["Area", "Bedrooms", "Bathrooms", "House_Age", "Distance_From_City", "Price"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col], bins=30, color="#6C63FF", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Saved: {PLOTS_DIR}/02_feature_distributions.png")

    # ── Plot 3: Actual vs Predicted (one subplot per model) ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Actual vs Predicted Prices", fontsize=15, fontweight="bold")

    y_test_vals = y_test.values   # Convert Series to numpy array

    for ax, (name, metrics) in zip(axes, results.items()):
        y_pred = metrics["Predictions"]
        color  = list(COLORS.values())[list(COLORS.keys()).index(
            "Linear Regression" if "Linear" in name
            else ("KNN Regressor" if "KNN" in name else "SVR (RBF)")
        )]

        ax.scatter(y_test_vals, y_pred, alpha=0.5, color=color, edgecolors="white", s=30)

        # Perfect prediction line (y = x)
        min_val = min(y_test_vals.min(), y_pred.min())
        max_val = max(y_test_vals.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Perfect Fit")

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Price (Rs.)")
        ax.set_ylabel("Predicted Price (Rs.)")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Saved: {PLOTS_DIR}/03_actual_vs_predicted.png")

    # ── Plot 4: Model Comparison Bar Chart (R² & RMSE) ──────────────────────
    model_names = list(results.keys())
    r2_scores   = [results[m]["R2"]   for m in model_names]
    rmse_scores = [results[m]["RMSE"] for m in model_names]
    bar_colors  = [COLORS.get(m, "#888") for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    # R² bar chart
    bars1 = ax1.bar(model_names, r2_scores, color=bar_colors, edgecolor="white", width=0.5)
    ax1.set_title("R² Score (higher is better)", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R² Score")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # RMSE bar chart
    bars2 = ax2.bar(model_names, rmse_scores, color=bar_colors, edgecolor="white", width=0.5)
    ax2.set_title("RMSE (lower is better)", fontsize=12)
    ax2.set_ylabel("RMSE (Rs.)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"Rs.{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Saved: {PLOTS_DIR}/04_model_comparison.png")

    # ── Plot 5: Linear Regression Line (Area vs Price) ──────────────────────
    # Sort by Area for a clean regression line
    df_sorted = df[["Area", "Price"]].sort_values("Area")
    lr_model  = hpm.models["Linear Regression"]

    # Build a dummy feature matrix for Area-only regression line
    # We use median values for other features to isolate Area's effect
    medians = df.median(numeric_only=True)
    area_range = np.linspace(df["Area"].min(), df["Area"].max(), 300)

    # Construct full feature matrix using median for non-Area columns
    X_line = np.column_stack([
        area_range,
        np.full(300, medians["Bedrooms"]),
        np.full(300, medians["Bathrooms"]),
        np.full(300, medians["House_Age"]),
        np.full(300, medians["Distance_From_City"])
    ])

    # Scale using the saved scaler
    X_line_scaled = scaler.transform(X_line)
    y_line_pred   = lr_model.predict(X_line_scaled)

    plt.figure(figsize=(9, 5))
    plt.scatter(df["Area"], df["Price"], alpha=0.3, color="#4C9BE8",
                edgecolors="white", s=20, label="Actual Data Points")
    plt.plot(area_range, y_line_pred, color="#E84C4C", lw=2.5,
             label="Linear Regression Line")
    plt.title("Linear Regression: Area vs Price\n(other features held at median)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price (Rs.)")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_linear_regression_line.png", dpi=150)
    plt.close()
    print(f"  📊  Saved: {PLOTS_DIR}/05_linear_regression_line.png")

    print(f"\n  ✅  All plots saved to '{PLOTS_DIR}/' directory.\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("   HOUSE PRICE PREDICTION -- TRAINING PIPELINE")
    print("=" * 65)

    # ── Step 0: Generate dataset if CSV is missing ──────────────────────────
    if not os.path.exists(DATASET_PATH):
        print("\n[INFO] Dataset not found. Generating synthetic dataset ...")
        generate_dataset(DATASET_PATH)
    else:
        print(f"\n[INFO] Dataset found: {DATASET_PATH}")

    # ── Step 1: Load & Preprocess data ──────────────────────────────────────
    # load_and_preprocess returns 9 values including y_scaler for SVR
    (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        y_train_scaled, y_test_scaled,
        scaler, y_scaler, df
    ) = load_and_preprocess(DATASET_PATH)

    # ── Step 2: Initialise and Train Models ─────────────────────────────────
    print("\n[INFO] Initialising and Training Models ...\n")
    hpm = HousePriceModel()   # Create model object

    # Linear Regression & KNN trained on original y (raw rupee values)
    # SVR trained on y_train_scaled (SVR needs scaled target for convergence)
    hpm.train_mixed(X_train_scaled, y_train, y_train_scaled)

    # ── Step 3: Evaluate Models ─────────────────────────────────────────────
    print("\n[INFO] Evaluating Models on Test Set ...")
    # evaluate_mixed de-scales SVR predictions before computing metrics
    results = hpm.evaluate_mixed(X_test_scaled, y_test, y_test_scaled, y_scaler)
    hpm.print_comparison_table()   # Print the comparison table

    # ── Step 4: Save Models & Scalers ───────────────────────────────────────
    print("[INFO] Saving models to disk ...")
    hpm.save_models(MODEL_DIR)

    # Save both scalers (needed at prediction time in app.py)
    scaler_path   = os.path.join(MODEL_DIR, "scaler.pkl")
    y_scaler_path = os.path.join(MODEL_DIR, "y_scaler.pkl")
    joblib.dump(scaler,   scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    print(f"  [SAVED] Feature scaler -> {scaler_path}")
    print(f"  [SAVED] Target  scaler -> {y_scaler_path}")

    # ── Step 5: Generate Visualisations ─────────────────────────────────────
    print("\n[INFO] Generating Visualisation Plots ...")
    generate_visualisations(df, results, y_test, scaler, hpm)

    print("=" * 65)
    print("   [OK]  Training Pipeline Completed Successfully!")
    print("   [>>]  Run  python app.py  to launch the Gradio UI.")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
