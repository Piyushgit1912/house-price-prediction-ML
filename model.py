"""
model.py
---------
Defines the HousePriceModel class that wraps Linear Regression,
KNN Regressor, and SVR in a clean, reusable Object-Oriented structure.

Key design decision:
  - Linear Regression and KNN train on the original (unscaled) target y.
  - SVR trains on the SCALED target (y_scaled), because SVR is very
    sensitive to the magnitude of the output variable.
  - During evaluation, SVR predictions are de-scaled back to rupees
    so all models are compared fairly on the same scale.

Author  : ML Practical Project
Purpose : Model definition and evaluation for House Price Prediction
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

# Fix Unicode / emoji output on Windows (CP1252 → UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import joblib
from tabulate import tabulate

# Scikit-Learn: Algorithm imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Scikit-Learn: Metrics for evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ─────────────────────────────────────────────────────────────────────────────
# HousePriceModel Class
# ─────────────────────────────────────────────────────────────────────────────
class HousePriceModel:
    """
    A wrapper class for three ML regression models:
      - Linear Regression
      - K-Nearest Neighbors (KNN) Regressor
      - Support Vector Regression (SVR) with RBF kernel

    Attributes
    ----------
    models : dict
        Dictionary holding model instances keyed by their short names.
    results : dict
        Dictionary storing evaluation metrics for each trained model.
    """

    def __init__(self):
        """
        Initialise the three regression models with their default settings.
        KNN uses 5 nearest neighbours (a common default).
        SVR uses the RBF (Radial Basis Function) kernel.
        """
        self.models = {
            "Linear Regression": LinearRegression(),
            "KNN Regressor"    : KNeighborsRegressor(n_neighbors=5),
            "SVR (RBF)"        : SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        }

        # Will store metrics after evaluation
        self.results = {}

    # ─────────────────────────────────────────────────────────────────────────
    def train_mixed(self, X_train, y_train, y_train_scaled):
        """
        Train all three models with appropriate targets:
          - Linear Regression → trains on y_train (original scale)
          - KNN Regressor     → trains on y_train (original scale)
          - SVR (RBF)         → trains on y_train_scaled (scaled y)
                                SVR needs scaled target for good convergence.

        Parameters
        ----------
        X_train         : array-like – Scaled feature matrix for training.
        y_train         : array-like – Original price values (for LR & KNN).
        y_train_scaled  : array-like – Scaled price values (for SVR).
        """
        for name, model in self.models.items():
            print(f"  [TRAIN]  Training {name} ...")

            # SVR is trained on scaled y; others on original y
            if name == "SVR (RBF)":
                model.fit(X_train, y_train_scaled)
            else:
                model.fit(X_train, y_train)

            print(f"  [DONE ]  {name} trained successfully.")

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_mixed(self, X_test, y_test, y_test_scaled, y_scaler):
        """
        Evaluate all models on the test set.

        SVR predictions are de-scaled using y_scaler before computing
        metrics, so all models are compared in the same rupee scale.

        Parameters
        ----------
        X_test          : array-like – Scaled feature matrix for testing.
        y_test          : array-like – True prices (original scale).
        y_test_scaled   : array-like – True prices (scaled — for SVR eval).
        y_scaler        : StandardScaler – Fitted scaler for the target y.

        Returns
        -------
        dict : Evaluation metrics for all models.
        """
        for name, model in self.models.items():
            if name == "SVR (RBF)":
                # SVR predicts in scaled space → de-scale back to rupees
                y_pred_scaled = model.predict(X_test)
                y_pred = y_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).ravel()
            else:
                y_pred = model.predict(X_test)

            # Use original y_test for fair metric comparison
            mae  = mean_absolute_error(y_test, y_pred)
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)                    # RMSE = sqrt(MSE)
            r2   = r2_score(y_test, y_pred)        # R² score (1.0 is perfect)

            self.results[name] = {
                "MAE"        : round(mae, 2),
                "MSE"        : round(mse, 2),
                "RMSE"       : round(rmse, 2),
                "R2"         : round(r2, 4),       # Using "R2" key (no special char)
                "Predictions": y_pred              # Keep for visualisation
            }

        return self.results

    # ─────────────────────────────────────────────────────────────────────────
    def print_comparison_table(self):
        """
        Print a neatly formatted comparison table of all model metrics.
        Highlights the best-performing model based on R2 score.
        """
        if not self.results:
            print("[WARN]  No results yet. Please call evaluate_mixed() first.")
            return

        print("\n" + "=" * 65)
        print("         MODEL PERFORMANCE COMPARISON TABLE")
        print("=" * 65)

        # Build rows for the tabulate table
        table_rows = []
        for name, metrics in self.results.items():
            table_rows.append([
                name,
                f"{metrics['MAE']:,.2f}",
                f"{metrics['MSE']:,.2f}",
                f"{metrics['RMSE']:,.2f}",
                f"{metrics['R2']:.4f}"
            ])

        headers = ["Model", "MAE (Rs.)", "MSE (Rs.2)", "RMSE (Rs.)", "R2 Score"]
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        # ── Find and display the best model based on R2 (higher is better)
        best_model = max(self.results, key=lambda m: self.results[m]["R2"])
        print(f"\n  [BEST]  Best Model  : {best_model}")
        print(f"          R2 Score    : {self.results[best_model]['R2']}")
        print(f"          RMSE        : Rs.{self.results[best_model]['RMSE']:,.2f}")
        print("=" * 65 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    def save_models(self, save_dir="models"):
        """
        Save all trained models to disk using Joblib.

        Parameters
        ----------
        save_dir : str
            Directory path where model .pkl files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)   # Create folder if missing

        # Map model names -> file names
        file_map = {
            "Linear Regression": "linear.pkl",
            "KNN Regressor"    : "knn.pkl",
            "SVR (RBF)"        : "svr.pkl"
        }

        for name, filename in file_map.items():
            path = os.path.join(save_dir, filename)
            joblib.dump(self.models[name], path)
            print(f"  [SAVED] {name} -> {path}")

    # ─────────────────────────────────────────────────────────────────────────
    def load_models(self, save_dir="models"):
        """
        Load previously saved models from disk.

        Parameters
        ----------
        save_dir : str
            Directory path containing the saved .pkl files.
        """
        file_map = {
            "Linear Regression": "linear.pkl",
            "KNN Regressor"    : "knn.pkl",
            "SVR (RBF)"        : "svr.pkl"
        }

        for name, filename in file_map.items():
            path = os.path.join(save_dir, filename)
            self.models[name] = joblib.load(path)
            print(f"  [LOAD]  {name} loaded from {path}")
