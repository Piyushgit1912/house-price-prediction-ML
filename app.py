"""
app.py
-------
Gradio web interface for the House Price Prediction project.

This module:
  - Loads all three pre-trained models and the scaler from disk
  - Accepts user input via interactive sliders
  - Returns predictions from all three models + an average
  - Displays results in a clean, readable format

Run AFTER train_model.py has been executed.

Usage:
    python app.py

Author  : ML Practical Project
Purpose : Gradio UI for House Price Prediction
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix Unicode / emoji output on Windows (CP1252 → UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import joblib
import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# Local Imports
# ─────────────────────────────────────────────────────────────────────────────
from model import HousePriceModel


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Load Saved Models & Scaler
# ═════════════════════════════════════════════════════════════════════════════
MODEL_DIR      = "models"
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")
Y_SCALER_PATH  = os.path.join(MODEL_DIR, "y_scaler.pkl")

# Guard: ensure models exist before loading
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "[ERROR] Trained models not found!\n"
        "        Please run  python train_model.py  first."
    )

# Load the HousePriceModel and restore saved weights
hpm = HousePriceModel()
hpm.load_models(MODEL_DIR)

# Load the fitted feature scaler (for X)
scaler = joblib.load(SCALER_PATH)

# Load the fitted target scaler (for y — used to de-scale SVR predictions)
y_scaler = joblib.load(Y_SCALER_PATH)
print("[OK]  All models and scalers loaded successfully.\n")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ Prediction Function
# ═════════════════════════════════════════════════════════════════════════════
def predict_price(area: float, bedrooms: int, bathrooms: int,
                  house_age: int, distance: float) -> str:
    """
    Accept user inputs and return price predictions from all three models.

    Steps:
      1. Assemble the input features into a numpy array
      2. Scale using the pre-fitted StandardScaler
      3. Predict using all three models
      4. Compute the average prediction
      5. Return a formatted HTML string for Gradio

    Parameters
    ----------
    area       : float  – House area in square feet
    bedrooms   : int    – Number of bedrooms
    bathrooms  : int    – Number of bathrooms
    house_age  : int    – Age of house in years
    distance   : float  – Distance from city centre in km

    Returns
    -------
    str : HTML-formatted prediction results table
    """
    # ── Assemble the input feature vector (same order as training) ──────────
    input_data = np.array([[area, bedrooms, bathrooms, house_age, distance]])

    # ── Scale the input using the same scaler used during training ──────────
    # This is CRUCIAL — raw input must be scaled before feeding to KNN / SVR
    input_scaled = scaler.transform(input_data)

    # ── Get predictions from all three models ───────────────────────────────
    lr_pred  = hpm.models["Linear Regression"].predict(input_scaled)[0]
    knn_pred = hpm.models["KNN Regressor"].predict(input_scaled)[0]

    # SVR was trained on scaled target → de-scale its prediction back to rupees
    svr_pred_scaled = hpm.models["SVR (RBF)"].predict(input_scaled)[0]
    svr_pred = y_scaler.inverse_transform([[svr_pred_scaled]])[0][0]

    # ── Calculate the average of all three predictions ────────────────────────
    avg_pred = (lr_pred + knn_pred + svr_pred) / 3

    # ── Format prices as readable Indian Rupee strings ──────────────────────
    def fmt(val):
        return f"₹{val:,.0f}"

    # ── Build an HTML output card ────────────────────────────────────────────
    html = f"""
    <div style="
        font-family: 'Segoe UI', sans-serif;
        max-width: 580px;
        margin: auto;
    ">
        <h2 style="
            text-align: center;
            color: #1a1a2e;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 14px 20px;
            border-radius: 10px 10px 0 0;
            margin: 0;
        ">🏠 House Price Predictions</h2>

        <table style="
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
        ">
            <thead>
                <tr style="background:#f0f0f0; color:#333;">
                    <th style="padding:12px 16px; text-align:left;">Algorithm</th>
                    <th style="padding:12px 16px; text-align:right;">Predicted Price</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #ddd; background:#fff;">
                    <td style="padding:12px 16px;">
                        <span style="color:#4C9BE8; font-weight:600;">📈 Linear Regression</span>
                    </td>
                    <td style="padding:12px 16px; text-align:right; font-weight:600; color:#4C9BE8;">
                        {fmt(lr_pred)}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #ddd; background:#f9f9f9;">
                    <td style="padding:12px 16px;">
                        <span style="color:#F4845F; font-weight:600;">🔵 KNN Regressor</span>
                    </td>
                    <td style="padding:12px 16px; text-align:right; font-weight:600; color:#F4845F;">
                        {fmt(knn_pred)}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #ddd; background:#fff;">
                    <td style="padding:12px 16px;">
                        <span style="color:#5CB85C; font-weight:600;">🔺 SVR (RBF Kernel)</span>
                    </td>
                    <td style="padding:12px 16px; text-align:right; font-weight:600; color:#5CB85C;">
                        {fmt(svr_pred)}
                    </td>
                </tr>
                <tr style="background:linear-gradient(135deg, #667eea22, #764ba222);">
                    <td style="padding:14px 16px; font-size:16px;">
                        <strong>⭐ Average Prediction</strong>
                    </td>
                    <td style="padding:14px 16px; text-align:right; font-size:18px;">
                        <strong style="color:#764ba2;">{fmt(avg_pred)}</strong>
                    </td>
                </tr>
            </tbody>
        </table>

        <div style="
            background:#f8f8ff;
            border-left: 4px solid #764ba2;
            padding: 10px 16px;
            margin-top: 12px;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            color: #555;
        ">
            ℹ️ Predictions are based on models trained on synthetic data.
            Results are for educational purposes only.
        </div>
    </div>
    """
    return html


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Gradio Interface Definition
# ═════════════════════════════════════════════════════════════════════════════
def build_interface() -> gr.Blocks:
    """
    Build and return the Gradio Blocks interface.

    The interface includes:
      - Sliders for each input feature
      - A Predict button
      - An HTML output card with predictions from all three models

    Returns
    -------
    gr.Blocks : The assembled Gradio application object
    """

    # Custom CSS for a polished look
    custom_css = """
    body {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    }
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .gr-block {
        border-radius: 12px !important;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(css=custom_css, title="House Price Predictor") as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            text-align: center;
            padding: 24px 20px 10px;
            font-family: 'Segoe UI', sans-serif;
        ">
            <h1 style="
                font-size: 2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 4px;
            ">🏠 House Price Predictor</h1>
            <p style="color:#aaa; font-size:0.95rem; margin-top:4px;">
                Compare predictions from Linear Regression, KNN, and SVR
            </p>
            <hr style="border-color:#333; margin-top:16px;">
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):

                gr.Markdown("### 📋 Enter House Details")

                # Input sliders ─────────────────────────────────────────────
                area = gr.Slider(
                    minimum=500, maximum=5000, value=1500, step=50,
                    label="🏗️  Area (sq ft)",
                    info="Total built-up area of the house"
                )

                bedrooms = gr.Slider(
                    minimum=1, maximum=6, value=3, step=1,
                    label="🛏️  Number of Bedrooms",
                    info="Total number of bedrooms"
                )

                bathrooms = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1,
                    label="🚿  Number of Bathrooms",
                    info="Total number of bathrooms"
                )

                house_age = gr.Slider(
                    minimum=0, maximum=50, value=5, step=1,
                    label="🏚️  House Age (years)",
                    info="Age of the house in years (0 = new)"
                )

                distance = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="📍  Distance from City Centre (km)",
                    info="Distance from the nearest city centre"
                )

                # Predict button ─────────────────────────────────────────────
                predict_btn = gr.Button(
                    "🔍  Predict Price",
                    variant="primary",
                    size="lg"
                )

        # Output section ─────────────────────────────────────────────────────
        gr.Markdown("### 📊 Prediction Results")
        output_html = gr.HTML(
            label="Prediction Output",
            value="""
            <div style="
                text-align:center;
                padding:30px;
                color:#888;
                font-family:'Segoe UI',sans-serif;
                border: 2px dashed #444;
                border-radius: 10px;
            ">
                ⬆️ Fill in the house details above and click <strong>Predict Price</strong>
            </div>
            """
        )

        # ── Wire up the button to the prediction function ───────────────────
        predict_btn.click(
            fn=predict_price,
            inputs=[area, bedrooms, bathrooms, house_age, distance],
            outputs=output_html
        )

        # ── Footer note ──────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            text-align:center;
            padding: 16px;
            color:#666;
            font-size:0.8rem;
            font-family:'Segoe UI',sans-serif;
        ">
            Built with 🐍 Python · Scikit-Learn · Gradio &nbsp;|&nbsp;
            College ML Practical Project
        </div>
        """)

    return demo


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[INFO] Launching House Price Predictor ...")
    interface = build_interface()
    interface.launch(
        share=False,       # Set to True to get a public shareable link
        inbrowser=True,    # Auto-open browser tab
        server_name="127.0.0.1",
        # server_port not set → Gradio auto-finds the next free port
    )
