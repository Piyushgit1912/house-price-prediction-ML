# 🏡 House Price Prediction System

An end-to-end Machine Learning project designed to analyze real estate data, evaluate multiple regression models, and predict housing prices with interactive visualizations and a web interface.

---

## 📌 Project Overview
Predicting property prices accurately is critical for real estate buyers, sellers, and agents. This project implements and compares multiple supervised learning regression algorithms to identify the best-performing model based on evaluation metrics like **MAE**, **MSE**, and **R² Score**.

---

## 🚀 Features
- **Exploratory Data Analysis (EDA):** Correlation heatmaps, feature distribution plots, and target regression lines.
- **Model Training & Comparison:** Implementation of Linear Regression, K-Nearest Neighbors (KNN), and Support Vector Regressor (SVR).
- **Scalable Pipeline:** Feature scaling using `StandardScaler` with serialized artifacts (`.pkl`).
- **Interactive Web App:** User-friendly prediction interface built with Python.

---

## 📂 Project Structure

```text
house_price_prediction/
│
├── models/                         # Serialized ML models and scalers
│   ├── knn.pkl
│   ├── linear.pkl
│   ├── svr.pkl
│   ├── scaler.pkl
│   └── y_scaler.pkl
│
├── plots/                          # Visualizations & EDA outputs
│   ├── 01_correlation_heatmap.png
│   ├── 02_feature_distributions.png
│   ├── 03_actual_vs_predicted.png
│   ├── 04_model_comparison.png
│   └── 05_linear_regression_line.png
│
├── app.py                          # Web application interface
├── model.py                        # Model inference and helper routines
├── train_model.py                  # Training pipeline and visualization generator
├── house_price.csv                 # Real estate dataset
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

---

## 📊 Models Implemented & Evaluated
1. **Linear Regression:** Baseline regression benchmark.
2. **K-Nearest Neighbors (KNN) Regressor:** Instance-based non-linear regression.
3. **Support Vector Regressor (SVR):** Non-linear kernel regression with scaled inputs and targets.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

### 2. Set up a virtual environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

---

## 💻 Usage

### Retrain Models & Generate Plots
```bash
python train_model.py
```

### Launch the Application
```bash
python app.py
```
*(Or `streamlit run app.py` if configured as a Streamlit application)*

---

## 📈 Visualizations
- **Feature Correlation Heatmap:** Analyzes multi-collinearity among housing features.
- **Feature Distributions:** Inspects skewness and variance across input variables.
- **Actual vs Predicted Plots:** Evaluates model residual patterns and accuracy.

---

## 👤 Author
- **Piyush Upadhyay**
- GitHub: [@your-username](https://github.com/)
- LinkedIn: [Your Profile](https://linkedin.com/)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
