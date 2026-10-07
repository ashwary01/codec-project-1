# 📊 ChurnVision AI — Customer Churn Prediction & Retention Intelligence

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ChurnVision AI** is an enterprise-grade Machine Learning platform designed to predict customer churn, identify key risk factors, and empower retention teams with real-time what-if scenario simulations and automated playbook recommendations.

---

## 🌟 Key Features

### 1. 🌐 Interactive Web Dashboard (`http://localhost:8085`)
- **Executive Data EDA**: Interactive visualizations exploring overall churn rates, contract length impact, monthly charge distributions, and support ticket correlations.
- **Multi-Model Studio**: Real-time side-by-side performance benchmark of 5 classification algorithms (*Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, K-Nearest Neighbors*) complete with Confusion Matrices, F1-Scores, and ROC-AUC curves.
- **Single Customer Risk Predictor**: Calculate instant Churn Risk Probability Scores (0% – 100%) and Risk Categories (*Critical, High, Medium, Low*).
- **SHAP-Style Factor Breakdown**: Explains exact positive and negative drivers contributing to each customer's risk score.
- **What-If Retention Simulator**: Adjust contract terms, add tech support, or apply monthly fee discounts in real-time to observe live risk reduction.
- **Batch CSV Scoring**: Upload customer CSV lists or generate 500+ record batches to filter high-risk cohorts and export CSV reports.

### 2. 🐍 End-to-End Machine Learning Pipeline
- **Dataset Generator (`generate_dataset.py`)**: Synthesizes realistic 5,000-record Telecom/SaaS customer datasets with features including tenure, contract type, internet service, support calls, and monthly charges.
- **Model Training Engine (`train_model.py`)**: Data preprocessing (One-Hot Encoding, StandardScaler), 80/20 stratified train/test split, model fitting, metric evaluation, and feature importance extraction.
- **Inference CLI (`predict_churn.py`)**: Standalone scoring module for command-line inference.
- **Jupyter Notebook (`Customer_Churn_Prediction.ipynb`)**: Fully documented notebook for step-by-step EDA, data cleaning, and model evaluation.

---

## 📂 Repository Structure

```
codec1/
├── generate_dataset.py           # Synthetic dataset generator script
├── train_model.py                # Scikit-Learn training & evaluation pipeline
├── predict_churn.py              # Single-customer risk scoring CLI & module
├── build_notebook.py             # Script to build standalone Jupyter Notebook
├── Customer_Churn_Prediction.ipynb # Interactive Jupyter Notebook
├── server.py                     # Light-weight Python HTTP & REST API server
├── telecom_churn.csv             # Generated 5,000 customer dataset
├── model_results.json            # Model evaluation metrics & feature importances
├── index.html                    # Dashboard web app structure
├── styles.css                    # Modern glassmorphism CSS design system
├── app.js                        # Dashboard logic & Chart.js controllers
├── ml_engine.js                  # Client-side JS machine learning engine
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

---

## 📈 Model Benchmark Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** *(Best)* | **80.6%** | **68.8%** | **60.2%** | **0.642** | **0.845** | Trained |
| **Gradient Boosting** | 78.7% | 64.2% | 59.5% | 0.618 | 0.827 | Trained |
| **Decision Tree** | 78.9% | 67.6% | 51.9% | 0.587 | 0.817 | Trained |
| **Random Forest** | 79.2% | 70.0% | 49.1% | 0.577 | 0.833 | Trained |
| **K-Nearest Neighbors** | 75.6% | 59.9% | 47.1% | 0.527 | 0.777 | Trained |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.8 or higher
- Required packages: `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`

Install dependencies:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```

### 2. Generate Dataset & Train Models
```bash
# 1. Generate 5,000 customer dataset
python generate_dataset.py

# 2. Train classification models
python train_model.py

# 3. Test CLI single customer predictor
python predict_churn.py
```

### 3. Launch Web Dashboard
```bash
python server.py
```
Open your browser and navigate to **`http://localhost:8085`**.

---

## 🔑 Top Churn Drivers
1. **Contract Type**: Month-to-month contracts increase churn risk log-odds significantly (+1.75). Switching to 1-Year or 2-Year contracts reduces churn probability by over 50%.
2. **Customer Service Calls**: High support call volume ($\ge 3$ calls in 6 months) indicates friction and drives churn risk.
3. **Tech Support & Online Security**: Customers lacking Tech Support packages demonstrate double the churn rate compared to subscribed customers.
4. **Tenure**: Customer churn is front-loaded in the first 12 months; onboarding engagement is critical.

---

## 📜 License
This project is licensed under the MIT License.
