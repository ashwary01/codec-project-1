"""
Customer Churn Prediction - Model Training & Evaluation Pipeline
Trains multiple classification algorithms on telecom churn data, evaluates performance, 
extracts feature importances, and saves evaluation metrics.
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def run_training_pipeline(csv_path='telecom_churn.csv'):
    print("1. Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Clean & preprocess TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Feature columns
    feature_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'CustomerServiceCalls'
    ]
    
    X = df[feature_cols].copy()
    y = (df['Churn'] == 'Yes').astype(int)
    
    # Categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Features: {len(feature_cols)} ({len(cat_cols)} categorical, {len(num_cols)} numerical)")
    
    # One-hot encoding categorical features
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1
    best_model_obj = None
    
    print("\n2. Training models & evaluating performance...")
    for name, model in models.items():
        # Scale for algorithms that require scaled inputs
        if name in ["Logistic Regression", "K-Nearest Neighbors"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else preds
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds).tolist()
        
        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm
        }
        
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    print(f"\nBest Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Feature importances (using Random Forest or Logistic Regression weights)
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    top_features = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp[:12]]
    
    summary = {
        "dataset_size": len(df),
        "test_size": len(y_test),
        "churn_rate": round(float((y == 1).mean()), 4),
        "models": results,
        "best_model": best_model_name,
        "feature_importances": top_features,
        "feature_names": feature_names
    }
    
    with open('model_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("Saved evaluation results to 'model_results.json'.")
    return summary

if __name__ == '__main__':
    run_training_pipeline()
