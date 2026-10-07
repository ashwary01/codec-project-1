"""
Customer Churn Predictor CLI & Module
Allows instant single-customer risk calculation or CSV batch scoring.
"""

import sys
import json
import numpy as np

def calculate_churn_risk(customer_data):
    """
    Calculates churn risk score (0-100%) and key risk contributors using domain coefficients.
    customer_data: dict with customer attributes
    """
    tenure = float(customer_data.get('tenure', 12))
    monthly = float(customer_data.get('MonthlyCharges', 65.0))
    contract = customer_data.get('Contract', 'Month-to-month')
    internet = customer_data.get('InternetService', 'Fiber optic')
    tech_support = customer_data.get('TechSupport', 'No')
    security = customer_data.get('OnlineSecurity', 'No')
    payment = customer_data.get('PaymentMethod', 'Electronic check')
    calls = int(customer_data.get('CustomerServiceCalls', 1))
    senior = int(customer_data.get('SeniorCitizen', 0))
    partner = customer_data.get('Partner', 'No')
    
    # Calculate log-odds score
    score_factors = []
    base_logit = -0.5
    
    # Tenure effect
    tenure_effect = -0.045 * tenure
    score_factors.append({'factor': 'Tenure Length', 'impact': round(tenure_effect, 2), 'type': 'neutral' if abs(tenure_effect)<0.2 else ('positive' if tenure_effect<0 else 'negative')})
    
    # Contract effect
    if contract == 'Month-to-month':
        c_effect = 1.75
    elif contract == 'One year':
        c_effect = -0.45
    else: # Two year
        c_effect = -0.85
    score_factors.append({'factor': f'Contract ({contract})', 'impact': round(c_effect, 2), 'type': 'negative' if c_effect > 0 else 'positive'})
    
    # Internet Service
    if internet == 'Fiber optic':
        net_effect = 0.85
    elif internet == 'DSL':
        net_effect = 0.15
    else:
        net_effect = -0.50
    score_factors.append({'factor': f'Internet ({internet})', 'impact': round(net_effect, 2), 'type': 'negative' if net_effect > 0 else 'positive'})
    
    # Tech Support & Security
    ts_effect = -0.65 if tech_support == 'Yes' else 0.40
    score_factors.append({'factor': f'Tech Support ({tech_support})', 'impact': round(ts_effect, 2), 'type': 'positive' if ts_effect < 0 else 'negative'})
    
    sec_effect = -0.55 if security == 'Yes' else 0.30
    score_factors.append({'factor': f'Online Security ({security})', 'impact': round(sec_effect, 2), 'type': 'positive' if sec_effect < 0 else 'negative'})
    
    # Monthly Charges
    monthly_effect = 0.012 * (monthly - 60)
    score_factors.append({'factor': f'Monthly Charges (${monthly})', 'impact': round(monthly_effect, 2), 'type': 'negative' if monthly_effect > 0 else 'positive'})
    
    # Payment Method
    pm_effect = 0.45 if payment == 'Electronic check' else -0.20
    score_factors.append({'factor': f'Payment ({payment})', 'impact': round(pm_effect, 2), 'type': 'negative' if pm_effect > 0 else 'positive'})
    
    # Customer Service Calls
    calls_effect = 0.40 * (calls - 1)
    score_factors.append({'factor': f'Support Calls ({calls})', 'impact': round(calls_effect, 2), 'type': 'negative' if calls_effect > 0 else 'positive'})
    
    total_logit = base_logit + tenure_effect + c_effect + net_effect + ts_effect + sec_effect + monthly_effect + pm_effect + calls_effect
    prob = 1.0 / (1.0 + np.exp(-total_logit))
    prob_percent = round(prob * 100, 1)
    
    if prob_percent >= 70:
        risk_level = "Critical Churn Risk"
        badge = "danger"
    elif prob_percent >= 45:
        risk_level = "High Churn Risk"
        badge = "warning"
    elif prob_percent >= 25:
        risk_level = "Medium Risk"
        badge = "info"
    else:
        risk_level = "Low Risk (Loyal)"
        badge = "success"
        
    # Recommendations
    recs = []
    if contract == 'Month-to-month':
        recs.append("Offer 15% discount for upgrading to an annual contract.")
    if tech_support == 'No' and internet != 'No':
        recs.append("Bundle complimentary Tech Support for 3 months.")
    if payment == 'Electronic check':
        recs.append("Encourage automated credit card billing with a $5 bill credit.")
    if calls >= 3:
        recs.append("Assign a dedicated Customer Success Specialist to resolve open grievances.")
    if not recs:
        recs.append("Customer is healthy! Send periodic loyalty appreciation rewards.")
        
    return {
        "churn_probability": prob_percent,
        "risk_level": risk_level,
        "badge": badge,
        "factors": score_factors,
        "recommendations": recs
    }

if __name__ == '__main__':
    sample_customer = {
        'tenure': 4,
        'Contract': 'Month-to-month',
        'MonthlyCharges': 89.50,
        'InternetService': 'Fiber optic',
        'TechSupport': 'No',
        'OnlineSecurity': 'No',
        'PaymentMethod': 'Electronic check',
        'CustomerServiceCalls': 4
    }
    
    res = calculate_churn_risk(sample_customer)
    print("--- CUSTOMER CHURN PREDICTION REPORT ---")
    print(f"Churn Risk Probability: {res['churn_probability']}%")
    print(f"Risk Status: {res['risk_level']}")
    print("\nTop Contributing Risk Factors:")
    for f in res['factors']:
        print(f" - {f['factor']}: {f['impact']:+}")
    print("\nActionable Retention Strategy:")
    for r in res['recommendations']:
        print(f" -> {r}")
