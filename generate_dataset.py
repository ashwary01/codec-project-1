"""
Customer Churn Dataset Generator
Generates realistic customer churn dataset for Telecom / SaaS industry.
Features include customer demographics, service subscriptions, account information, and churn status.
"""

import pandas as pd
import numpy as np

def generate_churn_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    customer_ids = [f"{np.random.randint(1000,9999)}-{chr(np.random.randint(65,91))}{chr(np.random.randint(65,91))}" for _ in range(n_samples)]
    
    gender = np.random.choice(['Male', 'Female'], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.52, 0.48])
    dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])
    
    # Tenure in months (1 to 72)
    tenure = np.random.randint(1, 73, size=n_samples)
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10])
    
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.44, 0.44, 0.12])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for net in internet_service:
        if net == 'No':
            online_security.append('No internet service')
            online_backup.append('No internet service')
            device_protection.append('No internet service')
            tech_support.append('No internet service')
            streaming_tv.append('No internet service')
            streaming_movies.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(['Yes', 'No'], p=[0.38, 0.62]))
            tech_support.append(np.random.choice(['Yes', 'No'], p=[0.33, 0.67]))
            streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.49, 0.51]))
            streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))
            
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.60, 0.40])
    payment_method = np.random.choice([
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ], size=n_samples, p=[0.34, 0.22, 0.22, 0.22])
    
    # Calculate realistic monthly charges based on services
    base_charge = np.where(phone_service == 'Yes', 20.0, 10.0)
    internet_charge = np.where(internet_service == 'Fiber optic', 45.0, np.where(internet_service == 'DSL', 25.0, 0.0))
    add_ons = (
        (np.array(online_security) == 'Yes').astype(int) * 8 +
        (np.array(online_backup) == 'Yes').astype(int) * 7 +
        (np.array(device_protection) == 'Yes').astype(int) * 7 +
        (np.array(tech_support) == 'Yes').astype(int) * 9 +
        (np.array(streaming_tv) == 'Yes').astype(int) * 12 +
        (np.array(streaming_movies) == 'Yes').astype(int) * 12
    )
    
    noise = np.random.normal(0, 3.0, size=n_samples)
    monthly_charges = np.round(np.clip(base_charge + internet_charge + add_ons + noise, 18.5, 118.9), 2)
    
    # Total charges roughly tenure * monthly_charges with slight variance
    total_charges = np.round(np.clip(monthly_charges * tenure + np.random.normal(0, 50.0, size=n_samples), 18.5, 8600.0), 2)
    
    # Customer Service Calls in past 6 months
    customer_service_calls = np.random.poisson(lam=1.5, size=n_samples)
    
    # Churn probability logit based on domain knowledge:
    # High churn factors: Month-to-month contract, Fiber optic without tech support, short tenure, high monthly charges, high customer service calls, electronic check.
    # Low churn factors: Two-year contract, long tenure, tech support, online security, automatic payment.
    
    logit = (
        -0.5
        - 0.05 * tenure
        + 1.8 * (contract == 'Month-to-month')
        - 0.8 * (contract == 'Two year')
        - 0.4 * (contract == 'One year')
        + 0.9 * (internet_service == 'Fiber optic')
        - 0.7 * (np.array(tech_support) == 'Yes')
        - 0.6 * (np.array(online_security) == 'Yes')
        + 0.015 * (monthly_charges - 60)
        + 0.45 * (payment_method == 'Electronic check')
        + 0.35 * (customer_service_calls >= 3)
        + 0.30 * (senior_citizen == 1)
        - 0.35 * (partner == 'Yes')
    )
    
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(str)
    churn = np.where(churn == 'True', 'Yes', 'No')
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'CustomerServiceCalls': customer_service_calls,
        'Churn': churn
    })
    
    return df

if __name__ == '__main__':
    df = generate_churn_dataset(5000)
    output_path = 'telecom_churn.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully with {len(df)} rows and saved to '{output_path}'.")
    print(f"Overall Churn Rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")
