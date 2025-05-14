import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../data/processed/processed_customer_data_v1.csv')

scaler = MinMaxScaler()

df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')

# Define reference_date
reference_date = pd.to_datetime('2024-05-01')

# Recency: Days since last login or product usage
df['recency_days'] = (reference_date - df['last_active_date']).dt.days

# Frequency: Login count, feature usage count
df['frequency_score'] = (df['logins_last_30_days'] * 0.7) + (df['feature_usage_score'] * 10 * 0.3)

# Monetary: Monthly spend or LTV (months x monthly plan)
df['months_active'] = (reference_date.year - df['signup_date'].dt.year) * 12 + (reference_date.month - df['signup_date'].dt.month)
df['ltv_amount'] = df['monthly_spend'] * df['months_active']

# Engagement score: Composite of usage, tickets and recency
# Normalize frequency (higher is better)
df['frequency_scaled'] = scaler.fit_transform(df[['frequency_score']])

# Normalize recency (lower is better, so invert the scale)
df['recency_scaled'] = 1 - scaler.fit_transform(df[['recency_days']])

# Normalize support tickets (lower is better, so invert the scale)
df['support_scaled'] = 1 - scaler.fit_transform(df[['support_tickets']])

df['engagement_score'] = (
    0.4 * df['frequency_scaled'] +    # Weight for frequency (login count + feature usage)
    0.3 * df['recency_scaled'] +      # Weight for recency (last login)
    0.2 * df['support_scaled']        # Weight for support tickets (less is better)
)

print("Reference Date:\n", reference_date)
print("Recency:\n", df['recency_days'].head(10))
print("Frequency:\n", df['frequency_score'].head(10))
print("Months active:\n", df['months_active'].head(10))
print("LTV:\n", df['ltv_amount'].head(10))
print("Engagement score:\n", df['engagement_score'].head(10))

# Save the processed data
df.to_csv('../data/processed/processed_customer_data_v2.csv', index=False)
