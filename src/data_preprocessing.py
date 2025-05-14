import pandas as pd

df = pd.read_csv('../data/raw/raw_customer_data.csv')

# Change Data Types:
df['subscription_plan'] = df['subscription_plan'].astype('category')
df['industry'] = df['industry'].astype('category')
df['is_churned'] = df['is_churned'].astype('bool')
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['last_active_date'] = pd.to_datetime(df['last_active_date'])

# Save the processed data
df.to_csv('../data/processed/processed_customer_data_v1.csv', index=False)

print("\nData types after conversion:\n")
print(df.info())
