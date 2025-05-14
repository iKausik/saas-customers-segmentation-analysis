import pandas as pd
import numpy as np

df = pd.read_csv('../data/processed/processed_customer_data_v2.csv')

# 
# Customer Segmentation:
# 
df['segment'] = 'Unclassified'

# Power Users: High engagement & recently active
df['segment'] = np.where(
    (df['engagement_score'] > 0.75) & (df['recency_days'] < 30),
    'Power User',
    df['segment']
)

# At Risk: High engagement but haven’t logged in for 30–120 days
df['segment'] = np.where(
    (df['engagement_score'] > 0.6) & (df['recency_days'] >= 30) & (df['recency_days'] < 120),
    'At Risk',
    df['segment']
)

# Sleeping Giants: High monetary value but low engagement
df['segment'] = np.where(
    (df['engagement_score'] < 0.5) & (df['ltv_amount'] > df['ltv_amount'].quantile(0.75)),
    'Sleeping Giant',
    df['segment']
)

# New Users: Recently signed up (last 2 months)
df['segment'] = np.where(
    df['months_active'] <= 2,
    'New User',
    df['segment']
)

# Already Churned: Directly using existing column
df['segment'] = np.where(
    df['is_churned'] == True,
    'Already Churned',
    df['segment']
)

# Nurture and Grow: Active but not engaging deeply
df['segment'] = np.where(
    (df['segment'] == 'Unclassified') &
    (df['engagement_score'] > 0.3) & 
    (df['recency_days'] < 60),
    'Nurture and Grow', 
    df['segment']
)

# Enterprise vs Self-Serve: Append info to segment
# Differentiating between Self-Serve and Enterprise, keeping the plan info intact
df['segment'] = np.where(
    df['subscription_plan'] == 'Enterprise',
    'Enterprise - ' + df['segment'],
    'Self-Serve - ' + df['segment']
)

print(df['segment'].value_counts())

# Save the updated data
df.to_csv('../data/processed/processed_customer_data_v3.csv', index=False)
