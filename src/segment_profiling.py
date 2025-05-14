import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/processed/processed_customer_data_v3.csv')

# Define monochrome palette and set style
mono_palette = sns.color_palette("Blues_r")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlepad": 15,
    "axes.labelpad": 10
})


# Plot 1: Average Feature Usage by Segment
plt.figure()
segment_usage = df.groupby("segment")["feature_usage_score"].mean().sort_values()
sns.barplot(x=segment_usage.values, y=segment_usage.index, palette=mono_palette)
plt.title("Average Feature Usage Score by Segment")
plt.xlabel("Feature Usage Score")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/figures/feature_usage_by_segment.png")
plt.show()

# Plot 2: Average Monthly Spend by Segment
plt.figure()
segment_spend = df.groupby("segment")["monthly_spend"].mean().sort_values()
sns.barplot(x=segment_spend.values, y=segment_spend.index, palette=mono_palette)
plt.title("Average Monthly Spend by Segment")
plt.xlabel("Monthly Spend ($)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/figures/monthly_spend_by_segment.png")
plt.show()

# Plot 3: Engagement Score by Segment
plt.figure()
sns.violinplot(data=df, x="engagement_score", y="segment", palette=mono_palette, density_norm="width", cut=0)
plt.title("Engagement Score Distribution by Segment")
plt.xlabel("Engagement Score")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/figures/engagement_score_by_segment.png")
plt.show()

# Plot 4: Support Tickets by Segment
plt.figure()
sns.stripplot(data=df, x="support_tickets", y="segment", palette=mono_palette, jitter=True, alpha=0.5)
plt.title("Support Tickets by Segment")
plt.xlabel("Support Tickets (Last 30 Days)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/figures/support_tickets_by_segment.png")
plt.show()

# Plot 5: Monthly Spend by Industry
industry_spend = df.groupby("industry")["monthly_spend"].mean().sort_values()
plt.figure()
sns.barplot(x=industry_spend.values, y=industry_spend.index,  palette=mono_palette)
plt.title("Average Monthly Spend by Industry")
plt.xlabel("Monthly Spend ($)")
plt.ylabel("Industry")
plt.tight_layout()
plt.savefig("../reports/figures/monthly_spend_by_industry.png")
plt.show()

# Plot 6: Engagement Score by Team Size Bucket
df["team_size_bucket"] = pd.cut(df["team_size"], bins=[0, 10, 50, 100, 200, 500],
                                labels=["1-10", "11-50", "51-100", "101-200", "201-500"])
team_engagement = df.groupby("team_size_bucket", observed=False)["engagement_score"].mean().reset_index()
plt.figure()
sns.pointplot(data=team_engagement, x="team_size_bucket", y="engagement_score", color=mono_palette[2])
plt.title("Average Engagement Score by Team Size Bucket")
plt.xlabel("Team Size")
plt.ylabel("Engagement Score")
plt.tight_layout()
plt.savefig("../reports/figures/engagement_by_team_size.png")
plt.show()

# Plot 7: Segment Distribution
plt.figure()
segment_dist = df["segment"].value_counts()
sns.barplot(x=segment_dist.values, y=segment_dist.index, palette=mono_palette)
plt.title("Customer Segment Distribution")
plt.xlabel("Number of Customers")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/figures/segment_distribution.png")
plt.show()

# Plot 8: LTV by Segment
plt.figure()
segment_ltv = df.groupby("segment")["ltv_amount"].mean().sort_values()
sns.barplot(x=segment_ltv.values, y=segment_ltv.index, palette=mono_palette)
plt.title("Average Customer Lifetime Value by Segment")
plt.xlabel("Lifetime Value ($)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../reports/figures/ltv_by_segment.png")
plt.show()
