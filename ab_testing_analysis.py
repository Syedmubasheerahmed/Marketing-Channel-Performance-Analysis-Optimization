
# A/B TESTING PROJECT: Channel A vs Channel B
import pandas as pd
import numpy as np
from scipy.stats import zscore
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

print("=" * 80)
print("MARKETING CHANNEL A/B TESTING ANALYSIS")
print("=" * 80)

# STEP 1: LOAD DATA
print("\n[STEP 1] LOADING DATA")
print("-" * 80)
data = pd.read_csv("marketing_data.csv")
print("Data loaded successfully!")
print(f"  Total rows: {len(data)}")
print(f"  Total columns: {len(data.columns)}\n")
print("First 10 rows:")
print(data.head(10))

# STEP 2: CALCULATE METRICS
print("\n[STEP 2] CALCULATING METRICS")
print("-" * 80)

# Create new columns for metrics
data['conversion_rate'] = data['conversions'] / data['clicks']
data['ctr'] = data['clicks'] / data['impressions']
data['cpc'] = data['cost'] / data['clicks']
data['cpa'] = data['cost'] / data['conversions']
print("\n✓ New metrics created:")
print("  • conversion_rate = conversions ÷ clicks")
print("  • CTR = clicks ÷ impressions")
print("  • CPC = cost ÷ clicks")
print("  • CPA = cost ÷ conversions")

# STEP 3: SUMMARY STATISTICS BY CHANNEL
print("\n[STEP 3] SUMMARY STATISTICS BY CHANNEL")
print("-" * 80)
channel_a = data[data['campaign'] == 'Channel A']
channel_b = data[data['campaign'] == 'Channel B']
print("\n📊 CHANNEL A:")
print(f"  Total clicks: {channel_a['clicks'].sum():,}")
print(f"  Total conversions: {channel_a['conversions'].sum():,}")
print(f"  Total cost: ${channel_a['cost'].sum():,.2f}")
print(f"  Average conversion rate: {channel_a['conversion_rate'].mean():.2%}")
print(f"  Average CPC: ${channel_a['cpc'].mean():.2f}")
print(f"  Average CPA: ${channel_a['cpa'].mean():.2f}")
print("\n📊 CHANNEL B:")
print(f"  Total clicks: {channel_b['clicks'].sum():,}")
print(f"  Total conversions: {channel_b['conversions'].sum():,}")
print(f"  Total cost: ${channel_b['cost'].sum():,.2f}")
print(f"  Average conversion rate: {channel_b['conversion_rate'].mean():.2%}")
print(f"  Average CPC: ${channel_b['cpc'].mean():.2f}")
print(f"  Average CPA: ${channel_b['cpa'].mean():.2f}")

# STEP 4: A/B TEST (Two-Proportion Z-Test)
print("\n[STEP 4] A/B TEST - TWO-PROPORTION Z-TEST")
print("-" * 80)

# Get totals for each channel
a_conversions = channel_a['conversions'].sum()
a_clicks = channel_a['clicks'].sum()
b_conversions = channel_b['conversions'].sum()
b_clicks = channel_b['clicks'].sum()

# Calculate conversion rates
a_conv_rate = a_conversions / a_clicks
b_conv_rate = b_conversions / b_clicks

print(f"\nChannel A conversion rate: {a_conv_rate:.4f} ({a_conv_rate*100:.2f}%)")
print(f"Channel B conversion rate: {b_conv_rate:.4f} ({b_conv_rate*100:.2f}%)")
print(f"Difference: {abs(b_conv_rate - a_conv_rate)*100:.2f}%")

# Run z-test
count = np.array([a_conversions, b_conversions])
nobs = np.array([a_clicks, b_clicks])
z_statistic, p_value = proportions_ztest(count, nobs)

print(f"\n--- Z-TEST RESULTS ---")
print(f"Z-statistic: {z_statistic:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significance level (α): 0.05")

if p_value < 0.05:
    print(f"\n✓ RESULT: STATISTICALLY SIGNIFICANT DIFFERENCE")
    print(f"   One channel IS genuinely better (not just luck)")
    winner = "Channel B" if b_conv_rate > a_conv_rate else "Channel A"
    print(f"   WINNER: {winner}")
else:
    print(f"\n✗ RESULT: NO SIGNIFICANT DIFFERENCE")
    print(f"   Can't say one is better; difference might be random")

# Calculate 95% Confidence Interval
se = np.sqrt(a_conv_rate*(1-a_conv_rate)/a_clicks + 
             b_conv_rate*(1-b_conv_rate)/b_clicks)
diff = b_conv_rate - a_conv_rate
ci_lower = diff - 1.96 * se
ci_upper = diff + 1.96 * se

print(f"\n--- 95% CONFIDENCE INTERVAL ---")
print(f"Point estimate: {diff:.4f} ({diff*100:.2f}%)")
print(f"True difference is between {ci_lower*100:.2f}% and {ci_upper*100:.2f}%")

# STEP 5: FIND OUTLIERS (Unusual Days)
print("\n[STEP 5] ANOMALY DETECTION")
print("-" * 80)

data['z_score'] = zscore(data['conversion_rate'])
outliers = data[np.abs(data['z_score']) > 2]

print(f"\nDays with unusual conversion rates: {len(outliers)}")
if len(outliers) > 0:
    print("\nUnusual days (|z-score| > 2):")
    print(outliers[['campaign', 'date', 'conversions', 'clicks', 'conversion_rate', 'z_score']])
else:
    print("No significant outliers found.")

# STEP 6: CORRELATION ANALYSIS
print("\n[STEP 6] CORRELATION ANALYSIS")
print("-" * 80)

correlation = data[['cost', 'conversions']].corr()
corr_value = correlation.iloc[0, 1]

print(f"\nCorrelation between Cost and Conversions: {corr_value:.4f}")

if abs(corr_value) > 0.7:
    print("  → Strong relationship (spending more = more conversions)")
elif abs(corr_value) > 0.3:
    print("  → Moderate relationship")
else:
    print("  → Weak relationship (spending more ≠ guaranteed more conversions)")
# STEP 7: CREATE VISUALIZATIONS

print("\n[STEP 7] CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('A/B Testing Analysis: Channel A vs Channel B', fontsize=16, fontweight='bold')

# Chart 1: Conversion Rate Comparison
channels = ['Channel A', 'Channel B']
rates = [a_conv_rate, b_conv_rate]
colors = ['#1f77b4', '#ff7f0e']

axes[0, 0].bar(channels, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Conversion Rate', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Conversion Rate Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0, max(rates) + 0.05)
for i, v in enumerate(rates):
    axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold', fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Chart 2: Cost vs Conversions
axes[0, 1].scatter(channel_a['cost'], channel_a['conversions'], 
                   label='Channel A', alpha=0.6, s=100, color='#1f77b4', edgecolors='black')
axes[0, 1].scatter(channel_b['cost'], channel_b['conversions'], 
                   label='Channel B', alpha=0.6, s=100, color='#ff7f0e', edgecolors='black')
axes[0, 1].set_xlabel('Cost ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Conversions', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Cost vs Conversions (Scatter Plot)', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Chart 3: Distribution (Box Plot)
bp = axes[1, 0].boxplot([channel_a['conversion_rate'], channel_b['conversion_rate']], 
                        labels=channels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_ylabel('Conversion Rate', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Chart 4: CPA Comparison
cpas = [channel_a['cpa'].mean(), channel_b['cpa'].mean()]
axes[1, 1].bar(channels, cpas, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Cost Per Acquisition ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Cost Per Acquisition Comparison', fontsize=12, fontweight='bold')
for i, v in enumerate(cpas):
    axes[1, 1].text(i, v + 0.1, f'${v:.2f}', ha='center', fontweight='bold', fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ab_test_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Charts saved as 'ab_test_analysis.png'")
plt.show()

# STEP 8: FINAL SUMMARY & RECOMMENDATIONS

print("\n[STEP 8] FINAL SUMMARY & RECOMMENDATIONS")
print("=" * 80)

summary = f"""

A/B TEST ANALYSIS - FINAL REPORT  Channel A vs Channel B Marketing Comparison


CAMPAIGN PERFORMANCE:

  CHANNEL A:
     Total clicks: {a_clicks:,}
     Total conversions: {a_conversions:,}
     Conversion rate: {a_conv_rate*100:.2f}%
     Cost per acquisition: ${channel_a['cpa'].mean():.2f}
     Total cost: ${channel_a['cost'].sum():,.2f}

  CHANNEL B:
     Total clicks: {b_clicks:,}
     Total conversions: {b_conversions:,}
     Conversion rate: {b_conv_rate*100:.2f}%
     Cost per acquisition: ${channel_b['cpa'].mean():.2f}
     Total cost: ${channel_b['cost'].sum():,.2f}

STATISTICAL TEST RESULTS:

  Test type: Two-Proportion Z-Test
  Z-statistic: {z_statistic:.4f}
  P-value: {p_value:.6f}
  Significance level (α): 0.05
  
  Result: {'✓ SIGNIFICANT' if p_value < 0.05 else '✗ NOT SIGNIFICANT'}
  
  Interpretation:
  P-value < 0.05 means: Difference is REAL (not random chance)
  P-value ≥ 0.05 means: Can't confirm; difference might be random

CONFIDENCE INTERVAL:

  Point estimate: {diff*100:.2f}%
  95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]
  
  Meaning: We're 95% confident the true difference lies in this range

WINNER:

  {('Channel B is BETTER' if b_conv_rate > a_conv_rate else 'Channel A is BETTER')}
  Conversion rate difference: {abs(b_conv_rate - a_conv_rate)*100:.2f}%
  
  CPA difference: ${abs(channel_b['cpa'].mean() - channel_a['cpa'].mean()):.2f}
  (Lower CPA is better)

KEY INSIGHTS:

  1. Conversion Rate:
     Channel A: {a_conv_rate*100:.2f}%  |  Channel B: {b_conv_rate*100:.2f}%

  2. Cost Efficiency:
     Channel A CPA: ${channel_a['cpa'].mean():.2f}  |  Channel B CPA: ${channel_b['cpa'].mean():.2f}

  3. Cost-to-Click:
     Channel A CPC: ${channel_a['cpc'].mean():.2f}  |  Channel B CPC: ${channel_b['cpc'].mean():.2f}

  4. Correlation Insight:
     Spend-to-Conversion correlation: {corr_value:.4f}
     {'Strong' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.3 else 'Weak'} relationship

 BUSINESS RECOMMENDATIONS:

  1. ALLOCATE BUDGET WISELY
    Focus on the channel with better conversion rate
    Consider cost per acquisition when deciding

  2. INVESTIGATE SUCCESS FACTORS
    Why is {'Channel B' if b_conv_rate > a_conv_rate else 'Channel A'} performing better?
    Analyze audience, messaging, and creative elements

  3. OPTIMIZE THE WEAKER CHANNEL
    Test new ad copy, visuals, or targeting
    Compare performance again in 30 days

  4. CONTINUOUS TESTING
    A/B test different variations within each channel
    Collect more data before making final decisions
    Re-run analysis monthly to track trends

  5. NEXT STEPS
    Run test for minimum 30 days (more data = better confidence)
    Collect at least 5,000+ clicks per channel
    Monitor anomalies and unusual days
    Document learnings for future campaigns
Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

print(summary)

# Save to file
with open('ab_test_report.txt', 'w') as f:
    f.write(summary)
print("\n✓ Report saved as 'ab_test_report.txt'")

# Save processed data
data.to_csv('marketing_data_processed.csv', index=False)
print("✓ Processed data saved as 'marketing_data_processed.csv'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
