# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns

np.random.seed(27)

plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

sns.set(style="whitegrid", font="DeJavu Serif")

# Load wave 1 data
data_w1 = pd.read_csv("./data/Depression_covid19_UK_wave1_data.csv")

# Restrict population
data_w1 = data_w1[data_w1.Ethnicity==3]

# Load wave 6 data
data_w6 = pd.read_csv("./data/Depression_covid19_UK_wave6_data.csv")
data_w6 = data_w6[data_w6.Ethnicity==3]

# Match subjects across waves
data_w6 = data_w6[data_w6.pid.isin(data_w1.pid.unique())]
data_w1 = data_w1[data_w1.pid.isin(data_w6.pid.unique())]


# Calculate observed PHQ-9 totals (depression)
phq_columns = [f'Dep_{i}' for i in range(1, 10)]  # PHQ-9 has 9 items
data_w1['Dep_Total'] = data_w1[phq_columns].sum(axis=1)
data_w6['Dep_Total'] = data_w6[phq_columns].sum(axis=1)

# Load posterior predictive for wave 1 (depression model)
idata_w1 = az.from_netcdf("idata_wave1_depression_ordered.nc")
y_hat_w1 = idata_w1.posterior_predictive['y_hat'].values
n_samples = y_hat_w1.shape[0] * y_hat_w1.shape[1]
n_obs_w1 = y_hat_w1.shape[2]
pred_scores_w1 = y_hat_w1.reshape(n_samples, n_obs_w1)

# Load posterior predictive for wave 6 (depression model)
idata_w6 = az.from_netcdf("idata_wave6_depression_ordered.nc")
y_hat_w6 = idata_w6.posterior_predictive['y_hat'].values
n_obs_w6 = y_hat_w6.shape[2]
pred_scores_w6 = y_hat_w6.reshape(n_samples, n_obs_w6)

# Create long format for wave 1
datas_w1 = []
for d in data_w1.columns[18:]:
    df = data_w1.drop(data_w1.columns[18:], axis=1)
    df["Score"] = data_w1[d]
    df["Question"] = np.repeat(d, len(df))
    datas_w1.append(df)
data_long_w1 = pd.concat(datas_w1)
data_long_w1 = data_long_w1.sort_values("Score")
data_long_w1.reset_index(inplace=True, drop=True)

# Create long format for wave 6
datas_w6 = []
for d in data_w6.columns[18:]:
    df = data_w6.drop(data_w6.columns[18:], axis=1)
    df["Score"] = data_w6[d]
    df["Question"] = np.repeat(d, len(df))
    datas_w6.append(df)
data_long_w6 = pd.concat(datas_w6)
data_long_w6 = data_long_w6.sort_values("Score")
data_long_w6.reset_index(inplace=True, drop=True)

# Function to calculate posterior predictive proportions
def calculate_posterior_proportions(pred_scores, data_long):
    obs_metadata = pd.DataFrame({
        'question': data_long['Question'],
        'income': data_long['Income'],
        'gender': data_long['Gender'],
        'ID': data_long['pid']
    })
    
    dep_questions = [f'Dep_{i}' for i in range(1, 10)]  # Depression questions
    is_dep = obs_metadata['question'].isin(dep_questions).values
    
    unique_ids = obs_metadata['ID'].unique()
    subject_id_map = {id_: i for i, id_ in enumerate(unique_ids)}
    subject_idx = np.array([subject_id_map[pid] for pid in obs_metadata['ID']])
    
    n_subjects = len(unique_ids)
    dep_scores = pred_scores[:, is_dep]
    dep_subject_idx = subject_idx[is_dep]
    
    dep_totals = np.zeros((n_samples, n_subjects))
    for s in range(n_samples):
        dep_totals[s] = np.bincount(dep_subject_idx, weights=dep_scores[s], minlength=n_subjects)
    
    subject_metadata = obs_metadata[['ID', 'income', 'gender']].drop_duplicates().set_index('ID')
    subject_incomes = subject_metadata['income'].values
    subject_genders = subject_metadata['gender'].values
    
    return dep_totals, subject_incomes, subject_genders

# Calculate posterior for both waves
dep_w1, incomes_w1, genders_w1 = calculate_posterior_proportions(pred_scores_w1, data_long_w1)
dep_w6, incomes_w6, genders_w6 = calculate_posterior_proportions(pred_scores_w6, data_long_w6)

# Calculate observed proportions (using PHQ-9 depression thresholds with 5 categories)
def calculate_observed_proportions(data):
    results = {}
    for gender in ['Female', 'Male']:
        gender_data = data[data['Gender'] == gender]
        results[gender] = {}
        for income in sorted(gender_data['Income'].unique()):
            income_data = gender_data[gender_data['Income'] == income]
            props = []
            # PHQ-9 thresholds: None (0-4), Mild (5-9), Moderate (10-14), Moderate-Severe (15-19), Severe (20-27)
            for low, high in [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]:
                in_category = (income_data['Dep_Total'] >= low) & (income_data['Dep_Total'] <= high)
                props.append(in_category.mean())
            results[gender][income] = props
    return results

observed_w1 = calculate_observed_proportions(data_w1)
observed_w6 = calculate_observed_proportions(data_w6)

# Define thresholds (5 categories for depression)
thresholds = ['None', 'Mild', 'Moderate', 'Mod-Severe', 'Severe']
threshold_ranges = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
income_levels = sorted(data_w1['Income'].unique())

# Function to get posterior proportions by group (using 90% HDI)
def get_group_props(dep_totals, incomes, genders, gender, income_levels):
    gender_mask = genders == gender
    n_thresholds = 5
    n_incomes = len(income_levels)
    props = np.zeros((n_thresholds, n_incomes))
    props_lower = np.zeros((n_thresholds, n_incomes))
    props_upper = np.zeros((n_thresholds, n_incomes))
    
    for inc_idx, income in enumerate(income_levels):
        inc_mask = gender_mask & (incomes == income)
        subject_indices = np.where(inc_mask)[0]
        
        if len(subject_indices) > 0:
            for thresh_idx, (low, high) in enumerate(threshold_ranges):
                in_category = (dep_totals[:, subject_indices] >= low) & (dep_totals[:, subject_indices] <= high)
                props_sample = in_category.mean(axis=1)
                props[thresh_idx, inc_idx] = props_sample.mean()
                # Use 90% HDI (5th and 95th percentiles)
                props_lower[thresh_idx, inc_idx] = np.percentile(props_sample, 5)
                props_upper[thresh_idx, inc_idx] = np.percentile(props_sample, 95)
    
    return props, props_lower, props_upper

# Get posterior proportions
w1_female_props, w1_female_lower, w1_female_upper = get_group_props(dep_w1, incomes_w1, genders_w1, 'Female', income_levels)
w1_male_props, w1_male_lower, w1_male_upper = get_group_props(dep_w1, incomes_w1, genders_w1, 'Male', income_levels)
w6_female_props, w6_female_lower, w6_female_upper = get_group_props(dep_w6, incomes_w6, genders_w6, 'Female', income_levels)
w6_male_props, w6_male_lower, w6_male_upper = get_group_props(dep_w6, incomes_w6, genders_w6, 'Male', income_levels)


# =============================================
# CREATE CSV SUMMARY OF PROBABILITIES BY INCOME AND GENDER
# =============================================

summary_data = []

# For each wave
for wave_name, wave_props, wave_observed in [('Wave1', 
                                               {'Female': (w1_female_props, w1_female_lower, w1_female_upper),
                                                'Male': (w1_male_props, w1_male_lower, w1_male_upper)},
                                               observed_w1),
                                              ('Wave6',
                                               {'Female': (w6_female_props, w6_female_lower, w6_female_upper),
                                                'Male': (w6_male_props, w6_male_lower, w6_male_upper)},
                                               observed_w6)]:
    
    for gender in ['Female', 'Male']:
        props, lower, upper = wave_props[gender]
        
        for inc_idx, income in enumerate(income_levels):
            for thresh_idx, (threshold, (low, high)) in enumerate(zip(thresholds, threshold_ranges)):
                
                # Posterior estimates
                posterior_mean = props[thresh_idx, inc_idx]
                posterior_lower = lower[thresh_idx, inc_idx]
                posterior_upper = upper[thresh_idx, inc_idx]
                
                # Observed proportion
                observed_prop = wave_observed[gender][income][thresh_idx]
                
                # Sample size for this group
                if wave_name == 'Wave1':
                    n_obs = len(data_w1[(data_w1['Gender'] == gender) & (data_w1['Income'] == income)])
                else:
                    n_obs = len(data_w6[(data_w6['Gender'] == gender) & (data_w6['Income'] == income)])
                
                # Calculate difference (simple subtraction since both are scalars)
                difference = posterior_mean - observed_prop
                
                summary_data.append({
                    'Wave': wave_name,
                    'Gender': gender,
                    'Income': income,
                    'Threshold': threshold,
                    'PHQ9_Range': f"{low}-{high}",
                    'N_Observations': n_obs,
                    'Posterior_Probability_Mean': posterior_mean.round(2) * 100,
                    'Posterior_Probability_Lower_90HDI': posterior_lower.round(2) * 100,
                    'Posterior_Probability_Upper_90HDI': posterior_upper.round(2) * 100,
                    'Observed_Proportion': observed_prop.round(2) * 100,
                    'Difference_Posterior_vs_Observed': difference.round(2) * 100
                })

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data)

# Save to CSV
# summary_df.to_csv('depression_probabilities_by_income_gender.csv', index=False)

# Also create a pivot table version for easier reading
pivot_data = []
for wave in ['Wave1', 'Wave6']:
    for gender in ['Female', 'Male']:
        for threshold in thresholds:
            subset = summary_df[(summary_df['Wave'] == wave) & 
                               (summary_df['Gender'] == gender) & 
                               (summary_df['Threshold'] == threshold)]
            
            for income in income_levels:
                row_data = subset[subset['Income'] == income].iloc[0]
                pivot_data.append({
                    'Wave': wave,
                    'Gender': gender,
                    'Threshold': threshold,
                    'Income': income,
                    'Posterior_Prob': f"{row_data['Posterior_Probability_Mean']:.3f}",
                    '90HDI': f"[{row_data['Posterior_Probability_Lower_90HDI']:.3f}, {row_data['Posterior_Probability_Upper_90HDI']:.3f}]",
                    'Observed': f"{row_data['Observed_Proportion']:.3f}",
                    'N': row_data['N_Observations']
                })

pivot_df = pd.DataFrame(pivot_data)
pivot_df.to_csv('depression_probabilities_pivot_format.csv', index=False)

print("CSV files saved for DEPRESSION (PHQ-9):")
print("1. depression_probabilities_by_income_gender.csv - Long format with all estimates")
print("2. depression_probabilities_pivot_format.csv - More readable pivot format")
print("\nPreview of the summary data:")
print(summary_df.head(20))

# =============================================
# CREATE 4-PANEL FIGURE for Depression
# =============================================

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

x = np.arange(len(income_levels))
width = 0.12
posterior_offsets = np.linspace(-0.22, 0.22, len(thresholds))
observed_offsets = np.linspace(-0.22, 0.22, len(thresholds))

# Panel A: Wave 1 - Females
ax1 = axes[0, 0]
for thresh_idx, (threshold, color) in enumerate(zip(thresholds, colors)):
    post_means = w1_female_props[thresh_idx]
    post_lower = w1_female_lower[thresh_idx]
    post_upper = w1_female_upper[thresh_idx]
    yerr_lower = np.maximum(post_means - post_lower, 0)
    yerr_upper = np.maximum(post_upper - post_means, 0)
    
    ax1.bar(x + posterior_offsets[thresh_idx] - width/2, post_means, width, 
            label=f'{threshold} (Posterior)', color=color, alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    ax1.errorbar(x + posterior_offsets[thresh_idx] - width/2, post_means, 
                 yerr=[yerr_lower, yerr_upper], fmt='none', 
                 ecolor='black', capsize=2, alpha=0.5, linewidth=1)
    
    obs_means = [observed_w1['Female'][inc][thresh_idx] for inc in income_levels]
    ax1.bar(x + observed_offsets[thresh_idx] + width/2, obs_means, width, 
            label=f'{threshold} (Observed)', color='grey', alpha=0.5, 
            edgecolor='black', linewidth=0.5, hatch='//')
ax1.set_xlabel('Income Level', fontsize=12)
ax1.set_ylabel('Proportion', fontsize=12)
ax1.set_title('A: Wave-1 - Females (Depression)', fontsize=14, fontweight='bold', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(income_levels)
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3, axis='y')

# Panel B: Wave 1 - Males
ax2 = axes[0, 1]
for thresh_idx, (threshold, color) in enumerate(zip(thresholds, colors)):
    post_means = w1_male_props[thresh_idx]
    post_lower = w1_male_lower[thresh_idx]
    post_upper = w1_male_upper[thresh_idx]
    yerr_lower = np.maximum(post_means - post_lower, 0)
    yerr_upper = np.maximum(post_upper - post_means, 0)
    
    ax2.bar(x + posterior_offsets[thresh_idx] - width/2, post_means, width, 
            color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.errorbar(x + posterior_offsets[thresh_idx] - width/2, post_means, 
                 yerr=[yerr_lower, yerr_upper], fmt='none', 
                 ecolor='black', capsize=2, alpha=0.5, linewidth=1)
    
    obs_means = [observed_w1['Male'][inc][thresh_idx] for inc in income_levels]
    ax2.bar(x + observed_offsets[thresh_idx] + width/2, obs_means, width, 
            color='grey', alpha=0.5, edgecolor='black', linewidth=0.5, hatch='//')
ax2.set_xlabel('Income Level', fontsize=12)
ax2.set_ylabel('Proportion', fontsize=12)
ax2.set_title('B: Wave-1 - Males (Depression)', fontsize=14, fontweight='bold', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels(income_levels)
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right', fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3, axis='y')

# Panel C: Wave 6 - Females
ax3 = axes[1, 0]
for thresh_idx, (threshold, color) in enumerate(zip(thresholds, colors)):
    post_means = w6_female_props[thresh_idx]
    post_lower = w6_female_lower[thresh_idx]
    post_upper = w6_female_upper[thresh_idx]
    yerr_lower = np.maximum(post_means - post_lower, 0)
    yerr_upper = np.maximum(post_upper - post_means, 0)
    
    ax3.bar(x + posterior_offsets[thresh_idx] - width/2, post_means, width, 
            color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.errorbar(x + posterior_offsets[thresh_idx] - width/2, post_means, 
                 yerr=[yerr_lower, yerr_upper], fmt='none', 
                 ecolor='black', capsize=2, alpha=0.5, linewidth=1)
    
    obs_means = [observed_w6['Female'][inc][thresh_idx] for inc in income_levels]
    ax3.bar(x + observed_offsets[thresh_idx] + width/2, obs_means, width, 
            color='grey', alpha=0.5, edgecolor='black', linewidth=0.5, hatch='//')
ax3.set_xlabel('Income Level', fontsize=12)
ax3.set_ylabel('Proportion', fontsize=12)
ax3.set_title('C: Wave-6 - Females (Depression)', fontsize=14, fontweight='bold', loc='left')
ax3.set_xticks(x)
ax3.set_xticklabels(income_levels)
ax3.set_ylim(0, 1)
ax3.legend(loc='upper right', fontsize=8, ncol=2)
ax3.grid(True, alpha=0.3, axis='y')

# Panel D: Wave 6 - Males
ax4 = axes[1, 1]
for thresh_idx, (threshold, color) in enumerate(zip(thresholds, colors)):
    post_means = w6_male_props[thresh_idx]
    post_lower = w6_male_lower[thresh_idx]
    post_upper = w6_male_upper[thresh_idx]
    yerr_lower = np.maximum(post_means - post_lower, 0)
    yerr_upper = np.maximum(post_upper - post_means, 0)
    
    ax4.bar(x + posterior_offsets[thresh_idx] - width/2, post_means, width, 
            color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.errorbar(x + posterior_offsets[thresh_idx] - width/2, post_means, 
                 yerr=[yerr_lower, yerr_upper], fmt='none', 
                 ecolor='black', capsize=2, alpha=0.5, linewidth=1)
    
    obs_means = [observed_w6['Male'][inc][thresh_idx] for inc in income_levels]
    ax4.bar(x + observed_offsets[thresh_idx] + width/2, obs_means, width, 
            color='grey', alpha=0.5, edgecolor='black', linewidth=0.5, hatch='//')
ax4.set_xlabel('Income Level', fontsize=12)
ax4.set_ylabel('Proportion', fontsize=12)
ax4.set_title('D: Wave-6 - Males (Depression)', fontsize=14, fontweight='bold', loc='left')
ax4.set_xticks(x)
ax4.set_xticklabels(income_levels)
ax4.set_ylim(0, 1)
ax4.legend(loc='upper right', fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("depression_posterior_vs_observed_wave1_wave6.png", dpi=300, bbox_inches='tight')
plt.savefig("./tiff_images/depression_posterior_vs_observed_wave1_wave6.tiff", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("DEPRESSION SUMMARY STATISTICS BY INCOME AND GENDER")
print("="*60)
for wave in ['Wave1', 'Wave6']:
    print(f"\n{wave}:")
    for gender in ['Female', 'Male']:
        print(f"\n  {gender}:")
        subset = summary_df[(summary_df['Wave'] == wave) & (summary_df['Gender'] == gender)]
        for threshold in thresholds:
            thresh_subset = subset[subset['Threshold'] == threshold]
            print(f"    {threshold}: Mean posterior probability = {thresh_subset['Posterior_Probability_Mean'].mean():.3f}")
            
# Additional summary by income level
print("\n" + "="*60)
print("DEPRESSION - SEVERE PROBABILITY BY INCOME")
print("="*60)
for wave in ['Wave1', 'Wave6']:
    print(f"\n{wave} - Severe Depression:")
    for gender in ['Female', 'Male']:
        subset = summary_df[(summary_df['Wave'] == wave) & 
                           (summary_df['Gender'] == gender) & 
                           (summary_df['Threshold'] == 'Severe')]
        print(f"\n  {gender}:")
        for _, row in subset.iterrows():
            print(f"    Income {row['Income']}: {row['Posterior_Probability_Mean']:.3f} "
                  f"(90% HDI: {row['Posterior_Probability_Lower_90HDI']:.3f}-{row['Posterior_Probability_Upper_90HDI']:.3f})")