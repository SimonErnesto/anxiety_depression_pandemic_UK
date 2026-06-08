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

thresholds = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]

# Function to assign GAD threshold based on score
def get_gad_threshold(score):
    for low, high in thresholds:
        if low <= score <= high:
            return f"{low} to {high}"
    return None

# Create threshold columns for both datasets
data_w1['PHQ_Threshold'] = data_w1['Dep_Total'].apply(get_gad_threshold)
data_w6['PHQ_Threshold'] = data_w6['Dep_Total'].apply(get_gad_threshold)

# Create a dictionary to store all results
results = {}

# Process each wave
for wave_name, data in [('Wave1', data_w1), ('Wave6', data_w6)]:
    
    # 1. Value counts for Income
    income_counts = data['Income'].value_counts().reset_index()
    income_counts.columns = ['Income', 'Count']
    income_counts['Wave'] = wave_name
    income_counts['Statistic'] = 'value_counts'
    
    # 2. Statistics for Age_year
    age_stats = pd.DataFrame({
        'Variable': ['Age_year'],
        'Mean': [data['Age_year'].mean().round(2)],
        'SD': [data['Age_year'].std().round(2)],
        'Min': [data['Age_year'].min()],
        'Max': [data['Age_year'].max()],
        'Wave': wave_name,
        'Statistic': 'descriptive_stats'
    })
    
    # 3. Statistics for GAD_Total
    gad_stats = pd.DataFrame({
        'Variable': ['Dep_Total'],
        'Mean': [data['Dep_Total'].mean().round(2)],
        'SD': [data['Dep_Total'].std().round(2)],
        'Min': [data['Dep_Total'].min()],
        'Max': [data['Dep_Total'].max()],
        'Wave': wave_name,
        'Statistic': 'descriptive_stats'
    })
    
    # 4. Value counts for GAD thresholds
    threshold_counts = data['PHQ_Threshold'].value_counts().reset_index()
    threshold_counts.columns = ['PHQ_Threshold', 'Count']
    threshold_counts['Wave'] = wave_name
    threshold_counts['Statistic'] = 'threshold_counts'
    
    # 5. Value counts for Gender
    gender_counts = data['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    gender_counts['Wave'] = wave_name
    gender_counts['Statistic'] = 'gender_counts'
    
    # Store in results dictionary
    results[wave_name] = {
        'income': income_counts,
        'age_stats': age_stats,
        'gad_stats': gad_stats,
        'thresholds': threshold_counts,
        'genders': gender_counts
    }

# Combine all results into one dataframe
combined_dfs = []

for wave_name in results:
    combined_dfs.append(results[wave_name]['income'])
    combined_dfs.append(results[wave_name]['age_stats'])
    combined_dfs.append(results[wave_name]['gad_stats'])
    combined_dfs.append(results[wave_name]['thresholds'])
    combined_dfs.append(results[wave_name]['genders'])

final_df = pd.concat(combined_dfs, ignore_index=True)

# Reorder columns for better readability
final_df = final_df[['Wave', 'Statistic', 'Income', 'Variable', 'PHQ_Threshold', 
                      'Count', 'Mean', 'SD', 'Min', 'Max']]

# Save to CSV
final_df.to_csv('descriptive_statistics.csv', index=False)

# Display the dataframe
print("Summary Statistics DataFrame:")
print(final_df)

print("\n" + "="*50)
print("\nPreview of PHQ threshold distribution:")
print("\nWave 1:")
print(data_w1['PHQ_Threshold'].value_counts().sort_index())
print("\nWave 6:")
print(data_w6['PHQ_Threshold'].value_counts().sort_index())