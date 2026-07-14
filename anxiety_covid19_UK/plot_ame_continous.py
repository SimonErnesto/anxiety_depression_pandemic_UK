# -*- coding: utf-8 -*-
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.special import expit

plt.rcParams['font.family'] = "Sans Serif"
plt.rcParams['font.serif'] = "Arial"
plt.rcParams['font.size'] = 10

def income_on_gad(gender_idx, income_j, income_k, age_values, age_mean, age_std, 
                  alpha_2, c, b, delta, kappa_k, score_idx=0):
    n_samples = 1000
    n_obs = len(age_values)
    
    delta_cumsum = np.cumsum(delta, axis=0)
    income_j_pos = delta_cumsum[income_j, :]
    income_k_pos = delta_cumsum[income_k, :]
    
    ade_per_obs = np.zeros((n_samples, n_obs))
    
    for i_age, age_val in enumerate(age_values):
        age_z_val = (age_val - age_mean) / age_std
        
        eta_j = alpha_2[gender_idx, :, :] + c[gender_idx, :, :] * age_z_val + b[gender_idx, :, :] * income_j_pos[None, :]
        eta_k = alpha_2[gender_idx, :, :] + c[gender_idx, :, :] * age_z_val + b[gender_idx, :, :] * income_k_pos[None, :]
        
        if score_idx == 0:
            p_j = expit(kappa_k[:, 0, :] - eta_j)
            p_k = expit(kappa_k[:, 0, :] - eta_k)
        elif score_idx == 3:
            p_j = 1 - expit(kappa_k[:, 2, :] - eta_j)
            p_k = 1 - expit(kappa_k[:, 2, :] - eta_k)
        else:
            p_j = expit(kappa_k[:, score_idx, :] - eta_j) - expit(kappa_k[:, score_idx-1, :] - eta_j)
            p_k = expit(kappa_k[:, score_idx, :] - eta_k) - expit(kappa_k[:, score_idx-1, :] - eta_k)
        
        ade_per_obs[:, i_age] = np.mean(p_k - p_j, axis=0)
    
    return ade_per_obs.T

# Hardcode age statistics (or load from a metadata file)
age_stats = {
    "wave1": {"mean": 50.28, "std": 15, "min": 18, "max": 83},
    "wave6": {"mean": 51.72, "std": 15.01, "min": 20.0, "max": 84}
}


waves = ["wave1", "wave6"]
wave_labels = ["Wave-1", "Wave-6"]
genders = ["Female", "Male"]
incomes = ["£0-£300", "£301-£490", "£491-£740", "£741-£1,111", "£1,112+"]
colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
lines = ["-", "--", ":", "-."]

all_results = {}

for wave_idx, wave_name in enumerate(waves):
    print(f"\nProcessing {wave_name}...")
    
    idata = az.from_netcdf(f"./idata_{wave_name}_anxiety_ordered.nc")
    
    alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values
    c = az.extract(idata.posterior.c, num_samples=1000)["c"].values
    b = az.extract(idata.posterior.b, num_samples=1000)["b"].values
    delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values
    kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values
    
    G = alpha_2.shape[0]
    
    age_mean = age_stats[wave_name]["mean"]
    age_std = age_stats[wave_name]["std"]
    age_min = age_stats[wave_name]["min"]
    age_max = age_stats[wave_name]["max"]
    ages = np.linspace(age_min, age_max, 64)
    
    results_gender = []
    for gender_idx in range(G):
        print(f"  Computing for {genders[gender_idx]}...")
        iop_results = np.zeros((4, len(ages), 1000))
        
        for k_idx, k in enumerate(range(1, 5)):
            iop_results[k_idx, :, :] = income_on_gad(
                gender_idx, 0, k, ages, age_mean, age_std, 
                alpha_2, c, b, delta, kappa_k, score_idx=0
            )
        
        results_gender.append(iop_results)
    
    all_results[wave_name] = results_gender

print("\nCreating 4-panel plot with SD (posterior uncertainty)...")
fig_width_in = 7.5   # PLOS Max Width
fig_height_in = 7.5  # Keeps the aspect ratios balanced
dpi = 600 

fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in))

letters = ["A. ", "B. ", "C. ", "D. "]
l = -1

for wave_idx, wave_name in enumerate(waves):
    for gender_idx, gender in enumerate(genders):
        ax = axes[wave_idx, gender_idx]
        l += 1
        results = all_results[wave_name][gender_idx]
        ages = np.linspace(age_stats[wave_name]["min"], age_stats[wave_name]["max"], 64)
        
        for k_idx in range(4):
            effects = results[k_idx, :, :]
            
            mean_effect = np.mean(effects, axis=1)
            sd_effect = np.std(effects, axis=1, ddof=1)  # SD, not divided by anything
            
            ax.plot(ages, mean_effect, color=colors[k_idx], ls=lines[k_idx], 
                   linewidth=2.5, label=f'{incomes[0]} → {incomes[k_idx+1]}')
            ax.fill_between(ages, mean_effect - sd_effect, 
                           mean_effect + sd_effect, 
                           color=colors[k_idx], alpha=0.25)
        
        ax.set_xlabel('Age (years)', fontsize=10)
        ax.set_ylabel('Probability Change', fontsize=10)
        ax.set_ylim(-0.1, 0.25)
        ax.set_title(f'{letters[l]}{wave_labels[wave_idx]} {gender}\n "No Anxiety" (Score=0)', 
                    fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('anxiety_income_effects_score0.png', dpi=300, bbox_inches='tight')
plt.savefig('./tiff_images/anxiety_income_effects_score0.tiff', 
            dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.show()

