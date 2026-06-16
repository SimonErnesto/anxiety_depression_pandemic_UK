# -*- coding: utf-8 -*-
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from scipy.special import logit, expit
import pandas as pd

plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

sns.set(style="whitegrid", font="DeJavu Serif")

waves = ["wave1", "wave6"]

# Hardcode age statistics (or load from a metadata file)
age_stats = {
    "wave1": {"mean": 50.28, "std": 15, "min": 18, "max": 83},
    "wave6": {"mean": 51.72, "std": 15.01, "min": 20.0, "max": 84}
}

for wa in range(len(waves)):
    wave_name = waves[wa]
    
    # Load posterior from NetCDF
    idata = az.from_netcdf(f"./idata_{wave_name}_anxiety_ordered.nc")
    
    # Extract coordinates from the NetCDF file
    gender_labels = list(idata.posterior.coords['gender'].values)
    question_labels = list(idata.posterior.coords['question'].values)
    income_labels = list(np.sort(idata.posterior.coords['income'].values))
    
    # Get wave label from coordinates or hardcode
    wave_label = "Wave1" if "wave1" in wave_name else "Wave6"
    
    # Get age statistics
    age_mean = age_stats[wave_name]["mean"]
    age_std = age_stats[wave_name]["std"]
    age_min = age_stats[wave_name]["min"]
    age_max = age_stats[wave_name]["max"]
    
    # Create age array for plotting
    ages = np.linspace(age_min, age_max, 100)
    age_z = (ages - age_mean) / age_std
    
    # Extract parameters
    alpha_1 = az.extract(idata.posterior.alpha_1, num_samples=1000)["alpha_1"].values
    a = az.extract(idata.posterior.a, num_samples=1000)["a"].values
    kappa_j = az.extract(idata.posterior.kappa_j, num_samples=1000)["kappa_j"].values
    kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values
    alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values
    c = az.extract(idata.posterior.c, num_samples=1000)["c"].values
    b = az.extract(idata.posterior.b, num_samples=1000)["b"].values
    delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values
    
    S = 4  # 4 score levels
    G = 2  # 2 genders
    I = 5  # 5 income levels
    Q = len(question_labels)
    
    ####################### Define Functions ############################
    #####################################################################
    def logistic_pdf(x):
        return np.exp(x) / (1 + np.exp(x))**2
        
    def pordlog(a):
        pa = expit(a)
        p_cum = np.concatenate(([0.], pa, [1.]))
        return p_cum[1:] - p_cum[:-1]
    
    def age_on_income(gender_idx, beta_var, var_values):
        """
        Compute average marginal effect(AME) for age on income (ordered logistic)
        """
        n_samples = 1000
        n_obs = len(var_values)
        n_categories = 5
        
        beta = beta_var[gender_idx,:]
        alpha = alpha_1[gender_idx,:]
        
        ame_per_obs = np.zeros((n_samples, n_obs, n_categories))
        
        for i, x_val in enumerate(var_values):
            eta_val = alpha + beta * x_val
            
            for s in range(n_categories):
                if s == 0:
                    term = logistic_pdf(kappa_j[0,:] - eta_val)
                    ame_per_obs[:, i, s] = -beta * term
                elif s == n_categories - 1:
                    term = logistic_pdf(kappa_j[3,:] - eta_val)
                    ame_per_obs[:, i, s] = beta * term
                else:
                    term1 = logistic_pdf(kappa_j[s-1,:] - eta_val)
                    term2 = logistic_pdf(kappa_j[s,:] - eta_val)
                    ame_per_obs[:, i, s] = beta * (term1 - term2)
        
        return np.mean(ame_per_obs, axis=1).T
    
    def age_on_gad(gender_idx, question_idx, income_idx, beta_var, var_values):
        """
        Compute average marginal effects (AME) for continuous variables
        """
        n_samples = 1000
        n_obs = len(var_values)
        n_categories = 4
        
        kappa_q = kappa_k[question_idx, :, :]
        beta = beta_var[gender_idx, question_idx, :]
        alpha = alpha_2[gender_idx, question_idx, :]
        b_val = b[gender_idx, question_idx, :]
        
        delta_cumsum = np.cumsum(delta, axis=0)
        income_effect = delta_cumsum[income_idx, :]
        
        ame_per_obs = np.zeros((n_samples, n_obs, n_categories))
        
        for i, x_val in enumerate(var_values):
            eta_val = alpha + beta * x_val + b_val * income_effect
            
            for s in range(n_categories):
                if s == 0:
                    term = logistic_pdf(kappa_q[0, :] - eta_val)
                    ame_per_obs[:, i, s] = -beta * term
                elif s == n_categories - 1:
                    term = logistic_pdf(kappa_q[2, :] - eta_val)
                    ame_per_obs[:, i, s] = beta * term
                else:
                    term1 = logistic_pdf(kappa_q[s-1, :] - eta_val)
                    term2 = logistic_pdf(kappa_q[s, :] - eta_val)
                    ame_per_obs[:, i, s] = beta * (term1 - term2)
        
        return np.mean(ame_per_obs, axis=1).T
    
    def income_on_gad(gender_idx, question_idx, income_j, income_k, age_values):
        """
        Compute average discrete effect (ADE) for income levels
        """
        n_samples = 1000
        
        kappa_q = kappa_k[question_idx, :, :]
        alpha = alpha_2[gender_idx, question_idx, :]
        bA = c[gender_idx, question_idx, :]
        b_val = b[gender_idx, question_idx, :]
        
        delta_cumsum = np.cumsum(delta, axis=0)
        income_j_pos = delta_cumsum[income_j, :]
        income_k_pos = delta_cumsum[income_k, :]
        
        ade_per_obs = np.zeros((4, n_samples))
        
        eta_j = alpha + bA * age_values.mean() + b_val * income_j_pos
        eta_k = alpha + bA * age_values.mean() + b_val * income_k_pos
        
        for s in range(4):
            if s == 0:
                p_j = expit(kappa_q[0, :] - eta_j)
                p_k = expit(kappa_q[0, :] - eta_k)
            elif s == 3:
                p_j = 1 - expit(kappa_q[2, :] - eta_j)
                p_k = 1 - expit(kappa_q[2, :] - eta_k)
            else:
                p_j = expit(kappa_q[s, :] - eta_j) - expit(kappa_q[s-1, :] - eta_j)
                p_k = expit(kappa_q[s, :] - eta_k) - expit(kappa_q[s-1, :] - eta_k)
            
            ade_per_obs[s,:] = p_k - p_j
        
        return ade_per_obs
    
    #################### Create Arrays for Plotting ####################
    ####################################################################
    
    aoi_ames = np.zeros((2, 5, 1000))
    for g in tqdm(range(G), desc=f"{wave_label}: Age on Income"):
        aoi_ames[g,:,:] = age_on_income(g, a, age_z) / age_std
    
    aop_ames = np.zeros((2, 7, 5, 4, 1000))
    for q in tqdm(range(Q), desc=f"{wave_label}: Age on GAD"):
        for i in range(I):
            for g in range(G):
                aop_ames[g,q,i,:,:] = age_on_gad(g, q, i, c, age_z) / age_std
    
    iop_ades = np.zeros((4, 2, 7, 4, 1000))
    for k_idx, k in enumerate(tqdm(range(1, 5), desc=f"{wave_label}: Income on GAD")):
        for q in range(Q):
            for g in range(G):
                ade_result = income_on_gad(g, q, 0, k, age_z)
                iop_ades[k_idx, g, q, :, :] = ade_result
    
    iop_ades1 = iop_ades[3,:,:,:,:]
    iop_ades2 = iop_ades[:,:,:,0,:]
    
    ###################### Plot Figure #####################
    ########################################################
    
    sex_levels = gender_labels
    income_levels = income_labels
    score_levels = ["Score 0", "Score 1", "Score 2", "Score 3"]
    income_comparisons = ["Inc1→Inc2", "Inc1→Inc3", "Inc1→Inc4", "Inc1→Inc5"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#556B2F', '#8E4585']
    
    # Panel 1: Age effects on income
    aoi_summary = np.array([[
        [np.mean(aoi_ames[g, i, :]), 
         np.percentile(aoi_ames[g, i, :], 5),
         np.percentile(aoi_ames[g, i, :], 95)]
        for i in range(5)
    ] for g in range(2)])
    
    ax1 = axes[0, 0]
    x_pos = np.arange(5)
    width = 0.35
    
    for gender_idx in range(2):
        means = aoi_summary[gender_idx, :, 0]
        errors = [[means[i] - aoi_summary[gender_idx, i, 1] for i in range(5)],
                  [aoi_summary[gender_idx, i, 2] - means[i] for i in range(5)]]
        
        ax1.bar(x_pos + width*gender_idx - width/2, means, width, 
                color=colors[gender_idx], label=sex_levels[gender_idx],
                yerr=errors, capsize=5, alpha=0.8)
    
    ax1.set_xlabel('Income Level')
    ax1.set_ylabel('AME')
    ax1.set_title('A. Age Effects by Income Level')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(income_levels, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-0.002, 0.007)
    
    # Panel 2: Age effects on GAD
    aop_avg_questions = np.mean(aop_ames, axis=1)
    aop_avg_income = np.mean(aop_avg_questions, axis=1)
    
    aop_summary = np.array([[
        [np.mean(aop_avg_income[g, s, :]),
         np.percentile(aop_avg_income[g, s, :], 5),
         np.percentile(aop_avg_income[g, s, :], 95)]
        for s in range(4)
    ] for g in range(2)])
    
    ax2 = axes[0, 1]
    x_pos = np.arange(4)
    width = 0.35
    
    for gender_idx in range(2):
        means = aop_summary[gender_idx, :, 0]
        errors = [[means[i] - aop_summary[gender_idx, i, 1] for i in range(4)],
                  [aop_summary[gender_idx, i, 2] - means[i] for i in range(4)]]
        
        ax2.bar(x_pos + width*gender_idx - width/2, means, width, 
                color=colors[gender_idx], label=sex_levels[gender_idx],
                yerr=errors, capsize=5, alpha=0.8)
    
    ax2.set_xlabel('GAD-7 Score')
    ax2.set_ylabel('AME')
    ax2.set_title('B. Age Effects by Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(score_levels, rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(-0.004, 0.007)
    
    # Panel 3: Income effects on GAD
    ax3 = axes[1, 0]
    iop_avg_questions = np.mean(iop_ades1, axis=1)
    
    x_pos = np.arange(4)
    width = 0.35
    
    for gender_idx in range(2):
        means = []
        errors_lower = []
        errors_upper = []
        
        for score_idx in range(4):
            ade_samples = iop_avg_questions[gender_idx, score_idx, :]
            mean_val = np.mean(ade_samples)
            lower = np.percentile(ade_samples, 5)
            upper = np.percentile(ade_samples, 95)
            
            means.append(mean_val)
            errors_lower.append(mean_val - lower)
            errors_upper.append(upper - mean_val)
        
        errors = [errors_lower, errors_upper]
        
        ax3.bar(x_pos + width*gender_idx - width/2, means, width, 
                color=colors[gender_idx], label=sex_levels[gender_idx],
                yerr=errors, capsize=5, alpha=0.8)
    
    ax3.set_xlabel('GAD-7 Score')
    ax3.set_ylabel('ADE (Inc5 - Inc1)')
    ax3.set_title('C. Income Effects by Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(score_levels, rotation=45, ha='right')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: All income differences for Score 0
    ax4 = axes[1, 1]
    iop_avg_questions2 = np.mean(iop_ades2, axis=2).swapaxes(0,1)
    
    x_pos = np.arange(4)
    width = 0.35
    
    for gender_idx in range(2):
        means = []
        errors_lower = []
        errors_upper = []
        
        for income_diff_idx in range(4):
            ade_samples = iop_avg_questions2[gender_idx, income_diff_idx, :]
            mean_val = np.mean(ade_samples)
            lower = np.percentile(ade_samples, 5)
            upper = np.percentile(ade_samples, 95)
            
            means.append(mean_val)
            errors_lower.append(mean_val - lower)
            errors_upper.append(upper - mean_val)
        
        errors = [errors_lower, errors_upper]
        
        ax4.bar(x_pos + width*gender_idx - width/2, means, width, 
                color=colors[gender_idx], label=sex_levels[gender_idx],
                yerr=errors, capsize=5, alpha=0.8)
    
    ax4.set_xlabel('Income Comparison')
    ax4.set_ylabel('ADE on Score=0')
    ax4.set_title('D. Income Effects by Group Pairs')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(income_comparisons, rotation=45, ha='right')
    ax4.legend()
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'effects_{wave_label}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./tiff_images/effects_{wave_label}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    
    ## Compute effects in log-odds
    mediator_effect = b
    direct_effect = c
    indirect_effect = a[:,None,:] * b
    total_effect = c + a[:,None,:] * b    
    
    # Compute from marginal effects
    a_mfx = aoi_ames[:,4,:] * (age_max - age_min)
    b_mfx = iop_ades[3,:,:,0,:]
    c_mfx = aop_ames[:,:,4,0,:] * (age_max - age_min)
    
    mediator_effect_mfx = b_mfx
    direct_effect_mfx = c_mfx
    indirect_effect_mfx = a_mfx[:,None,:] * b_mfx
    total_effect_mfx = c_mfx + a_mfx[:,None,:] * b_mfx
    
    r = mediator_effect_mfx.shape[0] * mediator_effect_mfx.shape[1]
    effs = list(np.repeat("Mediator", r)) + list(np.repeat("Direct", r)) + \
           list(np.repeat("Indirect", r)) + list(np.repeat("Total", r))
    
    sex1 = np.repeat(gender_labels[0], Q)
    sex2 = np.repeat(gender_labels[1], Q)
    sexs = np.array([list(sex1) + list(sex2) for e in range(4)]).flatten()
    
    quests = np.array([question_labels for g in range(G) for e in range(4)]).flatten()
    
    effects = pd.DataFrame({"Mean":effs, "SD":effs, "HDI_5":effs, "HDI_95":effs, 
                            "Effect":effs, "Sex":sexs, "Question":quests})
    
    effects["Question"] = effects["Question"].str.replace("Dep_", "Q").str.replace("GAD_", "Q")
    
    med_mean = mediator_effect_mfx.mean(axis=2).round(3)
    med_sd = mediator_effect_mfx.std(axis=2).round(3)
    med_hdi = np.array([az.hdi(mediator_effect_mfx[g].T, hdi_prob=0.9) for g in range(G)]).T.round(3)
    
    dir_mean = direct_effect_mfx.mean(axis=2).round(3)
    dir_sd = direct_effect_mfx.std(axis=2).round(3)
    dir_hdi = np.array([az.hdi(direct_effect_mfx[g].T, hdi_prob=0.9) for g in range(G)]).T.round(3)
    
    ind_mean = indirect_effect_mfx.mean(axis=2).round(5)
    ind_sd = indirect_effect_mfx.std(axis=2).round(5)
    ind_hdi = np.array([az.hdi(indirect_effect_mfx[g].T, hdi_prob=0.9) for g in range(G)]).T.round(5)
    
    tot_mean = total_effect_mfx.mean(axis=2).round(3)
    tot_sd = total_effect_mfx.std(axis=2).round(3)
    tot_hdi = np.array([az.hdi(total_effect_mfx[g].T, hdi_prob=0.9) for g in range(G)]).T.round(3)
    
    effects["Mean"] = np.array([med_mean, dir_mean, ind_mean, tot_mean]).flatten()
    effects["SD"] = np.array([med_sd, dir_sd, ind_sd, tot_sd]).flatten()
    effects["HDI_5"] = np.array([med_hdi[0], dir_hdi[0], ind_hdi[0], tot_hdi[0]]).flatten()
    effects["HDI_95"] = np.array([med_hdi[1], dir_hdi[1], ind_hdi[1], tot_hdi[1]]).flatten()
    
    effects.to_csv(f"{wave_label}_effects_summary.csv", index=False)
    
    # Average effects
    med_mean = mediator_effect_mfx.mean(axis=(1,2)).round(3)
    med_sd = mediator_effect_mfx.std(axis=(1,2)).round(3)
    med_hdi = az.hdi(mediator_effect_mfx.T, hdi_prob=0.9).T.round(3)
    
    dir_mean = direct_effect_mfx.mean(axis=(1,2)).round(3)
    dir_sd = direct_effect_mfx.std(axis=(1,2)).round(3)
    dir_hdi = az.hdi(direct_effect_mfx.T, hdi_prob=0.9).T.round(3)
    
    ind_mean = indirect_effect_mfx.mean(axis=(1,2)).round(5)
    ind_sd = indirect_effect_mfx.std(axis=(1,2)).round(5)
    ind_hdi = az.hdi(indirect_effect_mfx.T, hdi_prob=0.9).T.round(5)
    
    tot_mean = total_effect_mfx.mean(axis=(1,2)).round(3)
    tot_sd = total_effect_mfx.std(axis=(1,2)).round(3)
    tot_hdi = az.hdi(total_effect_mfx.T, hdi_prob=0.9).T.round(3)
    
    r = mediator_effect_mfx.shape[0] 
    effs = list(np.repeat("Mediator", r)) + list(np.repeat("Direct", r)) + \
           list(np.repeat("Indirect", r)) + list(np.repeat("Total", r))
    
    sex1 = np.repeat(gender_labels[0], 1)
    sex2 = np.repeat(gender_labels[1], 1)
    sexs = np.array([list(sex1) + list(sex2) for e in range(4)]).flatten()
    
    effects_ave = pd.DataFrame({"Effect":effs, "Sex":sexs, 
                                "Mean":effs, "SD":effs, "HDI_5":effs, "HDI_95":effs})
    
    effects_ave["Mean"] = np.array([med_mean, dir_mean, ind_mean, tot_mean]).flatten()
    effects_ave["SD"] = np.array([med_sd, dir_sd, ind_sd, tot_sd]).flatten()
    effects_ave["HDI_5"] = np.array([med_hdi[0], dir_hdi[0], ind_hdi[0], tot_hdi[0]]).flatten()
    effects_ave["HDI_95"] = np.array([med_hdi[1], dir_hdi[1], ind_hdi[1], tot_hdi[1]]).flatten()
    
    effects_ave.to_csv(f"{wave_label}_average_effects_summary.csv", index=False)