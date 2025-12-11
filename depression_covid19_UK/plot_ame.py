# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from scipy.special import logit, expit

plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

sns.set(style="whitegrid", font="DeJavu Serif")
# sns.set_style("whitegrid")

waves = ["wave1", "wave6"]


for wa in range(len(waves)):
    
    data = pd.read_csv("./data/depression_covid19_UK_"+waves[wa]+"_data.csv")
    
    wave = data.Wave.values[0]
    wave = wave.replace("W", "Wave")
    
    datas = []
    for d in data.columns[9:]:
        df = data.drop(data.columns[9:], axis=1)
        df["Score"] = data[d]
        df["Question"] = np.repeat(d, len(df))
        datas.append(df)
    
    data = pd.concat(datas)
    
    data = data.sort_values(["Income", "Score"])
    
    scores = data.Score.values
    
    age = data.Age_year.values
    age_z = (age - age.mean()) / age.std()
    
    S = len(data.Score.unique())
    G = len(data.Gender.unique())
    I = len(data.Income.unique())
    Q = len(data.Question.unique())
    
    g_idx = pd.factorize(data.Gender)[0]
    i_idx = pd.factorize(data.Income)[0]
    q_idx = pd.factorize(data.Question)[0]
    
    idata = az.from_netcdf("./idata_"+waves[wa]+"_depression_ordered.nc")
    
    # Extract parameters for the income model
    alpha_1 = az.extract(idata.posterior.alpha_1, num_samples=1000)["alpha_1"].values  # shape: (1000, 2) for 2 genders
    a = az.extract(idata.posterior.a, num_samples=1000)["a"].values        # a shape: (1000, 2) for 2 genders
    kappa_j = az.extract(idata.posterior.kappa_j, num_samples=1000)["kappa_j"].values  # shape: (1000, 4) for 4 cutpoints (5 income levels)
    kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values #(9, 3, 1000)
    alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values #(2, 9, 1000)
    c = az.extract(idata.posterior.c, num_samples=1000)["c"].values # c (2, 9, 1000)
    b = az.extract(idata.posterior.b, num_samples=1000)["b"].values # b (2, 9, 1000)
    delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values #(5, 1000)
    
    
    ####################### Define Functions ############################
    #####################################################################
    def logistic_pdf(x):
      return np.exp(x) / (1 + np.exp(x))**2
        
    def pordlog(a):
        pa = expit(a)
        p_cum = np.concatenate(([0.], pa, [1.]))
        return p_cum[1:] - p_cum[:-1]
    
    def calculate_income_probabilities(age_z, gender, kappa_j, alpha_1, a):
        """Calculate income probabilities for each age and sex"""
        results = []
        
        for gender_idx in range(len(gender)):  # 0 and 1 for sexes
            ages = np.sort(np.unique(age))
            ages_z = (ages - ages.mean()) / ages.std()
            for i in tqdm(range(len(np.unique(ages)))):
                # Calculate probabilities for each posterior sample
                n_samples = alpha_1.shape[1]  # 1000 samples
                probs_all = np.zeros((n_samples, 5))  # 5 income levels
                
                for s in range(n_samples):
                    eta = alpha_1[gender_idx, s] + a[gender_idx, s] * ages_z[i]
                    cuts = kappa_j[:, s] - eta  # kappa_j shape: (4, 1000)
                    probs_all[s, :] = pordlog(cuts)
                
                # Calculate summary statistics for each income level
                for income_level in range(5):
                    prob_samples = probs_all[:, income_level]
                    
                    results.append({
                        'age': ages[i],
                        'sex': gender[gender_idx],
                        'income_level': "Inc"+str(income_level+1),
                        'mean': np.mean(prob_samples),
                        'sd': np.std(prob_samples),
                        'hdi_5%': az.hdi(prob_samples, hdi_prob=0.9)[0],
                        'hdi_95%': az.hdi(prob_samples, hdi_prob=0.9)[1]
                    })
        
        return pd.DataFrame(results)
    
    
    def age_on_income(gender_idx, beta_var, var_values):
        """
        Compute average marginal effect(AME) for age on income (ordered logistic)
        
        gender_idx: which gender to compute for (0 or 1)
        beta_var: a coefficient for age
        var_values: standardised age (age_z) values to compute over
        """
        n_samples = 1000
        n_obs = len(var_values)
        n_categories = 5  # 5 income levels (4 cutpoints)
        
        # Get parameters for this gender
        beta = beta_var[gender_idx,:]  # shape: (1000,)
        alpha = alpha_1[gender_idx,:]   # shape: (1000,)
        
        ame_per_obs = np.zeros((n_samples, n_obs, n_categories))
        
        for i, x_val in enumerate(var_values):
            # Compute linear predictor for this observation
            eta_val = alpha + beta * x_val  # shape: (1000,)
            
            # Compute marginal effect for each income category
            for s in range(n_categories):
                if s == 0:
                    # P(income=0) = Λ(κ₀ - η)
                    term = logistic_pdf(kappa_j[0,:] - eta_val)
                    ame_per_obs[:, i, s] = -beta * term
                    
                elif s == n_categories - 1:
                    # P(income=4) = 1 - Λ(κ₃ - η)
                    term = logistic_pdf(kappa_j[3,:] - eta_val)
                    ame_per_obs[:, i, s] = beta * term
                    
                else:
                    # P(income=s) = Λ(κ_s - η) - Λ(κ_{s-1} - η)
                    term1 = logistic_pdf(kappa_j[s-1,:] - eta_val)
                    term2 = logistic_pdf(kappa_j[s,:] - eta_val)
                    ame_per_obs[:, i, s] = beta * (term1 - term2)
        
        # Average across observations
        return np.mean(ame_per_obs, axis=1).T  # shape: (5, 1000)
    
    
    def age_on_phq(gender_idx, question_idx, income_idx, beta_var, var_values):
        """
        Compute average marginal effects (AME) for continuous variables, holding income constant
        
        gender_idx: which gender to compute for (0 or 1)
        question_idx: which question to compute for (0-8)  
        income_idx: which income level to hold constant (0-4)
        beta_var: which beta coefficient to use (e.g., c for age)
        var_values: values of the variable to compute AME for
        """
        n_samples = 1000
        n_obs = len(var_values)
        n_categories = 4  # 3 cutpoints for 4 score categories
        
        # Get the relevant parameters
        kappa_q = kappa_k[question_idx, :, :]  # shape: (3, 1000)
        beta = beta_var[gender_idx, question_idx, :]  # shape: (1000,)
        alpha = alpha_2[gender_idx, question_idx, :]  # shape: (1000,)
        b_val = b[gender_idx, question_idx, :]  # shape: (1000,)
        
        # Get the specific income effect (not average!)
        delta_cumsum = np.cumsum(delta, axis=0)  # shape: (5, 1000)
        income_effect = delta_cumsum[income_idx, :]  # shape: (1000,)
        
        ame_per_obs = np.zeros((n_samples, n_obs, n_categories))
        
        for i, x_val in enumerate(var_values):
            # Compute linear predictor for this observation with FIXED income
            eta_val = alpha + beta * x_val + b_val * income_effect  # shape: (1000,)
            
            # Compute marginal effect for each category
            for s in range(n_categories):
                if s == 0:
                    # P(y=0) = Λ(κ₀ - η)
                    # ∂P/∂x = -β · λ(κ₀ - η)
                    term = logistic_pdf(kappa_q[0, :] - eta_val)
                    ame_per_obs[:, i, s] = -beta * term
                    
                elif s == n_categories - 1:
                    # P(y=3) = 1 - Λ(κ₂ - η)  
                    # ∂P/∂x = β · λ(κ₂ - η)
                    term = logistic_pdf(kappa_q[2, :] - eta_val)
                    ame_per_obs[:, i, s] = beta * term
                    
                else:
                    # P(y=s) = Λ(κ_s - η) - Λ(κ_{s-1} - η)
                    # ∂P/∂x = β · [λ(κ_{s-1} - η) - λ(κ_s - η)]
                    term1 = logistic_pdf(kappa_q[s-1, :] - eta_val)
                    term2 = logistic_pdf(kappa_q[s, :] - eta_val)
                    ame_per_obs[:, i, s] = beta * (term1 - term2)
        
        # Average across observations
        return np.mean(ame_per_obs, axis=1).T  # shape: (4,1000)
    
    
     
    def income_on_phq(gender_idx, question_idx, income_j, income_k, age_values):
        """
        Compute average discrete effect (ADE) for income levels
        
        gender_idx: which gender to compute for (0 or 1)
        question_idx: which question to compute for (0-8)  
        income_j: base income level
        income_k: compared income level
        age_values: values of standardised age
        """
        n_samples = 1000
        n_obs = len(age_values)
        
        # Get parameters
        kappa_q = kappa_k[question_idx, :, :]  # shape: (3, 1000)
        alpha = alpha_2[gender_idx, question_idx, :]  # shape: (1000,)
        bA = c[gender_idx, question_idx, :]  # shape: (1000,)
        b_val = b[gender_idx, question_idx, :]  # shape: (1000,)
        
        # Get income positions
        delta_cumsum = np.cumsum(delta, axis=0)  # shape: (5, 1000)
        income_j_pos = delta_cumsum[income_j, :]  # shape: (1000,)
        income_k_pos = delta_cumsum[income_k, :]  # shape: (1000,)
        
        ade_per_obs = np.zeros((4, n_samples))
        
        # for i, age_val in enumerate(age_values):
        # Compute linear predictors for both income levels
        eta_j = alpha + bA * age_values.mean() + b_val * income_j_pos  # shape: (1000,)
        eta_k = alpha + bA * age_values.mean() + b_val * income_k_pos  # shape: (1000,)
        
        # Compute probablity differences for all categories
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
        
        # Average across observations
        return ade_per_obs  # shape: (4,1000)
    
    
    
    #################### Create Arrays for Plotting ####################
    ####################################################################
    
    # Calculate probabilities and create dataframe
    df_income_probs = calculate_income_probabilities(age_z, data.Gender.unique(), kappa_j, alpha_1, a)
    # Save to CSV
    df_income_probs.to_csv(wave+"_income_probabilities_summary.csv", index=False)
    
    
    aoi_ames = np.zeros((2,5,1000))
    for g in tqdm(range(G)):
        aoi_ames[g,:,:] = age_on_income(g, a, age_z) / age.std() #recover age scale
        
    aop_ames = np.zeros((2,9,5,4,1000))
    for q in tqdm(range(Q)):
        for i in range(I):
            for g in range(G):
                aop_ames[g,q,i,:,:] = age_on_phq(g, q, i, c, age_z) / age.std() #recover age scale
                              
    iop_ades = np.zeros((4, 2, 9, 4, 1000))
    for k_idx, k in enumerate(tqdm(range(1, 5))):  # k_idx: 0,1,2,3 for Q2,Q3,Q4,Q5
        for q in range(Q):
            for g in range(G):
                # Compute difference between lowest income (0) and current income level (k)
                ade_result = income_on_phq(g, q, 0, k, age_z)  # shape: (4, 1000)
                iop_ades[k_idx, g, q, :, :] = ade_result
                
    iop_ades1 = iop_ades[3,:,:,:,:] #keep only lowest minus highest difference
    iop_ades2 = iop_ades[:,:,:,0,:] #keep all differences but at score 0 only                
    
                 
                    
    ###################### Plot Figure #####################
    ########################################################
    
    sex_levels = data.Gender.unique()
    
    income_levels = ["£0-£300", "£301-£490", "£491-£740", "£741-£1,111", "£1,112+"]
    
    age_levels = ["18-31", "31-44", "44-57", "57-70", "70-83"]
    
    score_levels = ["Not at all (0)", "Several days (1)", 
                    "More than half the days (2)", "Nearly every day (1)"]
        
    income_comparisons = ["Inc1→Inc2", "Inc1→Inc3", "Inc1→Inc4", "Inc1→Inc5"]  # Comparisons from lowest income
    
    score_levels = ["Score 0", "Score 1", "Score 2", "Score 3"]
    
    # Create the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#556B2F', '#8E4585'] #plum for males, olive for females
    
    # Panel 1: Age effects on income (aoi_ames) - shape: (2, 5, 1000)
    aoi_summary = np.array([[
        [np.mean(aoi_ames[g, i, :]), 
         np.percentile(aoi_ames[g, i, :], 5),
         np.percentile(aoi_ames[g, i, :], 95)]
        for i in range(5)
    ] for g in range(2)])  # shape: (2, 5, 3)
    
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
    ax1.set_ylabel('AME of Age on Income Level')
    ax1.set_title('A. Age Effects on Income Distribution')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(income_levels, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Age effects on PHQ (aop_ames) - shape: (2, 9, 5, 4, 1000)
    # Average over questions and income: (2, 4, 1000)
    aop_avg_questions = np.mean(aop_ames, axis=1)  # Average over questions: (2, 5, 4, 1000)
    aop_avg_income = np.mean(aop_avg_questions, axis=1)  # Average over income: (2, 4, 1000)
    
    aop_summary = np.array([[
        [np.mean(aop_avg_income[g, s, :]),
         np.percentile(aop_avg_income[g, s, :], 5),
         np.percentile(aop_avg_income[g, s, :], 95)]
        for s in range(4)
    ] for g in range(2)])  # shape: (2, 4, 3)
    
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
    
    ax2.set_xlabel('PHQ-9 Score')
    ax2.set_ylabel('AME of Age on PHQ-9 Score')
    ax2.set_title('B. Age Effects on Depression Scores')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(score_levels, rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Income effects on PHQ - difference between lowest and highest income
    # iop_ades shape: (2, 9, 4, 1000) - gender, question, score, samples
    ax3 = axes[1, 0]
    
    # Average over questions: (2, 4, 1000)
    iop_avg_questions = np.mean(iop_ades1, axis=1)
    
    x_pos = np.arange(4)
    width = 0.35
    
    for gender_idx in range(2):
        means = []
        errors_lower = []
        errors_upper = []
        
        for score_idx in range(4):  # For each PHQ-9 score
            # Get the ADE for moving from lowest to highest income for this score
            ade_samples = iop_avg_questions[gender_idx, score_idx, :]  # shape: (1000,)
            
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
    
    ax3.set_xlabel('PHQ-9 Score')
    ax3.set_ylabel('ADE: Inc1→Inc5 Effect')
    ax3.set_title('C. Income Effects: Lowest → Highest Income')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(score_levels, rotation=45, ha='right')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3, axis='y')
    
    
    # Panel 4: All income differences for Score 0 only (iop_ades2)
    # iop_ades2 shape: (2, 9, 4, 1000) - gender, question, income_diff, samples
    ax4 = axes[1, 1]
    
    # Average over questions: (2, 4, 1000)
    iop_avg_questions2 = np.mean(iop_ades2, axis=2).swapaxes(0,1)
    
    x_pos = np.arange(4)
    width = 0.35
    
    for gender_idx in range(2):
        means = []
        errors_lower = []
        errors_upper = []
        
        for income_diff_idx in range(4):  # For each income comparison
            # Get the ADE for this income comparison for Score 0
            ade_samples = iop_avg_questions2[gender_idx, income_diff_idx, :]  # shape: (1000,)
            
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
    ax4.set_title('D. Income Effects on "Not at all (0)"')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(income_comparisons, rotation=45, ha='right')
    ax4.legend()
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='y')
    
    
    plt.tight_layout()
    plt.savefig('effects_'+wave+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    ## compute in log-odds
    a = a
    b = b
    c = c
    mediator_effect = b
    direct_effect = c
    indirect_effect = a[:,None,:] * b
    total_effect = c + a[:,None,:] * b    
    
    #compute from mfxs
    a = aoi_ames[:,4,:] * (age.max() - age.min()) #lowest[0] income
    b = iop_ades[3,:,:,0,:] #lowest to highest income [3] and Score=0 [0]
    c = aop_ames[:,:,4,0,:] * (age.max() - age.min())  #score = 0 averaged over income and from min to max age
    mediator_effect = b
    direct_effect = c
    indirect_effect = a[:,None,:] * b
    total_effect = c + a[:,None,:] * b  
    
    r = mediator_effect.shape[0] * mediator_effect.shape[1]
    effs = list(np.repeat("Mediator", r)) + list(np.repeat("Direct", r)) + list(np.repeat("Indirect", r)) +  list(np.repeat("Total", r))
    
    sex1 = np.repeat(data.Gender.unique()[0], Q)
    sex2 = np.repeat(data.Gender.unique()[1], Q)
    sexs = np.array([list(sex1) + list(sex2) for e in range(4)]).flatten()
    
    qs = data.Question.unique()
    quests = np.array([qs for g in range(G) for e in range(4)]).flatten()
    
    effects = pd.DataFrame({"Mean":effs, "SD":effs, "HDI_5":effs, "HDI_95":effs, 
                            "Effect":effs, "Sex":sexs, "Question":quests})
    
    effects["Question"] = effects["Question"].str.replace("Dep_", "Q")
    
    
    med_mean = mediator_effect.mean(axis=2).round(3)
    med_sd = mediator_effect.std(axis=2).round(3)
    med_hdi = np.array([az.hdi(mediator_effect[g].T, hdi_prob=0.9) for g in range(G)]).T.round(3)
    
    dir_mean = direct_effect.mean(axis=2).round(3)
    dir_sd = direct_effect.std(axis=2).round(3)
    dir_hdi = np.array([az.hdi(direct_effect[g].T, hdi_prob=0.9) for g in range(G)]).T.round(3)  
    
    ind_mean = indirect_effect.mean(axis=2).round(5)
    ind_sd = indirect_effect.std(axis=2).round(5)
    ind_hdi = np.array([az.hdi(indirect_effect[g].T, hdi_prob=0.9) for g in range(G)]).T.round(5) 
    
    tot_mean = total_effect.mean(axis=2).round(3)
    tot_sd = total_effect.std(axis=2).round(3)
    tot_hdi = np.array([az.hdi(total_effect[g].T, hdi_prob=0.9) for g in range(G)]).T.round(3) 
    
    effects["Mean"] = np.array([med_mean, dir_mean, ind_mean, tot_mean]).flatten()
    effects["SD"] = np.array([med_sd, dir_sd, ind_sd, tot_sd]).flatten()
    effects["HDI_5"] = np.array([med_hdi[0], dir_hdi[0], ind_hdi[0], tot_hdi[0]]).flatten()
    effects["HDI_95"] = np.array([med_hdi[1], dir_hdi[1], ind_hdi[1], tot_hdi[1]]).flatten()
    
    effects.to_csv(wave+"_effects_summary.csv", index=False)
    
    
    med_mean = mediator_effect.mean(axis=(1,2)).round(3)
    med_sd = mediator_effect.std(axis=(1,2)).round(3)
    med_hdi = az.hdi(mediator_effect.T, hdi_prob=0.9).T.round(3)
    
    dir_mean = direct_effect.mean(axis=(1,2)).round(3)
    dir_sd = direct_effect.std(axis=(1,2)).round(3)
    dir_hdi = az.hdi(direct_effect.T, hdi_prob=0.9).T.round(3)  
    
    ind_mean = indirect_effect.mean(axis=(1,2)).round(5)
    ind_sd = indirect_effect.std(axis=(1,2)).round(5)
    ind_hdi = az.hdi(indirect_effect.T, hdi_prob=0.9).T.round(5) 
    
    tot_mean = total_effect.mean(axis=(1,2)).round(3)
    tot_sd = total_effect.std(axis=(1,2)).round(3)
    tot_hdi = az.hdi(total_effect.T, hdi_prob=0.9).T.round(3) 
    
    r = mediator_effect.shape[0] 
    effs = list(np.repeat("Mediator", r)) + list(np.repeat("Direct", r)) + list(np.repeat("Indirect", r)) +  list(np.repeat("Total", r))
    
    sex1 = np.repeat(data.Gender.unique()[0], 1)
    sex2 = np.repeat(data.Gender.unique()[1], 1)
    sexs = np.array([list(sex1) + list(sex2) for e in range(4)]).flatten()
    
    effects_ave = pd.DataFrame({"Effect":effs, "Sex":sexs, 
                                "Mean":effs, "SD":effs, "HDI_5":effs, "HDI_95":effs})
    
    effects_ave["Mean"] = np.array([med_mean, dir_mean, ind_mean, tot_mean]).flatten()
    effects_ave["SD"] = np.array([med_sd, dir_sd, ind_sd, tot_sd]).flatten()
    effects_ave["HDI_5"] = np.array([med_hdi[0], dir_hdi[0], ind_hdi[0], tot_hdi[0]]).flatten()
    effects_ave["HDI_95"] = np.array([med_hdi[1], dir_hdi[1], ind_hdi[1], tot_hdi[1]]).flatten()
    
    effects_ave.to_csv(wave+"_average_effects_summary.csv", index=False)
    
    