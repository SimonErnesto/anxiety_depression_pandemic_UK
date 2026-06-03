# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

sns.set(style="whitegrid", font="DeJavu Serif")

waves = ["wave1", "wave6"]

for wa in range(len(waves)):
    
    # For wave6, we need to use the same filtering as when the model was trained
    if waves[wa] == "wave6":
        data_w1 = pd.read_csv("./data/anxiety_covid19_UK_wave1_data.csv")
        data_w1 = data_w1[data_w1.Ethnicity==3]
        data_w1 = data_w1[data_w1.Residence==0]
        data_w1 = data_w1[data_w1.Trauma==0]
        
        data = pd.read_csv("./data/anxiety_covid19_UK_"+waves[wa]+"_data.csv")
        data = data[data.Ethnicity==3]
        data = data[data.Residence==0]
        data = data[data.Trauma==0]
        
        data = data[data.pid.isin(data_w1.pid.unique())]
        
    else:  
        data = pd.read_csv("./data/anxiety_covid19_UK_"+waves[wa]+"_data.csv")
        data = data[data.Ethnicity==3]
        data = data[data.Residence==0]
        data = data[data.Trauma==0]

    wave = data.Wave.values[0]
    wave = wave.replace("W", "Wave")

    # Reshape data from wide to long format
    datas = []
    for d in data.columns[19:]:
        df = data.drop(data.columns[19:], axis=1)
        df["Score"] = data[d]
        df["Question"] = np.repeat(d, len(df))
        datas.append(df)
    
    data_long = pd.concat(datas)
    data_long = data_long.sort_values(["Income", "Score"])
    
    scores = data_long.Score.values
    
    # Load the saved model
    idata = az.from_netcdf("./idata_"+waves[wa]+"_anxiety_ordered.nc")
    
    # Check shape of posterior predictive samples
    pp_samples = idata.posterior_predictive['y_hat'].values
    n_chains, n_draws, n_obs_model = pp_samples.shape
    n_obs_data = len(scores)
    
    print(f"Wave: {waves[wa]}")
    print(f"Number of observations in model: {n_obs_model}")
    print(f"Number of observations in current data: {n_obs_data}")
    
    # CRITICAL: Check if they match
    if n_obs_model != n_obs_data:
        print(f"WARNING: Mismatch detected! Model expects {n_obs_model} observations but data has {n_obs_data}")
        print("This indicates the filtering criteria used when training the model differs from now.")
        
        # Option 1: Only use the first n_obs_model observations (if data is larger)
        if n_obs_data > n_obs_model:
            print(f"Truncating data to {n_obs_model} observations")
            scores = scores[:n_obs_model]
            data_long = data_long.iloc[:n_obs_model]
        else:
            # Option 2: Raise error - you need to investigate why data is smaller
            raise ValueError(f"Data has fewer observations ({n_obs_data}) than model ({n_obs_model}). "
                           f"Check filtering criteria.")
    
    # Continue with your analysis if shapes match
    age = data_long.Age_year.values
    age_z = (age - age.mean()) / age.std()
    
    S = len(data_long.Score.unique())
    G = len(data_long.Gender.unique())
    I = len(data_long.Income.unique())
    Q = len(data_long.Question.unique())
    
    g_idx = pd.factorize(data_long.Gender)[0]
    i_idx = pd.factorize(data_long.Income)[0]
    q_idx = pd.factorize(data_long.Question)[0]
    
    # Extract parameters
    alpha_1 = az.extract(idata.posterior.alpha_1)["alpha_1"].values
    a = az.extract(idata.posterior.a)["a"].values
    kappa_j = az.extract(idata.posterior.kappa_j)["kappa_j"].values
    kappa_k = az.extract(idata.posterior.kappa_k)["kappa_k"].values
    alpha_2 = az.extract(idata.posterior.alpha_2)["alpha_2"].values
    c = az.extract(idata.posterior.c)["c"].values
    b = az.extract(idata.posterior.b)["b"].values
    delta = az.extract(idata.posterior.delta)["delta"].values
    
    # Get posterior predictive samples and ensure correct shape
    n_chains, n_draws, n_obs = pp_samples.shape
    n_samples_total = n_chains * n_draws
    
    # Reshape to (total_samples, obs)
    pp_samples_flat = pp_samples.reshape(n_samples_total, n_obs)
    
    # Verify shape consistency
    print(f"Posterior predictive shape: {pp_samples_flat.shape}")
    print(f"Scores shape: {scores.shape}")
    
    # Compute metrics for each posterior sample
    mae_samples = np.zeros(n_samples_total)
    concordance_samples = np.zeros(n_samples_total)
    kappa_samples = np.zeros(n_samples_total)
    
    for i in range(n_samples_total):
        pred = pp_samples_flat[i, :]
        mae_samples[i] = mean_absolute_error(scores, pred)
        concordance_samples[i] = np.mean((pred == scores) | (abs(pred - scores) == 1))
        kappa_samples[i] = cohen_kappa_score(scores, pred, weights='quadratic')
    
    # Continue with the rest of your analysis...
    all_betas = np.concatenate([
        idata.posterior['alpha_2'].values.flatten(),
        idata.posterior['c'].values.flatten(), 
        idata.posterior['b'].values.flatten()
    ])
    
    summaries = pd.DataFrame({"":[], "mean":[], "sd":[], "hdi_5":[], "hdi_95":[], "comment":[]}) 
    summaries[""] = ["MAE", "concordance", "kappa", "coefficients"]
    summaries.set_index("", inplace=True, drop=True)
    
    mae_mean = mae_samples.mean().round(2)
    mae_sd = mae_samples.std().round(2)
    mae_hdi = az.hdi(mae_samples, hdi_prob=0.9).round(2)
    mae_comment = f"On average, our predictions are ±{mae_mean} categories away from the true values"
    
    conc_mean = concordance_samples.mean().round(2)
    conc_sd = concordance_samples.std().round(2)
    conc_hdi = az.hdi(concordance_samples, hdi_prob=0.9).round(2)
    conc_comment = f"{conc_mean*100}% of predictions are within one category of the true value"
    
    kap_mean = kappa_samples.mean().round(2)
    kap_sd = kappa_samples.std().round(2)
    kap_hdi = az.hdi(kappa_samples, hdi_prob=0.9).round(2)
    kap_comment = f"Our predictions agree with true values {kap_mean*100}% better than random chance. Note that Kappa penalizes more for larger disagreements (weighted by squared distance)"
    
    coef_mean = all_betas.mean().round(2)
    coef_sd = all_betas.std().round(2)
    coef_hdi = az.hdi(all_betas, hdi_prob=0.9).round(2)
    
    if np.any(np.abs(all_betas) > 2.5):
        coef_comment = "Warning: Some coefficients approaching prior boundaries (at 2.5%)"
    else:
        coef_comment = "95% of all coefficients well within prior ranges"
    
    summaries["mean"] = [mae_mean, conc_mean, kap_mean, coef_mean]
    summaries["sd"] = [mae_sd, conc_sd, kap_sd, coef_sd]
    summaries["hdi_5"] = [mae_hdi[0], conc_hdi[0], kap_hdi[0], coef_hdi[0]]
    summaries["hdi_95"] = [mae_hdi[1], conc_hdi[1], kap_hdi[1], coef_hdi[1]]
    summaries["comment"] = [mae_comment, conc_comment, kap_comment, coef_comment]
    summaries.to_csv("fit_summary_"+wave+".csv")
    
    print("\nMAE posterior distribution:")
    print(f"  Mean: {np.mean(mae_samples):.3f}")
    print(f"  90% HDI: {az.hdi(mae_samples, hdi_prob=0.9)}")
    print(mae_comment)
    
    print("\nConcordance posterior distribution:")
    print(f"  Mean: {np.mean(concordance_samples):.3f}")
    print(f"  90% HDI: {az.hdi(concordance_samples, hdi_prob=0.9)}")
    print(conc_comment)
    
    print("\nKappa posterior distribution:")
    print(f"  Mean: {np.mean(kappa_samples):.3f}")
    print(f"  90% HDI: {az.hdi(kappa_samples, hdi_prob=0.9)}")
    print(kap_comment)
    
    print(f"\nAll coefficients: mean = {np.mean(all_betas):.3f}, std = {np.std(all_betas):.3f}")
    print(f"90% of coefficients between: [{np.percentile(all_betas, 5):.3f}, {np.percentile(all_betas, 95):.3f}]")
    print(coef_comment)