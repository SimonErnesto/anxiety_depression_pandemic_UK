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
# sns.set_style("whitegrid")

waves = ["wave1", "wave6"]


for wa in range(len(waves)):
    
    data = pd.read_csv("./data/anxiety_covid19_UK_"+waves[wa]+"_data.csv")
    
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
    
    idata = az.from_netcdf("./idata_"+waves[wa]+"_anxiety_ordered.nc")
    
    
    # Extract parameters for the income model
    # Extract parameters for the income model
    alpha_1 = az.extract(idata.posterior.alpha_1)["alpha_1"].values  # shape: (1000, 2) for 2 genders
    a = az.extract(idata.posterior.a)["a"].values        # a shape: (1000, 2) for 2 genders
    kappa_j = az.extract(idata.posterior.kappa_j)["kappa_j"].values  # shape: (1000, 4) for 4 cutpoints (5 income levels)
    kappa_k = az.extract(idata.posterior.kappa_k)["kappa_k"].values #(9, 3, 1000)
    alpha_2 = az.extract(idata.posterior.alpha_2)["alpha_2"].values #(2, 9, 1000)
    c = az.extract(idata.posterior.c)["c"].values # c (2, 9, 1000)
    b = az.extract(idata.posterior.b)["b"].values # b (2, 9, 1000)
    delta = az.extract(idata.posterior.delta)["delta"].values #(5, 1000)
    
    # Get posterior predictive samples
    pp_samples = idata.posterior_predictive['y_hat'].values  # shape: (chain, draw, obs)
    n_chains, n_draws, n_obs = pp_samples.shape
    n_samples_total = n_chains * n_draws
    
    # Reshape to (total_samples, obs)
    pp_samples_flat = pp_samples.reshape(n_samples_total, n_obs)
    
    # Compute metrics for each posterior sample
    mae_samples = np.zeros(n_samples_total)
    concordance_samples = np.zeros(n_samples_total)
    kappa_samples = np.zeros(n_samples_total)
    
    for i in range(n_samples_total):
        pred = pp_samples_flat[i, :]
        mae_samples[i] = mean_absolute_error(scores, pred)
        concordance_samples[i] = np.mean((pred == scores) | (abs(pred - scores) == 1))
        kappa_samples[i] = cohen_kappa_score(scores, pred, weights='quadratic')
    
    # Check the distribution of all coefficients
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
    
    # Check if any coefficients are hitting prior boundaries
    if np.any(np.abs(all_betas) > 2.5):
        coef_comment = "Warning: Some coefficients approaching prior boundaries (at 2.5%)"
    else:
        coef_comment = "95% of all coefficients well within prior ranges"
    
    
    summaries["mean"] = [mae_mean, conc_mean, kap_mean, coef_mean]
    summaries["sd"] = [mae_sd, conc_sd, kap_sd, coef_sd]
    summaries["hdi_5"] = [mae_hdi[0], conc_hdi[0], kap_hdi[0], coef_hdi[0]]
    summaries["hdi_95"] = [mae_hdi[1], conc_hdi[1], kap_hdi[1], coef_hdi[1]]
    summaries["comment"] = [mae_comment, conc_comment, kap_comment, coef_comment]
    summaries.to_csv("fit_summary"+wave+".csv")
    
    # Now you have full posterior distributions of metrics!
    print("MAE posterior distribution:")
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
    
    print(f"All coefficients: mean = {np.mean(all_betas):.3f}, std = {np.std(all_betas):.3f}")
    print(f"90% of coefficients between: [{np.percentile(all_betas, 5):.3f}, {np.percentile(all_betas, 95):.3f}]")
    print(coef_comment)