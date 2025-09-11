# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

np.random.seed(27) # set numpy random seed

data = pd.read_csv("./data/depression_covid19_UK_wave6_data.csv")

datas = []
for d in data.columns[9:]:
    df = data.drop(data.columns[9:], axis=1)
    df["Score"] = data[d]
    df["Question"] = np.repeat(d, len(df))
    datas.append(df)

data = pd.concat(datas)

data = data.sort_values(["Income", "Score"])

# data = data[data.Question.isin(["Dep_2", "Dep_4"])]

# data.reset_index(inplace=True, drop=True)

Q = data.Question.unique()

age = data.Age_year.values
age_z = (age - age.mean()) / age.std()
gender = data.Gender.values
income = pd.factorize(data.Income.values)[0]
scores = data.Score.values

g_idx = pd.factorize(data.Gender)[0]
i_idx = pd.factorize(data.Income)[0]
q_idx = pd.factorize(data.Question)[0]

income_levels = len(data.Income.unique())

score_levels = len(data.Score.unique())

coords = {"gender": data.Gender.unique(),
          "income": data.Income.unique(),
          "question": data.Question.unique(),
          "cuts": np.arange(score_levels-1), 
          "index": data.index.values,}

with pm.Model(coords=coords) as mod:
    
    w = pm.Data("w", income, dims="index")
    y = pm.Data("y", scores, dims="index")
    
    al_w_z = pm.Normal("al_w_z", 0, 1, dims=("gender")) 
    al_w_l = pm.Normal("al_w_l", 0, 1)
    al_w_s = pm.HalfNormal("al_w_s", 1)
    alpha_w = pm.Deterministic("alpha_w", al_w_l + al_w_s*al_w_z) #intercept mediator α1s
    
    al_y_z = pm.Normal("al_y_z", 0, 1, dims=("gender","question")) 
    al_y_l = pm.Normal("al_y_l", 0, 1)
    al_y_s = pm.HalfNormal("al_y_s", 1)
    alpha_y = pm.Deterministic("alpha_y", al_y_l + al_y_s*al_y_z) #intercept mediator α2q,s

    bA_w_z = pm.Normal("bA_w_z", 0, 1, dims=("gender")) 
    bA_w_l = pm.Normal("bA_w_l", 0, 1)
    bA_w_s = pm.HalfNormal("bA_w_s", 1)
    bA_w = pm.Deterministic("bA_w", bA_w_l + bA_w_s*bA_w_z) #age slope mediator a
    
    bA_y_z = pm.Normal("bA_y_z", 0, 1, dims=("gender","question")) 
    bA_y_l = pm.Normal("bA_y_l", 0, 1)
    bA_y_s = pm.HalfNormal("bA_y_s", 1)
    bA_y = pm.Deterministic("bA_y", bA_y_l + bA_y_s*bA_y_z) #age slope direct c
    
    bI_z = pm.Normal("bI_z", 0, 1, dims=("gender","question")) 
    bI_l = pm.Normal("bI_l", 0, 1)
    bI_s = pm.HalfNormal("bI_s", 1)
    bI = pm.Deterministic("bI", bI_l + bI_s*bI_z) #income slope direct b
    
    kappa = pm.Normal("kappa", 0, 1,
    transform=pm.distributions.transforms.ordered, 
    shape=(income_levels- 1),  initval=np.arange(income_levels - 1)-2.5)
    
    eta = alpha_w[g_idx] + bA_w[g_idx]*age_z 
    
    w_hat = pm.OrderedLogistic("w_hat", cutpoints=kappa, eta=eta, observed=w, dims="index") #mediator
    
    a = pt.ones(income_levels-1)
    delta_z = pm.Dirichlet("delta_z", a)
    delta = pm.Deterministic("delta", pm.math.concatenate([[0], delta_z]), dims="income")
    
    kappa2 = pm.Normal('kappa2', mu=[1,2,3], sigma=0.5, dims=("question","cuts"),
                      transform=pm.distributions.transforms.ordered) #cutpoints/difficulty
    
    eta2 = alpha_y[g_idx, q_idx]  + bA_y[g_idx, q_idx]*age_z + bI[g_idx, q_idx]*delta.cumsum()[i_idx]
    
    y = pm.OrderedLogistic('y_hat', cutpoints=kappa2[q_idx], eta=eta2, observed=scores, dims="index")

with mod:
    ppc = pm.sample_prior_predictive(samples=100)


ax1,ax2 = az.plot_ppc(ppc, group="prior")
ax1.set_xlabel("ŵ", size=14)
ax2.set_xlabel("ŷ", size=14)
ax1.set_title("Weekly Household Income", size=14)
ax2.set_title("PHQ-9 Score", size=14)
plt.suptitle("Wave-1 Prior Predictive Checks Depression", size=18)
plt.tight_layout()
plt.savefig("Wave6_prior_predictives_ordered.png", dpi=300)
 
with mod:
    idata = pm.sample(2000, tune=2000, chains=4, nuts_sampler="numpyro", target_accep=0.95)


summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("idata_wave6_summary_ordered.csv")


with mod:
    preds = pm.sample_posterior_predictive(idata, extend_inferencedata=True)


az.to_netcdf(idata, "idata_wave6_depression_ordered.nc")


ax1,ax2 = az.plot_ppc(idata, num_pp_samples=1000)
ax1.set_xlabel("ŵ", size=14)
ax2.set_xlabel("ŷ", size=14)
ax1.set_title("Weekly Household Income", size=14)
ax2.set_title("PHQ-9 Score", size=14)
plt.suptitle("Wave-6 Posterior Predictive Checks Depression", size=18)
plt.tight_layout()
plt.savefig("Wave6_posterior_predictives_ordered.png", dpi=300)
