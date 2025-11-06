# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

np.random.seed(27) # set numpy random seed

data = pd.read_csv("./data/anxiety_covid19_UK_wave1_data.csv")

datas = []
for d in data.columns[9:]:
    df = data.drop(data.columns[9:], axis=1)
    df["Score"] = data[d]
    df["Question"] = np.repeat(d, len(df))
    datas.append(df)

data = pd.concat(datas)

data = data.sort_values(["Income", "Score"])

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
    
    al_1_z = pm.Normal("al_1_z", 0, 1, dims=("gender")) #intercept mediator
    al_1_l = pm.Normal("al_1_l", 0, 1)
    al_1_s = pm.HalfNormal("al_1_s", 1)
    alpha_1 = pm.Deterministic("alpha_1", al_1_l + al_1_s*al_1_z)
    
    al_2_z = pm.Normal("al_2_z", 0, 1, dims=("gender","question")) #intercept total
    al_2_l = pm.Normal("al_2_l", 0, 1)
    al_2_s = pm.HalfNormal("al_2_s", 1)
    alpha_2 = pm.Deterministic("alpha_2", al_2_l + al_2_s*al_2_z)

    a_z = pm.Normal("a_z", 0, 1, dims=("gender")) #age slope mediator: a
    a_l = pm.Normal("a_l", 0, 1)
    a_s = pm.HalfNormal("a_s", 1)
    a = pm.Deterministic("a", a_l + a_s*a_z)
    
    c_z = pm.Normal("c_z", 0, 1, dims=("gender","question")) #age slope direct: c
    c_l = pm.Normal("c_l", 0, 1)
    c_s = pm.HalfNormal("c_s", 1)
    c = pm.Deterministic("c", c_l + c_s*c_z)
    
    b_z = pm.Normal("b_z", 0, 1, dims=("gender","question")) #income slope
    b_l = pm.Normal("b_l", 0, 1)
    b_s = pm.HalfNormal("b_s", 1)
    b = pm.Deterministic("b", b_l + b_s*b_z)
    
    kappa_j = pm.Normal("kappa_j", 0, 1,
    transform=pm.distributions.transforms.ordered, 
    shape=(income_levels- 1),  initval=np.arange(income_levels - 1)-2.5)
    
    theta = alpha_1[g_idx] + a[g_idx]*age_z 
    
    w_hat = pm.OrderedLogistic("w_hat", cutpoints=kappa_j, eta=theta, observed=w, dims="index") #mediator
    
    a = pt.ones(income_levels-1)
    delta_z = pm.Dirichlet("delta_z", a)
    delta = pm.Deterministic("delta", pm.math.concatenate([[0], delta_z]), dims="income")
    
    kappa_k = pm.Normal('kappa_k', mu=[1,2,3], sigma=0.5, dims=("question","cuts"),
                      transform=pm.distributions.transforms.ordered) #cutpoints/difficulty
    
    eta = alpha_2[g_idx, q_idx]  + c[g_idx, q_idx]*age_z + b[g_idx, q_idx]*delta.cumsum()[i_idx]
    
    y = pm.OrderedLogistic('y_hat', cutpoints=kappa_k[q_idx], eta=eta, observed=scores, dims="index")

with mod:
    ppc = pm.sample_prior_predictive(samples=100)


ax1,ax2 = az.plot_ppc(ppc, group="prior")
ax1.set_xlabel("ŵ", size=14)
ax2.set_xlabel("ŷ", size=14)
ax1.set_title("Weekly Household Income", size=14)
ax2.set_title("PHQ-9 Score", size=14)
plt.suptitle("Wave-1 Prior Predictive Checks Anxiety", size=18)
plt.tight_layout()
plt.savefig("Wave1_prior_predictives_ordered.png", dpi=300)


with mod:
    idata = pm.sample(2000, tune=2000, chains=4, nuts_sampler="numpyro", 
                      target_accep=0.99, random_seed=27)


summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("idata_wave1_summary_ordered.csv")


with mod:
    preds = pm.sample_posterior_predictive(idata, extend_inferencedata=True)


az.to_netcdf(idata, "idata_wave1_anxiety_ordered.nc")


ax1,ax2 = az.plot_ppc(idata, num_pp_samples=1000)
ax1.set_xlabel("ŵ", size=14)
ax2.set_xlabel("ŷ", size=14)
ax1.set_title("Weekly Household Income", size=14)
ax2.set_title("GAD-7 Score", size=14)
plt.suptitle("Wave-1 Posterior Predictive Checks Anxiety", size=18)
plt.tight_layout()
plt.savefig("Wave1_posterior_predictives_ordered.png", dpi=300)