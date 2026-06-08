# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

np.random.seed(27)
data = pd.read_csv("./data/Depression_covid19_UK_wave6_data.csv")

#restrict population to the majority ethnicity only
data = data[data.Ethnicity==3]

data_w1 = pd.read_csv("./data/Depression_covid19_UK_wave1_data.csv")
data_w1 = data_w1[data_w1.Ethnicity==3]

data = data[data.pid.isin(data_w1.pid.unique())]
        
# Reshape data
datas = []
for d in data.columns[18:]:
    df = data.drop(data.columns[18:], axis=1)
    df["Score"] = data[d]
    df["Question"] = np.repeat(d, len(df))
    datas.append(df)
data = pd.concat(datas)

data = data.sort_values("Score")
data.reset_index(inplace=True, drop=True)


g_idx = data.Sex.values
i_idx = data["Income"].replace({'Inc1':0, 'Inc2':1, 'Inc3':2, 'Inc4':3, 'Inc5':4})
q_idx = data["Question"].replace({'Dep_1':0, 'Dep_2':1, 'Dep_3':2, 
                                  'Dep_4':3, 'Dep_5':4, 'Dep_6':5, 'Dep_7':6, 
                                  'Dep_8':7, 'Dep_9':8})

confounder_cols_1 = ["Education", "Employment", "Children"]

confounder_cols_2 = ["Education", "Employment", "Children", 
                     "Religion", "Loneliness", "Politics", "Housing"]

# Create the matrices X_1 and X_2
X_1 = data[confounder_cols_1].values.astype(float)
X_2 = data[confounder_cols_2].values.astype(float)

coords = {
    "gender": ["Female", "Male"],
    "income": data.Income.unique(),
    "question": data.Question.unique(),
    "cuts": np.arange(4 - 1),
    "index": data.index.values,
}

with pm.Model(coords=coords) as mod:
    w = pm.Data("w", i_idx, dims="index")
    y = pm.Data("y", data.Score.values, dims="index")
    age_z = pm.Data("age_z", (data.Age_year.values - data.Age_year.mean()) / data.Age_year.std(), dims="index")
    
    ### Main parameters
    alpha_1 = pm.Normal("alpha_1", 0, 1, dims="gender")
    alpha_2 = pm.Normal("alpha_2", 0, 1, dims=("gender", "question"))
    a = pm.Normal("a", 0, 1, dims="gender")
    c = pm.Normal("c", 0, 1, dims=("gender", "question"))
    b = pm.Normal("b", 0, 1, dims=("gender", "question"))
    
    # Outcome coefficients
    ### Confounder Vectors (One coefficient per confounder)
    U_1 = pm.Normal("U_1", mu=0, sigma=0.25, shape=len(confounder_cols_1)) #mediator confs
    U_2 = pm.Normal("U_2", mu=0, sigma=0.25, shape=len(confounder_cols_2)) #outcome confs
    
    ### Mediator Model (Income)
    kappa_j = pm.Normal("kappa_j", mu=[0.25, 0.5, 1,2], sigma=0.5,
        transform=pm.distributions.transforms.ordered,
        shape=(len(data.Income.unique()) - 1), initval=np.arange(len(data.Income.unique()) - 1) - 2.5)
    
    theta = (alpha_1[g_idx] + a[g_idx] * age_z + pt.dot(X_1, U_1))
             
    w_hat = pm.OrderedLogistic("w_hat", cutpoints=kappa_j, eta=theta, observed=w, dims="index")
    
    ### Outcome Model (Mental Health)
    delta_z = pm.Dirichlet("delta_z", pt.ones(len(data.Income.unique()) - 1))
    delta = pm.Deterministic("delta", pm.math.concatenate([[0], delta_z]), dims="income")
    
    kappa_k = pm.Normal('kappa_k', mu=[1, 2, 3], sigma=0.5, dims=("question", "cuts"),
                      transform=pm.distributions.transforms.ordered)
    
    eta = (alpha_2[g_idx, q_idx] + c[g_idx, q_idx] * age_z + 
           b[g_idx, q_idx] * delta.cumsum()[i_idx] + pt.dot(X_2, U_2))
           
    y_hat = pm.OrderedLogistic('y_hat', cutpoints=kappa_k[q_idx], eta=eta, observed=y, dims="index")

with mod:
    ppc = pm.sample_prior_predictive(samples=100)


ax1,ax2 = az.plot_ppc(ppc, group="prior")
ax1.set_xlabel("ŵ", size=14)
ax2.set_xlabel("ŷ", size=14)
ax1.set_title("Weekly Household Income", size=14)
ax2.set_title("PHQ-9 Score", size=14)
plt.suptitle("Wave-6 Prior Predictive Checks Depression", size=18)
plt.tight_layout()
plt.savefig("Wave6_prior_predictives_ordered.png", dpi=300)


with mod:
    idata = pm.sample(1000, tune=1000, chains=4, nuts_sampler="nutpie", 
                      target_accep=0.95, random_seed=27)


summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("idata_wave6_summary_ordered.csv")


with mod:
    preds = pm.sample_posterior_predictive(idata, extend_inferencedata=True)


az.to_netcdf(idata, "idata_wave6_Depression_ordered.nc")


ax1,ax2 = az.plot_ppc(idata, num_pp_samples=1000)
ax1.set_xlabel("ŵ", size=14)
ax2.set_xlabel("ŷ", size=14)
ax1.set_title("Weekly Household Income", size=14)
ax2.set_title("PHQ-9 Score", size=14)
plt.suptitle("Wave-6 Posterior Predictive Checks Depression", size=18)
plt.tight_layout()
plt.savefig("Wave6_posterior_predictives_ordered.png", dpi=300)