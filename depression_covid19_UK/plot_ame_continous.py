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

def logistic_pdf(x):
  return np.exp(x) / (1 + np.exp(x))**2
    
def pordlog(a):
    pa = logistic_pdf(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]

def age_on_phq(gender_idx, income_idx, beta_var, var_values):
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
    kappa_q = kappa_k   # shape: (3, 1000)
    beta = beta_var[gender_idx, :]  # shape: (1000,)
    alpha = alpha_2[gender_idx, :]  # shape: (1000,)
    b_val = b[gender_idx, :]  # shape: (1000,)
    
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



sns.set(style="whitegrid", font="DeJavu Serif")
# sns.set_style("whitegrid")

data = pd.read_csv("./data/depression_covid19_UK_wave1_data.csv")
idata = az.from_netcdf("./idata_wave1_depression_ordered.nc")

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
    
# Extract parameters for the income model
alpha_1 = az.extract(idata.posterior.alpha_1, num_samples=1000)["alpha_1"].values  # shape: (1000, 2) for 2 genders
a = az.extract(idata.posterior.a, num_samples=1000)["a"].values        # a shape: (1000, 2) for 2 genders
kappa_j = az.extract(idata.posterior.kappa_j, num_samples=1000)["kappa_j"].values  # shape: (4, 1000) for 4 cutpoints (5 income levels)
kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values.mean(axis=0) #(9, 3, 1000)
alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values.mean(axis=1) #(2, 9, 1000)
c = az.extract(idata.posterior.c, num_samples=1000)["c"].values.mean(axis=1) # c (2, 9, 1000)
b = az.extract(idata.posterior.b, num_samples=1000)["b"].values.mean(axis=1) # b (2, 9, 1000)
delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values #(5, 1000)


c_wave1 = az.extract(idata.posterior.c, num_samples=1000)["c"].values.mean(axis=1)

aop_ames_wave1 = np.zeros((2,5,4,1000))
for i in tqdm(range(I)):
    for g in range(G):
        aop_ames_wave1[g,i,:,:] = age_on_phq(g, i, c, age_z) / age.std() #recover age scale



data = pd.read_csv("./data/depression_covid19_UK_wave6_data.csv")
idata = az.from_netcdf("./idata_wave6_depression_ordered.nc")

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
    
# Extract parameters for the income model
alpha_1 = az.extract(idata.posterior.alpha_1, num_samples=1000)["alpha_1"].values  # shape: (1000, 2) for 2 genders
a = az.extract(idata.posterior.a, num_samples=1000)["a"].values        # a shape: (1000, 2) for 2 genders
kappa_j = az.extract(idata.posterior.kappa_j, num_samples=1000)["kappa_j"].values  # shape: (4, 1000) for 4 cutpoints (5 income levels)
kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values.mean(axis=0) #(9, 3, 1000)
alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values.mean(axis=1) #(2, 9, 1000)
c = az.extract(idata.posterior.c, num_samples=1000)["c"].values.mean(axis=1) # c (2, 9, 1000)
b = az.extract(idata.posterior.b, num_samples=1000)["b"].values.mean(axis=1) # b (2, 9, 1000)
delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values #(5, 1000)


c_wave6 = az.extract(idata.posterior.c, num_samples=1000)["c"].values.mean(axis=1)


aop_ames_wave6 = np.zeros((2,5,4,1000))
for i in tqdm(range(I)):
    for g in range(G):
        aop_ames_wave6[g,i,:,:] = age_on_phq(g, i, c, age_z) / age.std() #recover age scale


aop_ames_wave1 = aop_ames_wave1[:,:,0,:] #extract score=0 only
aop_ames_wave6 = aop_ames_wave6[:,:,0,:] #extract score=0 only

ages = np.sort(age)

aop_ames_wave1 = np.array([aop_ames_wave1*a for a in ages])
aop_ames_wave6 = np.array([aop_ames_wave6*a for a in ages])

### Plot
incomes = ["£0-£300", "£301-£490", "£491-£740", "£741-£1,111", "£1,112+"]


lines = ["-", "--", ":", "-.", (0, (1, 1))]

colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
for i in range(I):
    m = aop_ames_wave1[:,0,i,:].mean(axis=1) 
    s = aop_ames_wave1[:,0,i,].std(axis=1)
    ax1.plot(ages, m, color=colors[i], ls=lines[i], label=incomes[i])
    ax1.fill_between(ages, m-s, m+s, color=colors[i], alpha=0.2)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("Wave-1 Female (Score=0)")
    ax1.set_ylim(0, 1)

ax2 = axes[0, 1]
for i in range(I):
    m = aop_ames_wave1[:,1,i,:].mean(axis=1) 
    s = aop_ames_wave1[:,1,i,].std(axis=1)
    ax2.plot(ages, m, color=colors[i], ls=lines[i], label=incomes[i])
    ax2.fill_between(ages, m-s, m+s, color=colors[i], alpha=0.2)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title("Wave-1 Male (Score=0)")
    ax2.set_ylim(0, 1)
    
ax3 = axes[1, 0]
for i in range(I):
    m = aop_ames_wave6[:,0,i,:].mean(axis=1) 
    s = aop_ames_wave6[:,0,i,].std(axis=1)
    ax3.plot(ages, m, color=colors[i], ls=lines[i], label=incomes[i])
    ax3.fill_between(ages, m-s, m+s, color=colors[i], alpha=0.2)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_title("Wave-6 Female (Score=0)")
    ax3.set_ylim(0, 1)

ax4 = axes[1, 1]
for i in range(I):
    m = aop_ames_wave6[:,1,i,:].mean(axis=1) 
    s = aop_ames_wave6[:,1,i,].std(axis=1)
    ax4.plot(ages, m, color=colors[i], ls=lines[i], label=incomes[i])
    ax4.fill_between(ages, m-s, m+s, color=colors[i], alpha=0.2)
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_title("Wave-6 Male (Score=0)")
    ax4.set_ylim(0, 1)

plt.tight_layout()
plt.title("PHQ-9 Score=0 Probabilities")
plt.savefig("depression_effects_continous.png", dpi=300)
plt.show()
plt.close()


######################### Income Plotting ##################################
iop_ades_wave1 = np.zeros((4, 2, 4, 63, 1000))
iop_ades_wave2 = np.zeros((4, 2, 4, 63, 1000))
iops = [iop_ades_wave1, iop_ades_wave2]
waves = ["wave1", "wave6"]
ages = []
    
for w in range(2):
    wave = waves[w]
    data = pd.read_csv("./data/depression_covid19_UK_"+wave+"_data.csv")
    
    idata = az.from_netcdf("./idata_"+wave+"_depression_ordered.nc")

    # Extract parameters for the income model
    alpha_1 = az.extract(idata.posterior.alpha_1, num_samples=1000)["alpha_1"].values  # shape: (1000, 2) for 2 genders
    a = az.extract(idata.posterior.a, num_samples=1000)["a"].values        # a shape: (1000, 2) for 2 genders
    kappa_j = az.extract(idata.posterior.kappa_j, num_samples=1000)["kappa_j"].values  # shape: (4, 1000) for 4 cutpoints (5 income levels)
    kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values.mean(axis=0) #(9, 3, 1000)
    alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values.mean(axis=1) #(2, 9, 1000)
    c = az.extract(idata.posterior.c, num_samples=1000)["c"].values.mean(axis=1) # c (2, 9, 1000)
    b = az.extract(idata.posterior.b, num_samples=1000)["b"].values.mean(axis=1) # b (2, 9, 1000)
    delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values #(5, 1000)

    
    def income_on_phq(gender_idx, income_j, income_k, age_values):
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
        kappa_q = kappa_k  # shape: (3, 1000)
        alpha = alpha_2[gender_idx, :]  # shape: (1000,)
        bA = c[gender_idx, :]  # shape: (1000,)
        b_val = b[gender_idx, :]  # shape: (1000,)
        
        # Get income positions
        delta_cumsum = np.cumsum(delta, axis=0)  # shape: (5, 1000)
        income_j_pos = delta_cumsum[income_j, :]  # shape: (1000,)
        income_k_pos = delta_cumsum[income_k, :]  # shape: (1000,)
        
        ade_per_obs = np.zeros((4, age_values.shape[0], n_samples))
        
        # for i, age_val in enumerate(age_values):
        # Compute linear predictors for both income levels
        eta_j = alpha[:,None] + bA[:,None] * age_values + b_val[:,None] * income_j_pos[:,None]  # shape: (1000,)
        eta_k = alpha[:,None] + bA[:,None] * age_values + b_val[:,None] * income_k_pos[:,None]  # shape: (1000,)
        
        # Compute probablity differences for all categories
        for s in range(4):
            if s == 0:
                p_j = expit(kappa_q[0, :][:,None] - eta_j)
                p_k = expit(kappa_q[0, :][:,None] - eta_k)
            elif s == 3:
                p_j = 1 - expit(kappa_q[2, :][:,None] - eta_j)
                p_k = 1 - expit(kappa_q[2, :][:,None] - eta_k)
            else:
                p_j = expit(kappa_q[s, :][:,None] - eta_j) - expit(kappa_q[s-1, :][:,None] - eta_j)
                p_k = expit(kappa_q[s, :][:,None] - eta_k) - expit(kappa_q[s-1, :][:,None] - eta_k)
            
            ade_per_obs[s,:,:] = p_k.T - p_j.T
        
        # Average across observations
        return ade_per_obs  # shape: (4,1000)
        
    
    age_values = np.arange(age.min(), age.max())[:63]
    ages.append(age_values)
    age_values = np.unique(age_z)
    #age_values = (age_values - age_values.mean()) / age_values.std()
    
    for k_idx, k in enumerate(tqdm(range(1, 5))):  # k_idx: 0,1,2,3 for Q2,Q3,Q4,Q5
        for g in range(G):
            # Compute difference between lowest income (0) and current income level (k)
            ade_result = income_on_phq(g, 0, k, age_values)  # shape: (4, 1000)
            iops[w][k_idx, g, :, :, :] = ade_result
    iops[w] = iops[w][3,:,:,:,:] #keep only lowest minus highest difference       
    # iop_ades1 = iop_ades[3,:,:,:,:] #keep only lowest minus highest difference
    # iop_ades2 = iop_ades[:,:,0,:,:] #keep all differences but at score 0 only    
    

iop_ames_wave1 = iops[0]
iop_ames_wave6 = iops[1]


### Plot
incomes = ["£0-£300", "£301-£490", "£491-£740", "£741-£1,111", "£1,112+"]

scores = ["Score = 0", "Score = 1", "Score = 2", "Score = 3"]

lines = [":", "-.", "--", "-"]

colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
for k in range(S):
    m = iop_ames_wave1[0,k,:,:].mean(axis=1) 
    s = iop_ames_wave1[0,k,:,:].std(axis=1)
    sl = np.percentile(iop_ames_wave1[0,k,:,:], 5)
    su = np.percentile(iop_ames_wave1[0,k,:,:], 95)
    ax1.plot(ages[0], m, color=colors[k], ls=lines[k], lw=2.5, label=scores[k])
    ax1.fill_between(ages[0], m-s, m+s, color=colors[k], alpha=0.2)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("Wave-1 Female (£0-£300 → £1,112+)")
    ax1.set_ylabel("APE (Inc5 - Inc1)")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylim(-0.12, 0.3)

ax2 = axes[0, 1]
for k in range(S):
    m = iop_ames_wave1[1,k,:,:].mean(axis=1) 
    s = iop_ames_wave1[1,k,:,:].std(axis=1)
    ax2.plot(ages[0], m, color=colors[k], ls=lines[k], lw=2.5, label=scores[k])
    ax2.fill_between(ages[0], m-s, m+s, color=colors[k], alpha=0.2)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title("Wave-1 Male (£0-£300 → £1,112+)")
    ax2.set_ylabel("APE (Inc5 - Inc1)")
    ax2.set_xlabel("Age (years)")
    ax2.set_ylim(-0.12, 0.3)
    
ax3 = axes[1, 0]
for k in range(S):
    m = iop_ames_wave6[0,k,:,:].mean(axis=1) 
    s = iop_ames_wave6[0,k,:,:].std(axis=1)
    ax3.plot(ages[1], m, color=colors[k], ls=lines[k], lw=2.5, label=scores[k])
    ax3.fill_between(ages[1], m-s, m+s, color=colors[k], alpha=0.2)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_title("Wave-6 Female (£0-£300 → £1,112+)")
    ax3.set_ylabel("APE (Inc5 - Inc1)")
    ax3.set_xlabel("Age (years)")
    ax3.set_ylim(-0.12, 0.3)

ax4 = axes[1, 1]
for k in range(S):
    m = iop_ames_wave6[1,k,:,:].mean(axis=1) 
    s = iop_ames_wave6[1,k,:,:].std(axis=1)
    ax4.plot(ages[1], m, color=colors[k], ls=lines[k], lw=2.5, label=scores[k])
    ax4.fill_between(ages[1], m-s, m+s, color=colors[k], alpha=0.2)
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_title("Wave-6 Male (£0-£300 → £1,112+)")
    ax4.set_ylabel("APE (Inc5 - Inc1)")
    ax4.set_xlabel("Age (years)")
    ax4.set_ylim(-0.12, 0.3)

plt.tight_layout()
plt.savefig("depression_ades_continous_score.png", dpi=300)
plt.show()
plt.close()



iop_ades_wave1 = np.zeros((4, 2, 4, 63, 1000))
iop_ades_wave2 = np.zeros((4, 2, 4, 63, 1000))
iops = [iop_ades_wave1, iop_ades_wave2]
waves = ["wave1", "wave6"]
ages = []
    
for w in range(2):
    wave = waves[w]
    data = pd.read_csv("./data/depression_covid19_UK_"+wave+"_data.csv")
    
    idata = az.from_netcdf("./idata_"+wave+"_depression_ordered.nc")

    # Extract parameters for the income model
    alpha_1 = az.extract(idata.posterior.alpha_1, num_samples=1000)["alpha_1"].values  # shape: (1000, 2) for 2 genders
    a = az.extract(idata.posterior.a, num_samples=1000)["a"].values        # a shape: (1000, 2) for 2 genders
    kappa_j = az.extract(idata.posterior.kappa_j, num_samples=1000)["kappa_j"].values  # shape: (4, 1000) for 4 cutpoints (5 income levels)
    kappa_k = az.extract(idata.posterior.kappa_k, num_samples=1000)["kappa_k"].values.mean(axis=0) #(9, 3, 1000)
    alpha_2 = az.extract(idata.posterior.alpha_2, num_samples=1000)["alpha_2"].values.mean(axis=1) #(2, 9, 1000)
    c = az.extract(idata.posterior.c, num_samples=1000)["c"].values.mean(axis=1) # c (2, 9, 1000)
    b = az.extract(idata.posterior.b, num_samples=1000)["b"].values.mean(axis=1) # b (2, 9, 1000)
    delta = az.extract(idata.posterior.delta, num_samples=1000)["delta"].values #(5, 1000)

    
    def income_on_phq(gender_idx, income_j, income_k, age_values):
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
        kappa_q = kappa_k  # shape: (3, 1000)
        alpha = alpha_2[gender_idx, :]  # shape: (1000,)
        bA = c[gender_idx, :]  # shape: (1000,)
        b_val = b[gender_idx, :]  # shape: (1000,)
        
        # Get income positions
        delta_cumsum = np.cumsum(delta, axis=0)  # shape: (5, 1000)
        income_j_pos = delta_cumsum[income_j, :]  # shape: (1000,)
        income_k_pos = delta_cumsum[income_k, :]  # shape: (1000,)
        
        ade_per_obs = np.zeros((4, age_values.shape[0], n_samples))
        
        # for i, age_val in enumerate(age_values):
        # Compute linear predictors for both income levels
        eta_j = alpha[:,None] + bA[:,None] * age_values + b_val[:,None] * income_j_pos[:,None]  # shape: (1000,)
        eta_k = alpha[:,None] + bA[:,None] * age_values + b_val[:,None] * income_k_pos[:,None]  # shape: (1000,)
        
        # Compute probablity differences for all categories
        for s in range(4):
            if s == 0:
                p_j = expit(kappa_q[0, :][:,None] - eta_j)
                p_k = expit(kappa_q[0, :][:,None] - eta_k)
            elif s == 3:
                p_j = 1 - expit(kappa_q[2, :][:,None] - eta_j)
                p_k = 1 - expit(kappa_q[2, :][:,None] - eta_k)
            else:
                p_j = expit(kappa_q[s, :][:,None] - eta_j) - expit(kappa_q[s-1, :][:,None] - eta_j)
                p_k = expit(kappa_q[s, :][:,None] - eta_k) - expit(kappa_q[s-1, :][:,None] - eta_k)
            
            ade_per_obs[s,:,:] = p_k.T - p_j.T
        
        # Average across observations
        return ade_per_obs  # shape: (4,1000)
        
    age_values = np.arange(age.min(), age.max())[:63]
    ages.append(age_values)
    age_values = np.unique(age_z)
    
    for k_idx, k in enumerate(tqdm(range(1, 5))):  # k_idx: 0,1,2,3 for Q2,Q3,Q4,Q5
        for g in range(G):
            # Compute difference between lowest income (0) and current income level (k)
            ade_result = income_on_phq(g, 0, k, age_values)  # shape: (4, 1000)
            iops[w][k_idx, g, :, :, :] = ade_result
    iops[w] = iops[w][:,:,0,:,:].swapaxes(1,0) #keep all differences but at score 0 only      
    # iop_ades1 = iop_ades[3,:,:,:,:] #keep only lowest minus highest difference
    # iop_ades2 = iop_ades[:,:,0,:,:] #keep all differences but at score 0 only    
        

iop_ames_wave1 = iops[0]
iop_ames_wave6 = iops[1]


### Plot
incomes = ["£0-£300 → £301-£490", "£0-£300 → £491-£740", 
           "£0-£300 → £741-£1,111", "£0-£300 → £1,112+"]

scores = ["Score = 0", "Score = 1", "Score = 2", "Score = 3"]

lines = [":", "-.", "--", "-"]

colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
for k in range(S):
    m = iop_ames_wave1[0,k,:,:].mean(axis=1) 
    s = iop_ames_wave1[0,k,:,:].std(axis=1)
    sl = np.percentile(iop_ames_wave1[0,k,:,:], 5, axis=1)
    su = np.percentile(iop_ames_wave1[0,k,:,:], 95, axis=1)
    ax1.plot(ages[0], m, color=colors[k], ls=lines[k], lw=2.5, label=incomes[k])
    ax1.fill_between(ages[0], m-s, m+s, color=colors[k], alpha=0.2)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title('Wave-1 Female: "Not at all(0)"')
    ax1.set_ylabel("APE (Score = 0)")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylim(0, 0.3)

ax2 = axes[0, 1]
for k in range(S):
    m = iop_ames_wave1[1,k,:,:].mean(axis=1) 
    s = iop_ames_wave1[1,k,:,:].std(axis=1)
    sl = np.percentile(iop_ames_wave1[1,k,:,:], 5, axis=1)
    su = np.percentile(iop_ames_wave1[1,k,:,:], 95, axis=1)
    ax2.plot(ages[0], m, color=colors[k], ls=lines[k], lw=2.5, label=incomes[k])
    ax2.fill_between(ages[0], m-s, m+s, color=colors[k], alpha=0.2)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title('Wave-1 Male: "Not at all(0)"')
    ax2.set_ylabel("APE (Score = 0)")
    ax2.set_xlabel("Age (years)")
    ax2.set_ylim(0, 0.3)
    
ax3 = axes[1, 0]
for k in range(S):
    m = iop_ames_wave6[0,k,:,:].mean(axis=1) 
    s = iop_ames_wave6[0,k,:,:].std(axis=1)
    sl = np.percentile(iop_ames_wave6[0,k,:,:], 5, axis=1)
    su = np.percentile(iop_ames_wave6[0,k,:,:], 95, axis=1)
    ax3.plot(ages[1], m, color=colors[k], ls=lines[k], lw=2.5, label=incomes[k])
    ax3.fill_between(ages[1], m-s, m+s, color=colors[k], alpha=0.2)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_title('Wave-6 Female: "Not at all(0)"')
    ax3.set_ylabel("APE (Score = 0)")
    ax3.set_xlabel("Age (years)")
    ax3.set_ylim(0, 0.3)

ax4 = axes[1, 1]
for k in range(S):
    m = iop_ames_wave6[1,k,:,:].mean(axis=1) 
    s = iop_ames_wave6[1,k,:,:].std(axis=1)
    sl = np.percentile(iop_ames_wave6[1,k,:,:], 5, axis=1)
    su = np.percentile(iop_ames_wave6[1,k,:,:], 95, axis=1)
    ax4.plot(ages[1], m, color=colors[k], ls=lines[k], lw=2.5, label=incomes[k])
    ax4.fill_between(ages[1], m-s, m+s, color=colors[k], alpha=0.2)
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_title('Wave-6 Male: "Not at all(0)"')
    ax4.set_ylabel("APE (Score = 0)")
    ax4.set_xlabel("Age (years)")
    ax4.set_ylim(0, 0.3)

plt.tight_layout()
plt.savefig("depression_ades_continous_income.png", dpi=300)
plt.show()
plt.close()