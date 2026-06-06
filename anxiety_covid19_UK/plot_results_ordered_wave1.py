# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['font.size'] = 12

sns.set(style="whitegrid", font="DeJavu Serif")
# sns.set_style("whitegrid")

data = pd.read_csv("./data/anxiety_covid19_UK_wave1_data.csv")
#restrict population to the majority ethnicity only
data = data[data.Ethnicity==3]
#Restrict data to participants without PTSD
data = data[data.Residence==0]
# #Restrict data to urban population
data = data[data.Trauma==0]

data_w6 = pd.read_csv("./data/anxiety_covid19_UK_wave6_data.csv")
data_w6 = data_w6[data_w6.Ethnicity==3]
data_w6 = data_w6[data_w6.Residence==0]
data_w6 = data_w6[data_w6.Trauma==0]

# This guarantess that datasets from both waves have exactly the same people
data_w6 = data_w6[data_w6.pid.isin(data.pid.unique())]
data = data[data.pid.isin(data_w6.pid.unique())]

datas = []
for d in data.columns[20:]:
    df = data.drop(data.columns[20:], axis=1)
    df["Score"] = data[d]
    df["Question"] = np.repeat(d, len(df))
    datas.append(df)

data = pd.concat(datas)

data = data.sort_values(["Income", "Score"])


idata = az.from_netcdf("./idata_wave1_anxiety_ordered.nc")


y_pos = az.extract(idata.posterior.y_hat_probs).y_hat_probs.values

K = len(data.Score.unique())

for s in tqdm(range(K)):
    y_p = y_pos[:,s,:]
    y_m = y_p.mean(axis=1)
    y_s = y_p.std(axis=1)
    y_h5, y_h95 = az.hdi(y_p.T, hdi_prob=0.9).T
    data["m_"+str(s)] = y_m
    data["s_"+str(s)] = y_s
    data["h5_"+str(s)] = y_h5
    data["h95_"+str(s)] = y_h95


data = data.sort_values("Income")

income_legend = ["£0 - £300", "£301 - £490",  "£491 - £740 ", "£741 - £1,111", 
                 "£1,112 or more"]

score_legend = ["Not at all (0)", "Several days (1)", 
                "More than half the days (2)", "Nearly every day (3)"]


income_rep = {"Inc1":"£0 - £300", "Inc2":"£301 - £490",  
              "Inc3":"£491 - £740 ", "Inc4":"£741 - £1,111", 
                 "Inc5":"£1,112 or more"}

data.Income.replace(income_rep, inplace=True)

data["Question"] = data["Question"].str.replace("GAD_","Q")

male = data[data.Gender=="Male"]
female = data[data.Gender=="Female"]

male = male.drop(['pid', 'StartDate', 'Gender', 'Sex', 'Age_year', 'Wave', 'Date',
       'Income_2019',  'Education', 'Ethnicity', 'Religion',
       'Children', 'Residence', 'Loneliness', 'Trauma', 'Employment',
       'Politics'], axis=1)
male = male.groupby(["Income", "Question", "Score"], as_index=False, sort=False).mean()

female = female.drop(['pid', 'StartDate', 'Gender', 'Sex', 'Age_year', 'Wave', 'Date',
       'Income_2019',  'Education', 'Ethnicity', 'Religion',
       'Children', 'Residence', 'Loneliness', 'Trauma', 'Employment',
       'Politics'], axis=1)
female = female.groupby(["Income", "Question", "Score"], as_index=False, sort=False).mean()


means = ["m_0", "m_1", "m_2", "m_3"]

panels = ["A. ", "B. ", "C. ", "D. "]

fig, axs = plt.subplots(2,2, figsize=(12,6))
axs = [ axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
for k in tqdm(range(K)):
    df = male[male.Score==k]
    df = df.pivot_table(index='Income', columns='Question', values=means[k], sort=False)
    df.fillna(0, inplace=True) 
    df = df[sorted(df.columns)]
    sns.heatmap(df, vmin=0, vmax=1, ax=axs[k], cmap="plasma", cbar_kws={'label': 'Probability'})
    axs[k].set_yticklabels(["Inc1", "Inc2", "Inc3", "Inc4", "Inc5"], rotation=0)
    axs[k].set_title(panels[k]+score_legend[k], loc="left")
    axs[k].set_ylabel("Weekly Household Income", size=10)
plt.suptitle("Male", size=16)    
plt.tight_layout()
plt.savefig("male_probs_ordered_wave1.png", dpi=300)
plt.show()


fig, axs = plt.subplots(2,2, figsize=(12,6))
axs = [ axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
for k in tqdm(range(K)):
    df = female[female.Score==k]
    df = df.pivot_table(index='Income', columns='Question', values=means[k], sort=False)
    df = df[sorted(df.columns)]
    df.fillna(0, inplace=True) 
    sns.heatmap(df, vmin=0, vmax=1, ax=axs[k], cmap="plasma", cbar_kws={'label': 'Probability'})
    axs[k].set_yticklabels(["Inc1", "Inc2", "Inc3", "Inc4", "Inc5"], rotation=0)
    axs[k].set_title(panels[k]+score_legend[k], loc="left")
    axs[k].set_ylabel("Weekly Household Income", size=10)
plt.suptitle("Female", size=16)    
plt.tight_layout()
plt.savefig("female_probs_ordered_wave1.png", dpi=300)
plt.show()


male = male.drop("Question", axis=1)
female = female.drop("Question", axis=1)
male = male.groupby(["Income", "Score"], as_index=False, sort=False).mean()
female = female.groupby(["Income", "Score"], as_index=False, sort=False).mean()


dfs = [female, male]
titles = ["A. Female", "B. Male"]
fig, axs = plt.subplots(1,2, figsize=(12,6))
for d in range(2):
    title = titles[d]
    df = dfs[d]
    df = df[["Income","Score","m_0","m_1","m_2","m_3","s_0","s_1","s_2","s_3"]]
    # df = df.groupby(["Income", "Score"], as_index=False, sort=False).mean()
    datas = []
    for m,s in zip(df.columns[2:6], df.columns[6:]):
        da = df.drop(df.columns[2:], axis=1)
        da["Mean"] = df[m]
        da["SD"] = df[s]
        da["Score"] = np.repeat(m.replace("m_",""), len(da))
        datas.append(da.drop_duplicates("Income"))
    df = pd.concat(datas)
    df = df[sorted(df.columns)]
    sns.barplot(df, x="Score", y="Mean", hue="Income", ax=axs[d])
    for i, k in enumerate(df['Score'].unique()):
        subset = df[df['Score'] == k]
        axs[d].errorbar(x=np.array([i-0.32, i-0.16, i, i+0.16, i+0.32]), y=subset["Mean"], 
                     yerr=subset['SD'], fmt='none', ecolor='black', capsize=5)
    axs[d].set_title(title, size=16, loc="left")
    axs[d].set_ylabel("Probability", size=14)
    axs[d].set_xlabel("Answer", size=14)
    axs[d].set_ylim(0,1)
    axs[d].legend(title='Weekly Household Income')
    axs[d].grid(alpha=0.5)
    # axs[d].set_xticks(np.arange(4), score_legend, rotation=45)
plt.suptitle("Average Answer Probability", size=18)    
plt.tight_layout()
plt.savefig("score_probs_ordered_wave1.png", dpi=300)
plt.show()



data = pd.read_csv("./data/anxiety_covid19_UK_wave1_data.csv")
#restrict population to the majority ethnicity only
data = data[data.Ethnicity==3]
#Restrict data to participants without PTSD
data = data[data.Residence==0]
# #Restrict data to urban population
data = data[data.Trauma==0]

data_w6 = pd.read_csv("./data/anxiety_covid19_UK_wave6_data.csv")
data_w6 = data_w6[data_w6.Ethnicity==3]
data_w6 = data_w6[data_w6.Residence==0]
data_w6 = data_w6[data_w6.Trauma==0]

# This guarantess that datasets from both waves have exactly the same people
data_w6 = data_w6[data_w6.pid.isin(data.pid.unique())]
data = data[data.pid.isin(data_w6.pid.unique())]

datas = []
for d in data.columns[20:]:
    df = data.drop(data.columns[20:], axis=1)
    df["Score"] = data[d]
    df["Question"] = np.repeat(d, len(df))
    datas.append(df)

data = pd.concat(datas)

# income_rep = {"Inc1":"£0 - £300", "Inc2":"£301 - £490",  
#               "Inc3":"£491 - £740 ", "Inc4":"£741 - £1,111", 
#                  "Inc5":"£1,112 or more"}

# data.Income.replace(income_rep, inplace=True)

data = data.sort_values(["Income", "Score"])

pred_m = az.extract(idata.posterior_predictive)
pred_m = pred_m.rename_dims({"index":"ID"})

y_probs = az.extract(idata.posterior)["y_hat_probs"].values
m_probs = y_probs.mean(axis=2)

### plot average
qs_ids = []
for i in tqdm(data.Question.unique()):
    for k in data.Score.unique():
        m_p = m_probs[:,k]
        data["yp"] = m_p
        q_df = data[data.Question==i]
        qs_df = q_df[q_df.Score==k]
        qs_da = pred_m["y_hat"].sel(ID=qs_df.index.values)
        qs_ids.append(np.sum(qs_da.T, axis=1)/len(q_df))
        # qs_ids.append(np.sum(qs_da.T + (qs_df["yp"].values**2), axis=1)/len(q_df))

q_ids = np.array(qs_ids)    
q_preds = q_ids.reshape(7,4,y_pos.shape[2])

questions = data.Question.unique()
Q = len(questions)

for i in tqdm(range(Q)):
    num = questions[i]
    data_i = data[data.Question==questions[i]]
    prop = [len(data_i[data_i.Score==s])/len(data_i) for s in range(K)]
    pmeans = q_preds[i].mean(axis=1)
    lo = az.hdi(q_preds[i].T, hdi_prob=0.9).T[0] #pmeans - q_preds[i].std(axis=1) #
    up = az.hdi(q_preds[i].T, hdi_prob=0.9).T[1] #pmeans + q_preds[i].std(axis=1) #
    plt.plot(pmeans, color="purple", linewidth=2, label='Predictions Mean')
    plt.fill_between(np.arange(K), lo, up, color="purple", alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.title(questions[i]+' Posterior Probability', size=11)
    plt.grid(alpha=0.5)
    plt.legend(prop={'size': 10})
    plt.xticks(range(K))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./ordered/"+questions[i]+'_posterior_prob_wave1.png', dpi=300)
    plt.close()



### plot males
dfm = data[data.Gender=="Male"]
qs_ids = []
for i in tqdm(dfm.Question.unique()):
    for k in dfm.Score.unique():
        m_p = m_probs[:,k]
        data["yp"] = m_p
        q_df = dfm[dfm.Question==i]
        qs_df = q_df[q_df.Score==k]
        qs_da = pred_m["y_hat"].sel(ID=qs_df.index.values)
        qs_ids.append(np.sum(qs_da.T, axis=1)/len(q_df))
        # qs_ids.append(np.sum(qs_da.T + (qs_df["yp"].values**2), axis=1)/len(q_df))

q_ids = np.array(qs_ids)    
q_preds = q_ids.reshape(7,4,y_pos.shape[2])

questions = data.Question.unique()
Q = len(questions)

for i in tqdm(range(Q)):
    num = questions[i]
    data_i = dfm[dfm.Question==questions[i]]
    prop = [len(data_i[data_i.Score==s])/len(data_i) for s in range(K)]
    pmeans = q_preds[i].mean(axis=1)
    lo = az.hdi(q_preds[i].T, hdi_prob=0.9).T[0] #pmeans - q_preds[i].std(axis=1) #
    up = az.hdi(q_preds[i].T, hdi_prob=0.9).T[1] #pmeans + q_preds[i].std(axis=1) #
    plt.plot(pmeans, color="orangered", linewidth=2, label='Predictions Mean')
    plt.fill_between(np.arange(K), lo, up, color="orangered", alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.title(questions[i]+' Posterior Probability (Male)', size=11)
    plt.grid(alpha=0.5)
    plt.legend(prop={'size': 10})
    plt.xticks(range(K))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./ordered/"+questions[i]+'_posterior_prob_males_wave1.png', dpi=300)
    plt.close()


### plot females
dff = data[data.Gender=="Female"]
qs_ids = []
for i in tqdm(dff.Question.unique()):
    for k in dff.Score.unique():
        m_p = m_probs[:,k]
        data["yp"] = m_p
        q_df = dff[dff.Question==i]
        qs_df = q_df[q_df.Score==k]
        qs_da = pred_m["y_hat"].sel(ID=qs_df.index.values)
        qs_ids.append(np.sum(qs_da.T, axis=1)/len(q_df))
        # qs_ids.append(np.sum(qs_da.T + (qs_df["yp"].values**2), axis=1)/len(q_df))

q_ids = np.array(qs_ids)    
q_preds = q_ids.reshape(7,4,y_pos.shape[2])

questions = data.Question.unique()
Q = len(questions)

for i in tqdm(range(Q)):
    num = questions[i]
    data_i = dff[dff.Question==questions[i]]
    prop = [len(data_i[data_i.Score==s])/len(data_i) for s in range(K)]
    pmeans = q_preds[i].mean(axis=1)
    lo = az.hdi(q_preds[i].T, hdi_prob=0.9).T[0] #pmeans - q_preds[i].std(axis=1) #
    up = az.hdi(q_preds[i].T, hdi_prob=0.9).T[1] #pmeans + q_preds[i].std(axis=1) #
    plt.plot(pmeans, color="steelblue", linewidth=2, label='Predictions Mean')
    plt.fill_between(np.arange(K), lo, up, color="steelblue", alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.title(questions[i]+' Posterior Probability (Female)', size=11)
    plt.grid(alpha=0.5)
    plt.legend(prop={'size': 10})
    plt.xticks(range(K))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./ordered/"+questions[i]+'_posterior_prob_females_wave1.png', dpi=300)
    plt.close()
    

#### High income
dfh = data[data.Income=="Inc5"]
qs_ids = []
for i in tqdm(dfh.Question.unique()):
    for k in dfh.Score.unique():
        m_p = m_probs[:,k]
        data["yp"] = m_p
        q_df = dfh[dfh.Question==i]
        qs_df = q_df[q_df.Score==k]
        qs_da = pred_m["y_hat"].sel(ID=qs_df.index.values)
        qs_ids.append(np.sum(qs_da.T, axis=1)/len(q_df))
        # qs_ids.append(np.sum(qs_da.T + (qs_df["yp"].values**2), axis=1)/len(q_df))

q_ids = np.array(qs_ids)    
q_preds = q_ids.reshape(7,4,y_pos.shape[2])

questions = data.Question.unique()
Q = len(questions)

for i in tqdm(range(Q)):
    num = questions[i]
    data_i = dfh[dfh.Question==questions[i]]
    prop = [len(data_i[data_i.Score==s])/len(data_i) for s in range(K)]
    pmeans = q_preds[i].mean(axis=1)
    lo = az.hdi(q_preds[i].T, hdi_prob=0.9).T[0] #pmeans - q_preds[i].std(axis=1) #
    up = az.hdi(q_preds[i].T, hdi_prob=0.9).T[1] #pmeans + q_preds[i].std(axis=1) #
    plt.plot(pmeans, color="green", linewidth=2, label='Predictions Mean')
    plt.fill_between(np.arange(K), lo, up, color="green", alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.title(questions[i]+' Posterior Probability (High Income)', size=11)
    plt.grid(alpha=0.5)
    plt.legend(prop={'size': 10})
    plt.xticks(range(K))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./ordered/"+questions[i]+'_posterior_prob_highi_wave1.png', dpi=300)
    plt.close()



#### low income
dfl = data[data.Income=="Inc1"]
qs_ids = []
for i in tqdm(dfl.Question.unique()):
    for k in dfl.Score.unique():
        m_p = m_probs[:,k]
        data["yp"] = m_p
        q_df = dfl[dfl.Question==i]
        qs_df = q_df[q_df.Score==k]
        qs_da = pred_m["y_hat"].sel(ID=qs_df.index.values)
        qs_ids.append(np.sum(qs_da.T, axis=1)/len(q_df))
        # qs_ids.append(np.sum(qs_da.T + (qs_df["yp"].values**2), axis=1)/len(q_df))

q_ids = np.array(qs_ids)    
q_preds = q_ids.reshape(7,4,y_pos.shape[2])

questions = data.Question.unique()
Q = len(questions)

for i in tqdm(range(Q)):
    num = questions[i]
    data_i = dfl[dfl.Question==questions[i]]
    prop = [len(data_i[data_i.Score==s])/len(data_i) for s in range(K)]
    pmeans = q_preds[i].mean(axis=1)
    lo = az.hdi(q_preds[i].T, hdi_prob=0.9).T[0] #pmeans - q_preds[i].std(axis=1) #
    up = az.hdi(q_preds[i].T, hdi_prob=0.9).T[1] #pmeans + q_preds[i].std(axis=1) #
    plt.plot(pmeans, color="peru", linewidth=2, label='Predictions Mean')
    plt.fill_between(np.arange(K), lo, up, color="peru", alpha=0.2, label='90% HDI')
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':', label='Observed Score')
    plt.title(questions[i]+' Posterior Probability (Low Income)', size=11)
    plt.grid(alpha=0.5)
    plt.legend(prop={'size': 10})
    plt.xticks(range(K))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./ordered/"+questions[i]+'_posterior_prob_lowi_wave1.png', dpi=300)
    plt.close()



#### GAD score
qs_ids = []
for i in tqdm(data.Question.unique()):
    q_df = data[data.Question==i]
    for j in data.Income.unique():
        qj_df = q_df[q_df.Income==j]
        for g in data.Gender.unique():
            qg_df = qj_df[qj_df.Gender==g]
            for k in data.Score.unique():
                qs_df = qg_df[qg_df.Score==k]
                qs_da = pred_m["y_hat"].sel(ID=qs_df.index.values)
                # qs_ids.append(qg_da.mean(axis=0))
                try:
                    qs_ids.append(np.max(qs_da.T, axis=1)/len(qs_df))
                except:
                    qs_ids.append(np.repeat(0, qs_da.shape[1]))
                

q_preds = np.array(qs_ids).reshape(7,5,2,4,y_pos.shape[2])

m_ans = np.flip(q_preds.sum(axis=(3,0)).mean(axis=2))
s_ans = np.flip(q_preds.sum(axis=(3,0)).std(axis=2))
g_ans = np.array([["Female", "Male"] for i in  range(len(m_ans))]).flatten()
i_ans = np.repeat(data.Income.unique(), len(m_ans.T)).flatten()

dfi = pd.read_csv("./data/anxiety_covid19_UK_wave1_data.csv")
#restrict population to the majority ethnicity only
dfi = dfi[dfi.Ethnicity==3]
#Restrict data to participants without PTSD
dfi = dfi[dfi.Residence==0]
# #Restrict data to urban population
dfi = dfi[dfi.Trauma==0]

data_w6 = pd.read_csv("./data/anxiety_covid19_UK_wave6_data.csv")
data_w6 = data_w6[data_w6.Ethnicity==3]
data_w6 = data_w6[data_w6.Residence==0]
data_w6 = data_w6[data_w6.Trauma==0]

# This guarantess that datasets from both waves have exactly the same people
data_w6 = data_w6[data_w6.pid.isin(data.pid.unique())]
dfi = dfi[dfi.pid.isin(data_w6.pid.unique())]

dfi = dfi.sort_values(["Income", "Gender"])
income_rep = {"Inc1":"£0 - £300", "Inc2":"£301 - £490",  
              "Inc3":"£491 - £740 ", "Inc4":"£741 - £1,111", 
                 "Inc5":"£1,112+"}
dfi.Income.replace(income_rep, inplace=True)

df_ans = pd.DataFrame({"Mean":m_ans.flatten(), "SD":s_ans.flatten(), 
                       "Gender":g_ans, "Income":i_ans})
df_ans.Income.replace(income_rep, inplace=True)

colors = ['#556B2F', '#8E4585']
fig, axs = plt.subplots(2,1, figsize=(8,8))
sns.violinplot(dfi, x="Income", y="GAD_Total", hue="Gender", palette=colors, ax=axs[0])
axs[0].set_ylabel("Added Score")
axs[0].set_title("A. Observed GAD-7 Total Scores", loc="left", size=16)
sns.barplot(df_ans, x="Income", y="Mean", hue="Gender", palette=colors, ax=axs[1])
for i, j in enumerate(df_ans['Income'].unique()):
    subset = df_ans[df_ans['Income'] == j]
    axs[1].errorbar(x=np.array([i-0.2, i+0.2]), y=subset["Mean"], 
                 yerr=subset['SD'], fmt='none', ecolor='black', capsize=5)
axs[1].set_title("B. Averaged Posterior Predictive Distributions", loc="left", size=16)
axs[1].set_ylabel("Posterior Predictive Mean (Added Score)")
plt.tight_layout()
plt.savefig("violin_plots_wave1.png", dpi=300)
plt.show()
plt.close()




######## Full Summary
data = pd.read_csv("./data/anxiety_covid19_UK_wave1_data.csv")
#restrict population to the majority ethnicity only
data = data[data.Ethnicity==3]
#Restrict data to participants without PTSD
data = data[data.Residence==0]
# #Restrict data to urban population
data = data[data.Trauma==0]

data_w6 = pd.read_csv("./data/anxiety_covid19_UK_wave6_data.csv")
data_w6 = data_w6[data_w6.Ethnicity==3]
data_w6 = data_w6[data_w6.Residence==0]
data_w6 = data_w6[data_w6.Trauma==0]

# This guarantess that datasets from both waves have exactly the same people
data_w6 = data_w6[data_w6.pid.isin(data.pid.unique())]
data = data[data.pid.isin(data_w6.pid.unique())]

datas = []
for d in data.columns[20:]:
    df = data.drop(data.columns[20:], axis=1)
    df["Score"] = data[d]
    df["Question"] = np.repeat(d, len(df))
    datas.append(df)

data = pd.concat(datas)

data = data.sort_values(["Income", "Score"])

data.reset_index(inplace=True)

aves_summ = {"Score":[], "Gender":[], "Income":[], "Age-range":[],
                          "Mean":[], "SD":[], "HDI_5%":[], "HDI_95%":[]}


for k,r in tqdm(enumerate(["18-34", "35-49", "50-64", "65-83"])):
    if k == 0:
        a,b =[18,34] 
    if k == 1:
        a,b =[35,49] 
    if k == 0:
        a,b =[50,64]
    if k == 0:
        a,b =[65,83] 
        
    for i, j in enumerate(data.Income.unique()):
        m_inc_id = data[(data.Gender=="Male") & (data.Income==j) 
                        & (data.Age_year>a) & (data.Age_year<b)].index.values
        
        f_inc_id = data[(data.Gender=="Female") & (data.Income==j) 
                        & (data.Age_year>a) & (data.Age_year<b)].index.values
        
        yp_f = y_probs[f_inc_id].mean(axis=0)
        yp_m = y_probs[m_inc_id].mean(axis=0)
        
        for s in data.Score.unique():
            aves_summ["Score"].append("S"+str(s))
            aves_summ["Gender"].append("Female")
            aves_summ["Income"].append(j)
            aves_summ["Age-range"].append(r)
            aves_summ["Mean"].append(yp_f[s].mean())
            aves_summ["SD"].append(yp_f[s].std())
            aves_summ["HDI_5%"].append(az.hdi(yp_f[s], hdi_prob=0.9)[0])
            aves_summ["HDI_95%"].append(az.hdi(yp_f[s], hdi_prob=0.9)[1])
        for s in data.Score.unique():
            aves_summ["Score"].append("S"+str(s))
            aves_summ["Gender"].append("Male")
            aves_summ["Income"].append(j)
            aves_summ["Age-range"].append(r)
            aves_summ["Mean"].append(yp_m[s].mean())
            aves_summ["SD"].append(yp_m[s].std())
            aves_summ["HDI_5%"].append(az.hdi(yp_m[s], hdi_prob=0.9)[0])
            aves_summ["HDI_95%"].append(az.hdi(yp_m[s], hdi_prob=0.9)[1])
    

for i, j in tqdm(enumerate(data.Income.unique())):
    m_inc_id = data[(data.Gender=="Male") & (data.Income==j)].index.values
    f_inc_id = data[(data.Gender=="Female") & (data.Income==j)].index.values
    
    yp_f = y_probs[f_inc_id].mean(axis=0)
    yp_m = y_probs[m_inc_id].mean(axis=0)
    
    for s in data.Score.unique():
        aves_summ["Score"].append("S"+str(s))
        aves_summ["Gender"].append("Female")
        aves_summ["Income"].append(j)
        aves_summ["Age-range"].append("Average_Age")
        aves_summ["Mean"].append(yp_f[s].mean())
        aves_summ["SD"].append(yp_f[s].std())
        aves_summ["HDI_5%"].append(az.hdi(yp_f[s], hdi_prob=0.9)[0])
        aves_summ["HDI_95%"].append(az.hdi(yp_f[s], hdi_prob=0.9)[1])
    for s in data.Score.unique():
        aves_summ["Score"].append("S"+str(s))
        aves_summ["Gender"].append("Male")
        aves_summ["Income"].append(j)
        aves_summ["Age-range"].append("Average_Age")
        aves_summ["Mean"].append(yp_m[s].mean())
        aves_summ["SD"].append(yp_m[s].std())
        aves_summ["HDI_5%"].append(az.hdi(yp_m[s], hdi_prob=0.9)[0])
        aves_summ["HDI_95%"].append(az.hdi(yp_m[s], hdi_prob=0.9)[1])
    

for i, g in enumerate(data.Gender.unique()):
    g_id = data[data.Gender==g].index.values
    yp_g = y_probs[g_id].mean(axis=0)
    for s in data.Score.unique():
        aves_summ["Score"].append("S"+str(s))
        aves_summ["Gender"].append(g)
        aves_summ["Income"].append("Average_Income")
        aves_summ["Age-range"].append("Average_Age")
        aves_summ["Mean"].append(yp_g[s].mean())
        aves_summ["SD"].append(yp_g[s].std())
        aves_summ["HDI_5%"].append(az.hdi(yp_g[s], hdi_prob=0.9)[0])
        aves_summ["HDI_95%"].append(az.hdi(yp_g[s], hdi_prob=0.9)[1])
    

aves_summ = pd.DataFrame(aves_summ)  
    
aves_summ.to_csv("Wave1_summary_full.csv", index=False)    
    

