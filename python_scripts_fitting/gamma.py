# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:34:24 2020

@author: iwona
"""


import pandas as pd
import pystan
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set()

DATA_PATH = '../data/'
OUT_PATH = '../fitting_outputs/'
SEED = 4321
ITER = 2000
CHAINS = 4
MAX_VAL = 133 

##############################################################################
# this code fits the
# GAMMA model
# to each of the covariates
t0 = time.time()
##############################################################################
# LOAD AND PREPARE THE DATA
drop_columns = ['Unnamed: 0', 'Start_date', 'End_date']

d_ICUstay = pd.read_csv(DATA_PATH + 'ICU-stay.csv')
d_ICUstay = d_ICUstay[d_ICUstay['ICU-stay'] <= MAX_VAL]
d_onsetDeath = pd.read_csv(DATA_PATH + 'onset-to-death.csv')
d_onsetDeath = d_onsetDeath[d_onsetDeath['onset-to-death'] <= MAX_VAL]
d_onsetDiagnosis= pd.read_csv(DATA_PATH + 'onset-to-diagnosis.csv')
d_onsetDiagnosis = d_onsetDiagnosis[d_onsetDiagnosis['onset-to-diagnosis'] <= MAX_VAL]
d_onsetAdmiss = pd.read_csv(DATA_PATH + 'onset-to-hospital-admission.csv')
d_onsetAdmiss = d_onsetAdmiss[d_onsetAdmiss['onset-to-hospital-admission'] <= MAX_VAL]
d_onsetDischarge = pd.read_csv(DATA_PATH + 'onset-to-hospital-discharge.csv')
d_onsetDischarge = d_onsetDischarge[d_onsetDischarge['onset-to-hospital-discharge'] <= MAX_VAL]
d_onsetICU = pd.read_csv(DATA_PATH + 'onset-to-ICU-admission.csv')
d_onsetICU = d_onsetICU[d_onsetICU['onset-to-ICU-admission'] <= MAX_VAL]
d_adminDeath = pd.read_csv(DATA_PATH + 'hospital-admission-to-death.csv')
d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] <= MAX_VAL]
d_onsetDiagnosis_pcr= pd.read_csv(DATA_PATH + 'onset-to-diagnosis-pcr.csv')
d_onsetDiagnosis_pcr = d_onsetDiagnosis_pcr[d_onsetDiagnosis_pcr['onset-to-diagnosis-pcr'] <= MAX_VAL]

all_dfs = [d_ICUstay, d_onsetDeath, d_onsetDiagnosis,
           d_onsetAdmiss, d_onsetDischarge, d_onsetICU, d_adminDeath, d_onsetDiagnosis_pcr]


for df in all_dfs:
    df.dropna(inplace=True)
    
# add a state ID (int)
states = d_onsetDeath['State'].unique()
assert(len(states) == 27)
states.sort()
state_map = dict(zip(states, list(range(1, len(states)+1))))
states_id = list(range(1, len(states)+1))

columns = []
for df in all_dfs:
    df.dropna(inplace=True) # remove the rows with nan values
    try:
        df.drop(columns = drop_columns, inplace = True)
    except:
        print('')
    col = str(df.columns[1])
    columns.append(col)
    df['StateID'] = df['State'].map(state_map)


# i = 0
# plt.figure(figsize=(12,7))
# for df in all_dfs:
#     col = columns[i]
#     stdata = df[col].values# + 0.5
#     max_value = stdata.max()
#     # stdata = stdata[stdata < max_value]
#     bins = int(max_value + 0.5)
#     plt.subplot(3,3,i+1)
#     sns.distplot(stdata, kde=False, norm_hist=True, bins=bins)
#     plt.title(col)
#     i += 1
    
# plt.tight_layout()

# plt.figure(figsize=(10,5))
# j=1
# for i in [3,5]:
#     col = columns[i]
#     df=all_dfs[i].copy()
#     df = df[df[col] > 0]
#     stdata = df[col].values + 0.5
#     max_value = 60 # stdata.max()
#     # stdata = stdata[stdata < max_value]
#     bins = int(max_value + 0.5)
#     plt.subplot(1,2,j)
#     sns.distplot(stdata, kde=False, norm_hist=True, bins=bins)
#     plt.title(col)
#     j += 1
    
# plt.tight_layout()

############################################################################
# Fitting distribution to the whole country - national estimates

code_brazil_gamma = """
data {
    int N;
    vector[N] y;
}
parameters {
    real<lower=0> alpha;
    real<lower=0> beta;
}
model {
    alpha ~ normal(0,1);
    beta ~ normal(0,1);
    y ~ gamma(alpha, beta);
}
"""
model_brazil = pystan.StanModel(model_code=code_brazil_gamma)
print('National fits starting...')

def fit_brazil(values, list_of_params):
    """"Fit the distribution to the completely pooled Brazil data
    i.e. gives the nationwide estimates"""
    stdata = values
    stan_data = {'N': len(stdata), 'y': stdata}
    fit = model_brazil.sampling(data=stan_data, iter=ITER, seed=SEED, 
                                chains=CHAINS, n_jobs=-1)
    print(fit)                            
    df = fit.to_dataframe()
    df = df[list_of_params]
    return df

def get_national_posteriors(param_list):
    national_posteriors = {}
    for i in range(len(columns)):
        df = all_dfs[i]
        col = columns[i]
        print(col)
        vals = df[col].values
        # watch out here!!! we're shifting the data!!!!
        vals = vals + 0.5
        posterior = fit_brazil(vals, param_list)
        national_posteriors.update({col: posterior})
    return national_posteriors

national_posteriors_gamma = get_national_posteriors(['alpha', 'beta'])

############################################################################
# Fitting distribution to the partially pooled data

code_pp_gamma = """
data {
    int K; // number of states
    int N; // total number of observations
    real X[N]; // observations
    int state[N]; // index with the state number for each observation
}
parameters {
    real<lower=0> alpha[K];
    real<lower=0> beta[K];
    // hyperparameters
    real<lower=0> sigma_alpha;
    real<lower=0> sigma_beta;
    
}
model {
    // likelihood
    for (i in 1:N){
            X[i] ~ gamma(alpha[state[i]], beta[state[i]]);
    }
    // priors
    alpha ~ normal(INSERT_ALPHA,sigma_alpha);
    beta ~ normal(INSERT_BETA,sigma_beta);
    // hyperpriors
    sigma_alpha ~ normal(0,1);
    sigma_beta ~ normal(0,1);
}
"""

print('Sub-national fits starting...')

def fit_partial_pooling(stan_code, df, col, alpha, beta):
    stan_code = stan_code.replace('INSERT_ALPHA', str(alpha))
    stan_code = stan_code.replace('INSERT_BETA', str(beta))
    model = pystan.StanModel(model_code=stan_code)
    stan_pp_data = {'K': 27, 'N': df.shape[0], 
                    'X': df[col].values + 0.5,
                    'state': df['StateID'].values}
    fit = model.sampling(data=stan_pp_data, iter=ITER, seed=SEED, 
                          chains=CHAINS, n_jobs=-1,
                          control={'adapt_delta': 0.8})
    print(fit)
    posterior_df = fit.to_dataframe()
    params_columns = posterior_df.columns.str.startswith('alpha')+posterior_df.columns.str.startswith('beta') +posterior_df.columns.str.startswith('sigma_')
    posterior_df = posterior_df.loc[:,params_columns]
    return posterior_df


state_posteriors_gamma = {}
for i in range(len(columns)):
    df = all_dfs[i]
    col = columns[i]
    print(col)
    stan_code = code_pp_gamma
    alpha = national_posteriors_gamma[col]['alpha'].values.mean()
    beta = national_posteriors_gamma[col]['beta'].values.mean()
    posterior = fit_partial_pooling(stan_code, df, col, alpha, beta)
    # add national estimates
    posterior = pd.concat([posterior, national_posteriors_gamma[col]], axis=1, sort=False)
    state_posteriors_gamma.update({col: posterior})
    # save the output
    posterior.to_csv(OUT_PATH + col +'-samples-gamma.csv', index=False)
    state_posteriors_gamma.update({col: posterior})
    del posterior, df


print('Gamma model fits done')
print('Time elapsed: ', round((time.time()-t0)/60,1), ' minutes')
