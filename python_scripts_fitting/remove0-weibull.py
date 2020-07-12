# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:50:49 2020

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
OUT_PATH = ''  # CHANGE BEFORE RUNNING THE JOB!!!!!
SAVE_STR = 'remove0'

SEED = 4321
ITER = 2000
CHAINS = 4
MAX_VAL = 133
MIN_VAL = 1


##############################################################################
# this code fits the
# Weibull model
# to each of the covariates
t0 = time.time()
##############################################################################
# LOAD AND PREPARE THE DATA
drop_columns = ['Unnamed: 0', 'Start_date', 'End_date']

d_onsetAdmiss = pd.read_csv(DATA_PATH + 'onset-to-hospital-admission.csv')
d_onsetAdmiss = d_onsetAdmiss[d_onsetAdmiss['onset-to-hospital-admission'] <= MAX_VAL]
d_onsetAdmiss = d_onsetAdmiss[d_onsetAdmiss['onset-to-hospital-admission'] >= MIN_VAL]

d_onsetICU = pd.read_csv(DATA_PATH + 'onset-to-ICU-admission.csv')
d_onsetICU = d_onsetICU[d_onsetICU['onset-to-ICU-admission'] <= MAX_VAL]
d_onsetICU = d_onsetICU[d_onsetICU['onset-to-ICU-admission'] >= MIN_VAL]

d_adminDeath = pd.read_csv(DATA_PATH + 'hospital-admission-to-death.csv')
d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] <= MAX_VAL]
d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] >= MIN_VAL]


all_dfs = [d_onsetAdmiss, d_onsetICU, d_adminDeath]

for df in all_dfs:
    df.dropna(inplace=True)
    
# add a state ID (int)
states = d_onsetAdmiss['State'].unique()
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


############################################################################
# Fitting distribution to the whole country - national estimates


code_brazil_wei = """
data {
    int N;
    vector[N] y;
}
parameters {
    real<lower=0> alpha;
    real<lower=0> sigma;
}
model {
    alpha ~ normal(1,1);
    sigma ~ normal(1,0.5);
    y ~ weibull(alpha, sigma);
}

"""
model_brazil = pystan.StanModel(model_code=code_brazil_wei)
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

national_posteriors_wei = get_national_posteriors(['alpha', 'sigma'])

############################################################################
# Fitting distribution to the partially pooled data
code_pp_wei = """
data {
    int K; // number of states
    int N; // total number of observations
    real X[N]; // observations
    int state[N]; // index with the state number for each observation
}
parameters {
    real<lower=0> alpha[K];
    real<lower=0> sigma[K];
    // hyperparameters
    real<lower=0> sigma_alpha;
    real<lower=0> sigma_sigma;
}

model {
    // likelihood
    for (i in 1:N){
            X[i] ~ weibull(alpha[state[i]], sigma[state[i]]);
    }
    // priors
    alpha ~ normal(INSERT_ALPHA,sigma_alpha);
    sigma ~ normal(INSERT_SIGMA,sigma_sigma);
    // hyperpriors
    sigma_alpha ~ normal(0,1);
    sigma_sigma ~ normal(0,1);
}
"""
print('Sub-national fits starting...')

def fit_partial_pooling(stan_code, df, col, alpha, sigma):
    stan_code = stan_code.replace('INSERT_ALPHA', str(alpha))
    stan_code = stan_code.replace('INSERT_SIGMA', str(sigma))
    model = pystan.StanModel(model_code=stan_code)
    stan_pp_data = {'K': 27, 'N': df.shape[0], 
                    'X': df[col].values + 0.5,
                    'state': df['StateID'].values}
    fit = model.sampling(data=stan_pp_data, iter=ITER, seed=SEED, chains=CHAINS, n_jobs=-1,
                          control={'adapt_delta': 0.8})
    print(fit)
    posterior_df = fit.to_dataframe()
    params_columns = posterior_df.columns.str.startswith('alpha')+posterior_df.columns.str.startswith('sigma')
    posterior_df = posterior_df.loc[:,params_columns]
    return posterior_df

state_posteriors_wei = {}
for i in range(len(columns)):
    df = all_dfs[i]
    col = columns[i]
    print(col)
    stan_code = code_pp_wei
    alpha = national_posteriors_wei[col]['alpha'].values.mean()
    sigma = national_posteriors_wei[col]['sigma'].values.mean()
    posterior = fit_partial_pooling(stan_code, df, col, alpha, sigma)
    # add national estimates
    posterior = pd.concat([posterior, national_posteriors_wei[col]], axis=1, sort=False)
    state_posteriors_wei.update({col: posterior})
    # save the output
    posterior.to_csv(OUT_PATH + col +'-samples-wei-' + SAVE_STR + '.csv', index=False)
    del posterior, df

print('Weibull model fits done')
print('Time elapsed: ', round((time.time()-t0)/60,1), ' minutes')
