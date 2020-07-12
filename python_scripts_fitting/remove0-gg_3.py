# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:21:41 2020

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

DATA_PATH = '../../data/'
OUT_PATH = '../'  # CHANGE BEFORE RUNNING THE JOB!!!!!

CHOSEN_COLUMN = 0
SEED = 4321
ITER = 2000
CHAINS = 4
MAX_VAL = 133 
MIN_VAL = 1

##############################################################################
# this code fits the
# GG (generalised gamma) model
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
columns = [columns[CHOSEN_COLUMN]]
all_dfs = [all_dfs[CHOSEN_COLUMN]]
############################################################################
# Fitting distribution to the whole country - national estimates
code_brazil_gg = """
functions{
 real custom_lpdf(real x, real a, real d, real p)
    {
     
     real tmp = log(p) - d*log(a) - lgamma(d/p) + (d-1) * log(x) - pow(x/a,p);
     return tmp;
    }
}
data {
    int N;
    real y[N];
}
parameters {
    real<lower=0> a;
    real<lower=0> d;
    real<lower=0> p;
}
model {
      for (i in 1:N){
        y[i] ~ custom(a, d, p);
        }
      a ~ normal(0,1);
      d ~ normal(0,1);
      p ~ normal(0,1);
}

"""
model_brazil = pystan.StanModel(model_code=code_brazil_gg)
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

national_posteriors_gg = get_national_posteriors(['a', 'd', 'p'])

############################################################################
# Fitting distribution to the partially pooled data
code_pp_gg = """
functions{
    real custom_lpdf(real x, real a, real d, real p)
    {
     
     real tmp = log(p) - d*log(a) - lgamma(d/p) + (d-1) * log(x) - pow(x/a,p);
     return tmp;
    }
}
data {
    int K; // number of states
    int N; // total number of observations
    real X[N]; // observations
    int state[N]; // index with the state number for each observation
}
parameters {
    real<lower=0> a[K];
    real<lower=0> d[K];
    real<lower=1> p[K];
    // hyperparameters
    real<lower=0> sigma_a;
    real<lower=0> sigma_d;
    real<lower=0> sigma_p;
}

model {
    // likelihood
    for (i in 1:N){
            X[i] ~ custom(a[state[i]], d[state[i]], p[state[i]]);
    }
    // priors
    a ~ normal(INSERT_A,sigma_a);
    d ~ normal(INSERT_D,sigma_d);
    p ~ normal(INSERT_P,sigma_p);

    // hyperpriors
    sigma_a ~ normal(1,1);
    sigma_d ~ normal(1,1);
    sigma_p ~ normal(1,1);

}
"""

print('Sub-national fits starting...')

def fit_partial_pooling(stan_code, df, col, a, d, p):
    stan_code = stan_code.replace('INSERT_A', str(a))
    stan_code = stan_code.replace('INSERT_D', str(d))
    stan_code = stan_code.replace('INSERT_P', str(p))
    model = pystan.StanModel(model_code=stan_code)
    stan_pp_data = {'K': 27, 'N': df.shape[0], 
                    'X': df[col].values + 0.5,
                    'state': df['StateID'].values}
    fit = model.sampling(data=stan_pp_data, iter=ITER, seed=SEED, chains=CHAINS, n_jobs=-1,
                          control={'adapt_delta': 0.8})
    print(fit)
    posterior_df = fit.to_dataframe()
    params_columns = posterior_df.columns.str.startswith('a')+posterior_df.columns.str.startswith('d')+posterior_df.columns.str.startswith('p')+posterior_df.columns.str.startswith('sigma')
    posterior_df = posterior_df.loc[:,params_columns]
    return posterior_df


state_posteriors_gg = {}

for i in range(len(columns)):
    df = all_dfs[i]
    col = columns[i]
    print(col)
    stan_code = code_pp_gg
    a = national_posteriors_gg[col]['a'].values.mean()
    d = national_posteriors_gg[col]['d'].values.mean()
    p = national_posteriors_gg[col]['p'].values.mean()
    posterior = fit_partial_pooling(stan_code, df, col, a, d, p)
    # add national estimates
    posterior = pd.concat([posterior, national_posteriors_gg[col]], axis=1, sort=False)
    state_posteriors_gg.update({col: posterior})
    # save the output
    posterior.to_csv(OUT_PATH + col +'-samples-gg-remove0.csv', index=False)
    del posterior, df


print('GG model fits done')
print('Time elapsed: ', round((time.time()-t0)/60,1), ' minutes')
