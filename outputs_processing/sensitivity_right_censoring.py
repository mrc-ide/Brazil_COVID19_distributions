# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:22:47 2020

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

SEED = 4321
ITER = 2000
CHAINS = 4
MAX_VAL = 133
MIN_VAL = 1

##############################################################################
# this code fits the right-censored
# GAMMA model
##############################################################################
# LOAD AND PREPARE THE DATA
drop_columns = ['Unnamed: 0', 'Start_date', 'End_date']

d_ICUstay = pd.read_csv(DATA_PATH + 'ICU-stay.csv')
d_ICUstay = d_ICUstay[d_ICUstay['ICU-stay'] <= MAX_VAL]
d_onsetDeath = pd.read_csv(DATA_PATH + 'onset-to-death.csv')
d_onsetDeath = d_onsetDeath[d_onsetDeath['onset-to-death'] <= MAX_VAL]
d_onsetDeath = d_onsetDeath[d_onsetDeath['onset-to-death'] >= MIN_VAL]
d_adminDeath = pd.read_csv(DATA_PATH + 'hospital-admission-to-death.csv')
d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] <= MAX_VAL]

all_dfs_tmp = [d_ICUstay, d_onsetDeath, d_adminDeath]
all_dfs = []

for df in all_dfs_tmp:
    df.dropna(inplace=True)
    
# add a state ID (int)
states = d_onsetDeath['State'].unique()
assert(len(states) == 27)
states.sort()
state_map = dict(zip(states, list(range(1, len(states)+1))))
states_id = list(range(1, len(states)+1))

columns = []
for df in all_dfs_tmp:
    df.dropna(inplace=True) # remove the rows with nan values
    df['Start_date'] = df['Start_date'].astype('datetime64[ns]')
    df = df[df['Start_date'] < np.datetime64('2020-06-02')]
    df['End_date'] = df['End_date'].astype('datetime64[ns]')
    df = df[df['End_date'] >= np.datetime64('2020-04-01')]
    try:
        df.drop(columns = drop_columns, inplace = True)
    except:
        print('')
    col = str(df.columns[1])
    columns.append(col)
    df['StateID'] = df['State'].map(state_map)
    all_dfs.append(df)
    
del all_dfs_tmp

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

all_means = pd.DataFrame(columns = columns)
for col in columns:
    alphas = national_posteriors_gamma[col]['alpha']
    betas = national_posteriors_gamma[col]['beta']
    means = alphas / betas
    all_means[col] = means

all_means.to_csv('../results/MeansPosteriors/' + 'BrazilCensored.csv', index=False)
