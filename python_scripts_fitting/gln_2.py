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

DATA_PATH = '../../data/'
OUT_PATH = '../'  # CHANGE BEFORE RUNNING THE JOB!!!!!

CHOSEN_COLUMN = 2
SEED = 4321
ITER = 2000
CHAINS = 4
MAX_VAL = 133 

##############################################################################
# this code fits the
# GLN (generalised log-normal) model
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

############################################################################
columns = [columns[CHOSEN_COLUMN]]
all_dfs = [all_dfs[CHOSEN_COLUMN]]
############################################################################
# Fitting distribution to the whole country - national estimates
print('National fits starting...')

code_brazil_gln = """
functions{
 real custom_lpdf(real x, real mu, real sigma, real g)
    {
     real logK = log(g) - (g+1)/g*log(2)-log(sigma)-lgamma(1/g);
     real tmp = logK - log(x) - 0.5 * pow(fabs((log(x)-mu)/sigma),g);
     return tmp;
    }
}
data {
    int N;
    real y[N];
}
parameters {
    real<lower=0> mu;
    real<lower=0> sigma;
    real<lower=1> g;
}
model {
      for (i in 1:N){
        y[i] ~ custom(mu, sigma, g);
        }
      mu ~ normal(2,0.5);
      sigma ~ normal(0.5,0.5);
      g ~ normal(1.5,0.5);
}

"""

     
model_brazil = pystan.StanModel(model_code=code_brazil_gln)


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

national_posteriors_gln = get_national_posteriors(['mu', 'sigma', 'g'])

############################################################################
# Fitting distribution to the partially pooled data
code_pp_gln = """
functions{
  real custom_lpdf(real x, real mu, real sigma, real g)
    {
      real logK = log(g) - (g+1)/g*log(2)-log(sigma)-lgamma(1/g);
      real tmp = logK - log(x) - 0.5 * pow(fabs((log(x)-mu)/sigma),g);
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
    real<lower=0> mu[K];
    real<lower=0> sigma[K];
    real<lower=1> g[K];
    // hyperparameters
    real<lower=0> sigma_mu;
    real<lower=0> sigma_sigma;
    real<lower=0> sigma_g;
}

model {
    // likelihood
    for (i in 1:N){
            X[i] ~ custom(mu[state[i]], sigma[state[i]], g[state[i]]);
    }
    // priors
    mu ~ normal(INSERT_MU,sigma_mu);
    sigma ~ normal(INSERT_SIGMA,sigma_sigma);
    g ~ normal(INSERT_G,sigma_g);

    // hyperpriors
    sigma_mu ~ normal(2,0.5);
    sigma_sigma ~ normal(0.5,0.5);
    sigma_g ~ normal(1.5,0.5);

}
"""

print('Sub-national fits starting...')

def fit_partial_pooling(stan_code, df, col, mu, sigma, g):
    stan_code = stan_code.replace('INSERT_MU', str(mu))
    stan_code = stan_code.replace('INSERT_SIGMA', str(sigma))
    stan_code = stan_code.replace('INSERT_G', str(g))

    model = pystan.StanModel(model_code=stan_code)
    stan_pp_data = {'K': 27, 'N': df.shape[0], 
                    'X': df[col].values + 0.5,
                    'state': df['StateID'].values}
    fit = model.sampling(data=stan_pp_data, iter=ITER, seed=SEED, chains=CHAINS, n_jobs=-1,
                          control={'adapt_delta': 0.8})
    print(fit)
    posterior_df = fit.to_dataframe()
    params_columns = posterior_df.columns.str.startswith('mu')+posterior_df.columns.str.startswith('sigma')+posterior_df.columns.str.startswith('g')
    posterior_df = posterior_df.loc[:,params_columns]
    return posterior_df


state_posteriors_gln = {}
for i in range(len(columns)):
    df = all_dfs[i]
    col = columns[i]
    print(col)
    stan_code = code_pp_gln
    mu = national_posteriors_gln[col]['mu'].values.mean()
    sigma = national_posteriors_gln[col]['sigma'].values.mean()
    g = national_posteriors_gln[col]['g'].values.mean()
    posterior = fit_partial_pooling(stan_code, df, col, mu, sigma, g)
    # add national estimates
    posterior = pd.concat([posterior, national_posteriors_gln[col]], axis=1, sort=False)
    state_posteriors_gln.update({col: posterior})
    # save the output
    posterior.to_csv(OUT_PATH + col +'-samples-gln.csv', index=False)
del posterior, df
    

print('GLN model fits done')
print('Time elapsed: ', round((time.time()-t0)/60,1), ' minutes')
