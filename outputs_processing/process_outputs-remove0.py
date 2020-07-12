# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:28:20 2020

@author: iwona
"""


import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

sns.set()

##############################################################################
# Load, process, visualise for the separate models
# brazil data fits
DATA_PATH = '../data/'
OUT_PATH = 'results/'
MAX_VAL = 133 
MIN_VAL = 0 # or 1

##############################################################################

# load the data
drop_columns = ['Unnamed: 0', 'Start_date', 'End_date']

d_ICUstay = pd.read_csv(DATA_PATH + 'ICU-stay.csv')
d_ICUstay = d_ICUstay[d_ICUstay['ICU-stay'] <= MAX_VAL]
d_onsetDeath = pd.read_csv(DATA_PATH + 'onset-to-death.csv')
d_onsetDeath = d_onsetDeath[d_onsetDeath['onset-to-death'] <= MAX_VAL]
d_onsetDeath = d_onsetDeath[d_onsetDeath['onset-to-death'] >= MIN_VAL]
d_onsetDiagnosis= pd.read_csv(DATA_PATH + 'onset-to-diagnosis.csv')
d_onsetDiagnosis = d_onsetDiagnosis[d_onsetDiagnosis['onset-to-diagnosis'] <= MAX_VAL]
d_onsetAdmiss = pd.read_csv(DATA_PATH + 'onset-to-hospital-admission.csv')
d_onsetAdmiss = d_onsetAdmiss[d_onsetAdmiss['onset-to-hospital-admission'] <= MAX_VAL]
d_onsetAdmiss = d_onsetAdmiss[d_onsetAdmiss['onset-to-hospital-admission'] >= MIN_VAL]
d_onsetDischarge = pd.read_csv(DATA_PATH + 'onset-to-hospital-discharge.csv')
d_onsetDischarge = d_onsetDischarge[d_onsetDischarge['onset-to-hospital-discharge'] <= MAX_VAL]
d_onsetICU = pd.read_csv(DATA_PATH + 'onset-to-ICU-admission.csv')
d_onsetICU = d_onsetICU[d_onsetICU['onset-to-ICU-admission'] <= MAX_VAL]
d_onsetICU = d_onsetICU[d_onsetICU['onset-to-ICU-admission'] >= MIN_VAL]
d_adminDeath = pd.read_csv(DATA_PATH + 'hospital-admission-to-death.csv')
d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] <= MAX_VAL]
d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] >= 0]  # different here!!!
d_onsetDiagnosis_pcr= pd.read_csv(DATA_PATH + 'onset-to-diagnosis-pcr.csv')
d_onsetDiagnosis_pcr = d_onsetDiagnosis_pcr[d_onsetDiagnosis_pcr['onset-to-diagnosis-pcr'] <= MAX_VAL]

all_dfs = [d_ICUstay, d_onsetDeath, d_onsetDiagnosis,
           d_onsetAdmiss, d_onsetDischarge, d_onsetICU, d_adminDeath, d_onsetDiagnosis_pcr]

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

# print n samples and range of data
for df in all_dfs:
    col = str(df.columns[1])
    print(col, len(df[col].index), df[col].min(), '-', df[col].max())
    
# print n samples == 0
print('Number of samples == 0:')
for df in all_dfs:
    col = str(df.columns[1])
    n = len(df[df[col]==0].index)
    print(col, n)
    
    
ids = [1,3,5,6]
columns = [columns[1], columns[3], columns[5], columns[6]]
all_dfs = [all_dfs[1], all_dfs[3], all_dfs[5], all_dfs[6]]
    
# load the samples (models fits)

state_posteriors_gamma = {}
state_posteriors_wei = {}
state_posteriors_lognorm = {}
state_posteriors_gln = {}
state_posteriors_gg = {}

for i in range(len(columns)):
    col = columns[i]
    if (col == columns[0]) or (col == columns[1]) or (col == columns[2]):  # remove first day in onset-admission and onset-ICU and onset-death
        added = '' # in the other one we removed0 there
    else: # admission death
        added = '-remove0'
        
    print(col, added)
    try:
        state_posteriors_gamma.update({col: pd.read_csv(col + '-samples-gamma' + added + '.csv')})
    except:
        print(col, 'did not find samples for gamma')
        state_posteriors_gamma.update({col: state_posteriors_gamma[columns[0]] * np.nan})
        
    try:
        state_posteriors_wei.update({col: pd.read_csv(col + '-samples-wei' + added + '.csv')})
    except:
        print(col, 'did not find samples for wei')
        state_posteriors_wei.update({col: state_posteriors_wei[columns[0]] * np.nan})
        
    try:
        state_posteriors_lognorm.update({col: pd.read_csv(col + '-samples-lognormal' + added + '.csv')})
    except:
        print(col, 'did not find samples for lognorm')
        state_posteriors_lognorm.update({col: state_posteriors_lognorm[columns[0]] * np.nan})
    try:
        state_posteriors_gln.update({col: pd.read_csv(col + '-samples-gln' + added + '.csv')})
    except:
        print(col, 'did not find samples for gln')
        state_posteriors_gln.update({col: state_posteriors_gln[columns[0]] * np.nan})
        
    try:
        state_posteriors_gg.update({col: pd.read_csv(col + '-samples-gg' + added + '.csv').drop(columns = ['draw', 'accept_stat__', 'divergent__'])})
    except:
        print(col, 'did not find samples for gg')
        state_posteriors_gg.update({col: state_posteriors_gg[columns[0]] * np.nan})

##############################################################################

def gln_pdf(x, mu, sigma, g):
    """PDF of the generalised log-normal distribution"""
    k = g / (2**((g+1)/g) * sigma * scipy.special.gamma(1/g))
    return k/x * np.exp(-0.5 * np.abs((np.log(x)-mu)/sigma)**g)

def gln_lpdf(x, mu, sigma, g):
    """Log-PDF of the generalised log-normal distribution"""
    logk = np.log(g) - ( ((g+1)/g)*np.log(2) + np.log(sigma) + np.log(scipy.special.gamma(1/g)))
    return logk - np.log(x) - 0.5 * np.abs((np.log(x)-mu)/sigma)**g

def gln_cdf_help(x, mu, sigma, g):
    """CDF of the generalised log-normal distribution"""
    m = x
    result = scipy.integrate.quad(lambda x: gln_pdf(x,2,0.5,2.5), 0, m)
    tmp = result[0]
    return tmp
    
def gln_cdf(x, mu, sigma, g):
    c = np.vectorize(gln_cdf_help)
    return c(x, mu, sigma, g)
    
def gg_pdf(x, a, d, p):
    tmp = p/(a**d) * (x**(d-1)) * np.exp(-((x/a)**p))
    tmp = tmp / scipy.special.gamma(d/p)
    return tmp

def gg_lpdf(x, a, d, p):
    """Log-PDF of the generalised gamma distribution"""
    tmp = np.log(p) - d*np.log(a) - np.log(scipy.special.gamma(d/p)) + (d-1) * np.log(x) - (x/a)**p
    return tmp

def gg_cdf(x, a, d, p):
    """CDF of the generalised gamma distribution"""
    arg1 = a/p
    arg2 = np.power(x/a,p)
    phi = scipy.special.gammainc(arg1, arg2)
    tmp = phi / scipy.special.gamma(d/p)
    return tmp
    
##############################################################################
# Laplace approx & Bayes Factors to find the best fit

def fit_posterior(distr, col):
    if distr == 'gamma':
        df = state_posteriors_gamma[col].iloc[:,:-2]
        samples = df.values
        assert samples.shape[1] == 27 * 2 + 2
    elif distr == 'weibull':
        df = state_posteriors_wei[col].iloc[:,:-2]
        samples = df.values
        assert samples.shape[1] == 27 * 2 + 2

    elif distr == 'lognormal':
        df = state_posteriors_lognorm[col].iloc[:,:-2]
        samples = df.values
        assert samples.shape[1] == 27 * 2 + 2
    elif distr == 'gln':
        df = state_posteriors_gln[col].iloc[:,:-3]
        samples = df.values
        assert samples.shape[1] == 27 * 3 + 3
    elif distr == 'gg':
        df = state_posteriors_gg[col].iloc[:,:-3]
        samples = df.values
        assert samples.shape[1] == 27 * 3 + 3
    else:
        print('Invalid distribution')
        return {}
    cov = np.cov(np.array(samples), rowvar = False)
    theta_mean = samples.mean(axis = 0)
    # c = np.linalg.inv(np.linalg.cholesky(cov))
    # covinv = np.dot(c.T,c)
    return {'mu': theta_mean, 'cov': cov, 'covinv': {}, 'distr': distr}

columns_BF = columns

posterior_gamma = {}
posterior_wei = {}
posterior_lognorm = {}
posterior_gln = {}
posterior_gg = {}

for col in columns_BF:
    posterior_gamma.update({col: fit_posterior('gamma', col)})
    posterior_wei.update({col: fit_posterior('weibull', col)})
    posterior_lognorm.update({col: fit_posterior('lognormal', col)})
    posterior_gln.update({col: fit_posterior('gln', col)})
    posterior_gg.update({col: fit_posterior('gg', col)})

def Logf(posterior, col):
    col_id = columns.index(col)
    data = all_dfs[col_id]
    distr = posterior[col]['distr']
    theta = posterior[col]['mu']
    x = data[col].values + 0.5 
    state = data['StateID'].values # int index for the state
    k = 27 #len(data['StateID'].unique())
    
    log_prior = 0
    log_likelihood = 0
    
    if distr == 'gamma':
        sigma_alpha = theta[-2]
        sigma_beta = theta[-1]
        alpha_brazil = state_posteriors_gamma[col].alpha.values.mean()
        beta_brazil =  state_posteriors_gamma[col].beta.values.mean()
        log_prior += scipy.stats.norm(0,1).logpdf(sigma_alpha)
        log_prior += scipy.stats.norm(0,1).logpdf(sigma_beta)
        for i in range(k):
            log_prior += scipy.stats.norm(alpha_brazil,sigma_alpha).logpdf(theta[i]) # alphas
            log_prior += scipy.stats.norm(beta_brazil,sigma_beta).logpdf(theta[i+k]) # betas
        for i in range(len(x)):
            alpha = theta[state[i]-1]
            beta = theta[state[i]-1 + k]
            log_likelihood += scipy.stats.gamma(a=alpha, scale = 1.0/beta).logpdf(x[i])
           
    elif distr == 'weibull':
        sigma_alpha = theta[-2]
        sigma_sigma = theta[-1]
        alpha_brazil = state_posteriors_wei[col].alpha.values.mean()
        sigma_brazil = state_posteriors_wei[col].sigma.values.mean()
        log_prior += scipy.stats.norm(0,1).logpdf(sigma_alpha)
        log_prior += scipy.stats.norm(0,1).logpdf(sigma_sigma)
        for i in range(k):
            log_prior += scipy.stats.norm(alpha_brazil,sigma_alpha).logpdf(theta[i]) # alphas
            log_prior += scipy.stats.norm(sigma_brazil,sigma_sigma).logpdf(theta[i+k]) # sigmas
        for i in range(len(x)):
            alpha = theta[state[i]-1]
            sigma = theta[state[i]-1 + k]
            log_likelihood += scipy.stats.weibull_min(c=alpha, scale = sigma).logpdf(x[i])
       
    elif distr == 'lognormal':
        sigma_mu = theta[-2]
        sigma_sigma = theta[-1]
        mu_brazil = state_posteriors_lognorm[col].mu.values.mean()
        sigma_brazil = state_posteriors_lognorm[col].sigma.values.mean()
        log_prior += scipy.stats.norm(0,1).logpdf(sigma_mu)
        log_prior += scipy.stats.norm(0,1).logpdf(sigma_sigma)
        for i in range(k):
            log_prior += scipy.stats.norm(mu_brazil,sigma_mu).logpdf(theta[i]) # mu
            log_prior += scipy.stats.norm(sigma_brazil,sigma_sigma).logpdf(theta[i+k]) # sigmas
        for i in range(len(x)):
            mu = theta[state[i]-1]
            sigma = theta[state[i]-1 + k]
            log_likelihood += scipy.stats.lognorm(s=sigma, scale = np.exp(mu)).logpdf(x[i])  
            
    elif distr == 'gln':
        sigma_mu = theta[-3]
        sigma_sigma = theta[-2]
        sigma_g = theta[-1]
        mu_brazil = state_posteriors_gln[col].mu.values.mean()
        sigma_brazil = state_posteriors_gln[col].sigma.values.mean()
        g_brazil = state_posteriors_gln[col].g.values.mean()
        log_prior += scipy.stats.norm(2,0.5).logpdf(sigma_mu)
        log_prior += scipy.stats.norm(0.5,0.5).logpdf(sigma_sigma)
        log_prior += scipy.stats.norm(1.5,0.5).logpdf(sigma_g)

        for i in range(k):
            log_prior += scipy.stats.norm(mu_brazil,sigma_mu).logpdf(theta[i]) # mu
            log_prior += scipy.stats.norm(sigma_brazil,sigma_sigma).logpdf(theta[i+k]) # sigmas
            log_prior += scipy.stats.norm(g_brazil,sigma_g).logpdf(theta[i+2*k]) # g
        for i in range(len(x)):
            mu = theta[state[i]-1]
            sigma = theta[state[i]-1 + k]
            g = theta[state[i]-1 + 2*k]
            log_likelihood += gln_lpdf(x[i], mu, sigma, g)
            
    elif distr == 'gg':
        sigma_a = theta[-3]
        sigma_d = theta[-2]
        sigma_p = theta[-1]
        a_brazil = state_posteriors_gg[col].a.values.mean()
        d_brazil = state_posteriors_gg[col].d.values.mean()
        p_brazil = state_posteriors_gg[col].p.values.mean()
        log_prior += scipy.stats.norm(1,1).logpdf(sigma_a)
        log_prior += scipy.stats.norm(1,1).logpdf(sigma_d)
        log_prior += scipy.stats.norm(1,1).logpdf(sigma_p)

        for i in range(k):
            log_prior += scipy.stats.norm(a_brazil,sigma_a).logpdf(theta[i]) # p
            log_prior += scipy.stats.norm(d_brazil,sigma_d).logpdf(theta[i+k]) # d
            log_prior += scipy.stats.norm(p_brazil,sigma_p).logpdf(theta[i+2*k]) # p
        for i in range(len(x)):
            a = theta[state[i]-1]
            d = theta[state[i]-1 + k]
            p = theta[state[i]-1 + 2*k]
            log_likelihood += gg_lpdf(x[i], a, d, p) 
    else:
        print('Invalid distribution')
        return np.nan

    return log_likelihood + log_prior    

def LogLaplaceCovariance(posterior, col):
      result = 1/2 * len(posterior[col]['mu']) * np.log(2*np.pi) 
      result += 1/2 * np.log(np.linalg.det(posterior[col]['cov']))
      result += posterior[col]['Logf']
      # result += Logf(posterior, col)
      return result

# careful, this is slow
for col in columns_BF:
    print(col)
    posterior_gamma[col].update({'Logf': Logf(posterior_gamma, col)})
    posterior_gamma[col].update({'LogLaplace': LogLaplaceCovariance(posterior_gamma, col)})

    posterior_wei[col].update({'Logf': Logf(posterior_wei, col)})
    posterior_wei[col].update({'LogLaplace': LogLaplaceCovariance(posterior_wei, col)})

    posterior_lognorm[col].update({'Logf': Logf(posterior_lognorm, col)})
    posterior_lognorm[col].update({'LogLaplace': LogLaplaceCovariance(posterior_lognorm, col)})
    
    posterior_gln[col].update({'Logf': Logf(posterior_gln, col)})
    posterior_gln[col].update({'LogLaplace': LogLaplaceCovariance(posterior_gln, col)})
    
    posterior_gg[col].update({'Logf': Logf(posterior_gg, col)})
    posterior_gg[col].update({'LogLaplace': LogLaplaceCovariance(posterior_gg, col)})

columns_models = ['Gamma', 'Weibull', 'Lognormal', 'Gen Lognorm', 'Gen Gamma']
bayesfactorsBR = pd.DataFrame(columns = columns_models, index = columns_models)
BayesFactorsBrazil = {}
evidence_dict = {'Gamma': posterior_gamma, 'Weibull': posterior_wei, 'Lognormal': posterior_lognorm,
                  'Gen Lognorm': posterior_gln, 'Gen Gamma': posterior_gg}

all_BF = {}
for col in columns_BF:
    for c1 in columns_models:
        for c2 in columns_models:
            bayesfactorsBR.loc[c1,c2] = 2*(evidence_dict[c1][col]['LogLaplace'] - evidence_dict[c2][col]['LogLaplace'])
    all_BF.update({col: bayesfactorsBR})
    print(col)
    print(bayesfactorsBR, '\n')


geogr_location_map = {'North': ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],
                      'Northeast': ['AL', 'BA', 'CE', 'MA', 'PB', 'PI', 'PE', 'SE', 'RN'],
                      'Central-West': ['DF', 'GO', 'MS', 'MT'],
                      'Southeast': ['ES', 'MG', 'RJ', 'SP'],
                      'South': ['PR', 'RS', 'SC']}

states_geo_order = list(geogr_location_map.values())
states_geo_order = [val for sublist in states_geo_order for val in sublist]



##############################################################################
r = 1 # rounding
# functions with calculating mean and var for each distribution
def stats_gamma(df, dict_out=True):
    df = df.copy()
    df.columns = df.columns.str.replace(r'\d+', '')
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')

    alpha = df.alpha.values
    beta = df.beta.values
    mean = alpha / beta
    var = alpha / beta**2
    if not dict_out:
        return mean, var
    mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
    var_CI95 = tuple(np.round(np.quantile(var,[.025,.975]),r))
    return {'mean': round(mean.mean(),1), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),1), 'var_CI95': var_CI95}

def stats_weibull(df, dict_out=True):
    df = df.copy()
    df.columns = df.columns.str.replace(r'\d+', '')
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')

    alpha = df.alpha.values
    sigma = df.sigma.values
    mean = sigma * scipy.special.gamma(1+(1/alpha))
    var = sigma**2 * (scipy.special.gamma(1+2/alpha)-(scipy.special.gamma(1+1/alpha))**2)
    if not dict_out:
        return mean, var
    mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
    var_CI95 = tuple(np.round(np.quantile(var,[.025,.975]),r))
    return {'mean': round(mean.mean(),1), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),1), 'var_CI95': var_CI95}

def stats_lognormal(df, dict_out=True):
    df = df.copy()
    df.columns = df.columns.str.replace(r'\d+', '')
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')

    mu = df.mu.values
    sigma = df.sigma.values
    mean = np.exp(mu+0.5*sigma**2)
    var = (np.exp(sigma**2)-1)*np.exp(2*mu+(sigma)**2)
    if not dict_out:
        return mean, var
    mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
    var_CI95 = tuple(np.round(np.quantile(var,[.025,.975]),r))
    return {'mean': round(mean.mean(),1), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),1), 'var_CI95': var_CI95}

def inf_sum(r, mu, sig, g):
    result = 0
    for j in range(1,150): # 100 should be enough to get this infinite sum but test it; gamma(170) = inf
        tmp = (r * sig)**j
        tmp = tmp * (1 + (-1)**j) * 2**(j/g)
        tmp = tmp * (scipy.special.gamma((j+1)/g) / scipy.special.gamma(j+1))
        result += tmp
    return result
        
def gln_mean_var_help(mu, sig, g):
    inf_sum1 = inf_sum(1, mu, sig, g)
    mean = np.exp(1 * mu) * (1 + 1/(2 * scipy.special.gamma(1/g)) * inf_sum1)
    inf_sum2 = inf_sum(2, mu, sig, g)
    sec_moment = np.exp(2 * mu) * (1 + 1/(2 * scipy.special.gamma(1/g)) * inf_sum2)
    var = sec_moment - mean**2
    return (mean, var)

# https://link.springer.com/content/pdf/10.1007/s00180-011-0233-9.pdf, equation 4 for r-th raw moment
def stats_gln(df, dict_out=True):
    df = df.copy()
    df.columns = df.columns.str.replace(r'\d+', '')
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')
    
    mu = df.mu.values
    sigma = df.sigma.values
    g = df.g.values
    mean, var = gln_mean_var_help(mu, sigma, g)
    if not dict_out:
        return mean, var
    mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
    var_CI95 = tuple(np.round(np.quantile(var,[.025,.975]),r))
    return {'mean': round(mean.mean(),1), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),1), 'var_CI95': var_CI95}

def stats_gg(df, dict_out=True):
    df = df.copy()
    df.columns = df.columns.str.replace(r'\d+', '')
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')

    a = df.a.values
    d = df.d.values
    p = df.p.values
    gamdp = scipy.special.gamma(d/p)
    mean = a * ((scipy.special.gamma((d+1)/p)) / gamdp)
    var = a**2 * (((scipy.special.gamma((d+2)/p))/gamdp) - ((scipy.special.gamma((d+1)/p))/(gamdp))**2)
    if not dict_out:
        return mean, var
    mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
    var_CI95 = tuple(np.round(np.quantile(var,[.025,.975]),r))
    return {'mean': round(mean.mean(),1), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),1), 'var_CI95': var_CI95}

def stats(name, df, dict_out=True):
    if name == 'Gamma':
        return stats_gamma(df, dict_out)
    if name == 'Weibull':
        return stats_weibull(df, dict_out)
    if name == 'Log-normal':
        return stats_lognormal(df, dict_out)
    if name == 'Gen. Gamma':
        return stats_gg(df, dict_out)
    if name == 'Gen. Log-normal':
        return stats_gln(df, dict_out)

##############################################################################
# save the means posteriors
best_fit_map = {'onset-to-death': 'Gen. Gamma', 
                'Hospital-Admission-to-death': 'Weibull',
                'onset-to-hospital-admission': 'Gen. Gamma',
                'onset-to-ICU-admission': 'Gen. Gamma'}

models = {'Gamma': state_posteriors_gamma, 'Weibull': state_posteriors_wei,
          'Log-normal': state_posteriors_lognorm, 'Gen. Gamma': state_posteriors_gg,
          'Gen. Log-normal': state_posteriors_gln}


def save_all_means_per_column_Brazil():
    all_means = pd.DataFrame(columns = columns)
    for col in columns:
        model = models[best_fit_map[col]][col] # posterior samples
        
        if best_fit_map[col] in ['Gamma', 'Weibull', 'Log-normal']:
            means, var = stats(best_fit_map[col], model.iloc[:,-2:], dict_out=False)
        else:
            means, var = stats(best_fit_map[col], model.iloc[:,-3:], dict_out=False)
        all_means[col] = means
    all_means.to_csv(OUT_PATH + '/MeansPosteriors/' + 'BrazilWithAlternativeBottomCut.csv', index=False)
    
save_all_means_per_column_Brazil() 
   
