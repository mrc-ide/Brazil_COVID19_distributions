# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:39:47 2020

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
OUT_PATH = '../results/'
SAMPLES_PATH = '../fitting_outputs/'
MAX_VAL = 133 
MIN_VAL = 1  # or 1

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
# d_adminDeath = d_adminDeath[d_adminDeath['Hospital-Admission-to-death'] >= MIN_VAL]
d_onsetDiagnosis_pcr= pd.read_csv(DATA_PATH + 'onset-to-diagnosis-pcr.csv')
d_onsetDiagnosis_pcr = d_onsetDiagnosis_pcr[d_onsetDiagnosis_pcr['onset-to-diagnosis-pcr'] <= MAX_VAL]

all_dfs = [d_ICUstay, d_onsetDeath, d_onsetDiagnosis,
           d_onsetAdmiss, d_onsetDischarge, d_onsetICU, d_adminDeath, d_onsetDiagnosis_pcr]


# clean the data and prepare some the variables list 'columns'
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

##############################################################################
# get number of samples for national

# print n samples and range of data
for df in all_dfs:
    col = str(df.columns[1])
    print(col, len(df[col].index), df[col].min(), '-', df[col].max())
    

##############################################################################
# get number of samples for sub-national

columns_order = [1,6,0,3,4,5,7,2]
columns_ordered = [columns[i] for i in columns_order]
number_samples = pd.DataFrame(columns = columns_ordered, index = states)

for df in all_dfs:
    count = df.State.value_counts().to_dict()
    col = df.columns[1]
    number_samples[col] = number_samples.index.map(count)
    
number_samples.fillna(0, inplace=True)
number_samples.to_csv(OUT_PATH + 'number_of_samples.csv')

number_samples_trans = number_samples.transpose()
number_samples_trans.to_csv(OUT_PATH + 'number_of_samples_wide.csv')

##############################################################################

# load the samples (models fits)

state_posteriors_gamma = {}
state_posteriors_wei = {}
state_posteriors_lognorm = {}
state_posteriors_gln = {}
state_posteriors_gg = {}

for i in range(len(columns)):
    col = columns[i]
    if (col == columns[3]) or (col == columns[5]) or (col == columns[1]):  # remove first day in onset-admission and onset-ICU and onset-death
        added = '-remove0'
    else:
        added = ''
    state_posteriors_gamma.update({col: pd.read_csv(SAMPLES_PATH + col +'-samples-gamma' + added + '.csv')})
    state_posteriors_wei.update({col: pd.read_csv(SAMPLES_PATH + col +'-samples-wei' + added + '.csv')})
    state_posteriors_lognorm.update({col: pd.read_csv(SAMPLES_PATH + col +'-samples-lognormal' + added + '.csv')})
    try:
        state_posteriors_gln.update({col: pd.read_csv(SAMPLES_PATH + col + '-samples-gln' + added + '.csv')})
    except:
        print(col, 'did not find samples for gln')
        state_posteriors_gln.update({col: state_posteriors_gln[columns[0]] * np.nan})
    try:
        state_posteriors_gg.update({col: pd.read_csv(SAMPLES_PATH + col + '-samples-gg' + added + '.csv').drop(columns = ['draw', 'accept_stat__', 'divergent__'])})
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
    return {'mu': theta_mean, 'cov': cov, 'distr': distr}

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
    posterior_gg.update({col: fit_posterior('gg', col)})
    posterior_gln.update({col: fit_posterior('gln', col)})

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


##############################################################################
r = 3 # rounding
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
    return {'mean': round(mean.mean(),r), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),r), 'var_CI95': var_CI95}

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
    return {'mean': round(mean.mean(),r), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),r), 'var_CI95': var_CI95}

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
    return {'mean': round(mean.mean(),r), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),r), 'var_CI95': var_CI95}

def inf_sum(r, mu, sig, g):
    result = 0
    tmp = 0
    for j in range(1,151): # 100 should be enough to get this infinite sum but test it; gamma(170) = inf
        tmp = (r * sig)**j
        tmp = tmp * (1 + (-1)**j) * 2**(j/g)
        tmp = tmp * (scipy.special.gamma((j+1)/g) / scipy.special.gamma(j+1))
        result += tmp
    # this sometimes diverges, so check the last tmp and see if it was small enough
    if isinstance(tmp, float):
        if tmp >=  0.0000001:
            return np.nan
    if tmp.any() >= 0.0000001:
        result[tmp > 0.0000001] = np.nan
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
    mean = mean[~np.isnan(mean)]
    var = var[~np.isnan(var)]
    if mean.size < 1000:        
        return {'mean': np.nan, 'mean_CI95': np.nan,
            'var': np.nan, 'var_CI95': np.nan}
    if var.size < 1000:
        mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
        return {'mean': round(mean.mean(),r), 'mean_CI95': mean_CI95, 
                'var': np.nan, 'var_CI95': np.nan}
    mean_CI95 = tuple(np.round(np.quantile(mean,[.025,.975]),r))
    var_CI95 = tuple(np.round(np.quantile(var,[.025,.975]),r))
    return {'mean': round(mean.mean(),r), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),r), 'var_CI95': var_CI95}


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
    return {'mean': round(mean.mean(),r), 'mean_CI95': mean_CI95,
            'var': round(var.mean(),r), 'var_CI95': var_CI95}

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
# get tables with outputs

results = pd.DataFrame(columns = ['dataset', 'model', 'state',
                                  'mean', 'mean_CI95', 'var', 'var_CI95', 
                                  'param1', 'param1_CI95',
                                  'param2', 'param2_CI95', 
                                  'param3', 'param3_CI95'])
models = [state_posteriors_gamma, state_posteriors_wei, state_posteriors_lognorm,
          state_posteriors_gg, state_posteriors_gln]
models_names = ['Gamma', 'Weibull', 'Log-normal', 'Gen. Gamma', 'Gen. Log-normal']

for col in columns:
    print(col)
    for im in range(len(models_names)):
        m = models[im][col]
        name = models_names[im]
        # get stats for each state
        for i in range(1,28):
            # get columns for the i-th state
            str_it = '[' + str(i) + ']'
            state_cols = [col for col in m.columns if str_it in col]
            # get parameters means for i-th state
            parameter_means = round(m.loc[:,state_cols].mean(axis=0),r)
            parameter_CI95 = round(m.loc[:,state_cols].quantile([.025, .975],axis=0),r) # this is a df
            # get mean and variance for i-th state
            stats_state = stats(name, m.loc[:,state_cols])
            # append all to the summary table
            stats_state.update({'dataset': col, 'model': name, 'state': i,
                                'param1': parameter_means[0], 'param2': parameter_means[1],
                                'param1_CI95': tuple(parameter_CI95.iloc[:,0]),
                                'param2_CI95': tuple(parameter_CI95.iloc[:,1])})
            if len(state_cols) == 3:
                stats_state.update({'param3': parameter_means[2],
                                    'param3_CI95': tuple(parameter_CI95.iloc[:,2])})
            results = results.append(stats_state, ignore_index=True)
        # add brazil
        if im < 3: # gamma, weibull, lognormal have two parameters
            stats_nat = stats(name, m.iloc[:,-2:])
            parameter_CI95 = round(m.iloc[:,-2:].quantile([.025, .975],axis=0),r)
            stats_nat.update({'param1': round(m.iloc[:,-2].values.mean(),r),
                              'param1_CI95': tuple(parameter_CI95.iloc[:,0])})
            stats_nat.update({'param2': round(m.iloc[:,-1].values.mean(),r),
                              'param2_CI95': tuple(parameter_CI95.iloc[:,1])})
        else:
            stats_nat = stats(name, m.iloc[:,-3:]) # generalised have 3 parameters
            parameter_CI95 = round(m.iloc[:,-3:].quantile([.025, .975],axis=0),r)
            stats_nat.update({'param1': round(m.iloc[:,-3].values.mean(),r),
                              'param1_CI95': tuple(parameter_CI95.iloc[:,0])})
            stats_nat.update({'param2': round(m.iloc[:,-2].values.mean(),r),
                              'param2_CI95': tuple(parameter_CI95.iloc[:,1])})
            stats_nat.update({'param3': round(m.iloc[:,-1].values.mean(),r),
                              'param3_CI95': tuple(parameter_CI95.iloc[:,2])})
            
        stats_nat.update({'dataset': col, 'model': name, 'state': 'Brazil'})
        results = results.append(stats_nat, ignore_index=True)
        
state_map.update({'Brazil': 'Brazil'})
int_to_state = {value:key for key, value in state_map.items()}
results['state'] = results['state'].map(int_to_state)
        
results.to_csv(OUT_PATH + 'results_full_table.csv', index = False)

# or just read in
# results = pd.read_csv(OUT_PATH + 'results_full_table.csv')
##############################################################################
# get the onset to death parameters for the Rt estimation model


def glue_two_cols(df, col1, col2):
    vcol1 = df[col1].values
    vcol2 = df[col2].values
    glued_str = []
    for i in range(len(vcol1)):
        glued_str.append(str(vcol1[i]) + ' ' + str(vcol2[i]))
    df[col1] = glued_str
    df.drop(columns=col2, inplace=True)
    return df
    
oDeathParams = results[results['model'] == 'Gamma']
oDeathParams = oDeathParams[oDeathParams['dataset'] == 'onset-to-death']
oDeathParams.to_csv(OUT_PATH + 'OnsetDeathParams.csv', index=False)
oDeathParams.drop(columns = ['model', 'dataset', 'param3', 'param3_CI95'], inplace = True)
oDeathParams = glue_two_cols(oDeathParams, 'mean', 'mean_CI95')
oDeathParams = glue_two_cols(oDeathParams, 'var', 'var_CI95')
oDeathParams = glue_two_cols(oDeathParams, 'param1', 'param1_CI95')
oDeathParams = glue_two_cols(oDeathParams, 'param2', 'param2_CI95')
oDeathParams.to_csv(OUT_PATH + 'OnsetDeathParamsStatesForManuscript.csv', index=False)

##############################################################################
# create a table for the manuscript
best_fit_map = {'ICU-stay': 'Gamma',
                'onset-to-death': 'Gamma', 
                'Hospital-Admission-to-death': 'Gamma',
                'onset-to-diagnosis': 'Gen. Log-normal',
                'onset-to-hospital-discharge': 'Gen. Log-normal',
                'onset-to-hospital-admission': 'Gen. Log-normal',
                'onset-to-ICU-admission': 'Gen. Log-normal',
                'onset-to-diagnosis-pcr': 'Gen. Log-normal'}

def round_and_glue_val_and_ci95(val, ci95, r=2):
    """takes the value as a number, ci95 as a tuple,
    rounds each value with r-decimal places,
    ang returns a string of 'val (ci1, ci2)'"""
    val = round(val,r)
    ci95 = (round(ci95[0],r), round(ci95[1],r))
    val_str = str(val) + ' ' + str(ci95)
    return val_str

def cut_results_df(col):
    model = best_fit_map[col]
    df = results[(results['dataset'] == col) & (results['model'] == model) & (results['state'] == 'Brazil')]
    res = df.iloc[0,:].to_dict()
    res['mean'] = round_and_glue_val_and_ci95(res['mean'], res['mean_CI95'])
    res['var'] = round_and_glue_val_and_ci95(res['var'], res['var_CI95'])
    res['param1'] = round_and_glue_val_and_ci95(res['param1'], res['param1_CI95'])
    res['param2'] = round_and_glue_val_and_ci95(res['param2'], res['param2_CI95'])
    if ~np.isnan(res['param3']):
        res['param3'] = round_and_glue_val_and_ci95(res['param3'], res['param3_CI95'])
    return res

summary = pd.DataFrame(columns = results.columns)
for col in columns:
    summary = summary.append(cut_results_df(col), ignore_index=True)

summary.drop(columns = ['state', 'mean_CI95', 'var_CI95', 'param1_CI95', 'param2_CI95', 'param3_CI95'],
             inplace = True)
summary = summary.sort_values(by=['model'])
summary.to_csv(OUT_PATH + 'summaryBestModelsNational.csv', index=False)
##############################################################################
# save the means posteriors
best_fit_map = {'ICU-stay': 'Gamma',
                'onset-to-death': 'Gamma', 
                'Hospital-Admission-to-death': 'Gamma',
                'onset-to-diagnosis': 'Gen. Log-normal',
                'onset-to-hospital-discharge': 'Gen. Log-normal',
                'onset-to-hospital-admission': 'Gen. Log-normal',
                'onset-to-ICU-admission': 'Gen. Log-normal',
                'onset-to-diagnosis-pcr': 'Gen. Log-normal'}
models = {'Gamma': state_posteriors_gamma, 'Weibull': state_posteriors_wei,
          'Log-normal': state_posteriors_lognorm, 'Gen. Gamma': state_posteriors_gg,
          'Gen. Log-normal': state_posteriors_gln}
# for each state
def save_all_means_per_column(states, col):
    model = models[best_fit_map[col]][col] # posterior samples
    all_means = pd.DataFrame(columns = states)
    for state in states:
        state_id = str(state_map[state])
        str_it = '[' + state_id + ']'
        state_cols = [col for col in model.columns if str_it in col]
        means, var = stats(best_fit_map[col], model.loc[:,state_cols], dict_out=False)
        all_means[state] = means
        if col == 'onset-to-diagnosis':
            all_means['AC'] = np.nan
    all_means.to_csv(OUT_PATH + '/MeansPosteriors/' + col + '.csv', index=False)
        
for col in columns:
    save_all_means_per_column(states_geo_order, col)
        
# for the whole country
def save_all_means_per_column_Brazil():
    all_means = pd.DataFrame(columns = columns)
    for col in columns:
        model = models[best_fit_map[col]][col] # posterior samples        
        if best_fit_map[col] in ['Gamma', 'Weibull', 'Log-normal']:
            means, var = stats(best_fit_map[col], model.iloc[:,-2:], dict_out=False)
        else:
            means, var = stats(best_fit_map[col], model.iloc[:,-3:], dict_out=False)
        all_means[col] = means
    all_means.to_csv(OUT_PATH + '/MeansPosteriors/' + 'Brazil.csv', index=False)
    
save_all_means_per_column_Brazil()
##############################################################################
# get the distributions pf means for the soc-eco analysis
best_fit_map = {'ICU-stay': 'Gamma',
                'onset-to-death': 'Gamma', 
                'Hospital-Admission-to-death': 'Gamma',
                'onset-to-diagnosis': 'Gen. Log-normal',
                'onset-to-hospital-discharge': 'Gen. Log-normal',
                'onset-to-hospital-admission': 'Gen. Log-normal',
                'onset-to-ICU-admission': 'Gen. Log-normal',
                'onset-to-diagnosis-pcr': 'Gen. Log-normal'}

models = {'Gamma': state_posteriors_gamma, 'Weibull': state_posteriors_wei,
          'Log-normal': state_posteriors_lognorm, 'Gen. Gamma': state_posteriors_gg,
          'Gen. Log-normal': state_posteriors_gln}
int_to_state = {value:key for key, value in state_map.items()}

means_distributions = pd.DataFrame(columns = columns)

for i in range(1,28):
    stats_means = {'state': int_to_state[i], 'stat': 'mean'}
    stats_var = {'state': int_to_state[i], 'stat': 'var'}

    # get columns for the i-th state
    str_it = '[' + str(i) + ']'
    for col in columns:
        distr_name = best_fit_map[col]
        m = models[distr_name][col]
        state_cols = [col for col in m.columns if str_it in col]
        
        mean, var = stats(distr_name, m.loc[:,state_cols], dict_out=False)
        mean_mean = mean.mean() # this is a mean of a distribution of means
        mean_var = mean.var() # this is a variance of a distribution of means. SORRY!
            
        # append all to the summary table
        stats_means.update({col: mean_mean})
        stats_var.update({col: mean_var})
        
    means_distributions = means_distributions.append(stats_means, ignore_index=True)
    means_distributions = means_distributions.append(stats_var, ignore_index=True)


means_distributions.to_csv(OUT_PATH + 'meansDistributions.csv', index = False)

#############################################################################
def util_col_name(col):
    if col == 'ICU-stay':
        return col;
    if col == 'onset-to-ICU-admission':
        return 'Onset-to-ICU-admission';
    if col == 'onset-to-diagnosis-pcr':
        return 'Onset-to-diagnosis (PCR)'
    if col == 'onset-to-diagnosis':
        return 'Onset-to-diagnosis (non-PCR)'
    if col == 'onset-to-death':
        return 'Onset-to-death'
    return col.capitalize()
    
##############################################################################
# plots
def plot_model_fit_all_models(col_id, xmax=None, ylab = '', right_ylab='', x_lab=''):
    col = columns[col_id]
    alpha_plot = 1
    if xmax:
        max_value = xmax
    else:
        max_value = all_dfs[col_id][col].values.max()
    stdata = all_dfs[col_id][col].values + 0.5  
    stdata = stdata[stdata <= max_value+0.5]
    bins = range(0,max_value+1)
    sns.distplot(stdata, kde=False, norm_hist=True, bins=bins)
    # plt.hist(stdata, density=True, cumulative=True, bins=bins)
    x = np.linspace(0, max_value, 200)

    df = state_posteriors_gamma[col]
    alpha = df.alpha.mean()
    beta = df.beta.mean()
    y = scipy.stats.gamma(a=alpha, scale = 1.0/beta).pdf(x)
    plt.plot(x, y, label = 'gamma', color = 'red', alpha = alpha_plot)
        
    df = state_posteriors_wei[col]
    alpha = df.alpha.mean()
    sigma = df.sigma.mean()
    y = scipy.stats.weibull_min(c=alpha, scale = sigma).pdf(x)
    plt.plot(x, y, label = 'weibull', color = 'green', alpha = alpha_plot)
    
    df = state_posteriors_lognorm[col]
    mu = df.mu.mean()
    sigma = df.sigma.mean()
    y = scipy.stats.lognorm(s=sigma, scale = np.exp(mu)).pdf(x)
    plt.plot(x, y, label = 'lognorm', color = 'brown', alpha = alpha_plot)

    df = state_posteriors_gln[col]
    mu = df.mu.mean()
    sigma = df.sigma.mean()
    g = df.g.mean()
    y = gln_pdf(x, mu, sigma, g)
    plt.plot(x, y, label = 'generalised lognormal', color = 'blue', alpha = alpha_plot)
 
    df = state_posteriors_gg[col]
    a = df.a.mean()
    d = df.d.mean()
    p = df.p.mean()
    y = gg_pdf(x, a, d, p)
    plt.plot(x, y, label = 'generalised gamma', color = 'magenta', alpha = alpha_plot)
    
    plt.xlim([0, max_value])
    # plt.xlabel(util_col_name(col))
    plt.xlabel('Days')
    plt.title(util_col_name(col))
    plt.ylabel(ylab)
    plt.ylim([0,0.125])
    # plt.grid(None)  # this will make the grid for the primary y-axis disappear
    
    plt.gca()
    plt.twinx()
    
    
    if best_fit_map[col] == 'Gamma':
        df = state_posteriors_gamma[col]
        alpha = df.alpha.mean()
        beta = df.beta.mean()
        y = scipy.stats.gamma(a=alpha, scale = 1.0/beta).cdf(x)
    elif best_fit_map[col] == 'Gen. Log-normal':
        df = state_posteriors_gln[col]
        mu = df.mu.mean()
        sigma = df.sigma.mean()
        g = df.g.mean()
        y = gln_cdf(x, mu, sigma, g)
        
    plt.plot(x, y, label = 'gamma', color = 'black', alpha = 0.7, ls = '--')
    
    if right_ylab:
        plt.ylabel('Cumulative')
    
    plt.grid(None) # this will make the grid for the secondary y-axis disappear
    plt.ylim([0,1.05])
    # plt.show()

def plot_legend():
    x = np.linspace(0,10)
    plt.plot(x*0, x, label= 'Gamma', c='r', linewidth=4)
    plt.plot(x*0, x, label= 'Weibull', c='g', linewidth=4)
    plt.plot(x*0, x, label= 'Log-normal', c='brown',linewidth=4)
    plt.plot(x*0, x, label= 'Gen. log-normal', c='blue',linewidth=4)
    plt.plot(x*0, x, label= 'Gen. gamma', c='magenta',linewidth=4)
    plt.xlim([0.5,10])
    plt.legend(prop={'size': 12})
    plt.gca().axison = False

xmax=40
plt.figure(figsize=(10,10))
plt.subplot(3,3,1)
plot_model_fit_all_models(1, xmax=xmax, ylab='Probability') 
plt.subplot(3,3,2)
plot_model_fit_all_models(6, xmax=xmax)
plt.subplot(3,3,3)
plot_model_fit_all_models(0, xmax=xmax, right_ylab=True) 
plt.subplot(3,3,4)
plot_model_fit_all_models(3, xmax=xmax, ylab='Probability')
plt.subplot(3,3,5)
plot_model_fit_all_models(4, xmax=xmax)
plt.subplot(3,3,6)
plot_model_fit_all_models(5, xmax=xmax, right_ylab=True, x_lab='days')
plt.subplot(3,3,7)
plot_model_fit_all_models(7, xmax=xmax, ylab= 'Probability', x_lab='days')
plt.subplot(3,3,8)
plot_model_fit_all_models(2, xmax=xmax, ylab= '', right_ylab=True, x_lab='days')
plt.subplot(3,3,9)
plot_legend()
plt.tight_layout()
plt.savefig(OUT_PATH + 'modelFitsAll.pdf', format='pdf')





##############################################################################
# for onset to death (Gamma) plot fits to Brazil and 5 states
states5 = ['Brazil', 'SP', 'RJ', 'AM', 'MA', 'RO']
name_map = {'Brazil': 'Brazil', 'SP': 'São Paulo', 
            'RJ': 'Rio de Janeiro', 'AM': 'Amazonas',
            'MA': 'Maranhão', 'RO': 'Rondônia'}
state_map.update({'Brazil': 'Brazil'})

def plot_model_fit_gamma(state, col, show_x_label=False):
    # state is the letter code or 'Brazil'
    max_value=70
    state_id = str(state_map[state])
    col_id = columns.index(col)
    data = all_dfs[col_id]
    df = state_posteriors_gamma[col]
    if state == 'Brazil':
        alpha_hat = df['alpha'].mean() # national fit
        beta_hat = df['beta'].mean() # national fit
        alpha_s = df.loc[::10, 'alpha'].values
        beta_s = df.loc[::10, 'beta'].values
    else:
        alpha_hat = df['alpha[' + state_id + ']'].mean()
        beta_hat = df['beta[' + state_id + ']'].mean()
        alpha_s = df.loc[::10, 'alpha[' + state_id + ']'].values
        beta_s = df.loc[::10, 'beta[' + state_id + ']'].values
    if state == 'Brazil':
        stdata = data[col].values
    else:
        stdata = data[data['State'] == state][col].values
    stdata = stdata[stdata <= max_value]
    x = np.linspace(0, max_value+0.5, 200)
    y = scipy.stats.gamma(a=alpha_hat, scale = 1.0/beta_hat).pdf(x)
    sns.distplot(stdata, label = 'original data', kde=False, norm_hist=True, bins=max_value)
    plt.plot(x, y, label = 'fitted model', color = 'red', alpha = 1)
    for i in range(len(alpha_s)):
        alpha = alpha_s[i]
        beta = beta_s[i]
        y_s = scipy.stats.gamma(a=alpha, scale = 1.0/beta).pdf(x)
        plt.plot(x, y_s, alpha = 0.01, color='y')
    plt.xlim([0,max_value])
    col_replaced = util_col_name(col)
    
    if show_x_label:
        plt.xlabel(col_replaced + ' (days)')#,  fontsize=16)
        
    plt.annotate(name_map[state], xy=(0.6, 0.7), xycoords='axes fraction', fontsize=16)
    
    plt.ylabel('Probability')


def get_means_and_vars_gamma(state, col):
    df = state_posteriors_gamma[col]
    state_id = str(state_map[state])
    if state == 'Brazil':
        # mean, var = stats_gamma_full_output(df.iloc[:,-2:])
        mean, var = stats_gamma(df.iloc[:,-2:], dict_out=False)
        return mean, var
    else:
        str_it = '[' + state_id + ']'
        state_cols = [col for col in df.columns if str_it in col]
        # mean, var = stats_gamma_full_output(df.loc[:,state_cols])
        mean, var = stats_gamma(df.loc[:,state_cols], dict_out=False)
        return mean, var
    
    
j = 0
col = 'onset-to-death'
plt.figure(figsize=(10,10))
for i in [0,3,6,9,12]:
    state = states5[j]
    gs = gridspec.GridSpec(6,3, width_ratios=[2, 1, 1]) 
    # plt.subplot(6, 3, i+1)
    plt.subplot(gs[i])
    plot_model_fit_gamma(state, col)
    # plt.subplot(6, 3, i+2)
    plt.subplot(gs[i+1])
    mean, var = get_means_and_vars_gamma(state, col)
    sns.kdeplot(mean)
    # plt.title(name_map[state])#,fontsize = 22)
    # plt.xlabel('mean')
    plt.xlim([12,17])
    # plt.subplot(6, 3, i+3)
    plt.subplot(gs[i+2])
    sns.kdeplot(var)
    # plt.xlabel('variance')
    plt.xlim([50,140])
    j = j+1
i = 15 
state = states5[j]
gs = gridspec.GridSpec(6,3, width_ratios=[2, 1, 1]) 
# plt.subplot(6, 3, i+1)
plt.subplot(gs[i])
plot_model_fit_gamma(state, col, show_x_label=True)
# plt.subplot(6, 3, i+2)
plt.subplot(gs[i+1])
mean, var = get_means_and_vars_gamma(state, col)
sns.kdeplot(mean)
# plt.title(name_map[state])#,fontsize = 22)
plt.xlabel('Mean (days)')
plt.xlim([12,17])
# plt.subplot(6, 3, i+3)
plt.subplot(gs[i+2])
sns.kdeplot(var)
plt.xlabel('Variance (days$^2$)')
plt.xlim([50,140])
plt.tight_layout()

plt.savefig(OUT_PATH + 'onsetToDeathMeanVar.pdf', format='pdf')


##############################################################################
# plot boxplots with Geographical location

geogr_location_map = {'North': ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],
                      'Northeast': ['AL', 'BA', 'CE', 'MA', 'PB', 'PI', 'PE', 'SE', 'RN'],
                      'Central-West': ['DF', 'GO', 'MS', 'MT'],
                      'Southeast': ['ES', 'MG', 'RJ', 'SP'],
                      'South': ['PR', 'RS', 'SC']}

states_geo_order = list(geogr_location_map.values())
states_geo_order = [val for sublist in states_geo_order for val in sublist]

def plot_means_boxplots(states, col, colour = 'k', show_ylabel=False):
    model = models[best_fit_map[col]][col] # posterior samples
    all_means = pd.DataFrame(columns = states)
    for state in states:
        state_id = str(state_map[state])
        str_it = '[' + state_id + ']'
        state_cols = [col for col in model.columns if str_it in col]
        means, var = stats(best_fit_map[col], model.loc[:,state_cols], dict_out=False)
        all_means[state] = means
        # remove AC for onset-to-diagnosis
        if col == 'onset-to-diagnosis':
            all_means['AC'] = np.nan
    
    my_pal = ['b', 'b', 'b', 'b', 'b', 'b', 'b',
              'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue',
              'g', 'g', 'g', 'g',
              'orange', 'orange', 'orange', 'orange',
              'r', 'r', 'r'] 
    sns.boxplot(x="variable", y="value", data=pd.melt(all_means), showfliers=False,  palette=my_pal, boxprops=dict(alpha=1))
    plt.xlabel('')
    plt.grid()
    plt.title(util_col_name(col), fontsize = 14)
    

plt.figure(figsize=(7,7))
plt.subplot(1,2,1)
plot_means_boxplots(states, 'onset-to-death', 'r', show_ylabel=True)
plt.subplot(1,2,2)
plot_means_boxplots(states, 'ICU-stay', 'b')
plt.tight_layout

columns_order = [1,6,0,3,4,5,7,2]

plt.figure(figsize=(12,15)) # do not change that ortherwise annotations will break
for i in range(len(columns_order)):
    plt.subplot(8,1,i+1)
    plot_means_boxplots(states_geo_order, columns[columns_order[i]])
    plt.axvline(x=6.5, ls = '--', c= 'k')
    plt.axvline(x=15.5, ls = '--', c= 'k')
    plt.axvline(x=19.5, ls = '--', c= 'k')
    plt.axvline(x=23.5, ls = '--', c= 'k')
    if i == 7:
        y = 55
        plt.annotate('North', xy=(120, y), xycoords='figure pixels', fontsize=14)
        plt.annotate('Northeast', xy=(350, y), xycoords='figure pixels', fontsize=14)
        plt.annotate('Central-West', xy=(530,y), xycoords='figure pixels', fontsize=14)
        plt.annotate('Southeast', xy=(650, y), xycoords='figure pixels', fontsize=14)
        plt.annotate('South', xy=(770, y), xycoords='figure pixels', fontsize=14)
plt.tight_layout()
plt.savefig(OUT_PATH + 'boxPlotsMeansLocationAllStates.pdf', format='pdf')


