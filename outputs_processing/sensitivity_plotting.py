# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:20:23 2020

@author: iwona
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

PATH = '../results/MeanPosteriors'

cols_0 = ['onset-to-hospital-admission', 'onset-to-ICU-admission', 'Hospital-Admission-to-death', 'onset-to-death']
cols_censor = ['ICU-stay', 'onset-to-death', 'Hospital-Admission-to-death']

basePosteriors = pd.read_csv( 'results/MeansPosteriors/Brazil.csv')
basePosteriors = basePosteriors.melt()
basePosteriors['label'] = '1st day removed'
basePosteriors['label'][basePosteriors['variable']=='Hospital-Admission-to-death'] = '1st day not removed'

censoredPosteriors = pd.read_csv( 'results/MeansPosteriors/BrazilCensored.csv')
censoredPosteriors = censoredPosteriors.melt()
censoredPosteriors['label'] = 'Censoring corrected'


zeroesPosteriors = pd.read_csv( 'results/MeansPosteriors/BrazilWithAlternativeBottomCut.csv')
zeroesPosteriors = zeroesPosteriors.melt()
zeroesPosteriors['label'] = '1st day not removed'
zeroesPosteriors['label'][zeroesPosteriors['variable']=='Hospital-Admission-to-death'] = '1st day removed'

sensData = pd.concat([basePosteriors,censoredPosteriors,zeroesPosteriors], axis=0)

cols_remove = ['onset-to-diagnosis', 'onset-to-diagnosis-pcr', 'onset-to-hospital-discharge']#, 'onset-to-death']
sensData = sensData[~sensData['variable'].isin(cols_remove)]
sensData.replace({'variable':  {'onset-to-hospital-admission': 'Onset-to-hospital-admission',
                             'onset-to-death': 'Onset-to-death',
                             'onset-to-ICU-admission': 'Onset-to-ICU-admission',
                             'Hospital-Admission-to-death': 'Hospital-admission-to-death'}}, inplace=True)

sensGroupped = sensData.groupby(['variable','label']).agg({'value': ['mean']})
sensGroupped.columns = sensGroupped.columns.droplevel()
sensGroupped = sensGroupped.reset_index()



plt.figure(figsize=(12,5))
############################################################
# removing errors
plt.subplot(1,2,1)
dataErrors = pd.concat([basePosteriors,zeroesPosteriors], axis=0)
cols_remove = ['onset-to-diagnosis', 'onset-to-diagnosis-pcr', 'onset-to-hospital-discharge']
dataErrors = dataErrors[dataErrors['variable'].isin(cols_0)]
dataErrors.replace({'variable':  {'onset-to-hospital-admission': 'Onset-to-hospital-admission',
                             'onset-to-death': 'Onset-to-death',
                             'onset-to-ICU-admission': 'Onset-to-ICU-admission',
                             'Hospital-Admission-to-death': 'Hospital-admission-to-death'}}, inplace=True)
dataErrors = dataErrors.groupby(['variable','label']).agg({'value': ['mean']})
dataErrors = dataErrors.sort_values(by=['label'], ascending=False)
dataErrors.columns = dataErrors.columns.droplevel()
dataErrors = dataErrors.reset_index()
ax = sns.scatterplot(y="variable", x="mean", hue="label", data=dataErrors, 
                alpha=1, edgecolors='black', s=300)
plt.xlabel('')
plt.ylabel('')
h,l = ax.get_legend_handles_labels()
plt.legend(h[1:],l[1:], loc='upper right')
plt.xlabel('Days')

############################################################
# censoring corrected
plt.subplot(1,2,2)
dataCensoring = pd.concat([basePosteriors,censoredPosteriors], axis=0)
cols_remove = ['onset-to-diagnosis', 'onset-to-diagnosis-pcr', 'onset-to-hospital-discharge']#, 'onset-to-death']
dataCensoring = dataCensoring[dataCensoring['variable'].isin(cols_censor)]
dataCensoring.replace({'variable':  {'onset-to-hospital-admission': 'Onset-to-hospital-admission',
                             'onset-to-death': 'Onset-to-death',
                             'Hospital-Admission-to-death': 'Hospital-admission-to-death'}}, inplace=True)
dataCensoring['label'][dataCensoring['label']!='Censoring corrected'] = 'Censoring'
dataCensoring = dataCensoring.groupby(['variable','label']).agg({'value': ['mean']})
dataCensoring.columns = dataCensoring.columns.droplevel()
dataCensoring = dataCensoring.reset_index()
ax = sns.scatterplot(y="variable", x="mean", hue="label", data=dataCensoring, 
                alpha=1, edgecolors='black', s=300)
plt.xlabel('Days')
plt.ylabel('')
h,l = ax.get_legend_handles_labels()
plt.legend(h[1:],l[1:], loc='lower right')
plt.tight_layout()
plt.savefig('results/' + 'sensitivityPlot.pdf', format='pdf', bbox_inches='tight')




# 90th Percentile
def quant(x,p):
            return x.quantile(p)
def quant025(x):
    return quant(x,0.025)
def quant975(x):
    return quant(x,0.975)

df = sensData.copy()
quantiles = df.groupby(['variable','label']).quantile([0.025, 0.95])
quantiles.columns = quantiles.columns.droplevel()
quantiles = quantiles.reset_index()
quantiles.rename(columns = {'level_2': 'quantile'}, inplace=True)


sensGroupped['quantile'] = 'mean'
sensGroupped.rename(columns = {'mean': 'value'}, inplace=True)

all_sensitivity_outputs = pd.concat([quantiles, sensGroupped])
all_sensitivity_outputs.to_csv('results/' + 'sensitivityOutputs.csv', index=False)





