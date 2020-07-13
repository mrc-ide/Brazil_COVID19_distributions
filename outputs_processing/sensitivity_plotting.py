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

cols_0 = ['onset-to-hospital-admission', 'onset-to-ICU-admission', 'Hospital-Admission-to-death']
cols_censor = ['ICU-stay', 'onset-to-death', 'Hospital-Admission-to-death']

basePosteriors = pd.read_csv( 'results/MeansPosteriors/Brazil.csv')
basePosteriors = basePosteriors.melt()
basePosteriors['label'] = '1st day removed'
basePosteriors['label'][basePosteriors['variable']=='Hospital-Admission-to-death'] = '1st day included'

censoredPosteriors = pd.read_csv( 'results/MeansPosteriors/BrazilCensored.csv')
# censoredPosteriors = censoredPosteriors.add_suffix('_censored')
censoredPosteriors = censoredPosteriors.melt()
censoredPosteriors['label'] = 'Censoring'


zeroesPosteriors = pd.read_csv( 'results/MeansPosteriors/BrazilWithAlternativeBottomCut.csv')
# zeroesPosteriors = zeroesPosteriors.add_suffix('_alternativeBottom')
zeroesPosteriors = zeroesPosteriors.melt()
zeroesPosteriors['label'] = '1st day included'
zeroesPosteriors['label'][zeroesPosteriors['variable']=='Hospital-Admission-to-death'] = '1st day removed'

sensData = pd.concat([basePosteriors,censoredPosteriors,zeroesPosteriors], axis=0)

cols_remove = ['onset-to-diagnosis', 'onset-to-diagnosis-pcr', 'onset-to-hospital-discharge']#, 'onset-to-death']
sensData = sensData[~sensData['variable'].isin(cols_remove)]


sensGroupped = sensData.groupby(['variable','label']).agg({'value': ['mean']})
sensGroupped.columns = sensGroupped.columns.droplevel()
sensGroupped = sensGroupped.reset_index()


sns.scatterplot(y="variable", x="mean", hue="label", data=sensGroupped, alpha=0.7,
                edgecolors='black', s=300)
plt.xlabel('mean time (days)')
plt.ylabel('')
plt.xticks(np.arange(7, 16.5, 1.0))

plt.savefig('../results/' + 'sensitivityPlot.pdf', format='pdf', bbox_inches='tight')


# 90th Percentile
def quant(x,p):
            return x.quantile(p)
def quant025(x):
    return quant(x,0.025)
def quant975(x):
    return quant(x,0.975)

my_DataFrame.groupby(['AGGREGATE']).agg({'MY_COLUMN': [q50, q90, 'max']})


quantiles = df.groupby(['variable','label']).quantile([0.025, 0.95])
quantiles.columns = quantiles.columns.droplevel()
quantiles = quantiles.reset_index()
quantiles.rename(columns = {'level_2': 'quantile'}, inplace=True)


sensGroupped['quantile'] = 'mean'
sensGroupped.rename(columns = {'mean': 'value'}, inplace=True)

all_sensitivity_outputs = pd.concat([quantiles, sensGroupped])
all_sensitivity_outputs.to_csv('../results/' + 'sensitivityOutputs.csv', index=False)





