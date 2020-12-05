# Brazil_COVID19_distributions 




https://github.com/mrc-ide/Brazil_COVID19_distributions [![DOI](https://zenodo.org/badge/279160599.svg)](https://zenodo.org/badge/latestdoi/279160599)

## Abstract:

Knowing COVID-19 epidemiological distributions, such as the time from patient admission to death, is directly relevant to effective primary and secondary care planning, and moreover, the mathematical modelling of the pandemic generally.\cite{flaxman2020nature} Here we determine epidemiological distributions for patients hospitalised with COVID-19 using a large dataset (range N=21,000-157,000) from the Brazilian SIVEP-Gripe (*Sistema  de  Informação  de  Vigilância  Epidemiológica  da  Gripe*) database.\cite{SIVEP} We fit a set of probability distribution functions and estimate a symptom-onset-to-death mean of 15.2 days for Brazil, which is lower than earlier estimates of 17.8 days based on early Chinese data.\cite {verity_estimates_2020} A joint Bayesian subnational model is used to simultaneously describe the 26 states and one federal district of Brazil, and shows significant variation in the mean of the symptom-onset-to-death time, with ranges between 11.2-17.8 days across the different states. We find strong evidence in favour of specific probability distribution function choices: for example, the gamma distribution gives the best fit for onset-to-death and the generalised log-normal for onset-to-hospital-discharge. Our results show that epidemiological distributions have considerable geographical variation, and provide the first estimates of these distributions in a low and middle-income setting. At the subnational level, variation in COVID-19 outcome timings are found to be correlated with poverty, deprivation and segregation levels, and weaker correlation is observed for mean age, wealth and urbanicity.

[Inference of COVID-19 epidemiological distributions from Brazilian hospital dataJ. R. Soc. Interface.1720200596](https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2020.0596)

## Repository description:

This code and data were used for the analysis presented in "Inference of COVID-19 epidemiological distributions from Brazilian hospital data" (link to preprint).

The repository contains the following elements:

1. *data*

   Contains all data used for analysis, that is:

   * [SIVEP-Gripe hospitalisation datasets](https://opendatasus.saude.gov.br/dataset/bd-srag-2020) (latest download 7th July 2020)
   * socioeconomic indicators data: [GeoSES](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232074#pone.0232074.s002), [income_data](https://www.ibge.gov.br/estatisticas/sociais/populacao/9109-projecao-da-populacao.html?=&t=resultados), and urbanicity data

2.  *python_scripts_fitting*

   Contains the scripts for fitting Bayesian hierarchical models to the SIVEP-Gripe dataset, using [PyStan](https://mc-stan.org/users/interfaces/pystan). Models fitted are gamma, Weibull, log-normal, generalised log-normal and generalised gamma. The fitting might be very slow for some of the models given a huge amount of data, therefore we recommend running the fits in parallel.

3. *fitting_outputs*

   Contains the outputs of the MCMC sampling per dataset and per model, generated during the model fitting process.

4. *outputs_processing*

   Contains python and R scripts for analysis of the model fitting outputs, creating figures and collating the outputs in the figures.

5. *results*

   Contains the collated and processed results of the analysis.

## Running the code

In this example we show how to run the model fitting for all distributions (onset-to-death, onset-to-diagnosis, onset-to-diagnosis-pcr, onset-to-hospital-admission, onset-to-ICU-admission, ICU-stay) for gamma probability distribution function (PDF).

1. Run the python script *pythons_scripts_fitting/gamma.py*

2. The results of the hierarchical model fitting, that is a sampled parameter value for each state and the whole country per iteration, will be saved in *fitting_outputs* folder. The results will be save in the form *distribution_name-gamma-samples.csv*, that is *gamma.py* script will generate *ICU-stay-gamma-samples.csv*, *onset-to-death-gamma-samples.csv*, etc.

3. The format of each of the generated csv files is as follows:

   | alpha[1] | alpha[2] | ...  | beta[1] | beta[2] | ...  | sigma_alpha | sigma_beta | alpha | beta |
   | -------- | -------- | ---- | ------- | ------- | ---- | ----------- | ---------- | ----- | ---- |
   |          |          |      |         |         |      |             |            |       |      |
   |          |          |      |         |         |      |             |            |       |      |
   |          |          |      |         |         |      |             |            |       |      |

   Each row of the table shows values generated in each iteration (one row of the table) of the Markov Chain Monte Carlo sampling conducted during the model fitting process. Here alpha[1], beta[1], alpha[2] , beta[2], .. are state-level parameters, where [1], [2],....[27] relates to alphabetically sorted list of states in the analysis. sigma_alpha and sigma_beta are standard deviations of the normal distributions for the alpha and beta parameters priors. alpha and beta are country-level estimates, that is results of the model fitting to the fully pooled data.

   You can easily extract the estimates for each parameter of the model, e.g. by calculating a mean and quartiles of each column of this table.

4. Additional data processing can be done through running *outputs_processing/process_outputs.py* script, however at the moment it is not adjusted to generate results of a single model outputs. You can run it however once you have run all scripts contained in *python_scripts_fitting* folder, or by altering the *process_outputs.py* script.

5. Script *outputs_processing/process_outputs.py* does the following:

   - loads in the distribution data extracted from SIVEP-Gripe database

   - loads in the outputs of the model fitting for all PDFs and all distributions

   - calculates the Bayes Factors

   - creates tables with sample size per distribution and per state

   - creates tables with summarised results, i.e. mean, variance and parameters estimates for each PDF, including the 95% credible intervals

   - creates a number of plots, including all PDFs fits for each distribution, boxplots showing a distribution of mean times per state and others, as presented in the manuscript text

     All tables and plots generated by this script are saved in the *results* folder.
