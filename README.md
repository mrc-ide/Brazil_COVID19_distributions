# Brazil_COVID19_distributions

https://github.com/mrc-ide/Brazil_COVID19_distributions

This code and data were used for the analysis presented in "Inference of COVID-19 epidemiological distributions from Brazilian hospital data" (link to preprint).

The repository contains the following elements:

1. data

   Contains all data used for analysis, that is:

   * [SIVEP-Gripe hospitalisation datasets](https://opendatasus.saude.gov.br/dataset/bd-srag-2020) (latest download 7th July 2020)
   * socioeconomic indicators data: [GeoSES](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232074#pone.0232074.s002), [income_data](https://www.ibge.gov.br/estatisticas/sociais/populacao/9109-projecao-da-populacao.html?=&t=resultados), and urbanicity data

2.  python_scripts_fitting

   Contains the scripts for fitting Bayesian hierarchical models to the SIVEP-Gripe dataset, using [PyStan](https://mc-stan.org/users/interfaces/pystan). Models fitted are gamma, Weibull, log-normal, generalised log-normal and generalised gamma. The fitting might be very slow for some of the models given a huge amount of data, therefore we recommend running the fits in parallel.

3. fitting_outputs

   Contains the outputs of the MCMC sampling per dataset and per model, generated during the model fitting process.

4. outputs_processing

   Contains python and R scripts for analysis of the model fitting outputs, creating figures and collating the outputs in the figures.

5. results

   Contains the collated and processed results of the analysis.

