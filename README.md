# A Meta-Level Learning Algorithm for Sequential Hyper-Parameter Space Reduction in AutoML

This repository contains all data and code required to produce the results in the submission with title
"A Meta-Level Learning Algorithm for Sequential Hyper-Parameter Space Reduction in AutoML".

## Data Description

> **Note on datasets and analyses**: The algorithm in the paper takes as input performance and execution times of past
> runs (i.e., ``ML_results_{classification,regression}.csv``). Providing all datasets and code to analyze them is out
> of scope. 


All required data to produce the results for the paper are in ``data/data.zip``. A list of files along with
a description follows.

- ``ML_results_{classification,regression}.csv``: 
  performance and execution time results of machine learning configurations on classification/regression datasets. These
  were obtained by running [JADBio](https://jadbio.com/) on all datasets.
- ``metadata_{classification,regression}.csv``: 
  meta-features used to represent classification/regression datasets.
- ``datasets_{classification,regression}.csv``:
  list of classification/regression datasets, along with some of their characteristics. 
  Used for convenience in ``plots.py``.
- ``dataset_sources.csv``: 
  list of all classification/regression datasets and their sources. The file contains the following information:
  - The name of the dataset. For datasets from OpenML, a suffix of the form ``_v${VERSION}_did${DATASET_ID}`` 
  is appended to the dataset name, where ``VERSION`` is the version of the dataset, and ``DATASET_ID`` is its OpenML identifier.
  - The problem type (classification or regression).
  - The dataset source ([OpenML](https://www.openml.org/) or [BioDataome](http://dataome.mensxmachina.org/))

For the sake of convenience, all intermediate results produced by the scripts in this project are also provided in 
``results/results.zip``.

### Note on Regression Problems

To increase the number of regression problems, classification problems were obtained from 
[BioDataome](http://dataome.mensxmachina.org/) and turned into regression problems as follows:
1. JADBio was executed on each classification problem with default parameters and feature selection enforced, to find the most 
predictive features.
2. The first returned feature was used as the outcome (all datasets contain only continuous variables), while all remaining ones 
For the sake of convenience, all intermediate results produce by the scripts in this project are also provided in 
``results/results.zip``.
were used as predictors.

These datasets can be obtained by selecting all regression datasets from ``dataset_sources.csv`` from 
[BioDataome](http://dataome.mensxmachina.org/).

## Instructions

> **Note on requirements.txt**: The code has been tested on the package versions in requirements.txt and might not run 
> with other versions. We recommend using [virtual environments](https://docs.python.org/3/tutorial/venv.html) to 
> install dependencies.

### Produce results required for plots

First, unzip ``data/data.zip`` files and add them to the ``data`` folder. Next, run the following scripts to produce
all results required for the plots:

- ``{classification,regression}_threshold.py``: Produces all results for Figure 2 (SHSR with different thresholds).
- ``{classification,regression}_configuration_subsampling.py``: Produces all results for Figure 3 (SHSR on partial results).
- ``{classification,regression}_random_elimination.py``: Produces all results for Figure 4 (SHSR vs random elimination).

All results are stored in the ``results`` folder. Alternatively, this step can be skipped by unziping the results 
``results/results.zip`` and adding them to the ``results`` folder.

### Produce plots

Run the ``plots.py`` script to produce all plots of the paper. The plots are stored in the ``plots`` folder.