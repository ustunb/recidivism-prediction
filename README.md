This repository contains software to reproduce the recidivism prediction models from the paper [Interpretable Classification Models for Recidivism Prediction](http://arxiv.org/abs/1503.07810)

## Generating the Processed Datasets

The datasets are processed using raw data from the study:

`Recidivism of Prisoners Released in 1994 (ICPSR 3355)`

Given to the sensitive nature of the raw data, however, our agreement with ICPSR does not allow us to post the data online. However, the data is available to the general public, and you can obtain from ICPSR at [this link](:
http://www.icpsr.umich.edu/icpsrweb/NACJD/studies/3355/version/8). 

Once you have obtained the raw data, you can create the processed data sets we used to fit models by running `create_datasets.R`. This will create the following files in the subdirectory `/data/`  with the input variables, outcome variables and folds used to fit models in R and MATLAB:

- `arrest.RData` / `arrest.mat`
- `drug.RData` / `drug.mat`
- `general_violence.RData` / `general_violence.mat`
- `domestic_violence.RData` / `domestic_violence.mat`
- `sexual_violence.RData` / `sexual_violence.mat`
- `fatal_violence.RData` / `fatal_violence.mat`

## Fitting Models

### Software Requirements

- [slim-matlab](https://github.com/ustunb/slim-matlab)
- IBM ILOG CPLEX Optimization Studio (available through IBM Academic Initiative)
- MATLAB 2014b or later
- R 3.0.0 or later
- R packages: dplyr, R.matlab, glmnet, sgb, randomForest, e1071, rpart, C5.0 

We fit models for 6 datasets x 19 values of class weights. The full set of 19 class weights are specific to each dataset:

- `arrest` / `drug` / `general_violence`: [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9]

- `domestic_violence` / `sexual_violence`: [1.815 1.825 1.835 1.845 1.855 1.865 1.875 1.885 1.895 1.905 1.915 1.925 1.935 1.945 1.955 1.965 1.975 1.985 1.995]

- `fatal_violence`: [1.976 1.977 1.978 1.979 1.980 1.981 1.982 1.983 1.984 1.985 1.986 1.987 1.988 1.989 1.990 1.991 1.992 1.993 1.994]

### Fitting SLIM Models

To train SLIM models, install the latest version of [slim-matlab](https://github.com/ustunb/slim-matlab). You can then use the script `fit_slim.m` to train SLIM for a particular dataset and set of class weights in the same way as the paper. When the training completes, a .mat file containing all models and summary statistics will be produced and stored under the `results/` subdirectory.

### Fitting Other Models

To train models using C5.0R, C5.0T, CART, Lasso, Ridge, RF, SVM RBF, or SGB, you can use the R script `fit_models.R`. Running this script will fit models for a given dataset and set of class weights. When the training completes, a .R file containing all models and summary statistics will be produced and stored under the `/results/` subdirectory.