# Causal Prices Shapley Flow

This part of the repository contains the implementation of the Shapley Flow analysis.
The adapted functions for calculating Shapley Flow values, based on https://github.com/nathanwang000/Shapley-Flow is contained in flow_adapted.py.
- credit_flow contains the calculated Shapley Flow values as well as Foreground and Background Samples that were used for the calculation.
- jobs: Scripts for running the Gradient Boosting Tree (GBT) Models as well as Shapley Flow valuzes calculation on a HPC cluster
- models: Contains the trained GBT models
- notebooks: Contains several notebooks for reproducing the figures form the paper, as well as the model performances
- results: contains the r2-scores as well as the correlation of Shapley Flow values and model features for picture the causal graph
- scripts_ Contains the scripts for data preparation in order to create the models, data visualization, training of the GBT models, and the calculation of the Shapley Flow values
- utils: contains support functions that are used for the functions in scripts and notebooks