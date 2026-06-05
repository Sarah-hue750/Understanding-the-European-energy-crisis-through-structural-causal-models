# Causal Prices Shapley Flow

This part of the repository contains the implementation of the Shapley Flow analysis.
The adapted functions for calculating Shapley Flow values, based on https://github.com/nathanwang000/Shapley-Flow is contained in flow_adapted.py.

The repo is structured as follows
- data: In order to run the GBT models and the Shapley Flow analysis, the data dataset *data_selected_2018-2023* which can be collected via the instructions in *SCM* is supposed to be saved in folder *data*.
- credit_flow contains the calculated Shapley Flow values as well as Foreground and Background Samples that were used for the calculation.
- jobs: Scripts for running the Gradient Boosting Tree (GBT) Models as well as Shapley Flow valuzes calculation on a HPC cluster on CPUs.
- models: Contains the trained GBT models
- notebooks: Contains several notebooks for reproducing the figures form the paper, as well as the model performances, i.e. the R2-scores, the evaluation of the Shapley Flow values, and the results of the what-if analysis 
- results: contains the r2-scores as well as the correlation of Shapley Flow values and model features for picturing the causal graph
- scripts: Contains the scripts for data preparation in order to create the models, data visualization, training of the GBT models, and the calculation of the Shapley Flow values
- utils: contains support functions that are used for the functions in scripts and notebooks

In order to run the GBT models and the Shapley Flow analysis, the data dataset *data_selected_2018-2023* which can be collected via the instructions in *SCM* is supposed to be saved in folder *data*.

|   README.md
|   requirements.txt
|       
+---data
|
+---credit_flow
|   |   
|   \---what_if_scenarios
|       
+---figs_for_paper
|       
+---jobs
|       run_flow_ES_price.sh
|       run_flow_ES_price_what_if.sh
|       run_flow_FR_export.sh
|       run_flow_FR_price.sh
|       run_flow_FR_price_what_if.sh
|       run_model.sh
|       
+---models
|       xgb_ES_price_start_2018-01-01_end_2023-12-31_best.json
|       xgb_ES_price_start_2018-01-01_end_2023-12-31_best_hyperparameters.pkl
|       xgb_FR_export_start_2018-01-01_end_2023-12-31_best.json
|       xgb_FR_export_start_2018-01-01_end_2023-12-31_best_hyperparameters.pkl
|       xgb_FR_price_start_2018-01-01_end_2023-12-31_best.json
|       xgb_FR_price_start_2018-01-01_end_2023-12-31_best_hyperparameters.pkl
|       
+---notebooks
|       calculate_model_performance.ipynb
|       plot_r2_scores.ipynb
|       shapley_flow_evaluation.ipynb
|       what_if_analysis.ipynb
|       
+---results
|       correlation_dict_combined.pkl
|       r2_scores.pkl
|       
+---scripts
|   |   01_prepare_data.py
|   |   02_visualize_data.py
|   |   03_gbt_training.py
|   |   04_gbt_shapley_flow.py
|   |   04_gbt_shapley_flow_what_if.py
|           
+---utils
|   |   feature_configuration.py
|   |   helper_functions.py

Authors & Acknowledgements
-------------------
- Florian Immig - software developement
- Ulrich Oberhofer - software developement

#### Special Thanks

- to the Shapley-Flow Library: Jiaxuan Wang, Jenna Wiens, Scott Lundberg. Shapley Flow: A Graph-based Approach to Interpreting Model Predictions, 2021.
    - Paper: [https://proceedings.mlr.press/v130/wang21b/wang21b.pdf](https://proceedings.mlr.press/v130/wang21b/wang21b.pdf)
    - Library: [https://github.com/nathanwang000/Shapley-Flow](https://github.com/nathanwang000/Shapley-Flow)