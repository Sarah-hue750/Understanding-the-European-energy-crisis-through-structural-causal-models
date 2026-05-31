import sys
sys.path.append('../../.')

from pathlib import Path
# Resolve parent of this file (file's directory → parent)
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

import pandas as pd
import xgboost as xgb

from shapflow.flow import (
    CausalLinks,
    build_feature_graph,
    GraphExplainer,
    edge_credits2edge_credit,
    create_xgboost_f,
    translator,
)

from utils.helper_functions import read_csv_incl_timeindex, calculate_edge_credit

from flow_adapted import CausalLinks, build_feature_graph


import time
import dill
import tqdm
import multiprocess as mp
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Shapley flow edge credits for GBT model')
    parser.add_argument('--reduced_features', default=True, help='Whether to use reduced feature set'),
    parser.add_argument('--target', type=str, default='FR_price', help='Target variable to explain')    
    return parser.parse_args()

args = parse_args()


edges_FR_price = [('hour_sin', 'rl_BE'), ('hour_sin', 'rl_ES'), ('hour_sin', 'rl_DE_LU'), ('hour_sin', 'rl_IT_NORD'), ('hour_sin', 'load_da_FR'), ('hour_sin', 'nuclear_avail_rte_FR'), ('hour_sin', 'ssrd_FR'), ('hour_sin', 'wind_speed_100m_FR'), ('hour_sin', 'air_temp_era5_FR'), ('hour_cos', 'rl_BE'), ('hour_cos', 'rl_ES'), ('hour_cos', 'rl_DE_LU'), ('hour_cos', 'rl_IT_NORD'), ('hour_cos', 'load_da_FR'), ('hour_cos', 'nuclear_avail_rte_FR'), ('hour_cos', 'ssrd_FR'), ('hour_cos', 'wind_speed_100m_FR'), ('hour_cos', 'air_temp_era5_FR'), ('day_of_year_sin', 'rl_BE'), ('day_of_year_sin', 'rl_ES'), ('day_of_year_sin', 'rl_DE_LU'), ('day_of_year_sin', 'rl_IT_NORD'), ('day_of_year_sin', 'load_da_FR'), ('day_of_year_sin', 'nuclear_avail_rte_FR'), ('day_of_year_sin', 'ssrd_FR'), ('day_of_year_sin', 'wind_speed_100m_FR'), ('day_of_year_sin', 'air_temp_era5_FR'), ('day_of_year_cos', 'rl_BE'), ('day_of_year_cos', 'rl_ES'), ('day_of_year_cos', 'rl_DE_LU'), ('day_of_year_cos', 'rl_IT_NORD'), ('day_of_year_cos', 'load_da_FR'), ('day_of_year_cos', 'nuclear_avail_rte_FR'), ('day_of_year_cos', 'ssrd_FR'), ('day_of_year_cos', 'wind_speed_100m_FR'), ('day_of_year_cos', 'air_temp_era5_FR'), ('isworkingday_FR', 'rl_BE'), ('isworkingday_FR', 'rl_ES'), ('isworkingday_FR', 'rl_DE_LU'), ('isworkingday_FR', 'rl_IT_NORD'), ('isworkingday_FR', 'load_da_FR'), ('isworkingday_FR', 'nuclear_avail_rte_FR'), ('air_temp_era5_FR', 'river_temp_FR'), ('air_temp_era5_FR', 'rl_BE'), ('air_temp_era5_FR', 'rl_ES'), ('air_temp_era5_FR', 'rl_DE_LU'), ('air_temp_era5_FR', 'rl_IT_NORD'), ('air_temp_era5_FR', 'load_da_FR'), ('air_temp_era5_FR', 'solar_da_FR'), ('air_temp_era5_FR', 'filling_rate_FR'), ('rl_BE', 'price_da_FR'), ('rl_ES', 'price_da_FR'), ('rl_DE_LU', 'price_da_FR'), ('rl_IT_NORD', 'price_da_FR'), ('load_da_FR', 'price_da_FR'), ('nuclear_avail_rte_FR', 'price_da_FR'), ('run_off_gen_FR', 'price_da_FR'), ('solar_da_FR', 'price_da_FR'), ('wind_da_FR', 'price_da_FR'), ('carbon_price_FR', 'price_da_FR'), ('gas_price_FR', 'price_da_FR'), ('filling_rate_FR', 'price_da_FR'), ('ssrd_FR', 'solar_da_FR'), ('wind_speed_100m_FR', 'wind_da_FR'), ('ssrd_FR', 'rl_BE'), ('ssrd_FR', 'rl_ES'), ('ssrd_FR', 'rl_DE_LU'), ('ssrd_FR', 'rl_IT_NORD'), ('wind_speed_100m_FR', 'rl_BE'), ('wind_speed_100m_FR', 'rl_ES'), ('wind_speed_100m_FR', 'rl_DE_LU'), ('wind_speed_100m_FR', 'rl_IT_NORD'), ('river_temp_FR', 'nuclear_avail_rte_FR'), ('river_temp_FR', 'run_off_gen_FR'), ('river_temp_FR', 'filling_rate_FR'), ('river_flow_mean_FR', 'nuclear_avail_rte_FR'), ('river_flow_mean_FR', 'run_off_gen_FR'), ('river_flow_mean_FR', 'filling_rate_FR'), ('day_of_year_sin', 'river_temp_FR'), ('day_of_year_sin', 'river_flow_mean_FR'), ('day_of_year_sin', 'filling_rate_FR'), ('day_of_year_sin', 'gas_price_FR'), ('day_of_year_cos', 'river_temp_FR'), ('day_of_year_cos', 'river_flow_mean_FR'), ('day_of_year_cos', 'filling_rate_FR'), ('day_of_year_cos', 'gas_price_FR'), ('year', 'carbon_price_FR'), ('year', 'gas_price_FR'), ('year', 'rl_BE'), ('year', 'rl_ES'), ('year', 'rl_DE_LU'), ('year', 'rl_IT_NORD'), ('year', 'load_da_FR'), ('year', 'ssrd_FR'), ('year', 'wind_speed_100m_FR'), ('year', 'air_temp_era5_FR'), ('year', 'river_temp_FR'), ('year', 'river_flow_mean_FR'), ('year', 'solar_da_FR'), ('year', 'wind_da_FR'), ('year', 'nuclear_avail_rte_FR'), ('year', 'run_off_gen_FR'), ('nuclear_avail_esios_ES', 'price_da_FR'), ('year', 'nuclear_avail_esios_ES'), ('hour_sin', 'nuclear_avail_esios_ES'), ('hour_cos', 'nuclear_avail_esios_ES'), ('day_of_year_sin', 'nuclear_avail_esios_ES'), ('day_of_year_cos', 'nuclear_avail_esios_ES'), ('isworkingday_FR', 'nuclear_avail_esios_ES')]
edges_FR_export = [('hour_sin', 'rl_BE'), ('hour_sin', 'rl_ES'), ('hour_sin', 'rl_DE_LU'), ('hour_sin', 'rl_IT_NORD'), ('hour_sin', 'load_da_FR'), ('hour_sin', 'nuclear_avail_rte_FR'), ('hour_sin', 'ssrd_FR'), ('hour_sin', 'wind_speed_100m_FR'), ('hour_sin', 'air_temp_era5_FR'), ('hour_cos', 'rl_BE'), ('hour_cos', 'rl_ES'), ('hour_cos', 'rl_DE_LU'), ('hour_cos', 'rl_IT_NORD'), ('hour_cos', 'load_da_FR'), ('hour_cos', 'nuclear_avail_rte_FR'), ('hour_cos', 'ssrd_FR'), ('hour_cos', 'wind_speed_100m_FR'), ('hour_cos', 'air_temp_era5_FR'), ('day_of_year_sin', 'rl_BE'), ('day_of_year_sin', 'rl_ES'), ('day_of_year_sin', 'rl_DE_LU'), ('day_of_year_sin', 'rl_IT_NORD'), ('day_of_year_sin', 'load_da_FR'), ('day_of_year_sin', 'nuclear_avail_rte_FR'), ('day_of_year_sin', 'ssrd_FR'), ('day_of_year_sin', 'wind_speed_100m_FR'), ('day_of_year_sin', 'air_temp_era5_FR'), ('day_of_year_cos', 'rl_BE'), ('day_of_year_cos', 'rl_ES'), ('day_of_year_cos', 'rl_DE_LU'), ('day_of_year_cos', 'rl_IT_NORD'), ('day_of_year_cos', 'load_da_FR'), ('day_of_year_cos', 'nuclear_avail_rte_FR'), ('day_of_year_cos', 'ssrd_FR'), ('day_of_year_cos', 'wind_speed_100m_FR'), ('day_of_year_cos', 'air_temp_era5_FR'), ('isworkingday_FR', 'rl_BE'), ('isworkingday_FR', 'rl_ES'), ('isworkingday_FR', 'rl_DE_LU'), ('isworkingday_FR', 'rl_IT_NORD'), ('isworkingday_FR', 'load_da_FR'), ('isworkingday_FR', 'nuclear_avail_rte_FR'), ('air_temp_era5_FR', 'river_temp_FR'), ('air_temp_era5_FR', 'rl_BE'), ('air_temp_era5_FR', 'rl_ES'), ('air_temp_era5_FR', 'rl_DE_LU'), ('air_temp_era5_FR', 'rl_IT_NORD'), ('air_temp_era5_FR', 'load_da_FR'), ('air_temp_era5_FR', 'solar_da_FR'), ('air_temp_era5_FR', 'filling_rate_FR'), ('rl_BE', 'net_export_FR'), ('rl_ES', 'net_export_FR'), ('rl_DE_LU', 'net_export_FR'), ('rl_IT_NORD', 'net_export_FR'), ('load_da_FR', 'net_export_FR'), ('nuclear_avail_rte_FR', 'net_export_FR'), ('run_off_gen_FR', 'net_export_FR'), ('solar_da_FR', 'net_export_FR'), ('wind_da_FR', 'net_export_FR'), ('carbon_price_FR', 'net_export_FR'), ('gas_price_FR', 'net_export_FR'), ('filling_rate_FR', 'net_export_FR'), ('ssrd_FR', 'solar_da_FR'), ('wind_speed_100m_FR', 'wind_da_FR'), ('ssrd_FR', 'rl_BE'), ('ssrd_FR', 'rl_ES'), ('ssrd_FR', 'rl_DE_LU'), ('ssrd_FR', 'rl_IT_NORD'), ('wind_speed_100m_FR', 'rl_BE'), ('wind_speed_100m_FR', 'rl_ES'), ('wind_speed_100m_FR', 'rl_DE_LU'), ('wind_speed_100m_FR', 'rl_IT_NORD'), ('river_temp_FR', 'nuclear_avail_rte_FR'), ('river_temp_FR', 'run_off_gen_FR'), ('river_temp_FR', 'filling_rate_FR'), ('river_flow_mean_FR', 'nuclear_avail_rte_FR'), ('river_flow_mean_FR', 'run_off_gen_FR'), ('river_flow_mean_FR', 'filling_rate_FR'), ('day_of_year_sin', 'river_temp_FR'), ('day_of_year_sin', 'river_flow_mean_FR'), ('day_of_year_sin', 'filling_rate_FR'), ('day_of_year_sin', 'gas_price_FR'), ('day_of_year_cos', 'river_temp_FR'), ('day_of_year_cos', 'river_flow_mean_FR'), ('day_of_year_cos', 'filling_rate_FR'), ('day_of_year_cos', 'gas_price_FR'), ('year', 'carbon_price_FR'), ('year', 'gas_price_FR'), ('year', 'rl_BE'), ('year', 'rl_ES'), ('year', 'rl_DE_LU'), ('year', 'rl_IT_NORD'), ('year', 'load_da_FR'), ('year', 'ssrd_FR'), ('year', 'wind_speed_100m_FR'), ('year', 'air_temp_era5_FR'), ('year', 'river_temp_FR'), ('year', 'river_flow_mean_FR'), ('year', 'solar_da_FR'), ('year', 'wind_da_FR'), ('year', 'nuclear_avail_rte_FR'), ('year', 'run_off_gen_FR'), ('nuclear_avail_esios_ES', 'net_export_FR'), ('year', 'nuclear_avail_esios_ES'), ('hour_sin', 'nuclear_avail_esios_ES'), ('hour_cos', 'nuclear_avail_esios_ES'), ('day_of_year_sin', 'nuclear_avail_esios_ES'), ('day_of_year_cos', 'nuclear_avail_esios_ES'), ('isworkingday_FR', 'nuclear_avail_esios_ES')]
edges_ES = [('hour_sin', 'rl_FR'), ('hour_sin', 'rl_PT'), ('hour_sin', 'load_da_ES'), ('hour_sin', 'nuclear_avail_esios_ES'), ('hour_sin', 'ssrd_ES'), ('hour_sin', 'wind_speed_100m_ES'), ('hour_sin', 'air_temp_era5_ES'), ('hour_cos', 'rl_FR'), ('hour_cos', 'rl_PT'), ('hour_cos', 'load_da_ES'), ('hour_cos', 'nuclear_avail_esios_ES'), ('hour_cos', 'ssrd_ES'), ('hour_cos', 'wind_speed_100m_ES'), ('hour_cos', 'air_temp_era5_ES'), ('day_of_year_sin', 'rl_FR'), ('day_of_year_sin', 'rl_PT'), ('day_of_year_sin', 'load_da_ES'), ('day_of_year_sin', 'nuclear_avail_esios_ES'), ('day_of_year_sin', 'ssrd_ES'), ('day_of_year_sin', 'wind_speed_100m_ES'), ('day_of_year_sin', 'air_temp_era5_ES'), ('day_of_year_cos', 'rl_FR'), ('day_of_year_cos', 'rl_PT'), ('day_of_year_cos', 'load_da_ES'), ('day_of_year_cos', 'nuclear_avail_esios_ES'), ('day_of_year_cos', 'ssrd_ES'), ('day_of_year_cos', 'wind_speed_100m_ES'), ('day_of_year_cos', 'air_temp_era5_ES'), ('isworkingday_ES', 'rl_FR'), ('isworkingday_ES', 'rl_PT'), ('isworkingday_ES', 'load_da_ES'), ('isworkingday_ES', 'nuclear_avail_esios_ES'), ('air_temp_era5_ES', 'river_temp_ES'), ('air_temp_era5_ES', 'rl_FR'), ('air_temp_era5_ES', 'rl_PT'), ('air_temp_era5_ES', 'load_da_ES'), ('air_temp_era5_ES', 'solar_da_ES'), ('air_temp_era5_ES', 'filling_rate_ES'), ('rl_FR', 'price_da_ES'), ('rl_PT', 'price_da_ES'), ('load_da_ES', 'price_da_ES'), ('nuclear_avail_esios_ES', 'price_da_ES'), ('run_off_gen_ES', 'price_da_ES'), ('solar_da_ES', 'price_da_ES'), ('wind_da_ES', 'price_da_ES'), ('carbon_price_ES', 'price_da_ES'), ('gas_price_ES', 'price_da_ES'), ('filling_rate_ES', 'price_da_ES'), ('ssrd_ES', 'solar_da_ES'), ('wind_speed_100m_ES', 'wind_da_ES'), ('ssrd_ES', 'rl_FR'), ('ssrd_ES', 'rl_PT'), ('wind_speed_100m_ES', 'rl_FR'), ('wind_speed_100m_ES', 'rl_PT'), ('river_temp_ES', 'nuclear_avail_esios_ES'), ('river_temp_ES', 'run_off_gen_ES'), ('river_temp_ES', 'filling_rate_ES'), ('river_flow_mean_ES', 'nuclear_avail_esios_ES'), ('river_flow_mean_ES', 'run_off_gen_ES'), ('river_flow_mean_ES', 'filling_rate_ES'), ('day_of_year_sin', 'river_temp_ES'), ('day_of_year_sin', 'river_flow_mean_ES'), ('day_of_year_sin', 'filling_rate_ES'), ('day_of_year_sin', 'gas_price_ES'), ('day_of_year_cos', 'river_temp_ES'), ('day_of_year_cos', 'river_flow_mean_ES'), ('day_of_year_cos', 'filling_rate_ES'), ('day_of_year_cos', 'gas_price_ES'), ('year', 'carbon_price_ES'), ('year', 'gas_price_ES'), ('year', 'rl_FR'), ('year', 'rl_PT'), ('year', 'load_da_ES'), ('year', 'ssrd_ES'), ('year', 'wind_speed_100m_ES'), ('year', 'air_temp_era5_ES'), ('year', 'river_temp_ES'), ('year', 'river_flow_mean_ES'), ('year', 'solar_da_ES'), ('year', 'wind_da_ES'), ('year', 'nuclear_avail_esios_ES'), ('year', 'run_off_gen_ES'), ('nuclear_avail_rte_FR', 'price_da_ES'), ('year', 'nuclear_avail_rte_FR'), ('hour_sin', 'nuclear_avail_rte_FR'), ('hour_cos', 'nuclear_avail_rte_FR'), ('day_of_year_sin', 'nuclear_avail_rte_FR'), ('day_of_year_cos', 'nuclear_avail_rte_FR'), ('isworkingday_ES', 'nuclear_avail_rte_FR')]

target_names = ['price_da_FR', 'net_export_FR', 'price_da_ES']


reduced_features = args.reduced_features

periods = [('2018-01-01', '2023-12-31')]

file_path = 'data/dataset_all_features/data_selected_2018-2023.csv'
data_full = pd.read_csv(file_path, index_col=0, parse_dates=True)

targets = [args.target]


for target in targets:
    print(target)
    if target == 'FR_price':
        edges = edges_FR_price
    elif target == 'FR_export':
        edges = edges_FR_export
    elif target == 'ES_price':
        edges = edges_ES

    for start_date, end_date in periods:
        model_name = 'xgb_{}_start_{}_end_{}'.format(target, start_date, end_date)
        
        X_full = read_csv_incl_timeindex('./data/X_full_{}.csv'.format(model_name))
        X_train = read_csv_incl_timeindex('./data/X_train_{}.csv'.format(model_name))

        X_test = read_csv_incl_timeindex('./data/X_test_{}.csv'.format(model_name))
        if reduced_features:
            X_test_features_reduced = read_csv_incl_timeindex('./data/X_test_features_reduced_{}.csv'.format(model_name))

        if target == 'FR_price':
            additional_nuc_avail = 10000
            X_test = X_test[(X_test.index >= pd.to_datetime('2022-01-01 00:00:00', utc=True)) & (X_test.index < pd.to_datetime('2023-01-01 00:00:00', utc=True))]
            print(X_test['nuclear_avail_rte_FR'].head())
            X_test['nuclear_avail_rte_FR'] += additional_nuc_avail
            print(X_test['nuclear_avail_rte_FR'].head())
        elif target == 'ES_price':
            X_test = X_test[(X_test.index >= pd.to_datetime('2022-06-15 00:00:00', utc=True)) & (X_test.index < pd.to_datetime('2023-02-27 00:00:00', utc=True))]
            # substitute with real data from dataset_all_features; 
            # this is just for being able to calculate what-if Shapley flow edge credits 
            # for a change in gas price in Spain, which is not included in the test set of the model, but is included in the full dataset.
            X_test['gas_price_ES'] = data_full.loc[X_test.index, 'gas_price_MIBGAS']  
        print(X_test.shape)

        model = xgb.Booster()
        model.load_model("./models/{}_best.json".format(model_name))
        seed = 7
        
        n_bg = 96 # number of sampled background samples 
        nsamples = 1000 # number of foreground samples to explain
        nruns = 750 # number of runs for Shapley flow (number of permutations to sample for Shapley value estimation)

        # choose background samples from training set and foreground samples from test set (this is just for consistency with the Shapley flow framework)
        bg = X_train.sample(n=n_bg, random_state=seed) # background samples
        fg = X_test.sample(n=nsamples, random_state=seed) # foreground samples (samples to explain)

        bg.to_csv('./credit_flow/what_if/bg_{}.csv'.format(model_name), sep=',', index=True)
        fg.to_csv('./credit_flow/what_if/fg_{}.csv'.format(model_name), sep=',', index=True)

        causal_links = CausalLinks()
        categorical_feature_names = []
        display_translator = translator(X_full.columns, X_full, X_full)

        feature_names = list(X_test.columns)
        if reduced_features:
            feature_names_reduced = list(X_test_features_reduced.columns)
        
        for edge in edges:
            node_cause = edge[0]
            node_effect = edge[1]
            if node_effect not in target_names:
                if not node_cause in feature_names:
                    print('Error: node_cause not in feature_names: {}'.format(node_cause))
                #print('skip adding edge to target: {} -> {}'.format(node_cause, node_effect))
                causal_links.add_causes_effects(node_cause, node_effect)

        if reduced_features:
            causal_links.add_causes_effects(feature_names_reduced, 
                                            target, #target_name, 
                                            create_xgboost_f(feature_names_reduced, model))
            print(feature_names, '\n',feature_names_reduced)
        else:
            causal_links.add_causes_effects(feature_names, 
                            target, #target_name, 
                            create_xgboost_f(feature_names, model))
            print(feature_names, '\n',feature_names)

        causal_graph, r2_scores = build_feature_graph(X_full,
                                        causal_links=causal_links, 
                                        categorical_feature_names=categorical_feature_names,
                                        display_translator=display_translator,
                                        target_name=target,# target_name=target_name,
                                        method='xgboost')
        with open('./credit_flow/what_if/causal_graph_{}.pkl'.format(model_name), 'wb') as file:
            dill.dump(causal_graph, file)
        with open('./credit_flow/what_if/r2_scores_{}.pkl'.format(model_name), 'wb') as file:
            dill.dump(r2_scores, file)
        # causal_graph.draw(rankdir = 'TB')
        # g = causal_graph.to_graphviz('LR')

        # calculate multiple background result (same as in income.ipynb)
        # change this to a suitable value, depending on machine (e.g. 6, 12; on cluster 20)
        num_processes = 96 #20

        start = time.time()

        model.set_param('n_jobs', -1)
        model.set_param('device', 'cpu')

        pool = mp.Pool(num_processes)
        _args = [(causal_graph, bg[i:i+1], fg, nruns) for i in range(len(bg))]
        edge_credits = pool.starmap(calculate_edge_credit, tqdm.tqdm(_args, total=len(_args)))
        pool.close()
        pool.join()

        end = time.time()
        print(end - start)
        
        # need this for being able to draw shapley flow (need to call shap_values for one bg sample redundandly)
        model.set_param('n_jobs', 96) #40
        explainer = GraphExplainer(causal_graph, bg[0:1], nruns, silent=False)
        cf = explainer.shap_values(fg)
        # save credit flow to file
        cf.edge_credit = edge_credits2edge_credit(edge_credits, cf.graph)
        
        directory = './credit_flow/what_if'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open('{}/flow_{}.pkl'.format(directory, model_name), 'wb') as file:
            dill.dump(cf, file)