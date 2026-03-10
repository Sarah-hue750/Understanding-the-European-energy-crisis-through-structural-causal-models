import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
import matplotlib.pyplot as plt
from utils.helper_functions import read_csv_incl_timeindex



data_version = 'revision'
directory = './data/{}'.format(data_version)

X_FR = read_csv_incl_timeindex('{}/X_FR_full.csv'.format(directory))
y_FR_price = read_csv_incl_timeindex('{}/y_FR_price_full.csv'.format(directory))
y_FR_export = read_csv_incl_timeindex('{}/y_FR_export_full.csv'.format(directory))
X_ES = read_csv_incl_timeindex('{}/X_ES_full.csv'.format(directory))
y_ES_price = read_csv_incl_timeindex('{}/y_ES_price_full.csv'.format(directory))

# X_FR = pd.read_csv('{}/X_FR_full.csv'.format(directory), index_col=0, parse_dates=True)
# y_FR_price = pd.read_csv('{}/y_FR_price_full.csv'.format(directory), index_col=0, parse_dates=True)
# y_FR_export = pd.read_csv('{}/y_FR_export_full.csv'.format(directory), index_col=0, parse_dates=True)
# X_ES = pd.read_csv('{}/X_ES_full.csv'.format(directory), index_col=0, parse_dates=True)
# y_ES_price = pd.read_csv('{}/y_ES_price_full.csv'.format(directory), index_col=0, parse_dates=True)

file_path = './data/{}/dataset_all_features/data_selected_2018-2023.csv'.format(data_version)
dataset_all_features = pd.read_csv(file_path)
# add timestamp index
dataset_all_features.index = pd.to_datetime(dataset_all_features['timestamp'])

# X_FR.plot(subplots=True, layout=(12,2), figsize=(22,22))
# plt.tight_layout()
# plt.show()

# X_ES.plot(subplots=True, layout=(12,2), figsize=(22,22))
# plt.tight_layout()
# plt.show()

fig,ax = plt.subplots(2,1, figsize=(26,10))
y_FR_price.plot(ax = ax[0])
y_ES_price.plot(ax = ax[1])
fig.tight_layout()
fig.show()

y_FR_export.plot(figsize=(26,10))
plt.show()