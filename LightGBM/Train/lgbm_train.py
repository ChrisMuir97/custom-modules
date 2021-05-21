import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

import lightgbm as lgb
import pickle

## Parse args
parser = argparse.ArgumentParser("LGBM-R-Train")
parser.add_argument("--Training_Data", type=str, help="Training dataset")
parser.add_argument("--Lable_Col", type=str, help="Lable column in the dataset")
parser.add_argument("--Model_FileName", type=str, help="Name of the model file")
parser.add_argument("--Model_Path", type=str, help="Path to store the model file in Json format")
args = parser.parse_args()

## Load data from DataFrameDirectory to Pandas DataFrame
training_df = load_data_frame_from_directory(args.Training_Data).data

## Prepare training data
training_df_features = training_df[[c for c in training_df.columns if c!=args.Lable_Col]]
training_df_lable = training_df[args.Lable_Col]

## Training
lgbm = lgb.LGBMRegressor()
lgbm.fit(training_df_features, training_df_lable)

## Save model
with open(args.Model_Path +"/"+ args.Model_FileName + '.pkl', 'wb') as handle:
    pickle.dump(lgbm, handle, protocol=pickle.HIGHEST_PROTOCOL)