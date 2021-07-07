import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

## Parse args
parser = argparse.ArgumentParser("prophet_prep")
parser.add_argument("--Loaded_Data", type=str, help="Loaded dataset")
parser.add_argument("--Timeseries", type=str, help="Timeseries column.")
parser.add_argument("--Forecast", type=str, help="Forecast column")
parser.add_argument("--Prepared_Data", type=str, help="Prepared dataset")
args = parser.parse_args()

## Load data from DataFrameDirectory to Pandas DataFrame
training_df = load_data_frame_from_directory(args.Loaded_Data).data

## Prepare training data
training_df = training_df[[args.Timeseries , args.Forecast]].copy()
training_df = training_df.rename(columns = {args.Timeseries:'ds', args.Forecast:'y'})
training_df['ds'] = pd.to_datetime(training_df['ds'])

## Output prepared Pandas DataFrame to DataFrameDirectory
save_data_frame_to_directory(args.Prepared_Data, training_df)
