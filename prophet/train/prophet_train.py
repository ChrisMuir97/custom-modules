import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

from prophet import Prophet
import pickle

## Parse args
parser = argparse.ArgumentParser("ProphetTraining")
parser.add_argument("--Training_Data", type=str, help="Training dataset")
parser.add_argument("--Model_FileName", type=str, help="Name of the model file")
parser.add_argument("--Model_Path", type=str, help="Path to store model")
args = parser.parse_args()

## Load data from DataFrameDirectory to Pandas DataFrame
training_df = load_data_frame_from_directory(args.Training_Data).data

## Training
model = Prophet()
model.fit(training_df)

## Save model
with open(args.Model_Path +"/"+ args.Model_FileName + '.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
