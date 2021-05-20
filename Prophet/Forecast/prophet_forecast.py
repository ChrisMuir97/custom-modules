import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.core.run import Run
import logging

from prophet import Prophet
import pickle

## Parse args
parser = argparse.ArgumentParser("Prophet-Forecast")
parser.add_argument("--Model_FileName", type=str, help="Name of the model file")
parser.add_argument("--Model_Path", type=str, help="Path to store model")
parser.add_argument("--Periods", type=int, help="Number of periods to forecast forwards")
parser.add_argument("--Frequency", type=str, help="Frequency of forecast (D for daily, M for monthly)")
parser.add_argument("--Evaluation_Output", type=str, help="Evaluation result")
args = parser.parse_args()

## Load model
with open(args.Model_Path + "/" + args.Model_FileName + '.pkl', 'rb') as handle:
    model = pickle.load(handle)

## Create the forecast container
future = model.make_future_dataframe(periods=args.Periods, freq=args.Frequency)

## Run the forecast
forecast = model.predict(future)

## Check output directory
os.makedirs(args.Model_Path, exist_ok=True)

# Get run context
run = Run.get_context()

## Create the Forecast plot
fig1 = model.plot(forecast)
run.log_image("Forecast", plot=fig1)
#plt.savefig(args.Model_Path + "/" +'forecast.png')

## Create the components plots
fig2 = model.plot_components(forecast)
run.log_image("Forecast Components", plot=fig2)
#plt.savefig(args.Model_Path + "/" +'components.png')

## Get prediction
model_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(args.Periods).copy()
model_output['Date'] = model_output['ds'].dt.strftime('%Y-%m-%d')
model_output = model_output.drop('ds', axis=1)
model_output = model_output[ ['Date'] + [ col for col in model_output.columns if col != 'Date' ] ]
#model_output.to_csv(args.Model_Path + "/" + args.Model_FileName + '.csv')

## Output the forecast
save_data_frame_to_directory(args.Evaluation_Output, model_output)