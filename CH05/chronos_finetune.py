# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="7WioAefBtDdu" jupyter={"outputs_hidden": true} outputId="eb74ec47-16d5-484a-bd5a-5075ae493eda"
# !pip install "chronos[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"

# %% colab={"base_uri": "https://localhost:8080/"} id="SefQeC5C8Hc4" outputId="150403b6-8119-4928-9bfa-b887fc773ee5"
# !git clone https://github.com/amazon-science/chronos-forecasting.git

# %% id="SkG6weyA88Z1"
# !cd chronos-forecasting/

# %% id="F-JMDcEOGK1H"
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from gluonts.dataset.arrow import ArrowWriter
import torch


# %% colab={"base_uri": "https://localhost:8080/"} id="tZBSigv9tWeA" outputId="d25d4e06-6de4-4581-84ff-a0c1a4f030da"
def convert_csv_to_arrow(
    csv_url: str,
    arrow_path: Union[str, Path],
    store_id: int
):
    # Read CSV file from URL
    df = pd.read_csv(csv_url)

    # Filter data for the specified store
    store_data = df[df['Store'] == store_id]

    # Convert 'Date' column to datetime
    store_data['Date'] = pd.to_datetime(store_data['Date'])

    # Get the start time (assuming all data for a store starts at the same time)
    start_time = store_data['Date'].min()

    # Extract Weekly_Sales column and convert to list of numpy arrays
    time_series = [store_data['Weekly_Sales'].values]

    # Prepare dataset
    dataset = [
        {"start": start_time, "target": ts} for ts in time_series
    ]

    # Write dataset to Arrow format
    ArrowWriter(compression="lz4").write_to_file(dataset, path=arrow_path)

csv_url = "https://raw.githubusercontent.com/marcopeix/FoundationModelsForTimeSeriesForecasting/main/data/walmart_sales_small.csv"
convert_csv_to_arrow(csv_url, "sales_data_store1.arrow", store_id=1)


# %% colab={"base_uri": "https://localhost:8080/"} id="MYB6_hqquvL_" outputId="5338bcbe-85da-4863-967b-dfb01e27bb01"
# !ls

# %% id="wsiS0jGdvhdI"
import yaml

# Define the configuration as a dictionary
config = {
    'training_data_paths': [
        "/content/sales_data_store1.arrow"
    ],
    'probability': [0.9],
    'context_length': 64,
    'prediction_length': 8,
    'min_past': 60,
    'max_steps': 500,
    'save_steps': 250,
    'log_steps': 100,
    'per_device_train_batch_size': 32,
    'learning_rate': 0.001,
    'optim': 'adamw_torch_fused',
    'num_samples': 20,
    'shuffle_buffer_length': 100,
    'gradient_accumulation_steps': 1,
    'model_id': 'google/t5-efficient-small',
    'model_type': 'seq2seq',
    'random_init': False,
    'tie_embeddings': True,
    'output_dir': './output/',
    'tf32': True,
    'torch_compile': True,
    'tokenizer_class': 'MeanScaleUniformBins',
    'tokenizer_kwargs': {
        'low_limit': -15.0,
        'high_limit': 15.0
    },
    'n_tokens': 4096,
    'lr_scheduler_type': 'linear',
    'warmup_ratio': 0.0,
    'dataloader_num_workers': 1,
    'max_missing_prop': 0.9,
    'use_eos_token': True
}

# Specify the path to the YAML file
yaml_file_path = 'finetune_config.yaml'

# Write the dictionary to a YAML file
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

# %% colab={"base_uri": "https://localhost:8080/"} id="Iw4K6oHm97JD" outputId="eb0e9379-5392-4955-8151-1a6dae1cbe14"
import os

# List all files in the current working directory
files_in_directory = os.listdir()

# Print the absolute path for each file
for file_name in files_in_directory:
    absolute_path = os.path.abspath(file_name)
    print(f"File: {file_name} | Absolute Path: {absolute_path}")


# %% colab={"base_uri": "https://localhost:8080/"} id="1tC9Vy00-1aI" outputId="bc15dad1-701e-42eb-cbd2-c131f772d99f"
# !CUDA_VISIBLE_DEVICES=0 python chronos-forecasting/scripts/training/train.py --config /content/finetune_config.yaml

# %% colab={"base_uri": "https://localhost:8080/"} id="Z7ArG5-VC_Cf" outputId="71351c40-3cc4-42ae-f203-af335bfa1052"
# List all files in the current working directory
files_in_directory = os.listdir()

# Print the absolute path for each file
for file_name in files_in_directory:
    absolute_path = os.path.abspath(file_name)
    print(f"File: {file_name} | Absolute Path: {absolute_path}")


# %% colab={"base_uri": "https://localhost:8080/"} id="Ule2KtT4DKg3" outputId="368e491c-24fd-4d31-a825-0a141d2e4583"
# !ls /content/output/run-1/checkpoint-final/

# %% [markdown] id="32S1aLMfFm4X"
# ## Use finetuned model

# %% id="7FimZdylFpw9"
url = "https://raw.githubusercontent.com/marcopeix/FoundationModelsForTimeSeriesForecasting/main/data/walmart_sales_small.csv"

df = pd.read_csv(url, parse_dates=['Date'])
df = df.query('Store == 1')


# %% id="pc7pT8YTF6MF"
def cross_validation_chronos(df, h, n_windows=4, target_col):

    lows = []
    medians = []
    highs = []
    
    for i in range(n_windows, 0, -1):
        context = torch.tensor(df[target_col][:-(h * i)])

        predictions = pipeline.predict(
            context=context,
            prediction_length=h,
            num_samples=20,
        )
    
        low, median, high = np.quantile(predictions[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    
        lows.extend(low)
        medians.extend(median)
        highs.extend(high)

    return lows, medians, highs
            


# %% id="TLTmDvAxGgQ_"
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("/content/output/run-1/checkpoint-final/", device_map='cuda', torch_dtype=torch.bfloat16)

# %% id="2AjKk2SX_gwI"
lows, medians, highs = cross_validation_chronos(df, 
                                                h=8,
                                                n_windows=4,
                                                target_col='Weekly_Sales')

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="ItivuJDaEiCh" outputId="47f751fd-2bf7-437a-adb4-894a5750d1e1"
test_df = df[['Store', 'Date', 'Weekly_Sales']][-32:]

test_df['low'] = lows
test_df['chronos-finetuned'] = medians
test_df['high'] = highs

test_df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="JlmjOOfPHIJp" outputId="d91e43ec-6fcf-4a5f-cf42-f51fab826e5b"
# !pip install utilsforecast

# %% colab={"base_uri": "https://localhost:8080/", "height": 125} id="qQhRj1TEHWmW" outputId="d1e19bf2-a5a0-4214-b44e-b5e346ed3dfb"
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

evaluation = evaluate(
    test_df,
    metrics=[mae, smape],
    models=['chronos-finetuned'],
    target_col='Weekly_Sales',
    id_col='Store'
)

evaluation

# %% id="n9AKFnpVHpZ5"
