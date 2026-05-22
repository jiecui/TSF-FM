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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="11bde39e-dfcb-47df-b15f-4abe83cf9431"
# # Fine-tuning Lag-Llama

# %% colab={"base_uri": "https://localhost:8080/"} id="f314d613-aac8-499f-a632-3a9062e1293e" outputId="49790138-ba35-4ecc-c2c7-7c0d4122cb18"
# !git clone https://github.com/time-series-foundation-models/lag-llama/

# %% colab={"base_uri": "https://localhost:8080/"} id="9b3a4589-1ff7-46b3-8ad3-b2c2bc998183" outputId="a28e1c66-f9cc-45bb-99c2-ff1be319a770"
# cd /content/lag-llama

# %% id="1cb4157e-a745-4938-b8c8-06eeeb57eeaa" colab={"base_uri": "https://localhost:8080/"} outputId="78f70a19-3e19-4f2e-a983-a3f8863ec02e"
# !pip install -r requirements.txt --quiet

# %% colab={"base_uri": "https://localhost:8080/"} id="80c5f8e7-3687-4f4c-8e42-2d84882845e1" outputId="69d5f03f-9a3f-4bd5-b36c-ddc21d28ca84"
# !huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir /content/lag-llama

# %% id="bZIEvK8CL4d9" outputId="a00ea7a0-164a-4fa0-eb13-88245574773e" colab={"base_uri": "https://localhost:8080/"}
# !pip install gluonts==0.14.4

# %% id="12626ee4-7175-4854-8701-7cc6a4f514d8" colab={"base_uri": "https://localhost:8080/"} outputId="fa13c81d-9faf-4605-afd8-3fc65a4de2e7"
from itertools import islice

from matplotlib import pyplot as plt

from tqdm.autonotebook import tqdm

import torch

from gluonts.dataset.pandas import PandasDataset

import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator


# %% id="e4c2405a-be1c-4f7d-b598-580c974a68fb"
def get_lag_llama_predictions(dataset,
                              prediction_length,
                              context_length=32,
                              device="cuda",
                              num_samples=100,
                              predictor=None):

    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        num_parallel_samples=num_samples,
    )

    if predictor is None:
      lightning_module = estimator.create_lightning_module()
      transformation = estimator.create_transformation()
      predictor = estimator.create_predictor(transformation, lightning_module)

    forecasts = predictor.predict(
        dataset=dataset
    )

    forecasts = list(forecasts)

    return forecasts


# %% id="de228599-5404-4b62-87cd-4b9573583981"
device = 'cuda'
prediction_length = 8
context_length = 64

ckpt = torch.load("lag-llama.ckpt", map_location=device)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,

        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        time_feat=estimator_args["time_feat"],

        # Fine-tuning arguments
        nonnegative_pred_samples=True,
        aug_prob=0,
        lr=5e-4,
        batch_size=64,
        num_parallel_samples=20,
        trainer_kwargs = {"max_epochs": 50,}
    )

# %% id="RAwWDYacM86D" outputId="d459bb5c-3bf9-4875-ad5e-38b0f03fdc7e" colab={"base_uri": "https://localhost:8080/"}
url = 'https://raw.githubusercontent.com/marcopeix/FoundationModelsForTimeSeriesForecasting/main/data/walmart_sales_small.csv'

df = pd.read_csv(url)
df = df[df['Store'] == 1]

input_df = df[:-8]
test_df = df[-8:]

input_df['Weekly_Sales'] = input_df['Weekly_Sales'].astype('float32')

# %% id="YngUvVvpNwo2"
input_ds = PandasDataset.from_long_dataframe(input_df, target='Weekly_Sales', item_id='Store')

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["2190665459db4bf68e161c73ab77e022", "bac497aad9d344e99c4999902ad38c2d", "5f53500cdc2344be963e19f80c630ebc", "c591def1ab6842a6bd6b08f8205aecc6", "e96c668e42a7433cb6e81dbd34cfdce3", "2b634c25f83049dcb0123c74ef126895", "58b660fcad6d465dba4634934d684534", "4d6299eefa2348e3b1579557faa9ee5d", "e072d213eab34deca6228f216fbe45b2", "604616432c4d46a7afef2c143b79430d", "3511248fcd8342deb08ad16a141395b9"]} id="7b2b75bd-657d-438b-8596-f97509253c92" outputId="63f7e40c-19c0-4562-bd37-2fe8dfdea75b"
predictor = estimator.train(input_ds, cache_data=True, shuffle_buffer_length=1000)

# %% id="54a88620-5a56-463c-adf7-ef13018da432"
finetune_forecasts = get_lag_llama_predictions(
        dataset=input_ds,
        prediction_length=prediction_length,
        predictor=predictor,
    )

# %% colab={"base_uri": "https://localhost:8080/"} id="3413cbf1-7e58-4c16-84a1-4c92d9926908" outputId="a4b22d2a-f6f2-4eac-d9e6-e99d3b7cc2d0"
finetune_forecasts[0].samples.shape

# %% id="7e828749-354e-441a-bded-af2b254d6e14"
import numpy as np

def get_median_and_ci(data,
                      start_date,
                      horizon,
                      freq,
                      id,
                      confidence=0.95):

    n_samples, n_timesteps = data.shape

    # Calculate the median for each timestep
    medians = np.median(data, axis=0)

    # Calculate the lower and upper percentile for the given confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    # Calculate the lower and upper bounds for each timestep
    lower_bounds = np.percentile(data, lower_percentile, axis=0)
    upper_bounds = np.percentile(data, upper_percentile, axis=0)

    pred_dates = pd.date_range(start=start_date, periods=horizon, freq=freq)
    formatted_dates = pred_dates.strftime('%m-%d-%Y').tolist()

    # Create a DataFrame with the results
    df = pd.DataFrame({
        'Date': formatted_dates,
        'Store': id,
        'Lag-Llama': medians,
        f'Lag-Llama-lo-{int(confidence*100)}': lower_bounds,
        f'Lag-Llama-hi-{int(confidence*100)}': upper_bounds
    })

    return df


# %% id="vIJ6mEYERy1Q" outputId="e022362b-f818-4ba5-83b7-9a1ce1250b0d" colab={"base_uri": "https://localhost:8080/", "height": 300}
finetuned_preds = get_median_and_ci(finetune_forecasts[0].samples,
                                    start_date='2012-09-07',
                                    horizon=8,
                                    freq='W-FRI',
                                    id=1,
                                    confidence=0.80)

finetuned_preds


# %% id="7IJJsgHwSHWV"
def calculate_mae_smape(pred_df, test_df, target_col, pred_col):

    # Extract the relevant columns
    y_true = test_df[target_col].values
    y_pred = pred_df[pred_col].values

    # Calculate MAE
    mae = int(np.mean(np.abs(y_true - y_pred)))

    # Calculate sMAPE
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = round(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100,2)

    return mae, smape


# %% id="sthEzou0SV13" outputId="ba0248a9-e87a-4079-8d03-149e8ec0a93a" colab={"base_uri": "https://localhost:8080/"}
mae_finetuned, smape_finetuned = calculate_mae_smape(finetuned_preds, test_df, 'Weekly_Sales', 'Lag-Llama')

print(mae_finetuned, smape_finetuned)

# %% id="45BiTe_4SdfQ"
