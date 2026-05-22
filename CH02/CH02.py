# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: tsf-fm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter 2 Building a foundation model

# %% [markdown]
# ## Importing libraries and setting up the environment

# %%
import pandas as pd
from cycler import cycler
from datasetsforecast.m3 import M3
import warnings
from neuralforecast.models import NBEATS

warnings.filterwarnings("ignore")

# %%
import matplotlib.pyplot as plt

# %%
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "#000072",  # blue (for historical data)
        "#80c21d",  # green (for actual data)
        "#924eae",  # purple
        "#ff0000",  # red
        "#ff9100",  # orange
    ]
)

# %% [markdown]
# ## Load M3 dataset

# %%
Y_df, *_ = M3.load(directory="../data/", group="Monthly")
Y_df["ds"] = pd.to_datetime(Y_df["ds"])

print(Y_df.head())
print(f"Number of unique time series: {Y_df['unique_id'].nunique()}")

# %% [markdown]
# ## Plotting the first four series of the M3 dataset

# %%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

for i, ax in enumerate(axes.flatten()):
    id = f"M{i+1}"
    filtered_df = Y_df[Y_df["unique_id"] == id]

    ax.plot(filtered_df["ds"], filtered_df["y"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"M{i+1}")

plt.tight_layout()

plt.savefig("figures/CH02_F03_peixeiro.png", dpi=300)
plt.savefig("figures/CH02_F03_peixeiro.pdf", format="pdf", bbox_inches="tight")

# %% [markdown]
# ## Pretraining N-BEATS

# %% [markdown]
# ### Initialize the model

# %%
import os

# %%
# use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
horizon = 12
models = [
    NBEATS(
        input_size=2 * horizon,
        h=horizon,
        max_steps=1000,
        enable_progress_bar=False,
    )
]

# %% [markdown]
# ### Pretrain the model

# %%
from neuralforecast.core import NeuralForecast

# %%
nf = NeuralForecast(models=models, freq="M")
nf.fit(df=Y_df)

# %% [markdown]
# ### Save model

# %%
nf.save(path="./model", model_index=None, overwrite=True, save_dataset=False)

# %% [markdown]
# ## Transfer learning

# %% [markdown]
# ### Load model

# %%
pretrained_model = NeuralForecast.load(path="./model")

# %% [markdown]
# ### Load data
#

# %%
df = pd.read_csv("../data/AusAntidiabeticDrug.csv")
df["ds"] = pd.to_datetime(df["ds"])
df["ds"] = df["ds"] + pd.offsets.MonthEnd(0)
df.insert(0, "unique_id", 1)

df.head()

# %% [markdown]
# ### Show the data

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["ds"], df["y"])
ax.set_xlabel("Date")
ax.set_ylabel("Volume of anti-diabetic drug prescriptions")

plt.tight_layout()

plt.savefig("figures/CH02_F04_peixeiro.png", dpi=300)
plt.savefig("figures/CH02_F04_peixeiro.pdf", format="pdf", bbox_inches="tight")

# %% [markdown]
# ### Split data into train and test

# %%
input_df = df[:-12]
test_df = df[-12:]

# %% [markdown]
# ### Zero-shot forecasting

# %%
zero_shot_preds = pretrained_model.predict(input_df)
zero_shot_preds.head()


# %% [markdown]
# ## Fine-tuning the pretrained model

# %% [markdown]
# ### Setting the maximum number of steps

# %%
def set_max_steps(nf, max_steps):
    trainer_kwargs = {**{"max_steps": max_steps}}
    nf.models[0].trainer_kwargs = trainer_kwargs


set_max_steps(pretrained_model, 10)

# %% [markdown]
# ### Fine-tune the model

# %%
pretrained_model.fit(input_df)

# %% [markdown]
# ### Show the forecast

# %%
finetuned_preds = pretrained_model.predict()
finetuned_preds.head()

# %% [markdown]
# ### Train a data-specific model

# %%
horizon = 12

# %%
models = [
    NBEATS(
        input_size=2 * horizon,
        h=horizon,
        max_steps=100,
        enable_progress_bar=False,
    )
]

nf = NeuralForecast(models=models, freq="M")
nf.fit(df=input_df)

# %%
trained_preds = nf.predict()
trained_preds.head()

# %% [markdown]
# ## Evaluation of the models

# %% [markdown]
# ### Combining predictions in a single DataFrame

# %%
zero_shot_preds = zero_shot_preds.rename(columns={"NBEATS": "NBEATS_zeroshot"})
finetuned_preds = finetuned_preds.rename(columns={"NBEATS": "NBEATS_finetuned"})
trained_preds = trained_preds.rename(columns={"NBEATS": "NBEATS_trained"})

# test_df = pd.merge(test_df, zero_shot_preds, 'left', 'ds')
# test_df = pd.merge(test_df, finetuned_preds, 'left', 'ds')
# test_df = pd.merge(test_df, trained_preds, 'left', 'ds')

# Merge on both unique_id and ds to avoid duplicates
test_df = pd.merge(test_df, zero_shot_preds, on=["unique_id", "ds"], how="left")
test_df = pd.merge(test_df, finetuned_preds, on=["unique_id", "ds"], how="left")
test_df = pd.merge(test_df, trained_preds, on=["unique_id", "ds"], how="left")

test_df.head()

# %% [markdown]
# ### Show the forecast

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["ds"].iloc[-60:], df["y"].iloc[-60:])
ax.plot(test_df["ds"], test_df["y"], label="Actual")
ax.plot(test_df["ds"], test_df["NBEATS_zeroshot"], ls="--", label="Zero-shot")
ax.plot(test_df["ds"], test_df["NBEATS_finetuned"], ls="-.", label="Fine-tuned")
ax.plot(test_df["ds"], test_df["NBEATS_trained"], ls=":", label="Trained")

ax.set_xlabel("Date")
ax.set_ylabel("Volume of anti-diabetic drug prescriptions")

ax.legend(loc="best")

plt.tight_layout()

plt.savefig("figures/CH02_F05_peixeiro.png", dpi=300)
plt.savefig("figures/CH02_F05_peixeiro.pdf", format="pdf", bbox_inches="tight")

# %% [markdown]
# ### Evaluating the performance of the models

# %% [markdown]
# Mean absolute error (MAE) is defined as:
# $$MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|.$$
#
# Symmetric mean absolute percentage error (sMAPE) is defined as:
# $$sMAPE=2 \times \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i-\hat{y}_i|}{(|y_i|+|\hat{y}_i|)}.$$

# %%
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

evaluation = evaluate(
    test_df,
    metrics=[mae, smape],
    models=["NBEATS_zeroshot", "NBEATS_finetuned", "NBEATS_trained"],
    target_col="y",
)

evaluation = evaluation.drop(["unique_id"], axis=1)
evaluation = evaluation.set_index("metric")
evaluation

# %%
ax = evaluation.plot(kind="bar", figsize=(10, 6), rot=0)

# Adding labels and title
plt.xlabel("Metric")
plt.ylabel("Value")
plt.title("Comparison of MAE and sMAPE for Different Models")
plt.legend(title="Model")

# Show plot
plt.show()

# %% [markdown]
# ## Forecasting another frequency

# %% [markdown]
# ### Loading daily data

# %%
daily_df = pd.read_csv("../data/daily_min_temp.csv")
daily_df = daily_df.rename(columns={"Date": "ds", "Temp": "y"})
daily_df["ds"] = pd.to_datetime(daily_df["ds"])
daily_df.insert(0, "unique_id", 1)

daily_df.head()

# %% [markdown]
# ### Splitting data into train and test  

# %%
d_input_df = daily_df[:-12]
d_test_df = daily_df[-12:]

# %% [markdown]
# ### Zero-shot forecasting

# %%
pretrained_model = NeuralForecast.load(path="./model")

d_zero_shot_preds = pretrained_model.predict(d_input_df)
d_zero_shot_preds.head()

# %% [markdown]
# ### Training a data-specific model

# %%
models = [
    NBEATS(
        input_size=2 * horizon,
        h=horizon,
        max_steps=500,
        enable_progress_bar=False,
    )
]

nf = NeuralForecast(models=models, freq="D")
nf.fit(df=d_input_df)

# %% [markdown]
# ### Predicting with the data-specific model

# %%
d_trained_preds = nf.predict()
d_trained_preds.head()

# %% [markdown]
# ### Combining the forecasts

# %%
d_zero_shot_preds = d_zero_shot_preds.rename(columns={"NBEATS": "NBEATS_zero_shot"})
d_zero_shot_preds = d_zero_shot_preds.reset_index(drop=True)
d_trained_preds = d_trained_preds.rename(columns={"NBEATS": "NBEATS_trained"})

# d_test_df = pd.merge(d_test_df, d_trained_preds, "left", "ds")
d_test_df = pd.merge(d_test_df, d_trained_preds, on=["unique_id", "ds"], how="left")
d_test_df = pd.concat([d_test_df, d_zero_shot_preds["NBEATS_zero_shot"]], axis=1)

print(d_test_df)

# %% [markdown]
# ### Plotting the forecasts

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(daily_df["ds"].iloc[-60:], daily_df["y"].iloc[-60:])
ax.plot(d_test_df["ds"], d_test_df["y"], label="Actual")
ax.plot(d_test_df["ds"], d_test_df["NBEATS_zero_shot"], ls="--", label="Zero-shot")
ax.plot(d_test_df["ds"], d_test_df["NBEATS_trained"], ls=":", label="Trained")

ax.set_xlabel("Date")
ax.set_ylabel("Minimum temperature (Celsius)")

ax.legend(loc="best")

plt.tight_layout()
fig.autofmt_xdate()

plt.savefig("figures/CH02_F06_peixeiro.png", dpi=300)
plt.savefig("figures/CH02_F06_peixeiro.pdf", format="pdf", bbox_inches="tight")

# %% [markdown]
# ### Evaluate the forecasts using MASE and sMAPE

# %%
d_evaluation = evaluate(
    d_test_df,
    metrics=[mae, smape],
    models=["NBEATS_zero_shot", "NBEATS_trained"],
    target_col="y",
)

d_evaluation = d_evaluation.drop(["unique_id"], axis=1)
d_evaluation = d_evaluation.set_index("metric")
d_evaluation
