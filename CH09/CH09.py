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

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.losses.pytorch import *
from neuralforecast.utils import PredictionIntervals

from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

import warnings
warnings.filterwarnings('ignore')

# %%
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 14  
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14 
plt.rcParams['ytick.labelsize'] = 14 
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#000072", # blue (for historical data)
    "#80c21d", # green (for actual data)
    "#924eae", # purple
    "#ff0000", # red
    "#ff9100", # orange
])

# %%
data = pd.read_csv('../data/walmart_sales_small.csv', parse_dates=['Date'])
data.head()

# %% [markdown]
# ## Forecasting

# %%
prompt_prefix = """
    The dataset contains information on weekly sales in four different stores.
    Sales tend to increase in towards the end of the year.
"""

# %%
timellm = TimeLLM(
    h=8,
    input_size=16,
    d_llm=768,
    prompt_prefix=prompt_prefix,
    batch_size=16,
    windows_batch_size=16,
    max_steps=150
)

# %%
nf = NeuralForecast(
    models=[timellm],
    freq='W'
)

# %%
df = data.drop(['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'], axis=1)
df.head()

# %%
nf.fit(
    df=df,
    time_col='Date', 
    target_col='Weekly_Sales',
    id_col='Store'
)
preds = nf.predict()
preds.head()

# %%
preds = preds.reset_index()
preds.head()

# %%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,9))

for i, ax in enumerate(axes.flatten()):
    plot_df = df[df['Store'] == i+1][-50:]
    preds_df = preds[preds['Store'] == i+1]

    ax.plot(plot_df['Date'], plot_df['Weekly_Sales'], label='Actual')
    ax.plot(preds_df['Date'], preds_df['TimeLLM'], ls='--', label='TimeLLM')
    ax.legend()
    ax.set_title(f'Store {i+1}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Sales ($)')

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH09_F07_peixeiro2.png", dpi=300)
plt.savefig('figures/CH09_F07_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Cross-validation

# %%
store1_df = df.query("Store == 1")

# %%
cv_df = nf.cross_validation(
    df=store1_df,
    n_windows=4,
    step_size=8,
    refit=False,
    id_col='Store',
    target_col='Weekly_Sales',
    time_col='Date'
)
cv_df.head()

# %%
cv_df = cv_df.reset_index()

fig, ax = plt.subplots(figsize=(10,7))

ax.plot(store1_df['Date'], store1_df['Weekly_Sales'], label='Actual')
ax.plot(cv_df['Date'], cv_df['TimeLLM'], ls='--', label='TimeLLM')

ax.set_title(f"Store 1")
ax.set_xlabel('Date')
ax.set_ylabel('Sales volume ($)')
ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH09_F08_peixeiro2.png", dpi=300)
plt.savefig('figures/CH09_F08_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %%
test_df = cv_df.drop(['Date', 'cutoff'], axis=1)
evaluation = evaluate(
    test_df,
    metrics=[mae, smape],
    models=['TimeLLM'],
    target_col='Weekly_Sales',
    id_col='Store'
)
evaluation

# %% [markdown]
# ## Anomaly detection

# %%
df_anomaly = pd.read_csv('../data/nyc_taxi_anomaly_daily.csv', parse_dates=['timestamp'])
df_anomaly['unique_id'] = 1
df_anomaly.head()

# %%
len(df_anomaly)

# %%
horizon = 20
input_size = 30

prediction_intervals = PredictionIntervals()

prompt_prefix = """
    The dataset contains information on daily taxi rides in New York City.
    There is a weekly seasonality.
"""

timellm = TimeLLM(
    h=horizon,
    input_size=input_size,
    d_llm=768,
    prompt_prefix=prompt_prefix,
    batch_size=16,
    windows_batch_size=16,
    max_steps=10
)

nf = NeuralForecast(models=[timellm], freq='D')

anomaly_cv_df = nf.cross_validation(
    df=df_anomaly,
    n_windows=8,
    step_size=horizon,
    refit=True,
    prediction_intervals=prediction_intervals,
    level=[99],
    target_col='value',
    time_col='timestamp'
)

# %%
anomaly_cv_df = anomaly_cv_df.reset_index()
anomaly_cv_df.head()

# %%
anomaly_cv_df['anomaly'] = (
    (anomaly_cv_df['value'] < anomaly_cv_df['TimeLLM-lo-99']) | \
    (anomaly_cv_df['value'] > anomaly_cv_df['TimeLLM-hi-99'])
).astype(int)

anomaly_cv_df.head()

# %%
actual_anomaly_df = df_anomaly.loc[df_anomaly['is_anomaly'] == 1]
pred_anomaly_df = anomaly_cv_df.loc[anomaly_cv_df['anomaly'] == 1] 

fig, ax = plt.subplots(figsize=(12,7))

ax.plot(df_anomaly['timestamp'], df_anomaly['value'])
ax.plot(actual_anomaly_df['timestamp'], actual_anomaly_df['value'], 'o', color='#ff0000', label='Anomaly (actual)')
ax.plot(pred_anomaly_df['timestamp'], pred_anomaly_df['value'], 'D', color='#ff9100', label='Anomaly (predicted)')
ax.plot(anomaly_cv_df['timestamp'], anomaly_cv_df['TimeLLM'], ls='--', color="#80c21d")
ax.fill_between(anomaly_cv_df['timestamp'], anomaly_cv_df['TimeLLM-lo-99'], anomaly_cv_df['TimeLLM-hi-99'], color='#80c21d', alpha=0.2, label='99 confidence interval')
ax.set_xlabel('Time')
ax.set_ylabel('Number of customers')
ax.legend(loc=2)
plt.tight_layout()
plt.savefig("figures/CH09_F09_peixeiro2.png", dpi=300)
plt.savefig('figures/CH09_F09_peixeiro2.pdf', format='pdf', bbox_inches='tight')


# %%
def evaluate_anomaly_detection(df, preds_col, actual_col):
    tp = ((df[preds_col] == 1) & (df[actual_col] == 1)).sum()
    
    tn = ((df[preds_col] == 0) & (df[actual_col] == 0)).sum()
    
    fp = ((df[preds_col] == 1) & (df[actual_col] == 0)).sum()
    
    fn = ((df[preds_col] == 0) & (df[actual_col] == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f1_score


# %%
n_pad_values = 184 - len(anomaly_cv_df)
pred_anomalies = np.pad(anomaly_cv_df['anomaly'].values, (n_pad_values, 0), 'constant')

df_anomaly = df_anomaly[-184:]
df_anomaly['pred_anomaly'] = pred_anomalies

df_anomaly.head()

# %%
precision, recall, f1_score = evaluate_anomaly_detection(df_anomaly, 'pred_anomaly', 'is_anomaly')

print(f"Precision: {round(precision,2)}")
print(f"Recall: {round(recall,2)}")
print(f"F1-Score: {round(f1_score,2)}")

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,8))

x = ['TimeGPT', 'Chronos', 'Moirai', 'Llama-3.2', 'TimeLLM']
y = [0.50, 0.35, 0.53, 0.12, 0.15]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('F1-Score')
ax.set_ylim(0,  0.7)

for i, v in enumerate(y):
    ax.text(x=i, y=v+0.01, s=str(v), ha='center', fontsize=18)

plt.tight_layout()
plt.savefig("figures/CH09_codefig4_peixeiro2.png", dpi=300)

# %%
