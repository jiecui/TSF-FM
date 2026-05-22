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
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chronos import ChronosPipeline

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

# %% [markdown]
# ## Zero-shot forecasting with Chronos

# %%
df = pd.read_csv('../data/walmart_sales_small.csv', parse_dates=['Date'])
df.head()

# %%
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14,8))

for i, ax in enumerate(axes.flatten()):
    store_id = i+1
    data = df.query("Store == @store_id")
    
    ax.plot(data['Date'], data['Weekly_Sales'])
    ax.set_title(f"Store {store_id}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales volume ($)')

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig("figures/CH05_F08_peixeiro2.png", dpi=300)
plt.savefig('figures/CH05_F08_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %%
context = [torch.tensor(df.query("Store == @i")['Weekly_Sales'].to_numpy()) for i in range(1, 5)]

# %%
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",  # use "cpu" for CPU inference, "mps" for Apple Silicon, "cuda" for GPU supporting CUDA
    torch_dtype=torch.bfloat16,
)

predictions = pipeline.predict(
    context=context,
    prediction_length=8,
    num_samples=20,
)

# %%
start_date = pd.to_datetime('2012-10-26')
forecast_dates =  [start_date + pd.DateOffset(weeks=i) for i in range(1, 9)]

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14,8))

for i, ax in enumerate(axes.flatten()):
    store_id = i+1
    data = df.query("Store == @store_id")
    low, median, high = np.quantile(predictions[i].numpy(), [0.1, 0.5, 0.9], axis=0)
    
    ax.plot(data['Date'], data['Weekly_Sales'])
    ax.plot(forecast_dates, median, ls='--', color='#80c21d', label='Forecast')
    ax.fill_between(forecast_dates, low, high, color="#80c21d", alpha=0.2)
    
    ax.set_title(f"Store {store_id}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales volume ($)')
    ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig("figures/CH05_F10_peixeiro2.png", dpi=300)
plt.savefig('figures/CH05_F10_peixeiro2.pdf', format='pdf', bbox_inches='tight')


# %% [markdown]
# ## Cross-validation with Chronos

# %%
def cross_validation_chronos(df, h, n_windows, target_col):

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
            


# %%
df = df.query('Store == 1')

lows, medians, highs = cross_validation_chronos(df, 
                                                h=8, 
                                                n_windows=4, 
                                                target_col='Weekly_Sales')

# %%
test_df = df[['Store', 'Date', "Weekly_Sales"]][-32:]

test_df['low'] = lows
test_df['median'] = medians
test_df['high'] = highs

test_df.head()

# %%
fig, ax = plt.subplots(figsize=(10,7))

ax.plot(df['Date'], df['Weekly_Sales'])
ax.plot(test_df['Date'], test_df['median'], ls='--', color='green', label='Forecast')
ax.fill_between(test_df['Date'], test_df['low'], test_df['high'], color="#80c21d", alpha=0.2)

ax.set_title(f"Store 1")
ax.set_xlabel('Date')
ax.set_ylabel('Sales volume ($)')
ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig("figures/CH05_F11_peixeiro2.png", dpi=300)
plt.savefig('figures/CH05_F11_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %%
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

evaluation = evaluate(
    test_df,
    metrics=[mae, smape],
    models=['median'],
    target_col='Weekly_Sales',
    id_col='Store'
)

evaluation

# %% [markdown]
# ## Finetuned Chronos

# %%
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

x = ['Chronos', 'Chronos-finetuned']
y_mae = [77657, 63811]
y_smape = [2.44, 1.99]

ax1.bar(x, y_mae, width=0.4, label='MAE')
ax1.set_xlabel('Models')
ax1.set_ylabel('MAE ($)')
ax1.legend()

for i, v in enumerate(y_mae):
    ax1.text(x=i, y=v+300, s=str(v), ha='center')

ax2.bar(x, y_smape, width=0.4, label='sMAPE')
ax2.set_xlabel('Models')
ax2.set_ylabel('sMAPE (%)')
ax2.legend()

for i, v in enumerate(y_smape):
    ax2.text(x=i, y=v+.03, s=str(v), ha='center')

plt.tight_layout()
plt.savefig("figures/Ch05_codefig04.png", dpi=300)

# %% [markdown]
# ## Anomaly detection

# %%
df = pd.read_csv('../data/nyc_taxi_anomaly_daily.csv', parse_dates=['timestamp'])
df.head()


# %%
def anomaly_detection_chronos(df, h, n_windows, value_col, confidence=0.99):

    lows = []
    medians = []
    highs = []

    low_conf = (1 - confidence)/2
    
    for i in range(n_windows, 0, -1):
        context = torch.tensor(df[value_col][:-(h * i)])

        predictions = pipeline.predict(
            context=context,
            prediction_length=h,
            num_samples=20,
            limit_prediction_length=False
        )
    
        low, median, high = np.quantile(predictions[0].numpy(), [low_conf, 0.5, 1-low_conf], axis=0)
    
        lows.extend(low)
        medians.extend(median)
        highs.extend(high)

    df_test = df[-(n_windows*h):]
    
    df_test['low'] = lows
    df_test['median'] = medians
    df_test['high'] = highs

    df_test['anomaly'] = ((df_test[value_col] < df_test['low']) | (df_test[value_col] > df_test['high'])).astype(int)

    return df_test


# %%
anomaly_df = anomaly_detection_chronos(df, 
                                       h=23, 
                                       n_windows=8, 
                                       value_col='value', 
                                       confidence=0.99)
anomaly_df

# %%
actual_anomaly_df = df.loc[df['is_anomaly'] == 1]
pred_anomaly_df = anomaly_df.loc[anomaly_df['anomaly'] == 1] 

fig, ax = plt.subplots(figsize=(12,7))

ax.plot(df['timestamp'], df['value'])
ax.plot(actual_anomaly_df['timestamp'], actual_anomaly_df['value'], 'o', color='red', label='Anomaly (actual)')
ax.plot(pred_anomaly_df['timestamp'], pred_anomaly_df['value'], 'D', color='orange', label='Anomaly (predicted)')
ax.set_xlabel('Time')
ax.set_ylabel('Number of customers')
ax.legend(loc='best')

plt.tight_layout()
plt.savefig("figures/CH05_codefig05.png", dpi=300)


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
precision, recall, f1_score = evaluate_anomaly_detection(anomaly_df[-182:], 'anomaly', 'is_anomaly')

print(f"Precision: {round(precision,2)}")
print(f"Recall: {round(recall,2)}")
print(f"F1-Score: {round(f1_score,2)}")

# %%
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

x = ['Precision', 'Recall', 'F1-Score']
y_99_timegpt = [0.38, 0.75, 0.50]
y_99_chronos = [round(precision, 2), round(recall,2), round(f1_score,2)]

ax1.bar(x, y_99_timegpt, width=0.4, label='99% confidence interval')
ax1.set_title('TimeGPT')
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.legend()

for i, v in enumerate(y_99_timegpt):
    ax1.text(x=i, y=v+0.01, s=str(v), ha='center')

ax2.bar(x, y_99_chronos, width=0.4, label='99% confidence interval')
ax2.set_title('Chronos')
ax2.set_xlabel('Metrics')
ax2.set_ylabel('Score')
ax2.legend()

for i, v in enumerate(y_99_chronos):
    ax2.text(x=i, y=v+.01, s=str(v), ha='center')

plt.tight_layout()
plt.savefig("figures/CH05_codefig06.png", dpi=300)

# %% [markdown]
# ### Extras

# %% [markdown]
# #### Tokenization

# %%
chronos_df = df[['Store', 'Date', 'Weekly_Sales']]

# %%
chronos_df = chronos_df.query('Store == 1')
chronos_df


# %%
def mean_scaling(x):
    mean = np.mean(np.abs(x))

    return x/mean


# %%
y = chronos_df['Weekly_Sales'].values
y_scaled = mean_scaling(y)

# %%
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

ax1.plot(chronos_df['Date'], y, color='blue', label='Original')
ax1.set_ylabel('Weekly sales')
ax1.legend()

ax2.plot(chronos_df['Date'], y_scaled, ls='--', label='Scaled')
ax2.set_ylabel('Weekly sales (scaled)')
ax2.set_xlabel('Date')
ax2.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/Ch05_codefig07.png", dpi=300)


# %%
def uniform_binning(df, column, n_bins):
    
    bin_edges = pd.cut(df[column], bins=n_bins, labels=False)
    df[f'{column}_{n_bins}'] = bin_edges

    return df


# %%
chronos_df['sales_scaled'] = y_scaled

chronos_df = uniform_binning(chronos_df, 'sales_scaled', 100)
chronos_df

# %%
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

ax1.plot(chronos_df['Date'][:50], y_scaled[:50], ls='--', label='Scaled')
ax1.set_ylabel('Weekly sales (scaled)')
ax1.legend()

ax2.scatter(chronos_df['Date'][:50], chronos_df['sales_scaled_100'][:50], marker='s', s=5, label='Quantized')
ax2.set_ylabel('Bins')
ax2.set_xlabel('Date')
ax2.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/Ch05_codefig08.png", dpi=300)

# %%
chronos_df = uniform_binning(chronos_df, 'sales_scaled', 5)
chronos_df

# %%
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

ax1.scatter(binned_df['Date'][:50], chronos_df['sales_scaled_100'][:50], marker='s', s=5, label='Quantized')
ax1.set_ylabel('Bins')
ax1.set_xlabel('Date')
ax1.legend()

ax2.scatter(chronos_df['Date'][:50], chronos_df['sales_scaled_5'][:50], marker='s', s=5, label='Quantized')
ax2.set_ylabel('Bins')
ax2.set_xlabel('Date')
ax2.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/Ch05_codefig09.png", dpi=300)

# %% [markdown]
# #### Data augmentation

# %%
df = df.iloc[:, :3]
traffic_df = pd.read_csv("https://raw.githubusercontent.com/marcopeix/time-series-analysis/master/data/daily_traffic.csv", parse_dates=['date_time'])

sales_data = df.query("Store == 1").reset_index(drop=True)

n = min(len(sales_data), len(traffic_df))
sales_data = sales_data[:n]
traffic_data = traffic_df[:n]

def tsmixup(data1, data2, lambda_val=0.5):
    # lambda_val = np.random.beta(alpha, alpha)
    mixed_data = lambda_val * data1 + (1-lambda_val) * data2
    return mixed_data

mixed_series = tsmixup(
    mean_scaling(sales_data['Weekly_Sales']), 
    mean_scaling(traffic_data['traffic_volume']))

mixed_data = sales_data.copy()
mixed_data['mixed'] = mixed_series

# %%
fig, axs = plt.subplots(2, 2, figsize=(12,8))

axs[0,0].plot(sales_data['Date'], sales_data['Weekly_Sales'], label='Sales data')
axs[0,0].set_xlabel('Date')
axs[0,0].set_ylabel('Sales')
axs[0,0].legend()

axs[0,1].plot(traffic_data['date_time'], traffic_data['traffic_volume'], label='Traffic volume')
axs[0,1].set_xlabel('Date')
axs[0,1].set_ylabel('Volume')
axs[0,1].legend(loc=1)

axs[1,0].plot(mixed_data['Date'], mixed_data['mixed'], label='Mixed data')
axs[1,0].set_xlabel('Date')
axs[1,0].legend(loc=1)

fig.delaxes(axs[1,1])
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH05_codefig10.png", dpi=300)


# %%
