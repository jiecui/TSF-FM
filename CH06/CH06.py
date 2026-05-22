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

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

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
# ## Zero-shot forecasting with Moirai

# %%
df = pd.read_csv('../data/walmart_sales_small.csv', parse_dates=['Date'])

df.head()

# %%
df = df[['Store', 'Date', 'Weekly_Sales']]
df = df.set_index('Date')
df.head()

# %%
ds = PandasDataset.from_long_dataframe(df, target='Weekly_Sales', item_id='Store')

# %%
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
    prediction_length=8,
    context_length=len(df.query('Store == 1')),
    patch_size='auto',
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real
)

# %%
predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(ds)
forecasts = list(forecasts)


# %%
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
        'Moirai': medians,
        f'Moirai-lo-{int(confidence*100)}': lower_bounds,
        f'Moirai-hi-{int(confidence*100)}': upper_bounds
    })
    
    return df


# %%
preds = [
    get_median_and_ci(
        data=forecasts[i].samples,
        start_date='11-02-2012',
        horizon=8,
        freq='W-FRI',
        id=i+1
    )
    for i in range(4)
]

preds_df = pd.concat(preds, axis=0, ignore_index=True)
preds_df['Date'] = pd.to_datetime(preds_df['Date'])

preds_df

# %%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,9))

plot_df = df.reset_index(drop=False).copy()
plot_preds_df = preds_df.copy()

for i, ax in enumerate(axes.flatten()):
    df = plot_df[plot_df['Store'] == i+1].iloc[-100:]
    preds_df = plot_preds_df[plot_preds_df['Store'] == i+1]

    ax.plot(df['Date'], df['Weekly_Sales'])
    ax.plot(preds_df['Date'], preds_df['Moirai'], 'g--', label='Moirai')
    ax.fill_between(preds_df['Date'], preds_df['Moirai-lo-95'], preds_df['Moirai-hi-95'], alpha=0.2, color='#80c21d')
    ax.legend()
    ax.set_title(f'Store {i+1}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Sales ($)')

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH06_F06_peixeiro2.png", dpi=300)
plt.savefig('figures/CH06_F06_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Cross-validation with Moirai

# %%
df = pd.read_csv('../data/walmart_sales_small.csv', parse_dates=['Date'])
df = df.query("Store == 1")
df = df[['Store', 'Date', 'Weekly_Sales']]
df = df.set_index('Date')

df.head()

# %%
ds = PandasDataset.from_long_dataframe(df, 
                                       target="Weekly_Sales", 
                                       item_id="Store")

# %%
train, test_template = split(ds, offset=-32)

test_data = test_template.generate_instances(
    prediction_length=8,
    windows=4,
    distance=8
)

# %%
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    prediction_length=8,
    context_length=100,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(test_data.input)

# %%
forecasts = list(forecasts)

# %%
start_dates = ["2012-03-23", "2012-05-18", "2012-07-13", "2012-09-07"]

cv_preds = [
    get_median_and_ci(
        data=forecasts[i].samples,
        start_date=start_dates[i],
        horizon=8,
        freq='W-FRI',
        id=1
    )
    for i in range(4)
]

cv_preds_df = pd.concat(cv_preds, axis=0, ignore_index=True)
cv_preds_df['Date'] = pd.to_datetime(cv_preds_df['Date'])

cv_preds_df

# %%
cv_preds_df['Weekly_Sales'] = df.iloc[-32:]['Weekly_Sales'].values
cv_preds_df

# %%
fig, ax = plt.subplots(figsize=(10,7))

plot_df = df.reset_index(drop=False).copy()

ax.plot(plot_df['Date'], plot_df['Weekly_Sales'])
ax.plot(cv_preds_df['Date'], cv_preds_df['Moirai'], ls='--', color='green', label='Moirai')
ax.fill_between(cv_preds_df['Date'], cv_preds_df['Moirai-lo-95'], cv_preds_df['Moirai-hi-95'], color="#80c21d", alpha=0.2)

ax.set_title(f"Store 1")
ax.set_xlabel('Date')
ax.set_ylabel('Sales volume ($)')
ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH06_F07_peixeiro2.png", dpi=300)
plt.savefig('figures/CH06_F07_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %%
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

evaluation = evaluate(
    cv_preds_df,
    metrics=[mae, smape],
    models=['Moirai'],
    target_col='Weekly_Sales',
    id_col='Store'
)

evaluation

# %% [markdown]
# ## Forecasting with features

# %%
df = pd.read_csv('../data/walmart_sales_small.csv', parse_dates=['Date'])
df = df.query("Store == 1")
df = df.set_index('Date')

df.head()

# %%
ds = PandasDataset.from_long_dataframe(
    df,
    target='Weekly_Sales',
    item_id='Store',
    past_feat_dynamic_real=["Temperature", "Fuel_Price", "CPI", "Unemployment"],
    feat_dynamic_real=["Holiday_Flag"]
)

# %%
train, test_template = split(
    ds, offset=-32
)

test_data = test_template.generate_instances(
    prediction_length=8,
    windows=4,
    distance=8
)

# %%
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    prediction_length=8,
    context_length=100,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(test_data.input)
forecasts = list(forecasts)

# %%
start_dates = ["2012-03-23", "2012-05-18", "2012-07-13", "2012-09-07"]

cv_feat_preds = [
    get_median_and_ci(
        data=forecasts[i].samples,
        start_date=start_dates[i],
        horizon=8,
        freq='W-FRI',
        id=1
    )
    for i in range(4)
]

cv_feat_preds_df = pd.concat(cv_feat_preds, axis=0, ignore_index=True)
cv_feat_preds_df['Date'] = pd.to_datetime(cv_feat_preds_df['Date'])

cv_feat_preds_df['Weekly_Sales'] = df.iloc[-32:]['Weekly_Sales'].values
cv_feat_preds_df

# %%
fig, ax = plt.subplots(figsize=(10,7))

plot_df = df.reset_index(drop=False).copy()

ax.plot(plot_df['Date'], plot_df['Weekly_Sales'])
ax.plot(cv_feat_preds_df['Date'], cv_feat_preds_df['Moirai'], ls='--', color='green', label='Moirai')
ax.fill_between(cv_feat_preds_df['Date'], cv_feat_preds_df['Moirai-lo-95'], cv_feat_preds_df['Moirai-hi-95'], color="#80c21d", alpha=0.3)

ax.set_title(f"Store 1")
ax.set_xlabel('Date')
ax.set_ylabel('Sales volume ($)')
ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH06_F08_peixeiro2.png", dpi=300)
plt.savefig('figures/CH06_F08_peixeiro2.pdf', format='pdf', bbox_inches='tight')

# %%
evaluation = evaluate(
    cv_feat_preds_df,
    metrics=[mae, smape],
    models=['Moirai'],
    target_col='Weekly_Sales',
    id_col='Store'
)

evaluation

# %%
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

x = ['Moirai', 'Moirai + features']
y_mae = [84140, 82041]
y_smape = [2.64, 2.57]

ax1.bar(x, y_mae, width=0.4, label='MAE')
ax1.set_xlabel('Models')
ax1.set_ylabel('MAE ($)')
ax1.set_ylim(0, 100000)
ax1.legend()

for i, v in enumerate(y_mae):
    ax1.text(x=i, y=v+300, s=str(v), ha='center')

ax2.bar(x, y_smape, width=0.4, label='sMAPE')
ax2.set_xlabel('Models')
ax2.set_ylabel('sMAPE (%)')
ax2.set_ylim(0, 3)
ax2.legend()


for i, v in enumerate(y_smape):
    ax2.text(x=i, y=v+.03, s=str(v), ha='center')

plt.tight_layout()
plt.savefig("figures/CH06_codefig04.png", dpi=300)

# %% [markdown]
# ## Anomaly detection

# %%
df = pd.read_csv('../data/nyc_taxi_anomaly_daily.csv', parse_dates=['timestamp'])
df.head()


# %%
def anomaly_detection_moirai(df, 
                             target_col, 
                             id_col,
                             date_col,
                             h, 
                             n_windows,
                             freq,
                             confidence=0.99):

    ds = PandasDataset.from_long_dataframe(
        df,
        target=target_col,
        item_id=id_col,
    )
    
    train, test_template = split(
    ds, offset=-(n_windows*h)
    )
    
    test_data = test_template.generate_instances(
        prediction_length=h,
        windows=n_windows,
        distance=h
    )

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
        prediction_length=h,
        context_length=100,
        patch_size="auto",
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    
    predictor = model.create_predictor(batch_size=32)
    forecasts = predictor.predict(test_data.input)
    forecasts = list(forecasts)

    df_test = df[-(n_windows*h):]

    list_dates = df_test[date_col]
    start_dates = list(list_dates[::h])

    cv_preds = [
        get_median_and_ci(
            data=forecasts[i].samples,
            start_date=start_dates[i],
            horizon=h,
            freq=freq,
            id=0,
            confidence=confidence
        )
        for i in range(n_windows)
    ]

    cv_preds_df = pd.concat(cv_preds, axis=0, ignore_index=True)
    cv_preds_df['Date'] = pd.to_datetime(cv_preds_df['Date'])

    df_test.rename(columns={date_col: 'Date'}, inplace=True)
    df_test = pd.merge(df_test, cv_preds_df, on='Date')
    df_test['anomaly'] = ((df_test[target_col] < df_test['Moirai-lo-99']) | (df_test[target_col] > df_test['Moirai-hi-99'])).astype(int)

    df_test = df_test.drop('Store', axis=1)

    return df_test


# %%
df['unique_id'] = 0

anomaly_df = anomaly_detection_moirai(df,
                                      target_col="value",
                                      id_col="unique_id",
                                      date_col="timestamp",
                                      h=23, 
                                      n_windows=8, 
                                      freq='D', 
                                      confidence=0.99)
anomaly_df

# %%
actual_anomaly_df = df.loc[df['is_anomaly'] == 1]
pred_anomaly_df = anomaly_df.loc[anomaly_df['anomaly'] == 1] 

fig, ax = plt.subplots(figsize=(12,7))

ax.plot(df['timestamp'], df['value'])
ax.plot(actual_anomaly_df['timestamp'], actual_anomaly_df['value'], 'o', color='red', label='Anomaly (actual)')
ax.plot(pred_anomaly_df['Date'], pred_anomaly_df['value'], 'D', color='orange', label='Anomaly (predicted)')
ax.set_xlabel('Time')
ax.set_ylabel('Number of customers')
ax.legend(loc='best')

plt.tight_layout()
plt.savefig("figures/CH06_codefig05.png", dpi=300)


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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,8))

x = ['TimeGPT', 'Chronos', 'Moirai']
y = [0.50, 0.35, round(f1_score,2)]

ax.bar(x, y, width=0.4, label='99% confidence interval')
ax.set_xlabel('Models')
ax.set_ylabel('F1-Score')
ax.set_ylim(0,  0.7)
ax.legend()

for i, v in enumerate(y):
    ax.text(x=i, y=v+0.01, s=str(v), ha='center')

plt.tight_layout()
plt.savefig("figures/CH06_codefig06.png", dpi=300)

# %%
