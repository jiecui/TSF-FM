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

# %% colab={"base_uri": "https://localhost:8080/"} id="QTwzXXw1NH1q" outputId="debe81df-1426-4d54-cfc2-ae85941778fb"
# !pip install timesfm

# %% colab={"base_uri": "https://localhost:8080/"} id="7fSIYLv6NLAx" outputId="a2eb8459-94f7-45c7-8101-7eb321714f33"
import timesfm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# %% id="dW0QNy_2eQpo"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#000072", # blue
    "#80c21d", # green
    "#924eae", # purple
    "#ff0000", # red
    "#ff9100", # orange
])

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fN88dw8SRYyn" outputId="8c52810d-80e4-48b8-fb58-9aaa834f4f78"
df = pd.read_csv("https://raw.githubusercontent.com/marcopeix/FoundationModelsForTimeSeriesForecasting/main/data/walmart_sales_small.csv")
df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 209, "referenced_widgets": ["c87f195b18204e06af2e8da07694ca05", "1a250e020d6c464f8fe789280f7246ef", "0154ebdf836e4b37a9ab497796b11773", "e1d356eb17dc452d8ca9ab37d6940491", "9801f21b22b849598938659c3a363346", "82756f461ef84a62b8121fb5954031a0", "a4a4eb293f624629b7feacb6355c7399", "c15c52cb85bc4f7287eb87d001edb089", "645bfd08b5fa4283932274bba4516971", "898a14d0094140459393c2d2d21e8288", "d85e642d953245e7858b46d259883738", "d43262bb54044720bf2888393310d636", "00e196ce95e64274a488d9e2d8f552f2", "7d911f3a3fbf444fb0fdadb4ba9a06e6", "c9d1281de179400aa790daad98242167", "da8addd2683f4eb287a2e562e65c8f63", "ca0cff1d3d3b4287ac4eca310eac10bf", "38c5cdb631464de8b2abcd71c4c424b9", "3b59f3f80c8b4d849a0b32a427a134ec", "6dab4dd01bbc4506afda5da99cf7a72f", "c6fdf1ff31b349e9a674f94b81fd8f2a", "b7389af51a8b4b6db868bfb64830c38b", "9f1075f7e73b4d65af1b5b030ce92a48", "d8411fc176e546359f69736f862219c9", "f9c90317fbe54dbf985e13ee1015a2a7", "79f8f0be8c8648b38e4785a3d6573307", "a631d9a3f908428684ab7cfb57c01498", "fa3f1561ca04467b883d61a5644fd30c", "2dc4a9c335964e08a85f5b9774f8dccc", "17265deb6fd748b7831e850d623d094e", "c2911a8911534380977d93d7d3d3c65f", "2dc6f835e83b4ced8aa96fd8c4d849be", "0477ec157d9644c1909cb57b4bdcebd4", "190ae2d5dddd49c1959c453936ad01bf", "b87c027eff7f4cb19838a58ebc356592", "ab0195e10cc1457f85d26cb284d864fa", "f8ed609dc3aa4200aa6311bbba29b6d9", "5885343bf9384976879b157981b5a9b4", "9d57fa0780b94793bba1bdd01f241d2c", "78b4271b19f14c5d8bfd4b60f76b077c", "08683bbce14540969d0255f79f1a76a3", "6e5ff10e31f24234be5e88151031f515", "889e12e9bb3145e78d8ffba3607a4ec5", "76ec7a01585a4c7fb0148799a7ba6a8d", "940b1b02e08449cb897859a0020d546d", "f83566f541924ecab56f24aeb01e1899", "8a45896aad6345179e94547e7fa1afe5", "eb83748cde8f43198b42dac80060cc75", "dc7fe1e80d8b4ce9b83384074920e870", "63eb3556a7b942858625d928aaa59917", "3dc36ee33bad4ed2ad9a283ac2f6adbe", "cce6c714bbca444598559c694b00fb50", "06cdef830e744d51842a5ebdcc8f5ae7", "ae85ddc19ee14d6996a4d7177fbab949", "0b2e114c5649491b878a68f39ef809fa", "e16a66f52f9746de974967deaffcf42f", "fe8c083dcea343e2980dafbe2b35c48a", "a68503939cdd46c187e05e5bd6715d80", "983944ffa6ed4014914b4bb1190fb180", "d993b034d24a4f238d48be1932ee21ec", "3f426fb7c4504f8e870d28eb617da6ba", "97fc9194330848c08beb2c808e32e251", "94278b0c689643aba4e1560eb21e4559", "e69f3da0cfb347f2b8b90641a5987caa", "fd163c3b14c44dc4a2f3c8ce4de49ab2", "aaedea1b267246af8dd1161d7cc03a77"]} id="42aEU2yPQRzv" outputId="ceddb333-d293-49e7-a73f-d9a41c690577"
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu", # "gpu" if CUDA is available
          per_core_batch_size=32,
          horizon_len=8,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="EfK1SvOPSuso" outputId="bbd6f283-bda8-42cf-b425-de870e40f3b7"
df = df.rename(columns={"Store": "unique_id", "Date": "ds"})
df['ds'] = pd.to_datetime(df['ds'])
df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="6EULLtxVR0T5" outputId="167e1bd8-4f2f-4d1f-b0d7-8376ed446cdb"
preds_df = tfm.forecast_on_df(
    inputs=df,
    freq="W",
    value_name="Weekly_Sales",
    num_jobs=-1
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 244} id="dfJEaHIlS5H0" outputId="ab04b5fa-2149-416e-c56e-bd1a8baaf87e"
preds_df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 665} id="uo-j3FNXTRWk" outputId="a6e57776-dee4-488d-eaef-9ea18e09c0e2"
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

for i, ax in enumerate(axes.flatten()):
    store_id = i+1
    data = df.query("unique_id == @store_id")
    preds = preds_df.query("unique_id == @store_id")

    ax.plot(data['ds'], data['Weekly_Sales'])
    ax.plot(preds['ds'], preds['timesfm'], label='TimesFM')
    ax.fill_between(preds['ds'], preds['timesfm-q-0.1'], preds['timesfm-q-0.9'], color="#80c21d", alpha=0.2)

    ax.set_title(f"Store {store_id}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales volume ($)')
    ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()


# %% [markdown] id="wqyVGaRWX9Ed"
# ## Cross-validation with TimesFM

# %% id="24d5wGepX6t2"
def cross_validation_timesfm(df, h, n_windows, target_col, freq):
  all_preds = []

  for i in range(n_windows, 0, -1):
    input_df = df.iloc[:-(h*i)]

    preds_df = tfm.forecast_on_df(
    inputs=input_df,
    freq=freq,
    value_name=target_col,
    num_jobs=-1
    )

    all_preds.append(preds_df)

  preds = pd.concat(all_preds, axis=0, ignore_index=True)

  return preds


# %% id="6QNWSiSppHWW"
cv_df = df.query("unique_id == 1")

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="LQvpZb4bavzl" outputId="fa719157-2c72-4112-e452-d03004e4c703"
cv_preds = cross_validation_timesfm(
    df=cv_df,
    h=8,
    n_windows=4,
    target_col="Weekly_Sales",
    freq="W")

cv_preds.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 695} id="6hdrb6m1bhlT" outputId="a6971514-8fd1-48f8-bc67-db771dac651b"
fig, ax = plt.subplots(figsize=(10,7))

ax.plot(cv_df['ds'], cv_df['Weekly_Sales'])
ax.plot(cv_preds['ds'], cv_preds['timesfm'], ls='--', color='green', label='Forecast')
ax.fill_between(cv_preds['ds'], cv_preds['timesfm-q-0.1'], cv_preds['timesfm-q-0.9'], color="#80c21d", alpha=0.2)

ax.set_title(f"Store 1")
ax.set_xlabel('Date')
ax.set_ylabel('Sales volume ($)')
ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()

# %% id="1DIzN1EHcZ1i"
eval_df = cv_preds[['unique_id', 'ds', 'timesfm']]
eval_df['Weekly_Sales'] = cv_df['Weekly_Sales'][-32:].values

# %% colab={"base_uri": "https://localhost:8080/", "height": 125} id="8XFxSOfyf0qs" outputId="fc2e9802-4c9e-4359-c60f-8b9db547b9c6"
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

evaluation = evaluate(
    eval_df,
    metrics=[mae, smape],
    models=['timesfm'],
    target_col='Weekly_Sales',
    id_col='unique_id'
)

evaluation

# %% [markdown] id="0r60TgSzidz9"
# ## Forecasting with covariates

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="hh1WvAMmf9b-" outputId="a56fc04a-cdff-488b-eae7-8e74bbd3a0c2"
train = cv_df[:-32]
test = cv_df[-32:]

train.head()

# %% id="IJOtmcQKUMl1"
from collections import defaultdict

# Data pipelining
def get_batched_data_fn(
    batch_size: int = 2,
    context_len: int = 64,
    horizon_len: int = 32,
):
    examples = defaultdict(list)

    num_examples = 0
    for start in range(0, len(cv_df) - (context_len + horizon_len), horizon_len):
        num_examples += 1
        examples["inputs"].append(train["Weekly_Sales"][start:(context_end := start + context_len)].tolist())
        examples["Holiday_Flag"].append(train["Holiday_Flag"][start:context_end + horizon_len].tolist())
        examples["outputs"].append(train["Weekly_Sales"][context_end:(context_end + horizon_len)].tolist())

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn


# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["f6f8c52a8a664056b241cb798c4b2381", "e6c185ad6d014cabbe309b4316a2d160", "3a2c67d2a9914ee6ab586932212c6881", "b2dcdfba43b544d1a4bfe496795c90f7", "cd29520d78c24f5f9c73d802ccf29c7d", "f19276b5be9d4122aab395fa935a73ca", "b95b17a7f797415b8c81e964cdd2ca6d", "88a1bd03923743729f460408650015f4", "28dc60ce6f374ed3a851a1572eb165a1", "be4726c246fb4fccbbd4a5f65ea9a739", "58feef66944f4804b46bf65eba045c63"]} id="RRqMma5jZWOb" outputId="07e9985f-322d-415c-d15a-a4962fe122a3"
tfm_h32 = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu", # "gpu" if CUDA is available
          per_core_batch_size=32,
          horizon_len=32,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )

# %% colab={"base_uri": "https://localhost:8080/"} id="ikOUq-XFs_7E" outputId="bb8e40e1-7360-42c7-ba5c-791ed7495095"
input_data = get_batched_data_fn()

for i, example in enumerate(input_data()):
    cov_forecast, _ = tfm_h32.forecast_with_covariates(
        inputs=example["inputs"],
        dynamic_numerical_covariates={},
        dynamic_categorical_covariates={
            "Holiday_Flag": example["Holiday_Flag"],
        },
        static_numerical_covariates={},
        static_categorical_covariates={},
        freq=[1] * len(example["inputs"]),
        xreg_mode="xreg + timesfm",
        ridge=0.0,
        force_on_cpu=False,
        normalize_xreg_target_per_input=True,
    )
    print(f"Done with round {i}")

# %% colab={"base_uri": "https://localhost:8080/"} id="e9ntSkPWtAtT" outputId="9feb7768-46cd-4e66-d9cd-a9b1561a5ec0"
cov_forecast[0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 296} id="6hXY7u34V8V4" outputId="eb677d6c-b700-4f39-e8d5-a9770cb56bce"
no_cov_preds = tfm_h32.forecast_on_df(
    inputs=train,
    freq="W",
    value_name="Weekly_Sales",
    num_jobs=-1
)

no_cov_preds.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 695} id="UtXa-YbZvQzo" outputId="dd8ee98c-b06f-4ac9-993f-0ce7b0660ace"
fig, ax = plt.subplots(figsize=(10,7))

ax.plot(train['ds'], train['Weekly_Sales'])
ax.plot(test['ds'], test['Weekly_Sales'])
ax.plot(test['ds'], cov_forecast[0], ls='--', label='Forecast (w/ covariates)')
ax.plot(test['ds'], no_cov_preds['timesfm'], ls=':', label='Forecast w/o covariates')

ax.set_title(f"Store 1")
ax.set_xlabel('Date')
ax.set_ylabel('Sales volume ($)')
ax.legend(loc=1)

fig.autofmt_xdate()
plt.tight_layout()

# %% colab={"base_uri": "https://localhost:8080/", "height": 125} id="YqtW-x5pvkM3" outputId="004cc205-0340-4095-acce-af97f9dfdf25"
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

eval_df = test[['unique_id', 'ds', 'Weekly_Sales']]
eval_df['timesfm_cov'] = cov_forecast[0]
eval_df['timesfm'] = no_cov_preds['timesfm'].values

evaluation = evaluate(
    eval_df,
    metrics=[mae, smape],
    models=['timesfm', 'timesfm_cov'],
    target_col='Weekly_Sales',
    id_col='unique_id'
)

evaluation

# %% colab={"base_uri": "https://localhost:8080/", "height": 792} id="aDWQN1MXucUL" outputId="820e8729-1077-4be1-e383-a1b0fb122eb5"
plt.rcParams.update({'font.size': 15})
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

x = ['TimesFM', 'TimesFM + features']
y_mae = [64579, 81961]
y_smape = [2.02, 2.57]

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

# %% id="50d3bUZXMvrG"
