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
#     display_name: tsf-fm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter 1 Understanding foundation models

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
# Set random seed for reproducibility
np.random.seed(42)

# %%
NUM_POINTS = 50

x = np.linspace(0, 10, NUM_POINTS)
pos_y = 2 * x + np.random.normal(0, 2, NUM_POINTS)
neg_y = -2 * x + np.random.normal(0, 2, NUM_POINTS)
x = x.reshape(-1,1)

pos_lr = LinearRegression().fit(x, pos_y)
neg_lr = LinearRegression().fit(x, neg_y)

pos_pred = pos_lr.predict(x)
neg_pred = neg_lr.predict(x)

# %%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

ax1.scatter(x, pos_y, color="blue")
ax1.plot(x, pos_pred)
for xp, yp, y_predp in zip(x, pos_y, pos_pred):
    ax1.plot([xp, xp], [yp, y_predp], color="red", ls="--")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.scatter(x, neg_y, color="blue")
ax2.plot(x, neg_pred)
for xp, yp, y_predp in zip(x, neg_y, neg_pred):
    ax2.plot([xp, xp], [yp, y_predp], color="red", ls="--")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.suptitle("Figure 1.1")
plt.tight_layout()

plt.savefig("figures/CH01_F01_peixeiro.png", dpi=300)

# %%
