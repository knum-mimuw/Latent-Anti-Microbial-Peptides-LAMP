import os
import random
import multiprocessing
import itertools
import functools
from joblib import dump, load
from pathlib import Path
import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.spatial.distance import cdist

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

LATENT_DIM = 64
seed = 7
np.random.seed(seed)
random.seed(seed)

def rbf_kernel(X, Y, gamma):
    D = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * D)

def mmd(X, Y, gamma=None):
    # Combine for median trick if needed
    Z = np.vstack([X, Y])
    if gamma is None:
        dists = cdist(Z, Z, 'sqeuclidean')
        median_sq_dist = np.median(dists)
        gamma = 1 / (2 * median_sq_dist)

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    n = len(X)
    m = len(Y)

    result = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) \
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) \
        - 2 * K_XY.mean()
    return result

mmd_score = mmd(X_test, pca_sample)
print(f"MMD between X_test and PCA-sampled data: {mmd_score:.6f}")


class Flow(nn.Module):
    def __init__(self, dim: int = X_train.shape[1], h: int = 64):
        super().__init__()
        ALPHA = 0.15
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(alpha=ALPHA),
            nn.Linear(h, h), nn.ELU(alpha=ALPHA),
            nn.Linear(h, h), nn.ELU(alpha=ALPHA),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        delta_t = t_end - t_start
        return x_t + delta_t * self(t=t_start + delta_t / 2,
                                    x_t=x_t + self(x_t=x_t, t=t_start) * delta_t / 2)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

BATCH_SIZE = 2048
dataset = TensorDataset(X_train_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

flow = Flow().to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

ITER = 200

print("Starting training loop")

for i in range(ITER):
    for (x_1,) in loader:
        x_0 = torch.randn_like(x_1, device=device)
        t = torch.rand(len(x_1), 1, device=device)

        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        optimizer.zero_grad()
        loss = loss_fn(flow(t=t, x_t=x_t), dx_t)
        loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f"Loss at iteration {i}: {loss.item()}")

n_samples = X_test.shape[0]
z = np.random.randn(n_samples, LATENT_DIM)
z = torch.tensor(z, dtype=torch.float32) # The same 'z' we used for PCA.
device = torch.device("cpu")
flow_cpu = flow.to(device)

n_steps = 300
time_steps = torch.linspace(0, 1.0, n_steps + 1)
flow_sample = z.clone().to(device)

for i in range(n_steps):
    t_start = time_steps[i]
    t_end = time_steps[i + 1]
    flow_sample = flow_cpu.step(x_t=flow_sample, t_start=t_start, t_end=t_end)