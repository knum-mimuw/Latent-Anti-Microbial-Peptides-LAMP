import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Dict

from scipy.spatial.distance import cdist

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import pytorch_lightning as pl

LATENT_DIM = 64


def rbf_kernel(X, Y, gamma):
    """RBF kernel for MMD computation."""
    D = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * D)


def mmd(X, Y, gamma=None):
    """Maximum Mean Discrepancy between two distributions."""
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


class FlowMatchingLightning(pl.LightningModule):
    """PyTorch Lightning module for Flow Matching model."""
    
    def __init__(
        self,
        dim: int = 64,
        hidden_dim: int = 64,
        learning_rate: float = 1e-2,
        alpha: float = 0.15,
        n_inference_steps: int = 300,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_inference_steps = n_inference_steps
        
        # Build the flow network
        self.flow_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), 
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, dim)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        """Forward pass of the flow network."""
        return self.flow_net(torch.cat((t, x_t), -1))
    
    def training_step(self, batch, batch_idx):
        """Training step for flow matching."""
        x_1 = batch[0]  # Real data
        batch_size = x_1.shape[0]
        
        # Sample noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=self.device)
        
        # Linear interpolation
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target vector field
        dx_t = x_1 - x_0
        
        # Predicted vector field
        v_pred = self(t=t, x_t=x_t)
        
        # Compute loss
        loss = self.loss_fn(v_pred, dx_t)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x_1 = batch[0]
        batch_size = x_1.shape[0]
        
        # Sample noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=self.device)
        
        # Linear interpolation
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target vector field
        dx_t = x_1 - x_0
        
        # Predicted vector field
        v_pred = self(t=t, x_t=x_t)
        
        # Compute loss
        loss = self.loss_fn(v_pred, dx_t)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def euler_step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        """Single Euler integration step."""
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        delta_t = t_end - t_start
        return x_t + delta_t * self(t=t_start + delta_t / 2,
                                    x_t=x_t + self(x_t=x_t, t=t_start) * delta_t / 2)
    
    def sample(self, n_samples: int, device: Optional[torch.device] = None) -> Tensor:
        """Sample from the flow model using ODE integration."""
        if device is None:
            device = self.device
            
        # Start from noise
        z = torch.randn(n_samples, self.dim, device=device)
        
        # Integration steps
        time_steps = torch.linspace(0, 1.0, self.n_inference_steps + 1)
        x_t = z.clone()
        
        self.eval()
        with torch.no_grad():
            for i in range(self.n_inference_steps):
                t_start = time_steps[i]
                t_end = time_steps[i + 1]
                x_t = self.euler_step(x_t=x_t, t_start=t_start, t_end=t_end)
        
        return x_t
    
    def compute_mmd_with_data(self, real_data: np.ndarray, n_samples: Optional[int] = None) -> float:
        """Compute MMD between generated samples and real data."""
        if n_samples is None:
            n_samples = len(real_data)
            
        # Generate samples
        generated_samples = self.sample(n_samples=n_samples, device=self.device)
        generated_samples = generated_samples.cpu().numpy()
        
        # Compute MMD
        mmd_score = mmd(real_data, generated_samples)
        return mmd_score


# Legacy Flow class for backward compatibility
class Flow(nn.Module):
    """Legacy Flow class - use FlowMatchingLightning for new implementations."""
    
    def __init__(self, dim: int = 64, h: int = 64):
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


# Example usage and training script
def create_example_training_script():
    """Example of how to use the FlowMatchingLightning module."""
    
    # Set random seed for reproducibility
    seed = 7
    pl.seed_everything(seed)
    
    # Create synthetic data for demonstration
    n_samples = 10000
    dim = 64
    X_train = np.random.randn(n_samples, dim) # TODO podmienic
    X_val = np.random.randn(1000, dim) # TODO podmienic
    
    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    
    # Initialize model
    model = FlowMatchingLightning(
        dim=dim,
        hidden_dim=64,
        learning_rate=1e-2,
        n_inference_steps=300
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        check_val_every_n_epoch=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Generate samples
    samples = model.sample(n_samples=1000)
    print(f"Generated samples shape: {samples.shape}")
    
    return model, trainer


if __name__ == "__main__":
    # Run example training
    model, trainer = create_example_training_script()