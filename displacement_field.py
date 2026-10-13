"""
Smoothness-Regularized Neural Displacement Field for Single-Cell Alignment
COMPLETE FIXED VERSION - Self-contained with proper error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from typing import Tuple, Optional, Dict, List
import warnings
from tqdm import tqdm
import os

# Install required packages:
# pip install torch pytorch-lightning scikit-learn tqdm

class LipschitzConstrainedLinear(nn.Module):
    """
    Linear layer with configurable Lipschitz constraint
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 lipschitz_bound: float = 1.0, bias: bool = True):
        super().__init__()
        
        # Create base linear layer
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # Apply PyTorch spectral normalization
        self.base_layer = torch.nn.utils.spectral_norm(self.base_layer)
        
        # Store the desired Lipschitz bound for scaling
        self.lipschitz_bound = lipschitz_bound
        
        print(f"LipschitzConstrainedLinear: {in_features}→{out_features}, L={lipschitz_bound}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply spectral normalized layer then scale
        output = self.base_layer(x)
        return output * self.lipschitz_bound
    
    def get_spectral_norm(self) -> float:
        """Get current spectral norm of the weight matrix"""
        weight = self.base_layer.weight
        try:
            u, s, v = torch.linalg.svd(weight.view(weight.size(0), -1))
        except AttributeError:
            u, s, v = torch.svd(weight.view(weight.size(0), -1))
        return s[0].item() * self.lipschitz_bound

class SmoothDisplacementNet(nn.Module):
    """
    Neural network with built-in smoothness constraints
    """
    
    def __init__(self, input_dim: int, 
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 lipschitz_constant: float = 1.0,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lipschitz_constant = lipschitz_constant
        self.n_layers = n_layers
        
        # Build layers step by step to avoid scope issues
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(LipschitzConstrainedLinear(input_dim, hidden_dim, lipschitz_constant))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ELU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(n_layers - 1):
            self.layers.append(LipschitzConstrainedLinear(hidden_dim, hidden_dim, lipschitz_constant))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ELU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        output_lipschitz = min(2.0, lipschitz_constant * 1.5)
        self.output_layer = LipschitzConstrainedLinear(hidden_dim, input_dim, output_lipschitz)
        
        # Skip connection weight
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize for small, smooth displacements"""
        for layer in self.layers:
            if isinstance(layer, LipschitzConstrainedLinear):
                nn.init.xavier_normal_(layer.base_layer.weight, gain=0.1)
                if layer.base_layer.bias is not None:
                    nn.init.constant_(layer.base_layer.bias, 0)
        
        # Initialize output layer
        nn.init.normal_(self.output_layer.base_layer.weight, mean=0, std=0.01)
        if self.output_layer.base_layer.bias is not None:
            nn.init.constant_(self.output_layer.base_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        h = x
        
        # Pass through all layers
        for layer in self.layers:
            h = layer(h)
        
        # Output displacement
        displacement = self.output_layer(h)
        
        # Skip connection for stability
        displacement = displacement + self.skip_weight * x
        
        return displacement
    
    def compute_theoretical_lipschitz_bound(self) -> float:
        """Compute theoretical Lipschitz bound"""
        bound = 1.0
        
        # Count Lipschitz layers
        for layer in self.layers:
            if isinstance(layer, LipschitzConstrainedLinear):
                bound *= layer.lipschitz_bound
        
        # Output layer
        bound *= self.output_layer.lipschitz_bound
        
        # Skip connection adds 1
        bound += 1.0
        
        return bound
    
    def get_lipschitz_bounds(self) -> Dict[str, float]:
        """Get Lipschitz bounds for all layers"""
        bounds = {}
        
        layer_count = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LipschitzConstrainedLinear):
                bounds[f'layer_{layer_count}'] = layer.lipschitz_bound
                layer_count += 1
        
        bounds['output_layer'] = self.output_layer.lipschitz_bound
        bounds['theoretical_network_bound'] = self.compute_theoretical_lipschitz_bound()
        
        return bounds

class SimpleProgressBar(pl.Callback):
    """Simplified progress bar that avoids PyTorch Lightning version issues"""
    
    def __init__(self):
        super().__init__()
        self.train_progress_bar = None
        self.current_epoch = 0
        self.total_epochs = 0
        
    def on_train_start(self, trainer, pl_module):
        """Initialize training progress"""
        self.total_epochs = trainer.max_epochs
        print(f"Starting training for {self.total_epochs} epochs...")
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Start epoch progress bar"""
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()
            
        self.current_epoch = trainer.current_epoch + 1
        
        # Try to get total batches, with fallback
        try:
            total_batches = len(trainer.train_dataloader)
        except:
            total_batches = None
            
        self.train_progress_bar = tqdm(
            total=total_batches,
            desc=f'Epoch {self.current_epoch}/{self.total_epochs}',
            position=0,
            leave=True,
            dynamic_ncols=True
        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update progress bar after each batch"""
        if self.train_progress_bar is not None:
            # Update progress
            self.train_progress_bar.update(1)
            
            # Update metrics every 10 batches
            if batch_idx % 10 == 0:
                logs = trainer.logged_metrics
                metric_str = {}
                if 'train_loss' in logs:
                    metric_str['Loss'] = f"{logs['train_loss']:.4f}"
                if 'mse_loss' in logs:
                    metric_str['MSE'] = f"{logs['mse_loss']:.4f}"
                if 'val_loss' in logs and self.current_epoch > 1:
                    metric_str['Val'] = f"{logs['val_loss']:.4f}"
                
                if metric_str:
                    self.train_progress_bar.set_postfix(metric_str)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Close epoch progress bar"""
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()
            self.train_progress_bar = None
            
    def on_validation_start(self, trainer, pl_module):
        """Simple validation start message"""
        if self.current_epoch > 0:  # Skip for sanity check
            print(f"  Running validation...")
    
    def on_validation_end(self, trainer, pl_module):
        """Simple validation end message"""
        if self.current_epoch > 0:  # Skip for sanity check
            logs = trainer.logged_metrics
            if 'val_loss' in logs:
                print(f"  Validation loss: {logs['val_loss']:.4f}")
            
    def on_train_end(self, trainer, pl_module):
        """Clean up on training end"""
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()
        print("Training completed!")

class SmoothDisplacementField(pl.LightningModule):
    """PyTorch Lightning module with smoothness regularization"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 lipschitz_constant: float = 1.0,
                 neighbor_weight: float = 0.1,
                 tv_weight: float = 0.05,
                 gradient_penalty_weight: float = 0.02,
                 manifold_weight: float = 0.1,
                 n_neighbors: int = 5):
        super().__init__()
        self.save_hyperparameters()
        
        # Network
        self.network = SmoothDisplacementNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=3,
            lipschitz_constant=lipschitz_constant,
            dropout_rate=0.1
        )
        
        # Regularization weights
        self.neighbor_weight = neighbor_weight
        self.tv_weight = tv_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        self.manifold_weight = manifold_weight
        self.n_neighbors = n_neighbors
        
        # Neighbor graph (set during training)
        self.neighbor_graph = None
        
        print(f"SmoothDisplacementField initialized:")
        print(f"  Lipschitz constant: {lipschitz_constant}")
        print(f"  Theoretical bound: {self.network.compute_theoretical_lipschitz_bound():.3f}")
        
    def forward(self, x):
        return self.network(x)
    
    def _compute_neighbor_consistency_loss(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Neighbor consistency loss"""
        if self.neighbor_graph is None:
            return torch.tensor(0.0, device=x.device)
        
        batch_size = x.shape[0]
        total_loss = 0.0
        count = 0
        
        sample_size = min(100, batch_size)
        sample_indices = torch.randperm(batch_size)[:sample_size]
        
        for i in sample_indices:
            point = x[i:i+1].cpu().numpy()
            
            try:
                distances, neighbor_indices = self.neighbor_graph.kneighbors(point)
                
                batch_neighbors = []
                for nidx in neighbor_indices[0]:
                    if nidx < batch_size:
                        batch_neighbors.append(nidx)
                
                if len(batch_neighbors) > 1:
                    neighbor_preds = pred[batch_neighbors]
                    diff_matrix = neighbor_preds.unsqueeze(0) - neighbor_preds.unsqueeze(1)
                    consistency_loss = torch.mean(torch.norm(diff_matrix, dim=2))
                    total_loss += consistency_loss
                    count += 1
                    
            except Exception:
                continue
        
        return total_loss / max(count, 1)
    
    def _compute_total_variation_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Total variation regularization"""
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        
        diff = pred[1:] - pred[:-1]
        return torch.mean(torch.norm(diff, dim=1))
    
    def _compute_gradient_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient penalty for smoothness"""
        x_copy = x.clone().detach().requires_grad_(True)
        pred = self.network(x_copy)
        
        gradients = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=x_copy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_norm = torch.norm(gradients, dim=1)
        return torch.mean(gradient_norm ** 2)
    
    def _compute_manifold_preservation_loss(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Manifold preservation loss"""
        batch_size = x.shape[0]
        if batch_size < self.n_neighbors:
            return torch.tensor(0.0, device=x.device)
        
        n_pairs = min(50, batch_size // 2)
        indices = torch.randperm(batch_size)[:2*n_pairs].reshape(-1, 2)
        
        manifold_loss = 0.0
        
        for i, j in indices:
            orig_dist = torch.norm(x[i] - x[j])
            transformed_i = x[i] + pred[i]
            transformed_j = x[j] + pred[j]
            transformed_dist = torch.norm(transformed_i - transformed_j)
            
            dist_change = torch.abs(transformed_dist - orig_dist) / (orig_dist + 1e-8)
            manifold_loss += dist_change
        
        return manifold_loss / len(indices)
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        
        # Main loss
        mse_loss = F.mse_loss(y_pred, y_true)
        
        # Regularization losses
        neighbor_loss = self._compute_neighbor_consistency_loss(x, y_pred)
        tv_loss = self._compute_total_variation_loss(y_pred)
        gradient_penalty = self._compute_gradient_penalty(x)
        manifold_loss = self._compute_manifold_preservation_loss(x, y_pred)
        
        # Combined loss
        total_loss = (mse_loss + 
                     self.neighbor_weight * neighbor_loss +
                     self.tv_weight * tv_loss +
                     self.gradient_penalty_weight * gradient_penalty +
                     self.manifold_weight * manifold_loss)
        
        # Logging
        self.log_dict({
            'train_loss': total_loss,
            'mse_loss': mse_loss,
            'neighbor_loss': neighbor_loss,
            'tv_loss': tv_loss,
            'gradient_penalty': gradient_penalty,
            'manifold_loss': manifold_loss
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        val_loss = F.mse_loss(y_pred, y_true)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate * 3,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

class SmoothDisplacementFieldTrainer:
    """High-level trainer for smooth displacement fields"""
    
    def __init__(self, input_dim: int, neighbor_weight: float = 0.1, device: str = 'auto'):
        self.input_dim = input_dim
        self.neighbor_weight = neighbor_weight
        self.device = self._setup_device(device)
        self.scaler = StandardScaler()
        self.model = None
        self.neighbor_graph = None
        
    def _setup_device(self, device: str):
        if device == 'auto':
            return 'gpu' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _build_neighbor_graph(self, X: np.ndarray, n_neighbors: int = 10):
        """Build k-NN graph for neighbor consistency"""
        print(f"Building neighbor graph with {n_neighbors} neighbors...")
        self.neighbor_graph = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        )
        self.neighbor_graph.fit(X)
        
    def fit(self, X_source: np.ndarray, X_target: np.ndarray, 
            mnn_pairs: np.ndarray,
            epochs: int = 100,
            batch_size: int = 512,
            validation_split: float = 0.2,
            lipschitz_constant: float = 1.0,
            neighbor_weight: float = 0.1,
            tv_weight: float = 0.05,
            gradient_penalty_weight: float = 0.02,
            manifold_weight: float = 0.1) -> 'SmoothDisplacementFieldTrainer':
        """Train smooth displacement field"""
        
        print("Training Smooth Displacement Field...")
        print(f"Lipschitz constant: {lipschitz_constant}")
        print(f"Regularization weights:")
        print(f"  Neighbor consistency: {neighbor_weight}")
        print(f"  Total variation: {tv_weight}")
        print(f"  Gradient penalty: {gradient_penalty_weight}")
        print(f"  Manifold preservation: {manifold_weight}")
        
        # Preprocess data
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Extract MNN training data
        source_mnn = X_source_scaled[mnn_pairs[:, 0]]
        target_mnn = X_target_scaled[mnn_pairs[:, 1]]
        displacements = target_mnn - source_mnn
        
        print(f"Training data: {len(mnn_pairs)} MNN pairs")
        print(f"Mean displacement magnitude: {np.linalg.norm(displacements, axis=1).mean():.4f}")
        
        # Build neighbor graph
        self._build_neighbor_graph(source_mnn, n_neighbors=10)
        
        # Create data loaders
        n_total = len(source_mnn)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        
        indices = torch.randperm(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(source_mnn[train_indices]),
            torch.FloatTensor(displacements[train_indices])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(source_mnn[val_indices]),
            torch.FloatTensor(displacements[val_indices])
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize model
        self.model = SmoothDisplacementField(
            input_dim=self.input_dim,
            hidden_dim=max(32, self.input_dim * 2),
            learning_rate=0.001,
            lipschitz_constant=lipschitz_constant,
            neighbor_weight=neighbor_weight,
            tv_weight=tv_weight,
            gradient_penalty_weight=gradient_penalty_weight,
            manifold_weight=manifold_weight
        )
        
        # Set neighbor graph
        self.model.neighbor_graph = self.neighbor_graph
        
        # Setup trainer with progress bar
        progress_bar = SimpleProgressBar()
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            devices='auto',
            precision="16",
            #accumulate_grad_batches=4,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=20,
            check_val_every_n_epoch=5,
            deterministic=False,
            callbacks=[progress_bar]
        )
        
        # Train
        trainer.fit(self.model, train_loader, val_loader)
        
        print("Training completed!")
        print(f"Final network Lipschitz bound: {self.model.network.compute_theoretical_lipschitz_bound():.3f}")
        
        return self
    
    def predict(self, X: np.ndarray, batch_size: int = 4096, show_progress: bool = True) -> np.ndarray:
        """Predict displacements"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        predictions = []
        
        if show_progress:
            n_batches = (len(X_scaled) + batch_size - 1) // batch_size
            pbar = tqdm(total=n_batches, desc="Predicting", dynamic_ncols=True)
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                batch = X_scaled[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch)
                
                if torch.cuda.is_available() and self.device == 'gpu':
                    batch_tensor = batch_tensor.cuda()
                    if self.model.device != batch_tensor.device:
                        self.model = self.model.to(batch_tensor.device)
                
                batch_pred = self.model(batch_tensor)
                predictions.append(batch_pred.cpu().numpy())
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return np.vstack(predictions)
    
    def transform(self, X: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """Apply displacement field"""
        displacements = self.predict(X, show_progress=show_progress)
        X_scaled = self.scaler.transform(X)
        X_transformed_scaled = X_scaled + displacements
        return self.scaler.inverse_transform(X_transformed_scaled)
    
    def analyze_smoothness(self, X_sample: np.ndarray, n_points: int = 1000) -> Dict[str, float]:
        """Analyze smoothness properties"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if len(X_sample) > n_points:
            indices = np.random.choice(len(X_sample), n_points, replace=False)
            X_analysis = X_sample[indices]
        else:
            X_analysis = X_sample
        
        displacements = self.predict(X_analysis, show_progress=False)
        
        nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
        nbrs.fit(X_analysis)
        
        smoothness_scores = []
        displacement_magnitudes = []
        
        for i, point in enumerate(X_analysis):
            distances, neighbor_indices = nbrs.kneighbors([point])
            neighbor_displacements = displacements[neighbor_indices[0]]
            current_displacement = displacements[i]
            
            displacement_diffs = neighbor_displacements - current_displacement
            smoothness = np.mean(np.linalg.norm(displacement_diffs, axis=1))
            smoothness_scores.append(smoothness)
            
            magnitude = np.linalg.norm(current_displacement)
            displacement_magnitudes.append(magnitude)
        
        return {
            'mean_smoothness_score': np.mean(smoothness_scores),
            'std_smoothness_score': np.std(smoothness_scores),
            'mean_displacement_magnitude': np.mean(displacement_magnitudes),
            'displacement_consistency': 1.0 / (1.0 + np.mean(smoothness_scores)),
            'max_local_variation': np.max(smoothness_scores),
            'smoothness_coefficient': np.std(displacement_magnitudes) / (np.mean(displacement_magnitudes) + 1e-8)
        }

# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    print("Testing SmoothDisplacementFieldTrainer with synthetic data...")
    
    # Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_source = 1000
    n_target = 800
    dim = 20
    
    X_source = np.random.randn(n_source, dim)
    X_target = np.random.randn(n_target, dim) + 0.5  # Slight shift
    
    # Generate synthetic MNN pairs
    n_mnn = 200
    mnn_pairs = np.column_stack([
        np.random.choice(n_source, n_mnn, replace=False),
        np.random.choice(n_target, n_mnn, replace=False)
    ])
    
    # Initialize and train
    trainer = SmoothDisplacementFieldTrainer(input_dim=dim)
    
    trainer.fit(
        X_source=X_source,
        X_target=X_target,
        mnn_pairs=mnn_pairs,
        epochs=10,  # Short for testing
        batch_size=64,
        lipschitz_constant=1.0
    )
    
    # Test prediction and transformation
    X_aligned = trainer.transform(X_source[:100])
    print(f"Alignment completed. Transformed shape: {X_aligned.shape}")
    
    # Analyze smoothness
    smoothness_metrics = trainer.analyze_smoothness(X_source[:200])
    print("Smoothness Analysis:")
    for key, value in smoothness_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("Test completed successfully!")