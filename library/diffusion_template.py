import argparse
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# AutoML Helpers
from tracker import MLTracker
from hardware import HardwareOptimizer
from data_cleaner import DataCleaner

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for embedding diffusion timesteps t.
    """
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionDenoiser(nn.Module):
    """
    Simple MLP-based denoiser with residual blocks and sinusoidal step embeddings.
    Designed for 1D vector denoising (dimension = 8).
    """
    def __init__(self, data_dim=8, time_embed_dim=16, hidden_dim=64):
        super(DiffusionDenoiser, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_block = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        self.res1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_block = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        # x: [batch, data_dim], t: [batch]
        t_embed = self.time_mlp(t) # [batch, hidden_dim]
        x_embed = self.input_block(x) # [batch, hidden_dim]
        
        # Late fusion of time embeddings
        h = x_embed + t_embed
        
        # Residual processing
        h = h + torch.tanh(self.res1(h))
        h = h + torch.tanh(self.res2(h))
        
        out = self.output_block(h)
        return out

class DiffusionModel:
    """
    DDPM (Denoising Diffusion Probabilistic Model) scheduling logic.
    """
    def __init__(self, num_timesteps=100, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear Beta Schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        # Cumulative product of alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        """
        Forward process: Adds noise to clean data at timestep t.
        """
        # x_start: [batch, data_dim]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Diffusion Model Template")
    parser.add_argument("--run-name", type=str, default="diffusion-sweep-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Target batch size")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_clean_patterns(num_samples=1000, data_dim=8):
    """
    Generates synthetic target signals (e.g. perfect sine and cosine waves).
    """
    np.random.seed(42)
    X = []
    # 2 types of clean shapes
    for i in range(num_samples):
        t = np.linspace(0, 2*np.pi, data_dim)
        if i % 2 == 0:
            X.append(np.sin(t))
        else:
            X.append(np.cos(t))
            
    return np.array(X, dtype=np.float32)

def main():
    args = parse_args()
    data_dim = 8

    # 1. Prepare Data
    print("[Diffusion] Generating clean target pattern arrays (dim=8)...")
    X = generate_clean_patterns()

    # Train/Val Split
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_val = X[train_size:]

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # 2. Hardware configuration
    hw_opt = HardwareOptimizer(target_batch_size=args.batch_size)
    opts = hw_opt.suggest_optimal_parameters(model_param_count=10000)
    print(f"[Diffusion] Selected Device: {opts['device']}")
    print(f"[Diffusion] Configured Batch Size: {opts['batch_size']}")

    train_dataset = torch.utils.data.TensorDataset(X_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t)

    dl_args = hw_opt.get_dataloader_args()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts["batch_size"], shuffle=False, **dl_args)

    # 3. Model Definition
    model = DiffusionDenoiser(data_dim=data_dim).to(hw_opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())

    # Define DDPM Scheduler
    diffusion = DiffusionModel(num_timesteps=100, device=hw_opt.device)

    # 4. Telemetry Logging
    hyperparameters = {
        "learning_rate": args.lr,
        "batch_size": opts["batch_size"],
        "optimizer": "Adam",
        "precision": opts["optimizer_precision"],
        "param_count": param_count,
        "primary_metric": "loss", # We optimize MSE Loss (lower is better)
        "inference_latency_ms": 1.9 # mock latency
    }
    
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="Diffusion Network (DDPM MLP)",
        simplicity_class=4, # Class 4 Deep NN
        hyperparameters=hyperparameters,
        baseline_metric=1.0, # Pure noise reconstruction baseline
        server_url=args.server_url
    )

    # 5. Training loop
    print("[Diffusion] Starting Diffusion noise-reduction training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Step function for OOM guard
        def step_fn():
            model.train()
            running_loss = 0.0
            for batch_x, in train_loader:
                batch_x = batch_x.to(hw_opt.device)
                optimizer.zero_grad()
                
                # Sample random diffusion timesteps t
                t = torch.randint(0, diffusion.num_timesteps, (batch_x.size(0),), device=hw_opt.device).long()
                # Sample Gaussian noise
                noise = torch.randn_like(batch_x)
                
                # Add noise (forward process)
                x_noisy = diffusion.q_sample(x_start=batch_x, t=t, noise=noise)
                
                # Predict noise (reverse process)
                if hw_opt.use_amp:
                    with torch.cuda.amp.autocast():
                        predicted_noise = model(x_noisy, t)
                        loss = criterion(predicted_noise, noise)
                    hw_opt.scaler.scale(loss).backward()
                    hw_opt.scaler.step(optimizer)
                    hw_opt.scaler.update()
                else:
                    predicted_noise = model(x_noisy, t)
                    loss = criterion(predicted_noise, noise)
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * batch_x.size(0)
            return running_loss / len(train_loader.dataset)

        train_loss, success = hw_opt.run_with_oom_guard(step_fn)
        if not success:
            print(f"[Diffusion] OOM triggered. Retrying with batch size {hw_opt.current_batch_size}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hw_opt.current_batch_size, shuffle=True, **dl_args)
            train_loss, _ = hw_opt.run_with_oom_guard(step_fn)

        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, in val_loader:
                batch_x = batch_x.to(hw_opt.device)
                t = torch.randint(0, diffusion.num_timesteps, (batch_x.size(0),), device=hw_opt.device).long()
                noise = torch.randn_like(batch_x)
                x_noisy = diffusion.q_sample(x_start=batch_x, t=t, noise=noise)
                predicted_noise = model(x_noisy, t)
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_dataset)
        epoch_time = time.time() - start_time

        # Report to dashboard.
        # Since primary metric is MSE loss (direction: minimize), we pass val_loss as val_metric and train_loss as train_metric.
        triage = tracker.log_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metric=val_loss,
            train_metric=train_loss,
            lr=optimizer.param_groups[0]['lr'],
            additional_metrics={"epoch_time_seconds": epoch_time}
        )

        if triage and triage.get("status") in ("KILL", "SUSPEND"):
            print(f"[Diffusion] AutoML stop advice received: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3)

    tracker.log_verdict(
        status="CONVERGED",
        message=f"Diffusion model training complete. Final Val MSE Loss: {val_loss:.5f}",
        final_metrics={"param_count": param_count}
    )
    print("[Diffusion] Done.")

if __name__ == "__main__":
    main()
