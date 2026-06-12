import argparse
import sys
import time
import torch
import numpy as np
import pandas as pd

# Import local AutoML helpers
from tracker import MLTracker
from hardware import HardwareOptimizer
from data_cleaner import DataCleaner

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Model Template Boilerplate")
    parser.add_argument("--run-name", type=str, default="my-boilerplate-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Target batch size")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic tabular data for demonstrating data cleaning and run logging.
    """
    np.random.seed(42)
    # 5 features, 1 target
    X = np.random.randn(num_samples, 5)
    # Add a categorical column
    cat_col = np.random.choice(['A', 'B', 'C'], size=num_samples)
    
    # Target (classification)
    y = (X[:, 0] * 1.5 + X[:, 1] * -0.8 + np.random.randn(num_samples) * 0.5 > 0).astype(int)
    
    # Inject some NaN values to test data cleaning
    X[np.random.choice(num_samples, 50), 2] = np.nan
    
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["category"] = cat_col
    df["target"] = y
    return df

def train_epoch(model, dataloader, optimizer, criterion, hw, epoch):
    """
    Standard training epoch wrapped in OOM protection helper.
    """
    def step_fn():
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(hw.device), batch_y.to(hw.device)
            optimizer.zero_grad()
            
            # Execute step with Automatic Mixed Precision (AMP) if supported
            if hw.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                hw.scaler.scale(loss).backward()
                hw.scaler.step(optimizer)
                hw.scaler.update()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        return total_loss / len(dataloader.dataset), correct / total

    # Run execution inside OOM Guard
    res, success = hw.run_with_oom_guard(step_fn)
    if success:
        return res[0], res[1], True
    return 0.0, 0.0, False

def main():
    args = parse_args()

    # 1. Load Data
    print("[Skeleton] Generating synthetic dataset...")
    df = generate_synthetic_data()

    # 2. Split Data (80/20 train/validation holdout)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()

    # 3. Apply AutoML Data Cleaning (Fits stats on train only)
    print("[Skeleton] Cleaning and preprocessing data...")
    # Impute missing values
    train_df, val_df = DataCleaner.impute_missing(train_df, val_df, strategy="median")
    
    # Scale numeric feature columns
    numeric_cols = [f"feat_{i}" for i in range(5)]
    train_df, val_df = DataCleaner.scale_features(train_df, val_df, numeric_cols, method="standard")
    
    # One-hot encode category column
    # Combine temporarily for one-hot encoding consistency or align columns manually
    full_encoded = pd.get_dummies(pd.concat([train_df, val_df]), columns=["category"], drop_first=True)
    train_encoded = full_encoded.iloc[:train_size].copy()
    val_encoded = full_encoded.iloc[train_size:].copy()

    # Run split audit to check for data leaks
    audit_results = DataCleaner.run_leakage_audit(train_df, val_df, target_col="target")
    print(f"[Skeleton] Split Leakage Audit Passed: {audit_results['passed']}. Details: {audit_results['details']}")

    # 4. Check Class Imbalance
    imbalance_ratio = DataCleaner.calculate_imbalance_ratio(train_encoded, "target")
    print(f"[Skeleton] Train class imbalance ratio: {imbalance_ratio:.2f}:1")
    primary_metric = "accuracy" if imbalance_ratio <= 3.0 else "f1"
    baseline_metric = 0.50 # Balanced synthetic baseline

    # 5. Initialize Hardware Optimizer
    hw_opt = HardwareOptimizer(target_batch_size=args.batch_size)
    opts = hw_opt.suggest_optimal_parameters(model_param_count=10000) # mock count
    print(f"[Skeleton] Hardware Auto-Tuning Results:")
    print(f"  - Selected Device: {opts['device']}")
    print(f"  - Precision Mode: {opts['optimizer_precision']}")
    print(f"  - Configured Batch Size: {opts['batch_size']}")

    # Prepare PyTorch Tensors
    X_train = torch.tensor(train_encoded.drop(columns=["target"]).values.astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(train_encoded["target"].values.astype(np.int64), dtype=torch.long)
    X_val = torch.tensor(val_encoded.drop(columns=["target"]).values.astype(np.float32), dtype=torch.float32)
    y_val = torch.tensor(val_encoded["target"].values.astype(np.int64), dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # Fetch multiprocessing workers safe for Windows/Linux
    dl_args = hw_opt.get_dataloader_args()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts["batch_size"], shuffle=False, **dl_args)

    # 6. Define Model (A simple linear candidate class 3)
    num_features = X_train.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(num_features, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2)
    ).to(hw_opt.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 7. Initialize Dashboard Telemetry Tracker
    hyperparameters = {
        "learning_rate": args.lr,
        "batch_size": opts["batch_size"],
        "optimizer": "Adam",
        "precision": opts["optimizer_precision"],
        "primary_metric": primary_metric,
        "param_count": sum(p.numel() for p in model.parameters()),
        "inference_latency_ms": 1.2 # mock latency
    }
    
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="LinearClassifierSkeleton",
        simplicity_class=3, # class 3 shallow NN
        hyperparameters=hyperparameters,
        baseline_metric=baseline_metric,
        server_url=args.server_url
    )

    # 8. Training loop
    print(f"[Skeleton] Starting training sweep...")
    for epoch in range(1, args.epochs + 1):
        # Time the epoch to calculate telemetry metrics
        start_time = time.time()
        
        # Train epoch (with OOM protection)
        train_loss, train_acc, success = train_epoch(model, train_loader, optimizer, criterion, hw_opt, epoch)
        
        # If OOM triggered, train_epoch halves batch size. We can rebuild dataloaders and continue!
        if not success:
            print(f"[Skeleton] Retrying epoch {epoch} with reduced batch size: {hw_opt.current_batch_size}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hw_opt.current_batch_size, shuffle=True, **dl_args)
            train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion, hw_opt, epoch)

        # Evaluate validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(hw_opt.device), batch_y.to(hw_opt.device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_dataset)
        val_acc = correct / total
        epoch_time = time.time() - start_time

        # If primary metric requires F1 or Accuracy
        val_metric = val_acc # Let's log validation accuracy as primary here
        train_metric = train_acc
        
        # Log telemetry to AutoML Dashboard
        triage = tracker.log_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metric=val_metric,
            train_metric=train_metric,
            lr=optimizer.param_groups[0]['lr'],
            additional_metrics={"epoch_time_seconds": epoch_time}
        )

        # Handle early-stopping recommendations from the Dashboard Rules Engine
        if triage and triage.get("status") in ("KILL", "SUSPEND"):
            print(f"[Skeleton] Training halted early by AutoML Dashboard recommendation: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3) # short delay to make UI updates readable during demonstration

    # Log clean convergence
    tracker.log_verdict(
        status="CONVERGED",
        message=f"Converged cleanly. Final Val Metric: {val_metric:.4f}",
        final_metrics={"param_count": sum(p.numel() for p in model.parameters())}
    )
    print("[Skeleton] Sweep complete.")

if __name__ == "__main__":
    main()
