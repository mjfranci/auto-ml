import argparse
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# AutoML Helpers
from tracker import MLTracker
from hardware import HardwareOptimizer
from data_cleaner import DataCleaner

class ConvNet(nn.Module):
    """
    Standard customizable 2D CNN architecture.
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 16 x 14 x 14 for input 28x28
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32 x 7 x 7 for input 28x28
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML CNN Template")
    parser.add_argument("--run-name", type=str, default="cnn-sweep-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Target batch size")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_synthetic_images(num_samples=800, width=28, height=28):
    """
    Generates synthetic image tensors (MNIST size 28x28) with targets.
    """
    np.random.seed(42)
    # Generate 1-channel images
    X = np.random.randn(num_samples, 1, width, height).astype(np.float32)
    # Draw simple shapes (e.g. circles or squares) to give the CNN signal
    for i in range(num_samples):
        label = i % 2 # Binary target
        if label == 1:
            # Draw a square in the center
            X[i, 0, 8:20, 8:20] += 2.0
        else:
            # Draw a dot in the center
            X[i, 0, 12:16, 12:16] += 3.5

    y = (np.arange(num_samples) % 2).astype(np.int64)
    return X, y

def main():
    args = parse_args()

    # 1. Prepare Data
    print("[CNN] Generating synthetic image data (28x28 pixels)...")
    X, y = generate_synthetic_images()

    # 80/20 train/val split
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # Normalize image pixel ranges
    # Fit mean/std on training set to avoid leakage
    mean_train = X_train.mean()
    std_train = X_train.std() or 1.0

    X_train = (X_train - mean_train) / std_train
    X_val = (X_val - mean_train) / std_train

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Convert targets to dataframe to run data cleaner imbalance checks
    dummy_df = pd.DataFrame({"target": y_train})
    imbalance_ratio = DataCleaner.calculate_imbalance_ratio(dummy_df, "target")
    print(f"[CNN] Class imbalance ratio: {imbalance_ratio:.2f}:1")
    primary_metric = "accuracy" if imbalance_ratio <= 3.0 else "f1"

    # 2. Hardware Optimization
    hw_opt = HardwareOptimizer(target_batch_size=args.batch_size)
    opts = hw_opt.suggest_optimal_parameters(model_param_count=100000)
    print(f"[CNN] Hardware Auto-Tuning:")
    print(f"  - Device: {opts['device']}")
    print(f"  - Precision: {opts['optimizer_precision']}")
    print(f"  - Batch Size: {opts['batch_size']}")

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    dl_args = hw_opt.get_dataloader_args()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts["batch_size"], shuffle=False, **dl_args)

    # 3. Model Definition
    model = ConvNet(in_channels=1, num_classes=2).to(hw_opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())

    # 4. Telemetry Tracking
    hyperparameters = {
        "learning_rate": args.lr,
        "batch_size": opts["batch_size"],
        "optimizer": "Adam",
        "precision": opts["optimizer_precision"],
        "param_count": param_count,
        "primary_metric": primary_metric,
        "inference_latency_ms": 2.5 # mock latency
    }
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="CNN (ConvNet)",
        simplicity_class=4, # Deep Neural Network (class 4)
        hyperparameters=hyperparameters,
        baseline_metric=0.50, # Balanced binary baseline
        server_url=args.server_url
    )

    # 5. Training loop
    print(f"[CNN] Starting CNN training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train epoch step function
        def step_fn():
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for bx, by in train_loader:
                bx, by = bx.to(hw_opt.device), by.to(hw_opt.device)
                optimizer.zero_grad()
                
                if hw_opt.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                    hw_opt.scaler.scale(loss).backward()
                    hw_opt.scaler.step(optimizer)
                    hw_opt.scaler.update()
                else:
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * bx.size(0)
                _, pred = torch.max(outputs.data, 1)
                total += by.size(0)
                correct += (pred == by).sum().item()
            return running_loss / len(train_loader.dataset), correct / total

        res, success = hw_opt.run_with_oom_guard(step_fn)
        if success:
            train_loss, train_acc = res[0], res[1]
        else:
            print(f"[CNN] OOM recovery triggered. Reducing batch size to {hw_opt.current_batch_size}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hw_opt.current_batch_size, shuffle=True, **dl_args)
            res, _ = hw_opt.run_with_oom_guard(step_fn)
            train_loss, train_acc = res[0], res[1]

        # Evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(hw_opt.device), by.to(hw_opt.device)
                outputs = model(bx)
                loss = criterion(outputs, by)
                val_loss += loss.item() * bx.size(0)
                _, pred = torch.max(outputs.data, 1)
                total += by.size(0)
                correct += (pred == by).sum().item()

        val_loss /= len(val_dataset)
        val_acc = correct / total
        epoch_time = time.time() - start_time

        # Report to dashboard
        triage = tracker.log_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metric=val_acc,
            train_metric=train_acc,
            lr=optimizer.param_groups[0]['lr'],
            additional_metrics={"epoch_time_seconds": epoch_time}
        )

        if triage and triage.get("status") in ("KILL", "SUSPEND"):
            print(f"[CNN] AutoML early stop triggered: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3)

    tracker.log_verdict(
        status="CONVERGED",
        message=f"CNN training complete. Final Accuracy: {val_acc:.4f}",
        final_metrics={"param_count": param_count}
    )
    print("[CNN] Done.")

if __name__ == "__main__":
    main()
