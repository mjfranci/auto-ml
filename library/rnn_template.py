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

class VanillaRNN(nn.Module):
    """
    Standard customizable Vanilla RNN architecture for sequential data.
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_classes=2):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        # Take the output of the last timestep
        out = self.fc(out[:, -1, :])
        return out

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Vanilla RNN Template")
    parser.add_argument("--run-name", type=str, default="rnn-sweep-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Target batch size")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_sequence_data(num_samples=800, seq_len=10, input_size=1):
    """
    Generates synthetic sequence classification data (predicting if a sine wave frequency is high or low).
    """
    np.random.seed(42)
    X = []
    y = []
    
    for _ in range(num_samples):
        label = np.random.choice([0, 1])
        y.append(label)
        
        # High frequency vs low frequency waves
        freq = 2.0 if label == 1 else 0.5
        t = np.linspace(0, 5, seq_len)
        noise = np.random.randn(seq_len) * 0.1
        seq = np.sin(freq * t) + noise
        X.append(seq.reshape(seq_len, input_size))
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    args = parse_args()

    # 1. Prepare Data
    print("[RNN] Generating synthetic sequence data...")
    X, y = generate_sequence_data()

    # Train/Val split
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # Scale data (normalize along feature dimension)
    mean_train = X_train.mean()
    std_train = X_train.std() or 1.0
    X_train = (X_train - mean_train) / std_train
    X_val = (X_val - mean_train) / std_train

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Clean imbalance checks
    dummy_df = pd.DataFrame({"target": y_train})
    imbalance_ratio = DataCleaner.calculate_imbalance_ratio(dummy_df, "target")
    print(f"[RNN] Sequence class imbalance: {imbalance_ratio:.2f}:1")
    primary_metric = "accuracy" if imbalance_ratio <= 3.0 else "f1"

    # 2. Hardware Optimization
    hw_opt = HardwareOptimizer(target_batch_size=args.batch_size)
    opts = hw_opt.suggest_optimal_parameters(model_param_count=5000)
    print(f"[RNN] Hardware configuration:")
    print(f"  - Device: {opts['device']}")
    print(f"  - Precision: {opts['optimizer_precision']}")
    print(f"  - Batch Size: {opts['batch_size']}")

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    dl_args = hw_opt.get_dataloader_args()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts["batch_size"], shuffle=False, **dl_args)

    # 3. Model Definition
    model = VanillaRNN(input_size=1, hidden_size=32, num_layers=1, num_classes=2).to(hw_opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())

    # 4. Telemetry Logging
    hyperparameters = {
        "learning_rate": args.lr,
        "batch_size": opts["batch_size"],
        "optimizer": "Adam",
        "precision": opts["optimizer_precision"],
        "param_count": param_count,
        "primary_metric": primary_metric,
        "inference_latency_ms": 1.8 # mock latency
    }
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="RNN (Vanilla)",
        simplicity_class=4, # Deep/Sequential (Class 4)
        hyperparameters=hyperparameters,
        baseline_metric=0.50, # Balanced binary baseline
        server_url=args.server_url
    )

    # 5. Training Loop
    print("[RNN] Starting training sequence...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Step function for OOM guard
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
            print(f"[RNN] Resizing batch loader to {hw_opt.current_batch_size}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hw_opt.current_batch_size, shuffle=True, **dl_args)
            res, _ = hw_opt.run_with_oom_guard(step_fn)
            train_loss, train_acc = res[0], res[1]

        # Evaluate validation set
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

        # Send checkpoint telemetry
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
            print(f"[RNN] AutoML stopping rule triggered: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3)

    tracker.log_verdict(
        status="CONVERGED",
        message=f"RNN sweep converged. Accuracy: {val_acc:.4f}",
        final_metrics={"param_count": param_count}
    )
    print("[RNN] Complete.")

if __name__ == "__main__":
    main()
