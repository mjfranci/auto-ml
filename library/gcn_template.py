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

class GraphConv(nn.Module):
    """
    Pure PyTorch implementation of a Graph Convolutional Network (GCN) layer.
    Computes spectral graph convolution: D^{-1/2} A_tilde D^{-1/2} X W
    """
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x, adj):
        """
        x: Node features of shape [num_nodes, in_features]
        adj: Symmetric normalized adjacency matrix [num_nodes, num_nodes]
        """
        # Message passing step
        support = self.linear(x) # shape: [num_nodes, out_features]
        output = torch.spmm(adj, support) # shape: [num_nodes, out_features]
        return output

class GCN(nn.Module):
    """
    2-Layer Graph Convolutional Network.
    """
    def __init__(self, in_features, hidden_features, out_classes):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.gcn2 = GraphConv(hidden_features, out_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML GCN Template")
    parser.add_argument("--run-name", type=str, default="gcn-sweep-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden dimension size")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_synthetic_graph(num_nodes=300, num_features=16):
    """
    Generates a synthetic graph: node features, adjacency matrix, and node labels.
    Organized into 3 community structures.
    """
    np.random.seed(42)
    
    # Node features
    X = np.random.randn(num_nodes, num_features).astype(np.float32)
    
    # Group nodes into 3 communities
    labels = np.random.choice([0, 1, 2], size=num_nodes)
    
    # Populate adjacency matrix based on communities
    # High probability of intra-community links, low probability of inter-community links
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if labels[i] == labels[j]:
                prob = 0.08
            else:
                prob = 0.005
            if np.random.rand() < prob:
                A[i, j] = 1.0
                A[j, i] = 1.0
                
    # Add Self-loops (A_tilde = A + I)
    A_tilde = A + np.eye(num_nodes)
    
    # Compute Symmetric Normalized Adjacency: D^{-1/2} A_tilde D^{-1/2}
    row_sums = A_tilde.sum(axis=1)
    d_inv_sqrt = np.power(row_sums, -0.5, where=row_sums>0)
    d_inv_sqrt[row_sums <= 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    norm_A = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
    
    # Train/Val split masks (70/30)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_count = int(0.7 * num_nodes)
    train_mask[indices[:train_count]] = True
    val_mask[indices[train_count:]] = True
    
    return X, norm_A, labels, train_mask, val_mask

def main():
    args = parse_args()

    # 1. Prepare Graph Data
    print("[GCN] Generating synthetic community graph (300 nodes)...")
    X, norm_A, labels, train_mask, val_mask = generate_synthetic_graph()

    # Clean imbalance checks
    dummy_df = pd.DataFrame({"target": labels[train_mask]})
    imbalance_ratio = DataCleaner.calculate_imbalance_ratio(dummy_df, "target")
    print(f"[GCN] Train node imbalance: {imbalance_ratio:.2f}:1")
    primary_metric = "accuracy" if imbalance_ratio <= 3.0 else "f1"

    # Convert to PyTorch Tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    adj_t = torch.tensor(norm_A, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool)

    # 2. Hardware Optimization
    # GCN runs on full graph, so batch size auto-tuning is not applicable.
    # We still use HardwareOptimizer for device configuration.
    hw_opt = HardwareOptimizer()
    opts = hw_opt.suggest_optimal_parameters()
    print(f"[GCN] Selected Device: {opts['device']}")

    # Transfer data to device
    X_t = X_t.to(hw_opt.device)
    adj_t = adj_t.to(hw_opt.device)
    labels_t = labels_t.to(hw_opt.device)
    train_mask_t = train_mask_t.to(hw_opt.device)
    val_mask_t = val_mask_t.to(hw_opt.device)

    # 3. Model Definition
    num_features = X.shape[1]
    num_classes = 3
    model = GCN(in_features=num_features, hidden_features=args.hidden_dim, out_classes=num_classes).to(hw_opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())

    # 4. Telemetry Logging
    hyperparameters = {
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "optimizer": "Adam",
        "weight_decay": 5e-4,
        "param_count": param_count,
        "primary_metric": primary_metric,
        "inference_latency_ms": 1.5 # mock latency
    }
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="GCN (Graph Convolutional Network)",
        simplicity_class=4, # Neural Network (Class 4)
        hyperparameters=hyperparameters,
        baseline_metric=0.33, # 3-class balanced baseline
        server_url=args.server_url
    )

    # 5. Training loop
    print("[GCN] Starting GCN training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Step function for training
        def step_fn():
            model.train()
            optimizer.zero_grad()
            outputs = model(X_t, adj_t)
            # Calculate loss only on training nodes
            loss = criterion(outputs[train_mask_t], labels_t[train_mask_t])
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs[train_mask_t].data, 1)
            correct = (predicted == labels_t[train_mask_t]).sum().item()
            total = train_mask_t.sum().item()
            train_acc = correct / (total or 1)
            return loss.item(), train_acc

        # Execute training with OOM guard
        res, success = hw_opt.run_with_oom_guard(step_fn)
        if success:
            train_loss, train_acc = res[0], res[1]
        else:
            print("[GCN] OOM recovery triggered. Full graph GCN cannot scale down batch size. Halting.")
            tracker.log_verdict(status="KILLED", message="Out-of-memory error during GCN graph convolution.")
            sys.exit(1)

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_t, adj_t)
            val_loss = criterion(outputs[val_mask_t], labels_t[val_mask_t]).item()
            
            _, predicted = torch.max(outputs[val_mask_t].data, 1)
            correct = (predicted == labels_t[val_mask_t]).sum().item()
            total = val_mask_t.sum().item()
            val_acc = correct / total

        epoch_time = time.time() - start_time

        # Report telemetry
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
            print(f"[GCN] AutoML early stopping rule triggered: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3)

    tracker.log_verdict(
        status="CONVERGED",
        message=f"GCN training completed. Node Validation Accuracy: {val_acc:.4f}",
        final_metrics={"param_count": param_count}
    )
    print("[GCN] Complete.")

if __name__ == "__main__":
    main()
