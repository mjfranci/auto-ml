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

class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention from scratch.
    Equation: softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q: [batch, heads, seq_q, d_k], k: [batch, heads, seq_k, d_k], v: [batch, heads, seq_v, d_v]
        d_k = q.size(-1)
        # Compute scores: QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (replace 0s in attention mask with -1e4)
        if mask is not None:
            # Mask shape must be broadcastable to [batch, heads, seq_q, seq_k]
            scores = scores.masked_fill(mask == 0, -1e4)
            
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention layer.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        # Projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections & split into heads
        # Shape: [batch, heads, seq_len, d_k]
        q_proj = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_proj = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_proj = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        out, weights = self.attention(q_proj, k_proj, v_proj, mask)
        
        # Concatenate heads and project out
        # Shape: [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(out), weights

class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer Encoder Block containing:
    1. Multi-Head Attention + Residual Connection + LayerNorm
    2. Feed Forward Network + Residual Connection + LayerNorm
    """
    def __init__(self, d_model, num_heads, ffn_hidden, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi-Head Attention
        attn_out, weights = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out)) # residual + norm
        
        # 2. Feed Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out)) # residual + norm
        return x, weights

class TransformerClassifier(nn.Module):
    """
    A simple sequence classifier built on top of a Transformer Encoder Block.
    """
    def __init__(self, vocab_size, d_model=32, num_heads=4, ffn_hidden=64, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoderBlock(d_model, num_heads, ffn_hidden)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask=None):
        # Embedding
        x = self.embedding(ids) # [batch, seq_len, d_model]
        
        # Prepare mask for batch dimensions: reshape mask [batch, seq_len] to [batch, 1, 1, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            
        # Encoder Block
        x, weights = self.encoder(x, mask)
        
        # Average pooling
        pooled = x.mean(dim=1)
        # Linear classifier
        out = self.fc(pooled)
        return out, weights

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Attention Transformer Template")
    parser.add_argument("--run-name", type=str, default="transformer-sweep-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Target batch size")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_synthetic_vocab_data(num_samples=600, seq_len=10, vocab_size=50):
    """
    Generates synthetic integer sequences (indices in vocab) and targets.
    Target is 1 if index 5 is present in the first half of the sequence, 0 otherwise.
    """
    np.random.seed(42)
    X = np.random.randint(2, vocab_size, size=(num_samples, seq_len)).astype(np.int64)
    # Binary targets based on rule
    y = np.any(X[:, :seq_len//2] == 5, axis=1).astype(np.int64)
    
    # Generate mock mask (all ones for synthetic full padding sequence)
    mask = np.ones((num_samples, seq_len), dtype=np.float32)
    return X, mask, y

def main():
    args = parse_args()
    vocab_size = 50

    # 1. Prepare Data
    print("[Attention] Generating synthetic integer sequence data...")
    X, mask, y = generate_synthetic_vocab_data(vocab_size=vocab_size)

    # Split dataset
    train_size = int(0.8 * len(X))
    X_train, mask_train, y_train = X[:train_size], mask[:train_size], y[:train_size]
    X_val, mask_val, y_val = X[train_size:], mask[train_size:], y[train_size:]

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    mask_train_t = torch.tensor(mask_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.long)
    mask_val_t = torch.tensor(mask_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Imbalance checks
    dummy_df = pd.DataFrame({"target": y_train})
    imbalance_ratio = DataCleaner.calculate_imbalance_ratio(dummy_df, "target")
    print(f"[Attention] Sequence target imbalance: {imbalance_ratio:.2f}:1")
    primary_metric = "accuracy" if imbalance_ratio <= 3.0 else "f1"

    # 2. Hardware Optimization
    hw_opt = HardwareOptimizer(target_batch_size=args.batch_size)
    opts = hw_opt.suggest_optimal_parameters(model_param_count=15000)
    print(f"[Attention] Hardware Parameters:")
    print(f"  - Selected Device: {opts['device']}")
    print(f"  - Precision Mode: {opts['optimizer_precision']}")
    print(f"  - Batch Size: {opts['batch_size']}")

    train_dataset = torch.utils.data.TensorDataset(X_train_t, mask_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, mask_val_t, y_val_t)

    dl_args = hw_opt.get_dataloader_args()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts["batch_size"], shuffle=False, **dl_args)

    # 3. Model Definition
    model = TransformerClassifier(vocab_size=vocab_size, d_model=32, num_heads=4, ffn_hidden=64, num_classes=2).to(hw_opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())

    # 4. Telemetry Log
    hyperparameters = {
        "learning_rate": args.lr,
        "batch_size": opts["batch_size"],
        "optimizer": "Adam",
        "precision": opts["optimizer_precision"],
        "param_count": param_count,
        "primary_metric": primary_metric,
        "inference_latency_ms": 4.1 # mock latency
    }
    
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="Attention (Transformer Classifier)",
        simplicity_class=4, # Class 4 Deep/Attention
        hyperparameters=hyperparameters,
        baseline_metric=0.50, # Balanced binary baseline
        server_url=args.server_url
    )

    # 5. Training loop
    print("[Attention] Starting Transformer training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Step function for OOM guard
        def step_fn():
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for bx, bmask, by in train_loader:
                bx, bmask, by = bx.to(hw_opt.device), bmask.to(hw_opt.device), by.to(hw_opt.device)
                optimizer.zero_grad()
                
                if hw_opt.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = model(bx, bmask)
                        loss = criterion(outputs, by)
                    hw_opt.scaler.scale(loss).backward()
                    hw_opt.scaler.step(optimizer)
                    hw_opt.scaler.update()
                else:
                    outputs, _ = model(bx, bmask)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * bx.size(0)
                _, pred = torch.max(outputs.data, 1)
                total += by.size(0)
                correct += (pred == by).sum().item()
            return running_loss / len(train_loader.dataset), correct / (total or 1)

        res, success = hw_opt.run_with_oom_guard(step_fn)
        if success:
            train_loss, train_acc = res[0], res[1]
        else:
            print(f"[Attention] OOM triggered. Retrying with batch size {hw_opt.current_batch_size}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hw_opt.current_batch_size, shuffle=True, **dl_args)
            res, _ = hw_opt.run_with_oom_guard(step_fn)
            train_loss, train_acc = res[0], res[1]

        # Evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, bmask, by in val_loader:
                bx, bmask, by = bx.to(hw_opt.device), bmask.to(hw_opt.device), by.to(hw_opt.device)
                outputs, _ = model(bx, bmask)
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
            print(f"[Attention] AutoML halting advice received: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3)

    tracker.log_verdict(
        status="CONVERGED",
        message=f"Transformer classifier training complete. Accuracy: {val_acc:.4f}",
        final_metrics={"param_count": param_count}
    )
    print("[Attention] Complete.")

if __name__ == "__main__":
    main()
