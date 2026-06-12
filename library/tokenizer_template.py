import argparse
import sys
import time
import torch
import numpy as np
import pandas as pd

# AutoML Helpers
from tracker import MLTracker
from hardware import HardwareOptimizer
from data_cleaner import DataCleaner

# Try to import HuggingFace transformers, fallback to a simple word tokenizer if unavailable
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class SimpleFallbackTokenizer:
    """
    Fallback word tokenizer for systems without Hugging Face transformers installed.
    """
    def __init__(self, vocab_size=1000):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def fit(self, texts):
        word_counts = {}
        for text in texts:
            words = text.lower().replace(".", "").replace(",", "").split()
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
        
        # Sort words and insert into vocab
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words:
            if len(self.vocab) >= self.vocab_size:
                break
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)

    def __call__(self, text, padding=True, truncation=True, max_length=16):
        words = text.lower().replace(".", "").replace(",", "").split()
        input_ids = [self.vocab["[CLS]"]]
        for w in words:
            input_ids.append(self.vocab.get(w, self.vocab["[UNK]"]))
        input_ids.append(self.vocab["[SEP]"])

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        attention_mask = [1] * len(input_ids)

        if padding and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.vocab["[PAD]"]] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

def parse_args():
    parser = argparse.ArgumentParser(description="AutoML Tokenizer Template")
    parser.add_argument("--run-name", type=str, default="tokenizer-sweep-run", help="Name of this run")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Target batch size")
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased", help="Hugging Face tokenizer name")
    parser.add_argument("--server-url", type=str, default="http://localhost:3000", help="AutoML server URL")
    return parser.parse_args()

def generate_synthetic_texts(num_samples=600):
    """
    Generates synthetic text sentences with positive/negative sentiments.
    """
    positive_phrases = ["this is great", "i loved it", "absolutely wonderful", "highly recommend", "excellent work", "perfect fit"]
    negative_phrases = ["this is terrible", "i hated it", "completely awful", "do not recommend", "poor quality", "bad experience"]
    
    np.random.seed(42)
    texts = []
    y = []
    
    for _ in range(num_samples):
        label = np.random.choice([0, 1])
        y.append(label)
        
        # Construct sentence
        if label == 1:
            phrase = np.random.choice(positive_phrases) + " and " + np.random.choice(positive_phrases)
        else:
            phrase = np.random.choice(negative_phrases) + " and " + np.random.choice(negative_phrases)
            
        texts.append(phrase)
        
    return texts, np.array(y, dtype=np.int64)

def main():
    args = parse_args()

    # 1. Prepare Text Data
    print("[Tokenizer] Generating synthetic text sentiment data...")
    texts, labels = generate_synthetic_texts()

    # Split dataset
    train_size = int(0.8 * len(texts))
    texts_train, y_train = texts[:train_size], labels[:train_size]
    texts_val, y_val = texts[train_size:], labels[train_size:]

    # Class imbalance
    dummy_df = pd.DataFrame({"target": y_train})
    imbalance_ratio = DataCleaner.calculate_imbalance_ratio(dummy_df, "target")
    print(f"[Tokenizer] Sentiment class imbalance: {imbalance_ratio:.2f}:1")
    primary_metric = "accuracy" if imbalance_ratio <= 3.0 else "f1"

    # 2. Initialize Tokenizer (Hugging Face or Fallback)
    tokenizer = None
    if HF_AVAILABLE:
        try:
            print(f"[Tokenizer] Loading pre-trained HuggingFace tokenizer '{args.tokenizer_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            vocab_size = tokenizer.vocab_size
        except Exception as e:
            print(f"[Tokenizer] Error loading HF tokenizer: {e}. Falling back to Simple Word Tokenizer.")
            tokenizer = None
            
    if tokenizer is None:
        print("[Tokenizer] HuggingFace transformers library not available or load failed. Initializing Simple Fallback Tokenizer...")
        tokenizer = SimpleFallbackTokenizer(vocab_size=200)
        tokenizer.fit(texts_train)
        vocab_size = tokenizer.vocab_size

    # Tokenize input datasets
    print("[Tokenizer] Tokenizing and padding text sequences...")
    max_length = 16
    train_inputs = [tokenizer(t, padding=True, truncation=True, max_length=max_length) for t in texts_train]
    val_inputs = [tokenizer(t, padding=True, truncation=True, max_length=max_length) for t in texts_val]

    # Convert lists of dicts to tensors
    X_train_ids = torch.tensor([item["input_ids"] for item in train_inputs], dtype=torch.long)
    X_train_mask = torch.tensor([item["attention_mask"] for item in train_inputs], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val_ids = torch.tensor([item["input_ids"] for item in val_inputs], dtype=torch.long)
    X_val_mask = torch.tensor([item["attention_mask"] for item in val_inputs], dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # 3. Hardware Optimization
    hw_opt = HardwareOptimizer(target_batch_size=args.batch_size)
    opts = hw_opt.suggest_optimal_parameters(model_param_count=200000)
    print(f"[Tokenizer] Hardware Auto-Tuning:")
    print(f"  - Device: {opts['device']}")
    print(f"  - Precision: {opts['optimizer_precision']}")
    print(f"  - Batch Size: {opts['batch_size']}")

    train_dataset = torch.utils.data.TensorDataset(X_train_ids, X_train_mask, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_ids, X_val_mask, y_val_t)

    dl_args = hw_opt.get_dataloader_args()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, **dl_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opts["batch_size"], shuffle=False, **dl_args)

    # 4. Model Definition (Embedding layer + linear head classification)
    class EmbeddingClassifier(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim=32, num_classes=2):
            super(EmbeddingClassifier, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.fc = torch.nn.Linear(embedding_dim, num_classes)

        def forward(self, ids, mask):
            embedded = self.embedding(ids) # [batch, seq, dim]
            # Average pooling along the sequence length dimension, considering mask
            mask_expanded = mask.unsqueeze(-1) # [batch, seq, 1]
            summed = torch.sum(embedded * mask_expanded, dim=1)
            counts = torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0)
            pooled = summed / counts
            return self.fc(pooled)

    model = EmbeddingClassifier(vocab_size=vocab_size, embedding_dim=32, num_classes=2).to(hw_opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())

    # 5. Telemetry Tracking
    hyperparameters = {
        "learning_rate": args.lr,
        "batch_size": opts["batch_size"],
        "optimizer": "Adam",
        "precision": opts["optimizer_precision"],
        "param_count": param_count,
        "vocab_size": vocab_size,
        "primary_metric": primary_metric,
        "inference_latency_ms": 2.1
    }
    
    tracker = MLTracker(
        run_name=args.run_name,
        model_type="Tokenizer (Embedding Classifier)",
        simplicity_class=3, # Class 3 embedding classification
        hyperparameters=hyperparameters,
        baseline_metric=0.50, # Balanced binary baseline
        server_url=args.server_url
    )

    # 6. Training loop
    print("[Tokenizer] Starting text classification sweep...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Step function for OOM guard
        def step_fn():
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for ids, mask, target in train_loader:
                ids, mask, target = ids.to(hw_opt.device), mask.to(hw_opt.device), target.to(hw_opt.device)
                optimizer.zero_grad()
                
                if hw_opt.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(ids, mask)
                        loss = criterion(outputs, target)
                    hw_opt.scaler.scale(loss).backward()
                    hw_opt.scaler.step(optimizer)
                    hw_opt.scaler.update()
                else:
                    outputs = model(ids, mask)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * ids.size(0)
                _, pred = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()
            return running_loss / len(train_loader.dataset), correct / (total or 1)

        res, success = hw_opt.run_with_oom_guard(step_fn)
        if success:
            train_loss, train_acc = res[0], res[1]
        else:
            print(f"[Tokenizer] Rebuilding loader with batch size {hw_opt.current_batch_size}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hw_opt.current_batch_size, shuffle=True, **dl_args)
            res, _ = hw_opt.run_with_oom_guard(step_fn)
            train_loss, train_acc = res[0], res[1]

        # Evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for ids, mask, target in val_loader:
                ids, mask, target = ids.to(hw_opt.device), mask.to(hw_opt.device), target.to(hw_opt.device)
                outputs = model(ids, mask)
                loss = criterion(outputs, target)
                val_loss += loss.item() * ids.size(0)
                _, pred = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()

        val_loss /= len(val_dataset)
        val_acc = correct / total
        epoch_time = time.time() - start_time

        # Telemetry
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
            print(f"[Tokenizer] AutoML stopping rule triggered: {triage.get('status')}")
            tracker.log_verdict(status=f"AUTO_ML_{triage.get('status')}ED", message=triage.get("reason"))
            sys.exit(0)

        time.sleep(0.3)

    tracker.log_verdict(
        status="CONVERGED",
        message=f"Text classification sweep complete. Accuracy: {val_acc:.4f}",
        final_metrics={"param_count": param_count}
    )
    print("[Tokenizer] Done.")

if __name__ == "__main__":
    main()
