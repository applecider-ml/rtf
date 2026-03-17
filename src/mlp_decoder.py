"""
MLP classifier decoder: train a small neural network on frozen latent vectors.

Demonstrates that downstream consumers can build accurate classifiers
from the compressed representation without access to raw alerts.

Compares:
  - Linear probe (logistic regression) — baseline
  - 2-layer MLP — practical decoder
  - 3-layer MLP with dropout — stronger decoder

Usage:
    python src/mlp_decoder.py \
        --runs-dir runs \
        --output-dir analysis/decoders
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

COARSE_NAMES = ["SNIa", "SNcc", "Cataclysmic", "AGN", "TDE"]
FINE_NAMES = [
    "SN Ia",
    "SN Ib",
    "SN Ic",
    "SN II",
    "SN IIP",
    "SN IIn",
    "SN IIb",
    "Cataclysmic",
    "AGN",
    "TDE",
]


class MLPClassifier(nn.Module):
    """Simple MLP for classification from latent vectors."""

    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    hidden_dims=(256, 128),
    dropout=0.3,
    epochs=100,
    lr=1e-3,
    batch_size=256,
    device="cpu",
):
    """Train an MLP classifier on embeddings."""
    # Standardize
    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_va = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    input_dim = X_tr.shape[1]
    model = MLPClassifier(input_dim, num_classes, hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va.to(device))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, scaler


def evaluate_classifier(model, scaler, X_test, y_test, num_classes, device="cpu"):
    """Evaluate a trained MLP on test data."""
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        proba = F.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    try:
        roc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    except ValueError:
        roc = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc,
        "predictions": pred,
        "probabilities": proba,
    }


def linear_probe(X_train, y_train, X_test, y_test):
    """Baseline logistic regression."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1)
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    try:
        roc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    except ValueError:
        roc = float("nan")
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "roc_auc": roc}


def evaluate_run(run_path, output_dir, device="cpu"):
    """Run all decoder variants on a single trained model's embeddings."""
    summary_path = os.path.join(run_path, "summary.json")
    train_path = os.path.join(run_path, "train_embeddings.npz")
    test_path = os.path.join(run_path, "test_embeddings.npz")

    if not all(os.path.exists(p) for p in [summary_path, train_path, test_path]):
        return None

    with open(summary_path) as f:
        run_summary = json.load(f)

    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    X_train = train_data["embeddings"]
    X_test = test_data["embeddings"]

    run_name = os.path.basename(run_path)
    mode = run_summary["mode"]
    latent_dim = run_summary["latent_dim"]
    run_out = os.path.join(output_dir, run_name)
    os.makedirs(run_out, exist_ok=True)

    results = {
        "run": run_name,
        "mode": mode,
        "latent_dim": latent_dim,
        "compression_ratio": run_summary["compression_ratio"],
    }

    for task, y_key, num_classes, class_names in [
        ("coarse", "labels_coarse", 5, COARSE_NAMES),
        ("fine", "labels_fine", 10, FINE_NAMES),
    ]:
        y_train = train_data[y_key]
        y_test = test_data[y_key]

        # Use first 20% of train as val for MLP early stopping
        n_val = max(1, len(X_train) // 5)
        idx = np.random.RandomState(42).permutation(len(X_train))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        print(f"  {task} ({num_classes}-class):")

        # Linear probe
        lp = linear_probe(X_train, y_train, X_test, y_test)
        print(
            f"    Linear:     acc={lp['accuracy']:.3f}  bal={lp['balanced_accuracy']:.3f}  auc={lp['roc_auc']:.3f}"
        )

        # 2-layer MLP
        mlp2, scaler2 = train_mlp(
            X_train[tr_idx],
            y_train[tr_idx],
            X_train[val_idx],
            y_train[val_idx],
            num_classes,
            hidden_dims=(256, 128),
            dropout=0.3,
            epochs=100,
            device=device,
        )
        res2 = evaluate_classifier(mlp2, scaler2, X_test, y_test, num_classes, device)
        print(
            f"    MLP-2:      acc={res2['accuracy']:.3f}  bal={res2['balanced_accuracy']:.3f}  auc={res2['roc_auc']:.3f}"
        )

        # 3-layer MLP
        mlp3, scaler3 = train_mlp(
            X_train[tr_idx],
            y_train[tr_idx],
            X_train[val_idx],
            y_train[val_idx],
            num_classes,
            hidden_dims=(512, 256, 128),
            dropout=0.4,
            epochs=150,
            device=device,
        )
        res3 = evaluate_classifier(mlp3, scaler3, X_test, y_test, num_classes, device)
        print(
            f"    MLP-3:      acc={res3['accuracy']:.3f}  bal={res3['balanced_accuracy']:.3f}  auc={res3['roc_auc']:.3f}"
        )

        # Classification report for best model
        best = max(
            [("linear", lp), ("mlp2", res2), ("mlp3", res3)],
            key=lambda x: x[1]["balanced_accuracy"],
        )
        best_name, best_res = best

        results[f"{task}_linear"] = lp
        results[f"{task}_mlp2"] = {
            k: v for k, v in res2.items() if k not in ("predictions", "probabilities")
        }
        results[f"{task}_mlp3"] = {
            k: v for k, v in res3.items() if k not in ("predictions", "probabilities")
        }
        results[f"{task}_best"] = best_name

        # Save detailed report for best
        if "predictions" in best_res:
            report = classification_report(
                y_test,
                best_res["predictions"],
                target_names=class_names,
                output_dict=True,
            )
            with open(
                os.path.join(run_out, f"report_{task}_{best_name}.json"), "w"
            ) as f:
                json.dump(report, f, indent=2)

    with open(os.path.join(run_out, "decoder_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="MLP decoder evaluation")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_dirs = args.models or sorted(
        [
            d
            for d in os.listdir(args.runs_dir)
            if os.path.isdir(os.path.join(args.runs_dir, d))
        ]
    )

    all_results = []
    for d in run_dirs:
        run_path = os.path.join(args.runs_dir, d) if "/" not in d else d
        print(f"\n{'=' * 60}")
        print(f"Decoders for {d}")
        print(f"{'=' * 60}")
        result = evaluate_run(run_path, args.output_dir, args.device)
        if result:
            all_results.append(result)

    with open(os.path.join(args.output_dir, "all_decoder_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'=' * 100}")
    print("DECODER COMPARISON (coarse 5-class)")
    print(f"{'=' * 100}")
    print(
        f"{'mode':>6} {'dim':>5} "
        f"{'lin_acc':>7} {'lin_bal':>7} "
        f"{'mlp2_acc':>8} {'mlp2_bal':>8} "
        f"{'mlp3_acc':>8} {'mlp3_bal':>8} {'best':>6}"
    )
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: (x["mode"], x["latent_dim"])):
        lp = r["coarse_linear"]
        m2 = r["coarse_mlp2"]
        m3 = r["coarse_mlp3"]
        print(
            f"{r['mode']:>6} {r['latent_dim']:5d} "
            f"{lp['accuracy']:7.3f} {lp['balanced_accuracy']:7.3f} "
            f"{m2['accuracy']:8.3f} {m2['balanced_accuracy']:8.3f} "
            f"{m3['accuracy']:8.3f} {m3['balanced_accuracy']:8.3f} "
            f"{r['coarse_best']:>6}"
        )


if __name__ == "__main__":
    main()
