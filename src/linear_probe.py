"""
Run linear probe classification on embeddings from any trained model.

Usage:
    python src/linear_probe.py --runs-dir runs --output-dir analysis
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc = float("nan")
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "roc_auc": roc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_dirs = sorted([d for d in os.listdir(args.runs_dir)
                       if os.path.isdir(os.path.join(args.runs_dir, d))])

    results = []
    for d in run_dirs:
        run_path = os.path.join(args.runs_dir, d)
        train_path = os.path.join(run_path, "train_embeddings.npz")
        test_path = os.path.join(run_path, "test_embeddings.npz")
        summary_path = os.path.join(run_path, "summary.json")

        if not all(os.path.exists(p) for p in [train_path, test_path, summary_path]):
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        train_data = np.load(train_path, allow_pickle=True)
        test_data = np.load(test_path, allow_pickle=True)

        print(f"\n{d}: {summary['mode']} dim={summary['latent_dim']}")

        coarse = linear_probe(
            train_data["embeddings"], train_data["labels_coarse"],
            test_data["embeddings"], test_data["labels_coarse"],
        )
        fine = linear_probe(
            train_data["embeddings"], train_data["labels_fine"],
            test_data["embeddings"], test_data["labels_fine"],
        )

        entry = {
            "run": d, "mode": summary["mode"], "latent_dim": summary["latent_dim"],
            "compression_ratio": summary["compression_ratio"],
            "compressed_bytes": summary["compressed_bytes"],
            "coarse_5class": coarse, "fine_10class": fine,
            "recon_cont": summary["test_metrics"]["recon_cont"],
            "recon_band": summary["test_metrics"]["recon_band"],
            "band_acc": summary["test_metrics"]["band_acc"],
        }
        results.append(entry)

        print(f"  Coarse: acc={coarse['accuracy']:.3f} bal={coarse['balanced_accuracy']:.3f} auc={coarse['roc_auc']:.3f}")
        print(f"  Fine:   acc={fine['accuracy']:.3f} bal={fine['balanced_accuracy']:.3f} auc={fine['roc_auc']:.3f}")

    with open(os.path.join(args.output_dir, "linear_probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*100}")
    print("LINEAR PROBE RESULTS (all models)")
    print(f"{'='*100}")
    print(f"{'mode':>6} {'dim':>5} {'compress':>8} {'recon':>8} {'band%':>6} "
          f"{'c_acc':>6} {'c_bal':>6} {'c_auc':>6} {'f_acc':>6} {'f_bal':>6} {'f_auc':>6}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: (x["mode"], x["latent_dim"])):
        c, fi = r["coarse_5class"], r["fine_10class"]
        print(f"{r['mode']:>6} {r['latent_dim']:5d} {r['compression_ratio']:7.1f}x "
              f"{r['recon_cont']+r['recon_band']:8.4f} {r['band_acc']:6.3f} "
              f"{c['accuracy']:6.3f} {c['balanced_accuracy']:6.3f} {c['roc_auc']:6.3f} "
              f"{fi['accuracy']:6.3f} {fi['balanced_accuracy']:6.3f} {fi['roc_auc']:6.3f}")


if __name__ == "__main__":
    main()
