"""
Evaluate a trained RTF model at different detection counts.

Loads a single pretrained checkpoint and evaluates it by truncating
the test set to N=3,5,7,10,15,20,30,50,100,all detections, extracting
embeddings, and running linear probe classification at each stage.

This answers: "at what detection count does each modality start helping?"

Usage:
    python src/eval_early.py \
        --checkpoint runs/ae_dim64_meta_gp_randtrunc/best_model.pt \
        --summary runs/ae_dim64_meta_gp_randtrunc/summary.json \
        --data-dir data_gp \
        --output-dir analysis/early_classification
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import MetaNPZDataset, collate_fn
from model import LightCurveCompressor


def _get_extra_args(batch, device):
    images = batch["images"].to(device) if "images" in batch else None
    gp_features = batch["gp_features"].to(device) if "gp_features" in batch else None
    return images, gp_features


def extract_embeddings(model, loader, device):
    model.eval()
    all_emb, all_fine, all_coarse = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            images, gp_features = _get_extra_args(batch, device)
            emb = model.embed(x, pad_mask, images, gp_features=gp_features)
            all_emb.append(emb.cpu().numpy())
            all_fine.append(batch["label_fine"].numpy())
            all_coarse.append(batch["label_coarse"].numpy())
    return (
        np.concatenate(all_emb),
        np.concatenate(all_fine),
        np.concatenate(all_coarse),
    )


def linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1)
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "balanced_accuracy": bal, "roc_auc": auc}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RTF at different detection counts"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--data-dir", required=True, help="Path to preprocessed data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--detection-counts",
        nargs="+",
        type=int,
        default=[3, 5, 7, 10, 15, 20, 30, 50, 100],
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--image-backbone",
        default=None,
        help="Override image backbone (simple or zoobot)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override num_classes for classification head",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model config
    with open(args.summary) as f:
        summary = json.load(f)

    image_backbone = args.image_backbone or summary.get("image_backbone", "simple")
    num_classes = (
        args.num_classes
        if args.num_classes is not None
        else summary.get("num_classes", 0)
    )
    model = LightCurveCompressor(
        mode=summary["mode"],
        latent_dim=summary["latent_dim"],
        in_channels=summary.get("in_channels", 7),
        d_model=summary.get("d_model", 128),
        use_images=summary.get("use_images", False),
        image_backbone=image_backbone,
        freeze_backbone=summary.get("freeze_backbone", image_backbone == "zoobot"),
        gp_dim=summary.get("gp_dim", 0),
        num_classes=num_classes,
    ).to(args.device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    )
    model.eval()
    print(f"Loaded model: {summary['mode']} dim={summary['latent_dim']}")

    use_images = summary.get("use_images", False)
    use_gp = summary.get("use_gp", False)

    # Add "all" (no truncation) to detection counts
    detection_counts = args.detection_counts + [None]

    results = []
    for max_det in detection_counts:
        label = f"N={max_det}" if max_det else "all"
        print(f"\n{'=' * 40}")
        print(f"Evaluating at {label}")

        train_ds = MetaNPZDataset(
            os.path.join(args.data_dir, "train"),
            use_images=use_images,
            use_gp=use_gp,
            max_detections=max_det,
        )
        test_ds = MetaNPZDataset(
            os.path.join(args.data_dir, "test"),
            use_images=use_images,
            use_gp=use_gp,
            max_detections=max_det,
        )

        pin = args.device != "cpu"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=pin,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=pin,
        )

        train_emb, train_fine, train_coarse = extract_embeddings(
            model, train_loader, args.device
        )
        test_emb, test_fine, test_coarse = extract_embeddings(
            model, test_loader, args.device
        )

        coarse = linear_probe(train_emb, train_coarse, test_emb, test_coarse)
        fine = linear_probe(train_emb, train_fine, test_emb, test_fine)

        entry = {
            "max_detections": max_det,
            "label": label,
            "coarse": coarse,
            "fine": fine,
        }
        results.append(entry)
        print(
            f"  Coarse: acc={coarse['accuracy']:.3f} bal={coarse['balanced_accuracy']:.3f} auc={coarse['roc_auc']:.3f}"
        )
        print(
            f"  Fine:   acc={fine['accuracy']:.3f} bal={fine['balanced_accuracy']:.3f} auc={fine['roc_auc']:.3f}"
        )

    # Save results
    with open(os.path.join(args.output_dir, "early_classification.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'=' * 70}")
    print("EARLY CLASSIFICATION: accuracy vs detection count")
    print(f"{'=' * 70}")
    print(f"{'N':>5} {'c_acc':>7} {'c_bal':>7} {'c_auc':>7} {'f_acc':>7} {'f_bal':>7}")
    print("-" * 45)
    for r in results:
        c, fi = r["coarse"], r["fine"]
        print(
            f"{r['label']:>5} {c['accuracy']:7.3f} {c['balanced_accuracy']:7.3f} "
            f"{c['roc_auc']:7.3f} {fi['accuracy']:7.3f} {fi['balanced_accuracy']:7.3f}"
        )


if __name__ == "__main__":
    main()
