"""
train.py - FraudScan model training.

Trains: CombinedFeatureExtractor (TF-IDF + Indian hand-crafted features) -> LogisticRegression

Outputs to model/:
    model.pkl               - full sklearn pipeline
    evaluation_report.json  - precision, recall, F1, confusion matrix
    top_features.json       - top fraud/real LR coefficients for extension popup

Usage:
    python model/train.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).parent.parent))
from model.features import CombinedFeatureExtractor

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("model")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline():
    return Pipeline([
        ("features", CombinedFeatureExtractor(max_features=30000)),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=42,
        )),
    ])


def get_top_features(pipeline, n=20):
    clf = pipeline.named_steps["clf"]
    extractor = pipeline.named_steps["features"]
    all_names = extractor.get_feature_names_out()
    coefs = clf.coef_[0]

    top_fraud_idx = np.argsort(coefs)[::-1][:n]
    top_real_idx  = np.argsort(coefs)[:n]

    return {
        "top_fraud_signals": [
            {"feature": all_names[i], "coefficient": float(coefs[i])}
            for i in top_fraud_idx
        ],
        "top_real_signals": [
            {"feature": all_names[i], "coefficient": float(coefs[i])}
            for i in top_real_idx
        ],
    }


def evaluate(pipeline, X_val, y_val, X_test, y_test):
    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = pipeline.predict(X)
        report = classification_report(y, preds, output_dict=True, target_names=["real", "fraud"])
        cm = confusion_matrix(y, preds).tolist()
        results[split_name] = {
            "classification_report": report,
            "confusion_matrix": cm,
        }
        print(f"\n── {split_name.upper()} ──")
        print(classification_report(y, preds, target_names=["real", "fraud"]))
        print(f"Confusion matrix:\n{np.array(cm)}")
    return results


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    pipeline = build_pipeline()
    pipeline.fit(train, train["label"])
    print("Training complete.")

    eval_results = evaluate(pipeline, val, val["label"], test, test["label"])
    top_features = get_top_features(pipeline)

    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {OUT_DIR / 'model.pkl'}")

    with open(OUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    with open(OUT_DIR / "top_features.json", "w") as f:
        json.dump(top_features, f, indent=2)

    print(f"Evaluation + features saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()