"""
merge_datasets.py

Merges three sources:
  1. synthetic_indian_jobs.csv  - 300 fraud + 452 real (Groq-generated, Indian market)
  2. EMSCAD fraud rows (866)    - real human-written Western scam postings
  3. EMSCAD real rows (500)     - sampled real human-written Western job postings

Final dataset: ~1166 fraud, ~952 real

Usage:
    python data_pipeline/merge_datasets.py
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CSV      = RAW_DIR / "synthetic_indian_jobs.csv"
EMSCAD_CSV         = RAW_DIR / "emscad_fake_job_postings.csv"
EMSCAD_REAL_SAMPLE = 500


def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["description"]  = df["description"].fillna("")
    df["requirements"] = df["requirements"].fillna("")
    df["full_text"] = (
        df["title"].fillna("") + " "
        + df["company"].fillna("") + " "
        + df["description"] + " "
        + df["requirements"]
    ).str.strip()
    return df[["title", "company", "location", "salary", "contact", "full_text", "label", "source", "archetype"]]


def load_emscad(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    def build_rows(subset, label, source, archetype):
        subset = subset.copy()
        subset["description"]  = subset["description"].fillna("")
        subset["requirements"] = subset["requirements"].fillna("")
        subset["full_text"] = (
            subset["title"].fillna("") + " "
            + subset["company_profile"].fillna("") + " "
            + subset["description"] + " "
            + subset["requirements"]
        ).str.strip()
        subset["label"]     = label
        subset["source"]    = source
        subset["archetype"] = archetype
        subset["company"]   = ""
        subset["salary"]    = subset["salary_range"].fillna("")
        subset["contact"]   = ""
        subset["location"]  = subset["location"].fillna("")
        return subset[["title", "company", "location", "salary", "contact", "full_text", "label", "source", "archetype"]]

    fraud = build_rows(df[df["fraudulent"] == 1], label=1, source="emscad", archetype="emscad_fraud")
    real  = build_rows(
        df[df["fraudulent"] == 0].sample(n=EMSCAD_REAL_SAMPLE, random_state=42),
        label=0, source="emscad_real", archetype="legitimate"
    )
    return fraud, real


def split_and_save(df: pd.DataFrame):
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=42
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name}: {len(split)} rows  (fraud={(split['label']==1).sum()}, real={(split['label']==0).sum()})")

    metadata = {
        "total_rows": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "archetype_distribution": df["archetype"].value_counts().to_dict(),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "random_state": 42,
    }
    with open(OUT_DIR / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {OUT_DIR}/dataset_metadata.json")


def main():
    synthetic = load_synthetic(SYNTHETIC_CSV)
    print(f"Synthetic  : {len(synthetic)} rows  (fraud={(synthetic['label']==1).sum()}, real={(synthetic['label']==0).sum()})")

    emscad_fraud, emscad_real = load_emscad(EMSCAD_CSV)
    print(f"EMSCAD fraud: {len(emscad_fraud)} rows")
    print(f"EMSCAD real : {len(emscad_real)} rows (sampled {EMSCAD_REAL_SAMPLE})")

    combined = pd.concat([synthetic, emscad_fraud, emscad_real], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nCombined   : {len(combined)} rows  (fraud={(combined['label']==1).sum()}, real={(combined['label']==0).sum()})")

    print("\nSplitting...")
    split_and_save(combined)
    print(f"\nDone. Files in {OUT_DIR}/")


if __name__ == "__main__":
    main()