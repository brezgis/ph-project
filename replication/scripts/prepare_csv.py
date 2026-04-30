"""Combine WebText (human) and GPT-2 Small (machine) JSONL files into the
labeled CSV format the Kushnareva reference notebooks expect.

Each output row has columns: sentence, labels (0=human, 1=machine). Column
names match what the reference notebooks read directly (data['sentence'],
data['labels']) so no notebook-side renaming is needed.

Output files: data/processed/{train,valid,test}.csv

By default, takes all examples in each split. Pass --max-per-class to
subsample symmetrically — useful for tiny first-pass runs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"

SPLITS = ("train", "valid", "test")
HUMAN_PREFIX = "webtext"
MACHINE_PREFIX = "small-117M"


def load_jsonl_texts(path: Path) -> list[str]:
    """Load text fields from a JSONL file, skipping rows whose text is empty
    or whitespace-only. Such rows roundtrip as NaN through pandas CSV I/O,
    which breaks downstream notebook code that calls re.sub on each value.
    """
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def build_split(split: str, max_per_class: int | None, seed: int) -> pd.DataFrame:
    human = load_jsonl_texts(RAW_DIR / f"{HUMAN_PREFIX}.{split}.jsonl")
    machine = load_jsonl_texts(RAW_DIR / f"{MACHINE_PREFIX}.{split}.jsonl")

    rng = random.Random(seed)
    if max_per_class is not None:
        if len(human) > max_per_class:
            human = rng.sample(human, max_per_class)
        if len(machine) > max_per_class:
            machine = rng.sample(machine, max_per_class)

    rows = [(t, 0) for t in human] + [(t, 1) for t in machine]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["sentence", "labels"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap examples per class per split (default: use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling and shuffling.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        df = build_split(split, args.max_per_class, args.seed + hash(split) % 10_000)
        out_path = OUT_DIR / f"{split}.csv"
        df.to_csv(out_path, index=False)
        n_h = (df.labels == 0).sum()
        n_m = (df.labels == 1).sum()
        print(f"{split}: {len(df)} rows ({n_h} human, {n_m} machine) -> {out_path}")


if __name__ == "__main__":
    main()
