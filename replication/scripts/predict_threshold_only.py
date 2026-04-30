"""Train a logistic-regression classifier on Kushnareva threshold features.

This is a smaller-scope substitute for the full features_prediction.ipynb,
which concatenates threshold + barcode + template features. We don't have
the barcode/template features yet (blocked on ripserplusplus, see
ph-project-nmh), but the threshold features alone are enough to verify the
pipeline produces a real classifier accuracy on this hardware.

Inputs (produced by features_calculation_by_thresholds.ipynb):
- replication/outputs/features/{train,test}_all_heads_12_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_128_bert-base-uncased.npy
- replication/data/processed/{train,test}.csv  (for labels)

Reports accuracy and Matthews correlation. With 500/class training data and
threshold features only, expect well below the paper's ~85-95% (which used
20K/class and all three feature families) but substantially above 50%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
FEATURES_DIR = REPO_ROOT / "outputs" / "features"

FEATURE_FILE_TEMPLATE = (
    "{subset}_all_heads_12_layers_s_e_v_c_b0b1_lists_array_6"
    "_thrs_MAX_LEN_128_bert-base-uncased.npy"
)


def load_split(subset: str) -> tuple[np.ndarray, np.ndarray]:
    feat_path = FEATURES_DIR / FEATURE_FILE_TEMPLATE.format(subset=subset)
    csv_path = DATA_DIR / f"{subset}.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"missing {feat_path}\n"
            f"Run features_calculation_by_thresholds.ipynb with "
            f"subset = '{subset}' first."
        )
    features = np.load(feat_path, allow_pickle=True)
    labels = pd.read_csv(csv_path)["labels"].to_numpy()
    n_samples = features.shape[3]
    if n_samples != len(labels):
        raise ValueError(
            f"feature/label count mismatch on {subset}: "
            f"{n_samples} features vs {len(labels)} labels"
        )
    X = np.stack([features[:, :, :, i, :].flatten() for i in range(n_samples)])
    return X.astype(np.float32), labels.astype(np.int64)


def main() -> None:
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print(f"train: X={X_train.shape} y={y_train.shape} balance={np.bincount(y_train).tolist()}")
    print(f"test:  X={X_test.shape} y={y_test.shape} balance={np.bincount(y_test).tolist()}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    clf.fit(X_train_s, y_train)

    y_train_pred = clf.predict(X_train_s)
    y_test_pred = clf.predict(X_test_s)

    print()
    print(f"train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"test  accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"test  matthews corrcoef: {matthews_corrcoef(y_test, y_test_pred):.4f}")


if __name__ == "__main__":
    main()
