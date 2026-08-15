#!/usr/bin/env python3
"""Train and benchmark LightGBM on Kaggle's credit-card fraud dataset.

Designed for the AWS Lab 16 CPU compute node (t3.medium: 2 vCPU, 4 GiB RAM).
The script keeps the test set untouched until final evaluation, uses a separate
validation set for early stopping and threshold selection, and writes all
results to JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Avoid accidental thread oversubscription on the 2-vCPU lab instance.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark LightGBM for credit-card fraud detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "ml-benchmark" / "creditcard.csv",
        help="Path to creditcard.csv (default: ~/ml-benchmark/creditcard.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "ml-benchmark" / "benchmark_result.json",
        help="Output JSON path (default: ~/ml-benchmark/benchmark_result.json).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="LightGBM worker threads (default: at most 2).",
    )
    return parser.parse_args()


def require_valid_args(args: argparse.Namespace) -> tuple[Path, Path]:
    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Download it first with: kaggle datasets download "
            "-d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/"
        )
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return data_path, output_path


def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    # float32 halves feature memory versus pandas' default float64. This is
    # precise enough for LightGBM and is friendlier to the 4-GiB instance.
    dtypes: dict[str, Any] = {
        "Time": np.float32,
        "Amount": np.float32,
        "Class": np.uint8,
    }
    dtypes.update({f"V{i}": np.float32 for i in range(1, 29)})

    started = time.perf_counter()
    frame = pd.read_csv(data_path, dtype=dtypes, engine="c", low_memory=False)
    load_seconds = time.perf_counter() - started

    if "Class" not in frame.columns:
        raise ValueError("Dataset must contain the target column 'Class'.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame.isnull().values.any():
        raise ValueError("Dataset contains missing values; aborting for reproducibility.")

    target = frame.pop("Class").to_numpy(dtype=np.uint8, copy=False)
    feature_names = frame.columns.tolist()
    features = np.ascontiguousarray(frame.to_numpy(dtype=np.float32, copy=False))
    del frame

    if np.unique(target).size != 2:
        raise ValueError("Target column 'Class' must contain exactly two classes.")

    return features, target, feature_names, load_seconds


def choose_f1_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Choose the probability threshold that maximizes F1 on validation data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, 0.0

    denominator = precisions[:-1] + recalls[:-1]
    f1_values = np.divide(
        2.0 * precisions[:-1] * recalls[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def benchmark_inference(
    model: LGBMClassifier,
    x_test: np.ndarray,
    best_iteration: int,
    threads: int,
) -> dict[str, Any]:
    """Measure warmed-up single-row latency and 1,000-row throughput."""
    single_row = np.ascontiguousarray(x_test[:1])
    batch_size = min(1000, len(x_test))
    batch = np.ascontiguousarray(x_test[:batch_size])

    def predict(values: np.ndarray, num_threads: int) -> np.ndarray:
        return model.predict_proba(
            values,
            num_iteration=best_iteration,
            num_threads=num_threads,
        )[:, 1]

    # Warm-up removes one-time initialization from the reported measurements.
    for _ in range(20):
        predict(single_row, 1)
    for _ in range(3):
        predict(batch, threads)

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        single_durations: list[float] = []
        for _ in range(1000):
            started = time.perf_counter_ns()
            predict(single_row, 1)
            single_durations.append((time.perf_counter_ns() - started) / 1_000_000)

        batch_durations: list[float] = []
        for _ in range(30):
            started = time.perf_counter_ns()
            predict(batch, threads)
            batch_durations.append((time.perf_counter_ns() - started) / 1_000_000)
    finally:
        if gc_was_enabled:
            gc.enable()

    median_batch_ms = statistics.median(batch_durations)
    throughput = batch_size / (median_batch_ms / 1000.0)

    return {
        "single_row_latency_ms_median": round(statistics.median(single_durations), 6),
        "single_row_latency_ms_p95": round(
            float(np.percentile(single_durations, 95)), 6
        ),
        "single_row_repetitions": len(single_durations),
        "batch_size": batch_size,
        "batch_latency_ms_median": round(median_batch_ms, 6),
        "batch_throughput_rows_per_second": round(throughput, 2),
        "batch_repetitions": len(batch_durations),
        "warmup_enabled": True,
    }


def main() -> None:
    args = parse_args()
    data_path, output_path = require_valid_args(args)

    print(f"Loading dataset: {data_path}", flush=True)
    features, target, feature_names, load_seconds = load_dataset(data_path)

    split_started = time.perf_counter()
    # Final test set is held out and is never used for training, early stopping,
    # or threshold selection.
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=args.seed,
        stratify=target,
    )
    del features, target

    # 20% of the remaining 80% becomes validation: 64% train / 16% val / 20% test.
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=args.seed,
        stratify=y_train_full,
    )
    del x_train_full, y_train_full
    split_seconds = time.perf_counter() - split_started

    negative_count = int(np.count_nonzero(y_train == 0))
    positive_count = int(np.count_nonzero(y_train == 1))
    class_imbalance_ratio = negative_count / positive_count

    model_parameters: dict[str, Any] = {
        "objective": "binary",
        # Disable LightGBM's default binary_logloss metric. Early stopping below
        # should follow Average Precision only, which is much more informative
        # than log-loss or accuracy for this extremely imbalanced dataset.
        "metric": "None",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "random_state": args.seed,
        "n_jobs": args.threads,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    print(
        f"Rows: train={len(x_train):,}, validation={len(x_valid):,}, "
        f"test={len(x_test):,} | fraud ratio in train="
        f"{positive_count / len(y_train):.6%}",
        flush=True,
    )
    print("Training LightGBM...", flush=True)

    model = LGBMClassifier(**model_parameters)
    training_started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        eval_names=["validation"],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )
    training_seconds = time.perf_counter() - training_started

    raw_best_iteration = getattr(model, "best_iteration_", None)
    best_iteration = int(raw_best_iteration or model_parameters["n_estimators"])

    validation_probabilities = model.predict_proba(
        x_valid,
        num_iteration=best_iteration,
        num_threads=args.threads,
    )[:, 1]
    decision_threshold, validation_best_f1 = choose_f1_threshold(
        y_valid, validation_probabilities
    )

    test_probabilities = model.predict_proba(
        x_test,
        num_iteration=best_iteration,
        num_threads=args.threads,
    )[:, 1]
    test_predictions = (test_probabilities >= decision_threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, test_probabilities)),
        "pr_auc": float(average_precision_score(y_test, test_probabilities)),
        "majority_class_baseline_accuracy": float(np.mean(y_test == 0)),
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "f1_score": float(f1_score(y_test, test_predictions, zero_division=0)),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "decision_threshold": decision_threshold,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    print("Benchmarking inference...", flush=True)
    inference = benchmark_inference(
        model=model,
        x_test=x_test,
        best_iteration=best_iteration,
        threads=args.threads,
    )

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "data_load_seconds": load_seconds,
            "training_seconds": training_seconds,
            "best_iteration": best_iteration,
            "auc_roc": metrics["auc_roc"],
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "inference_latency_1_row_ms": inference[
                "single_row_latency_ms_median"
            ],
            "inference_throughput_1000_rows_per_second": inference[
                "batch_throughput_rows_per_second"
            ],
        },
        "timings_seconds": {
            "data_load": load_seconds,
            "data_preparation_and_split": split_seconds,
            "training": training_seconds,
        },
        "dataset": {
            "path": str(data_path),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "train_rows": len(x_train),
            "validation_rows": len(x_valid),
            "test_rows": len(x_test),
            "train_fraud_rows": positive_count,
            "train_normal_rows": negative_count,
        },
        "training": {
            "best_iteration": best_iteration,
            "validation_best_f1": validation_best_f1,
            "class_imbalance_ratio": class_imbalance_ratio,
            "imbalance_strategy": (
                "Unweighted training plus F1-optimal threshold selected only on "
                "the validation set"
            ),
            "parameters": model_parameters,
        },
        "test_metrics": metrics,
        "inference": inference,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "configured_threads": args.threads,
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print("\n=== BENCHMARK RESULT ===")
    print(f"Data load:                 {load_seconds:.4f} s")
    print(f"Training:                  {training_seconds:.4f} s")
    print(f"Best iteration:            {best_iteration}")
    print(f"Decision threshold:        {decision_threshold:.6f}")
    print(f"AUC-ROC:                   {metrics['auc_roc']:.6f}")
    print(f"PR-AUC:                    {metrics['pr_auc']:.6f}")
    print(
        "Majority baseline accuracy: "
        f"{metrics['majority_class_baseline_accuracy']:.6f}"
    )
    print(f"Accuracy:                  {metrics['accuracy']:.6f}")
    print(f"F1-score:                  {metrics['f1_score']:.6f}")
    print(f"Precision:                 {metrics['precision']:.6f}")
    print(f"Recall:                    {metrics['recall']:.6f}")
    print(
        "Inference latency (1 row): "
        f"{inference['single_row_latency_ms_median']:.6f} ms (median)"
    )
    print(
        "Throughput (1000 rows):    "
        f"{inference['batch_throughput_rows_per_second']:.2f} rows/s"
    )
    print(f"Result file:               {output_path}")


if __name__ == "__main__":
    main()
