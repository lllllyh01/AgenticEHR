"""Multi-task training + model container for the MIMIC health-summary tasks.

The shared feature matrix is built once; one calibrated XGBoost is trained per
task. Chronic-phenotype tasks drop their label-defining feature columns (the
"don't feed the answer" rule). At inference the container predicts every task,
producing the panel that becomes a multi-label ``HealthRiskProfile``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ...config import Config
from ...eval.model_eval import evaluate_predictions, evaluate_regression
from ...logging_utils import get_logger
from ...models.base import BaseModel
from ...models.xgboost_classifier import XGBoostClassifierModel
from ...models.xgboost_regression import XGBoostRegressionModel
from ..dataset import _load_event_label_records
from ..featurize import CountFeaturizer
from . import tasks as T

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
FEATURIZER_NAME = "featurizer.joblib"


# --------------------------------------------------------------------------- #
# Shared feature matrix                                                        #
# --------------------------------------------------------------------------- #
def _labels_path(cfg: Config, task_name: str) -> str:
    events_path = Path(cfg.get("data.mimic.events_path"))
    return str(events_path.parent / f"labels_{task_name}.parquet")


def build_matrix(cfg: Config):
    """Load events once, fit the featurizer, return (records, featurizer, X)."""
    events_path = cfg.get("data.mimic.events_path")
    # Any task's labels file provides patient ids / prediction time / demographics.
    anchor_labels = _labels_path(cfg, T.ALL_TASKS[0].name)
    records = _load_event_label_records(events_path, anchor_labels, "mimic")
    featurizer = CountFeaturizer(
        lookback_days=cfg.get("data.featurize.lookback_days", 3650),
        max_features=cfg.get("data.featurize.max_features", 400),
        numeric_codes=cfg.get("data.featurize.numeric_codes", "auto"),
    )
    X = featurizer.fit_transform(records)
    logger.info("Feature matrix: %d patients x %d features", *X.shape)
    return records, featurizer, X


def _load_y(cfg: Config, task_name: str, index: pd.Index) -> np.ndarray:
    df = pd.read_parquet(_labels_path(cfg, task_name))
    s = df.set_index(df["patient_id"].astype(str))["value"]
    return s.reindex(index).to_numpy()


def split_ids(index: pd.Index, seed: int, val_frac: float, test_frac: float):
    rng = np.random.default_rng(seed)
    pos = rng.permutation(len(index))
    n_test = int(len(pos) * test_frac)
    n_val = int(len(pos) * val_frac)
    return pos[n_test + n_val:], pos[n_test:n_test + n_val], pos[:n_test]  # train, val, test


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_all(cfg: Config, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    params = cfg.get("model.params", {})
    calibrate = cfg.get("model.calibrate", True)
    calib_method = cfg.get("model.calibration_method", "isotonic")

    _, featurizer, X = build_matrix(cfg)
    joblib.dump(featurizer, out / FEATURIZER_NAME)
    tr, va, te = split_ids(X.index, seed, 0.15, 0.15)

    manifest = {"featurizer": FEATURIZER_NAME, "feature_names": list(X.columns), "tasks": {}}
    for task in T.ALL_TASKS:
        y = _load_y(cfg, task.name, X.index)
        keep = [c for c in X.columns if c not in set(T.excluded_columns(list(X.columns), task))]
        Xb = X[keep]
        if task.kind == "regression":
            model = XGBoostRegressionModel(params=params)
            model.fit(Xb.iloc[tr], y[tr], X_val=Xb.iloc[va], y_val=y[va])
            metrics = evaluate_regression(y[te], model.predict(Xb.iloc[te])).to_dict()
            logger.info("  %-22s MAE=%.2f RMSE=%.2f R2=%.3f (n=%d, feats=%d)",
                        task.name, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["n"], len(keep))
        else:
            model = XGBoostClassifierModel(params=params, calibrate=calibrate, calibration_method=calib_method)
            model.fit(Xb.iloc[tr], y[tr].astype(int), X_val=Xb.iloc[va], y_val=y[va].astype(int))
            metrics = evaluate_predictions(y[te].astype(int), model.predict_proba(Xb.iloc[te])).to_dict()
            logger.info("  %-22s AUROC=%.3f AUPRC=%.3f (n=%d, pos=%d, feats=%d)",
                        task.name, metrics["auroc"], metrics["auprc"], metrics["n"],
                        int(y[te].sum()), len(keep))
        model_file = f"{task.name}.joblib"
        model.save(str(out / model_file))
        manifest["tasks"][task.name] = {
            "model": model_file, "columns": keep, "kind": task.kind, "group": task.group,
            "label": task.label, "positive_label": task.positive_label, "horizon": task.horizon,
            "metrics": metrics,
        }

    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("Saved %d task models + manifest to %s", len(manifest["tasks"]), out)
    return manifest


# --------------------------------------------------------------------------- #
# Inference container                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class TaskModel:
    spec: T.TaskSpec
    model: BaseModel
    columns: list[str]
    metrics: dict


class MultiTaskModel:
    """Loads the featurizer + all per-task models for panel inference."""

    def __init__(self, featurizer: CountFeaturizer, task_models: dict[str, TaskModel]):
        self.featurizer = featurizer
        self.task_models = task_models

    @classmethod
    def load(cls, model_dir: str) -> "MultiTaskModel":
        out = Path(model_dir)
        manifest = json.loads((out / MANIFEST_NAME).read_text())
        featurizer = joblib.load(out / manifest["featurizer"])
        task_models: dict[str, TaskModel] = {}
        for name, info in manifest["tasks"].items():
            model_cls = XGBoostRegressionModel if info["kind"] == "regression" else XGBoostClassifierModel
            task_models[name] = TaskModel(
                spec=T.get_task(name),
                model=model_cls.load(str(out / info["model"])),
                columns=info["columns"],
                metrics=info["metrics"],
            )
        logger.info("Loaded MultiTaskModel: %d tasks from %s", len(task_models), out)
        return cls(featurizer, task_models)

    def predict_panel(self, x_row: pd.DataFrame) -> dict[str, "object"]:
        """Return {task_name: ModelOutput} for a single featurized patient row."""
        out = {}
        for name, tm in self.task_models.items():
            out[name] = tm.model.predict_output(x_row[tm.columns])[0]
        return out
