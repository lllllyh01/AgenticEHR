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


def _events_dir(cfg: Config) -> Path:
    return Path(cfg.get("data.mimic.events_path")).parent


def _labels_path(cfg: Config, task_name: str) -> str:
    return str(_events_dir(cfg) / f"labels_{task_name}.parquet")


def _featurizer_name(window: int | None) -> str:
    return f"featurizer_{T.window_tag(window)}.joblib"


def _windows_used() -> list[int | None]:
    return sorted({t.window for t in T.ALL_TASKS}, key=lambda w: (w is not None, w or 0))


def _anchor_task(window: int | None) -> T.TaskSpec:
    return next(t for t in T.ALL_TASKS if t.window == window)


def _build_matrix(cfg: Config, window: int | None):
    """Featurize one observation window: (featurizer, X indexed by patient_id)."""
    events_path = str(_events_dir(cfg) / T.events_filename(window))
    anchor_labels = _labels_path(cfg, _anchor_task(window).name)
    records = _load_event_label_records(events_path, anchor_labels, "mimic")
    featurizer = CountFeaturizer(
        lookback_days=cfg.get("data.featurize.lookback_days", 3650),
        max_features=cfg.get("data.featurize.max_features", 400),
        numeric_codes=cfg.get("data.featurize.numeric_codes", "auto"),
    )
    X = featurizer.fit_transform(records)
    logger.info("Feature matrix (%s window): %d patients x %d features",
                T.window_tag(window), *X.shape)
    return featurizer, X


def _load_y_series(cfg: Config, task_name: str) -> pd.Series:
    df = pd.read_parquet(_labels_path(cfg, task_name))
    return df.set_index(df["patient_id"].astype(str))["value"]


def split_ids(ids: pd.Index, seed: int, val_frac: float, test_frac: float):
    """Deterministic patient-id split. Applied to one canonical id ordering and
    reused across all windows, so a patient is in the same fold everywhere."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = int(len(ids) * test_frac)
    n_val = int(len(ids) * val_frac)
    ids = pd.Index(ids)
    return ids[perm[n_test + n_val:]], ids[perm[n_test:n_test + n_val]], ids[perm[:n_test]]


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

    # Featurize each observation window; fit one featurizer per window.
    matrices = {w: _build_matrix(cfg, w) for w in _windows_used()}
    featurizers = {}
    for w, (featurizer, _) in matrices.items():
        name = _featurizer_name(w)
        joblib.dump(featurizer, out / name)
        featurizers[T.window_tag(w)] = name       # JSON-safe key (int/None can't be one)

    # One canonical patient-id split (from the discharge window), reused across
    # windows so a patient never lands in different folds for different tasks.
    canon = (matrices[None] if None in matrices else next(iter(matrices.values())))[1].index
    train_ids, val_ids, test_ids = split_ids(canon, seed, 0.15, 0.15)

    manifest = {"featurizers": featurizers, "tasks": {}}
    for task in T.ALL_TASKS:
        X = matrices[task.window][1]
        y = _load_y_series(cfg, task.name)
        keep = [c for c in X.columns if c not in set(T.excluded_columns(list(X.columns), task))]
        Xtr, Xva, Xte = X.loc[train_ids, keep], X.loc[val_ids, keep], X.loc[test_ids, keep]
        ytr, yva, yte = y.loc[train_ids], y.loc[val_ids], y.loc[test_ids]
        if task.kind == "regression":
            model = XGBoostRegressionModel(params=params)
            model.fit(Xtr, ytr.to_numpy(), X_val=Xva, y_val=yva.to_numpy())
            metrics = evaluate_regression(yte.to_numpy(), model.predict(Xte)).to_dict()
            logger.info("  %-22s [%s] MAE=%.2f RMSE=%.2f R2=%.3f (n=%d, feats=%d)",
                        task.name, T.window_tag(task.window), metrics["mae"], metrics["rmse"],
                        metrics["r2"], metrics["n"], len(keep))
        else:
            model = XGBoostClassifierModel(params=params, calibrate=calibrate, calibration_method=calib_method)
            model.fit(Xtr, ytr.astype(int).to_numpy(), X_val=Xva, y_val=yva.astype(int).to_numpy())
            metrics = evaluate_predictions(yte.astype(int).to_numpy(), model.predict_proba(Xte)).to_dict()
            logger.info("  %-22s [%s] AUROC=%.3f AUPRC=%.3f (n=%d, pos=%d, feats=%d)",
                        task.name, T.window_tag(task.window), metrics["auroc"], metrics["auprc"],
                        metrics["n"], int(yte.sum()), len(keep))
        model_file = f"{task.name}.joblib"
        model.save(str(out / model_file))
        manifest["tasks"][task.name] = {
            "model": model_file, "columns": keep, "kind": task.kind, "group": task.group,
            "label": task.label, "positive_label": task.positive_label, "horizon": task.horizon,
            "window": task.window, "metrics": metrics,
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
    """Loads the per-window featurizers + all per-task models for panel inference."""

    def __init__(self, featurizers: dict[str, CountFeaturizer], task_models: dict[str, TaskModel]):
        self.featurizers = featurizers      # window_tag -> fitted CountFeaturizer
        self.task_models = task_models

    def featurizer_for(self, window: int | None) -> CountFeaturizer:
        return self.featurizers[T.window_tag(window)]

    @classmethod
    def load(cls, model_dir: str) -> "MultiTaskModel":
        out = Path(model_dir)
        manifest = json.loads((out / MANIFEST_NAME).read_text())
        featurizers = {w: joblib.load(out / fname) for w, fname in manifest["featurizers"].items()}
        task_models: dict[str, TaskModel] = {}
        for name, info in manifest["tasks"].items():
            model_cls = XGBoostRegressionModel if info["kind"] == "regression" else XGBoostClassifierModel
            task_models[name] = TaskModel(
                spec=T.get_task(name),
                model=model_cls.load(str(out / info["model"])),
                columns=info["columns"],
                metrics=info["metrics"],
            )
        logger.info("Loaded MultiTaskModel: %d tasks, %d windows from %s",
                    len(task_models), len(featurizers), out)
        return cls(featurizers, task_models)
