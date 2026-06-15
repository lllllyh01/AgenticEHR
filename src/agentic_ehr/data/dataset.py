"""The dataset abstraction the rest of the system consumes.

``EHRDataset`` bundles the featurized matrix ``X``, labels ``y``, the fitted
featurizer, the raw :class:`PatientRecord` list (needed to build agent-facing
snapshots), and the task metadata. It can be built from the synthetic generator
or from FEMR / EHR-shot-style extracts on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .featurize import CountFeaturizer
from .schema import Event, PatientRecord, PatientSnapshot, TaskMetadata
from .synthetic import generate_records

logger = get_logger(__name__)


@dataclass
class DataSplit:
    X: pd.DataFrame
    y: np.ndarray
    records: list[PatientRecord]


class EHRDataset:
    def __init__(
        self,
        records: list[PatientRecord],
        featurizer: CountFeaturizer,
        task: TaskMetadata,
    ):
        self.records = records
        self.featurizer = featurizer
        self.task = task
        self._record_by_id = {r.patient_id: r for r in records}

    # ----- constructors -----------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "EHRDataset":
        source = cfg.get("data.source", "synthetic")
        task = TaskMetadata.from_config(cfg)
        featurizer = CountFeaturizer(
            lookback_days=cfg.get("data.featurize.lookback_days", 365),
            max_features=cfg.get("data.featurize.max_features", 200),
        )
        if source == "synthetic":
            records = generate_records(
                n_patients=cfg.get("data.synthetic.n_patients", 4000),
                positive_rate=cfg.get("data.synthetic.positive_rate", 0.18),
                seed=cfg.get("seed", 42),
            )
        elif source == "femr":
            records = _load_femr_records(cfg)
        else:
            raise ValueError(f"Unknown data.source: {source!r}")
        featurizer.fit(records)
        logger.info("Built EHRDataset: %d patients, source=%s", len(records), source)
        return cls(records, featurizer, task)

    # ----- access ------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return self.featurizer.feature_names_

    def matrix(self) -> tuple[pd.DataFrame, np.ndarray]:
        X = self.featurizer.transform(self.records)
        y = np.array([r.label if r.label is not None else -1 for r in self.records])
        return X, y

    def split(self, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
        """Deterministic train/val/test split returning :class:`DataSplit` objects."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.records))
        n_test = int(len(idx) * test_frac)
        n_val = int(len(idx) * val_frac)
        test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
        X, y = self.matrix()
        Xv = X.values  # row order matches self.records
        def make(ix):
            recs = [self.records[i] for i in ix]
            return DataSplit(
                X=pd.DataFrame(Xv[ix], columns=X.columns, index=[self.records[i].patient_id for i in ix]),
                y=y[ix],
                records=recs,
            )
        return make(train_idx), make(val_idx), make(test_idx)

    def record(self, patient_id: str) -> PatientRecord:
        return self._record_by_id[patient_id]

    def features_for(self, patient_id: str) -> pd.DataFrame:
        return self.featurizer.transform([self.record(patient_id)])

    def snapshot(self, patient_id: str) -> PatientSnapshot:
        return build_snapshot(self.record(patient_id), self.featurizer.lookback_days)


# --------------------------------------------------------------------------
# Snapshot builder: compact, non-identifying current-state view for the agent.
# --------------------------------------------------------------------------
def build_snapshot(rec: PatientRecord, lookback_days: int) -> PatientSnapshot:
    from datetime import timedelta

    cutoff = rec.prediction_time - timedelta(days=lookback_days)
    window = [e for e in rec.events if cutoff <= e.time <= rec.prediction_time]
    conditions = sorted({
        e.description or e.code for e in window if e.code.startswith("ICD")
    })
    obs: dict[str, object] = {}
    for e in window:
        if isinstance(e.value, (int, float)):
            label = e.description or e.code
            obs[label] = round(float(e.value), 1)  # latest wins (sorted)
    encounters = sum(1 for e in window if e.code.startswith("ENC"))
    return PatientSnapshot(
        patient_id=rec.patient_id,
        age=rec.demographics.get("age"),
        sex=rec.demographics.get("sex"),
        active_conditions=conditions,
        recent_observations=obs,
        n_recent_encounters=encounters,
    )


# --------------------------------------------------------------------------
# FEMR / EHR-shot loader. Configurable, tolerant of missing files so the demo
# never hard-fails; raises clear errors when 'femr' source is requested without
# the necessary extracts.
# --------------------------------------------------------------------------
def _load_femr_records(cfg) -> list[PatientRecord]:
    events_path = cfg.get("data.femr.events_path")
    labels_path = cfg.get("data.femr.labels_path")
    if not events_path or not labels_path:
        raise FileNotFoundError(
            "data.source=femr requires data.femr.events_path and data.femr.labels_path. "
            "Provide FEMR/EHR-shot extracts (see README) or use data.source=synthetic."
        )
    events_df = _read_table(events_path)
    labels_df = _read_table(labels_path)

    required_events = {"patient_id", "time", "code"}
    required_labels = {"patient_id", "label_time", "value"}
    _require_columns(events_df, required_events, events_path)
    _require_columns(labels_df, required_labels, labels_path)

    by_patient: dict[str, list[Event]] = {}
    for row in events_df.itertuples(index=False):
        by_patient.setdefault(str(row.patient_id), []).append(
            Event(
                time=pd.to_datetime(row.time).to_pydatetime(),
                code=str(row.code),
                value=getattr(row, "value", None),
                description=getattr(row, "description", None),
            )
        )

    records: list[PatientRecord] = []
    for row in labels_df.itertuples(index=False):
        pid = str(row.patient_id)
        events = sorted(by_patient.get(pid, []), key=lambda e: e.time)
        demo = {}
        # Optional demographic columns on the labels table.
        for col in ("age", "sex"):
            if hasattr(row, col):
                demo[col] = getattr(row, col)
        records.append(
            PatientRecord(
                patient_id=pid,
                events=events,
                prediction_time=pd.to_datetime(row.label_time).to_pydatetime(),
                label=int(row.value),
                demographics=demo,
            )
        )
    logger.info("Loaded %d FEMR/EHR-shot records", len(records))
    return records


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Extract not found: {path}")
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: set[str], path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
