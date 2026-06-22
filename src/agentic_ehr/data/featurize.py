"""Featurization: longitudinal events -> a fixed-width numeric feature vector.

This is a simplified, dependency-free version of FEMR's "count featurizer":
for each patient we count code occurrences within a lookback window before the
prediction time, and summarise the most recent numeric lab values. The vocab
(which codes become columns) is learned from the training split only.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .schema import PatientRecord

logger = get_logger(__name__)

# Default codes whose latest numeric value becomes a feature. Datasets with
# their own numeric concepts (e.g. MIMIC labs/vitals) override this via the
# ``numeric_codes`` constructor argument (wired from ``data.featurize.numeric_codes``).
_NUMERIC_CODES = ("LAB/HbA1c", "LAB/eGFR")


class CountFeaturizer:
    """Learns a code vocabulary from training records, then transforms records.

    Feature columns:
      * ``count__<code>``      number of occurrences in the lookback window
      * ``value__<code>``      most recent numeric value (for numeric codes)
      * ``age``, ``sex_female`` demographics
      * ``n_events`` total event count in window
    """

    def __init__(
        self,
        lookback_days: int = 365,
        max_features: int = 200,
        numeric_codes: tuple[str, ...] | list[str] | str | None = None,
    ):
        self.lookback_days = lookback_days
        self.max_features = max_features
        # Codes whose latest numeric value is exposed as a ``value__<code>`` column.
        # ``numeric_codes="auto"`` detects them from the data (any code that ever
        # carries a numeric value); otherwise an explicit whitelist is used.
        if numeric_codes == "auto":
            self.numeric_mode = "auto"
            self.numeric_code_whitelist: tuple[str, ...] = ()
        elif numeric_codes is not None:
            self.numeric_mode = "whitelist"
            self.numeric_code_whitelist = tuple(numeric_codes)
        else:
            self.numeric_mode = "whitelist"
            self.numeric_code_whitelist = _NUMERIC_CODES
        self.vocab_: list[str] = []
        self.numeric_codes_: list[str] = []
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, records: list[PatientRecord]) -> "CountFeaturizer":
        freq: dict[str, int] = {}
        for rec in records:
            cutoff = rec.prediction_time - timedelta(days=self.lookback_days)
            seen = {e.code for e in rec.events if cutoff <= e.time <= rec.prediction_time}
            for code in seen:
                freq[code] = freq.get(code, 0) + 1
        # Sort by frequency desc, breaking ties on code name so the column
        # order is deterministic regardless of set/dict iteration order (which
        # depends on PYTHONHASHSEED). This keeps training fully reproducible.
        self.vocab_ = [c for c, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][: self.max_features]
        if self.numeric_mode == "auto":
            numeric_detected = {
                e.code
                for rec in records
                for e in rec.events
                if isinstance(e.value, (int, float)) and e.value == e.value  # exclude NaN
            }
            self.numeric_codes_ = [c for c in self.vocab_ if c in numeric_detected]
        else:
            self.numeric_codes_ = [c for c in self.numeric_code_whitelist if c in self.vocab_]
        self.feature_names_ = (
            [f"count__{c}" for c in self.vocab_]
            + [f"value__{c}" for c in self.numeric_codes_]
            + ["age", "sex_female", "n_events"]
        )
        self._fitted = True
        logger.info(
            "Featurizer fitted: %d code columns, %d numeric value columns",
            len(self.vocab_), len(self.numeric_codes_),
        )
        return self

    def transform(self, records: list[PatientRecord]) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("CountFeaturizer must be fitted before transform().")
        rows = [self._transform_one(rec) for rec in records]
        return pd.DataFrame(rows, columns=self.feature_names_, index=[r.patient_id for r in records])

    def fit_transform(self, records: list[PatientRecord]) -> pd.DataFrame:
        return self.fit(records).transform(records)

    def _transform_one(self, rec: PatientRecord) -> dict[str, float]:
        cutoff = rec.prediction_time - timedelta(days=self.lookback_days)
        window = [e for e in rec.events if cutoff <= e.time <= rec.prediction_time]
        counts: dict[str, int] = {}
        latest_value: dict[str, float] = {}
        for e in window:
            counts[e.code] = counts.get(e.code, 0) + 1
            if e.code in self.numeric_codes_ and isinstance(e.value, (int, float)):
                latest_value[e.code] = float(e.value)  # window is time-sorted; last wins

        row: dict[str, float] = {}
        for c in self.vocab_:
            row[f"count__{c}"] = float(counts.get(c, 0))
        for c in self.numeric_codes_:
            # NaN for "not measured"; XGBoost handles missing natively.
            row[f"value__{c}"] = latest_value.get(c, np.nan)
        row["age"] = float(rec.demographics.get("age", np.nan))
        row["sex_female"] = 1.0 if rec.demographics.get("sex") == "female" else 0.0
        row["n_events"] = float(len(window))
        return row
