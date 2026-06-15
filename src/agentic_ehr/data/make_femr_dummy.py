#!/usr/bin/env python
"""Generate a FEMR / EHR-shot-format DUMMY dataset to exercise the real
`data.source=femr` loader end-to-end (no gated data required).

Writes two tables in exactly the schema `_load_femr_records` expects:

  events.csv : patient_id, time, code, value, description
  labels.csv : patient_id, label_time, value, age, sex

The events carry the same learnable signal as the synthetic generator, so the
XGBoost baseline has something real to fit — but here it round-trips through the
on-disk FEMR-format loader, not the in-memory generator.

This is a STAND-IN, not real EHR-shot patient data.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from agentic_ehr.data.synthetic import generate_records
from agentic_ehr.logging_utils import get_logger

logger = get_logger("make_femr_dummy")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="artifacts/femr_dummy")
    p.add_argument("--n-patients", type=int, default=4000)
    p.add_argument("--positive-rate", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = generate_records(
        n_patients=args.n_patients, positive_rate=args.positive_rate, seed=args.seed
    )

    events_path = out / "events.csv"
    labels_path = out / "labels.csv"

    n_events = 0
    with open(events_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "time", "code", "value", "description"])
        for rec in records:
            for e in rec.events:
                w.writerow([
                    rec.patient_id,
                    e.time.isoformat(),
                    e.code,
                    "" if e.value is None else e.value,
                    e.description or "",
                ])
                n_events += 1

    with open(labels_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "label_time", "value", "age", "sex"])
        for rec in records:
            w.writerow([
                rec.patient_id,
                rec.prediction_time.isoformat(),
                rec.label,
                rec.demographics.get("age", ""),
                rec.demographics.get("sex", ""),
            ])

    logger.info("Wrote %d patients, %d events", len(records), n_events)
    print(f"events: {events_path}  ({n_events} rows)")
    print(f"labels: {labels_path}  ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
