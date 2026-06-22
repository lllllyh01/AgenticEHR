"""Build a discharge-anchored MIMIC-IV cohort + multi-task feature/label tables.

Purpose: feed a health-summary agent. A "prediction step" first produces a panel
of signals about a patient from their record up to hospital **discharge**; the
LLM then turns that panel into a summary + advice.

Output uses the FEMR-style contract the existing pipeline consumes, so each task
rides the tested featurizer -> model -> explain -> agent path:

  events.parquet            patient_id (= index hadm_id), time, code, value, description
  labels_<task>.parquet     patient_id, label_time, value, age, sex

Anchor / cohort:
  * One index admission per adult subject: the first hospital admission that
    contains an ICU stay and that the patient survived to discharge.
  * Anchor = hospital ``dischtime``; "past EHR" = this admission + prior admissions.

Shared input features (one set, used by every task; train == inference):
  VITAL/<V>_<STAT>  vital-sign summary over the admission (chartevents)
  LAB/<NAME>        latest lab value in the admission (labevents)
  DX/<GROUP>        comorbidity present in PRIOR admissions (history)
  UTIL/N_PRIOR_ADM  number of prior admissions
  + age, sex (on the labels table)

Targets:
  forward : mortality_1y, readmission_30d, prolonged_stay, los_days (regression)
  chronic : dx_<group> for each ICD phenotype group

Run:  python -m agentic_ehr.data.mimic.build --root <MIMIC_ROOT> --out-dir artifacts/mimic
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...logging_utils import get_logger
from . import concepts as C
from . import io

logger = get_logger(__name__)


@dataclass
class BuildConfig:
    root: str
    out_dir: str = "artifacts/mimic"
    min_age: int = 18
    los_long_days: float = 7.0
    readmit_days: int = 30
    mortality_horizon_days: int = 365
    limit_subjects: int | None = None
    threads: int | None = None


# --------------------------------------------------------------------------- #
# Cohort: first ICU-containing admission per adult survivor, anchored at        #
# discharge.                                                                    #
# --------------------------------------------------------------------------- #
def _cohort(con, root: Path, cfg: BuildConfig) -> pd.DataFrame:
    sql = f"""
    WITH icu_adm AS (SELECT DISTINCT hadm_id FROM {io.table(root, 'icustays')}),
    base AS (
      SELECT
        a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.hospital_expire_flag,
        p.gender, p.dod,
        (CAST(date_part('year', a.admittime) AS INTEGER) - p.anchor_year + p.anchor_age) AS age_at_admit,
        date_diff('hour', a.admittime, a.dischtime) / 24.0 AS hosp_los_days,
        row_number() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS adm_rank
      FROM {io.table(root, 'admissions')} a
      JOIN {io.table(root, 'patients')} p ON a.subject_id = p.subject_id
      WHERE a.hadm_id IN (SELECT hadm_id FROM icu_adm)
    )
    SELECT subject_id, hadm_id, admittime, dischtime, gender, dod,
           age_at_admit, hosp_los_days,
           dischtime AS prediction_time
    FROM base
    WHERE adm_rank = 1
      AND age_at_admit >= {cfg.min_age}
      AND hospital_expire_flag = 0          -- survived to discharge (discharge anchor)
      AND hosp_los_days > 0
    ORDER BY subject_id
    """
    if cfg.limit_subjects:
        sql += f"\nLIMIT {int(cfg.limit_subjects)}"
    df = con.execute(sql).fetch_df()
    df["hadm_id"] = df["hadm_id"].astype("int64")
    logger.info("Cohort: %d index admissions", len(df))
    return df


# --------------------------------------------------------------------------- #
# Features                                                                      #
# --------------------------------------------------------------------------- #
def _vital_features(con, root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    con.register("cohort", cohort[["hadm_id", "admittime", "dischtime"]])
    # itemid -> vital code, with Fahrenheit conversion folded into the value.
    vital_case = "CASE " + " ".join(
        f"WHEN c.itemid IN ({','.join(map(str, v.itemids + v.fahrenheit_itemids))}) THEN '{v.code}'"
        for v in C.VITAL_SERIES
    ) + " END"
    fahr_ids = ",".join(str(i) for v in C.VITAL_SERIES for i in v.fahrenheit_itemids)
    val_expr = f"CASE WHEN c.itemid IN ({fahr_ids}) THEN {C.FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum END" if fahr_ids else "c.valuenum"
    all_ids = ",".join(str(i) for v in C.VITAL_SERIES for i in (v.itemids + v.fahrenheit_itemids))
    agg = con.execute(f"""
        SELECT hadm_id, vital,
               avg(val) AS mean, min(val) AS min, max(val) AS max,
               stddev_samp(val) AS std, arg_max(val, charttime) AS last
        FROM (
          SELECT co.hadm_id, {vital_case} AS vital, {val_expr} AS val, c.charttime
          FROM {io.table(root, 'chartevents')} c
          JOIN cohort co ON c.hadm_id = co.hadm_id
          WHERE c.itemid IN ({all_ids}) AND c.valuenum IS NOT NULL
            AND c.charttime >= co.admittime AND c.charttime <= co.dischtime
        ) GROUP BY hadm_id, vital
    """).fetch_df()
    con.unregister("cohort")
    if agg.empty:
        return pd.DataFrame(columns=["hadm_id", "code", "value", "description"])
    desc = {v.code: v.description for v in C.VITAL_SERIES}
    rows = []
    for r in agg.itertuples(index=False):
        for stat in C.VITAL_STATS:
            val = getattr(r, stat)
            if val is not None and not pd.isna(val):
                rows.append({"hadm_id": r.hadm_id, "code": f"VITAL/{r.vital}_{stat}",
                             "value": float(val), "description": f"{desc[r.vital]} ({stat})"})
    return pd.DataFrame(rows)


def _lab_features(con, root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    con.register("cohort", cohort[["hadm_id", "dischtime"]])
    lab_case = "CASE " + " ".join(
        f"WHEN l.itemid IN ({','.join(map(str, lab.itemids))}) THEN '{lab.code}'"
        for lab in C.LAB_PANEL
    ) + " END"
    all_ids = ",".join(str(i) for lab in C.LAB_PANEL for i in lab.itemids)
    agg = con.execute(f"""
        SELECT hadm_id, lab, arg_max(valuenum, charttime) AS last
        FROM (
          SELECT co.hadm_id, {lab_case} AS lab, l.valuenum, l.charttime
          FROM {io.table(root, 'labevents')} l
          JOIN cohort co ON l.hadm_id = co.hadm_id
          WHERE l.itemid IN ({all_ids}) AND l.valuenum IS NOT NULL
            AND l.charttime <= co.dischtime
        ) GROUP BY hadm_id, lab
    """).fetch_df()
    con.unregister("cohort")
    if agg.empty:
        return pd.DataFrame(columns=["hadm_id", "code", "value", "description"])
    desc = {lab.code: lab.description for lab in C.LAB_PANEL}
    agg["code"] = "LAB/" + agg["lab"].astype(str)
    agg["value"] = agg["last"].astype(float)
    agg["description"] = agg["lab"].map(desc)
    return agg[["hadm_id", "code", "value", "description"]]


def _icd_by_index(con, root: Path, cohort: pd.DataFrame, prior_only: bool):
    """Pull diagnoses for each index admission, from prior admissions only
    (history features) or from prior+index (chronic labels)."""
    con.register("cohort", cohort[["subject_id", "hadm_id", "admittime"]])
    cmp = "<" if prior_only else "<="
    df = con.execute(f"""
        SELECT co.hadm_id AS index_hadm, d.icd_code, d.icd_version
        FROM cohort co
        JOIN {io.table(root, 'admissions')} a ON a.subject_id = co.subject_id AND a.admittime {cmp} co.admittime
        JOIN {io.table(root, 'diagnoses_icd')} d ON d.hadm_id = a.hadm_id
    """).fetch_df()
    con.unregister("cohort")
    groups: dict[int, set[str]] = {}
    for r in df.itertuples(index=False):
        for g in C.classify_icd(r.icd_code, r.icd_version):
            groups.setdefault(int(r.index_hadm), set()).add(g)
    return groups


def _dx_history_features(groups: dict[int, set[str]]) -> pd.DataFrame:
    desc = {g.code: g.description for g in C.ICD_GROUPS}
    rows = []
    for hadm, gs in groups.items():
        for g in gs:
            rows.append({"hadm_id": hadm, "code": f"DX/{g}", "value": np.nan,
                         "description": f"History of {desc[g].lower()}"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["hadm_id", "code", "value", "description"])


def _utilization_features(con, root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    con.register("cohort", cohort[["subject_id", "hadm_id", "admittime"]])
    df = con.execute(f"""
        SELECT co.hadm_id,
               count(a.hadm_id) FILTER (WHERE a.admittime < co.admittime) AS n_prior
        FROM cohort co
        JOIN {io.table(root, 'admissions')} a ON a.subject_id = co.subject_id
        GROUP BY co.hadm_id
    """).fetch_df()
    con.unregister("cohort")
    df["code"] = "UTIL/N_PRIOR_ADM"
    df["value"] = df["n_prior"].astype(float)
    df["description"] = "Number of prior hospital admissions"
    return df[["hadm_id", "code", "value", "description"]]


# --------------------------------------------------------------------------- #
# Labels                                                                        #
# --------------------------------------------------------------------------- #
def _readmission(con, root: Path, cohort: pd.DataFrame, days: int) -> pd.Series:
    con.register("cohort", cohort[["subject_id", "hadm_id", "dischtime"]])
    df = con.execute(f"""
        SELECT co.hadm_id,
               max(CASE WHEN a.admittime > co.dischtime
                         AND a.admittime <= co.dischtime + INTERVAL '{days} days'
                        THEN 1 ELSE 0 END) AS readmit
        FROM cohort co
        JOIN {io.table(root, 'admissions')} a ON a.subject_id = co.subject_id
        GROUP BY co.hadm_id
    """).fetch_df()
    con.unregister("cohort")
    return df.set_index("hadm_id")["readmit"]


def _labels(con, root: Path, cohort: pd.DataFrame, cfg: BuildConfig,
            chronic_groups: dict[int, set[str]]) -> dict[str, pd.DataFrame]:
    base = pd.DataFrame({
        "patient_id": cohort["hadm_id"].astype("int64").astype(str),
        "label_time": cohort["prediction_time"],
        "age": cohort["age_at_admit"].astype(float),
        "sex": cohort["gender"].map({"F": "female", "M": "male"}).fillna("unknown"),
    }).reset_index(drop=True)

    dod = pd.to_datetime(cohort["dod"]).reset_index(drop=True)
    disch = pd.to_datetime(cohort["dischtime"]).reset_index(drop=True)
    mort = ((dod.notna()) & ((dod - disch).dt.days <= cfg.mortality_horizon_days)).astype(int)

    readmit = _readmission(con, root, cohort, cfg.readmit_days)
    readmit = cohort["hadm_id"].map(readmit).fillna(0).astype(int).reset_index(drop=True)

    los = cohort["hosp_los_days"].astype(float).reset_index(drop=True)

    out: dict[str, pd.DataFrame] = {
        "mortality_1y": base.assign(value=mort.values),
        "readmission_30d": base.assign(value=readmit.values),
        "prolonged_stay": base.assign(value=(los >= cfg.los_long_days).astype(int).values),
        "los_days": base.assign(value=los.values),
    }
    # Chronic phenotype targets.
    hadm_ids = cohort["hadm_id"].astype("int64").reset_index(drop=True)
    for grp in C.CHRONIC_TARGETS:
        vals = hadm_ids.map(lambda h: 1 if grp.code in chronic_groups.get(int(h), set()) else 0)
        out[f"dx_{grp.code.lower()}"] = base.assign(value=vals.values)
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #
def _events(parts: list[pd.DataFrame], cohort: pd.DataFrame) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    ev = pd.concat(parts, ignore_index=True)
    # All feature events are timestamped at the discharge anchor (within lookback).
    pt = cohort.set_index("hadm_id")["prediction_time"]
    ev["time"] = ev["hadm_id"].map(pt)
    ev = ev.rename(columns={"hadm_id": "patient_id"})
    ev["patient_id"] = ev["patient_id"].astype("int64").astype(str)
    return ev[["patient_id", "time", "code", "value", "description"]].sort_values(["patient_id", "code"])


def build(cfg: BuildConfig) -> dict[str, str]:
    root = io.resolve_root(cfg.root)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = io.connect(threads=cfg.threads)

    cohort = _cohort(con, root, cfg)
    if cohort.empty:
        raise RuntimeError("Empty cohort — check filters / MIMIC root.")

    logger.info("Vital features ...");   vitals = _vital_features(con, root, cohort)
    logger.info("Lab features ...");     labs = _lab_features(con, root, cohort)
    logger.info("Dx history (prior) ..."); prior_groups = _icd_by_index(con, root, cohort, prior_only=True)
    dx_hist = _dx_history_features(prior_groups)
    logger.info("Utilization ...");      util = _utilization_features(con, root, cohort)
    logger.info("Chronic labels (prior+index) ..."); chronic_groups = _icd_by_index(con, root, cohort, prior_only=False)

    events = _events([vitals, labs, dx_hist, util], cohort)
    events_path = out / "events.parquet"
    events.to_parquet(events_path, index=False)
    written = {"events": str(events_path)}

    for task, df in _labels(con, root, cohort, cfg, chronic_groups).items():
        p = out / f"labels_{task}.parquet"
        df.to_parquet(p, index=False)
        written[task] = str(p)
        stat = float(df["value"].mean())
        logger.info("  %-18s n=%d  %s=%.3f", task, len(df),
                    "mean" if task == "los_days" else "pos_rate", stat)

    logger.info("Done. %d events / %d patients -> %s",
                len(events), events["patient_id"].nunique(), out)
    con.close()
    return written


def main() -> int:
    p = argparse.ArgumentParser(description="Build discharge-anchored MIMIC-IV cohort + tasks.")
    p.add_argument("--root", required=True, help="MIMIC-IV root (dir containing hosp/ and icu/)")
    p.add_argument("--out-dir", default="artifacts/mimic")
    p.add_argument("--min-age", type=int, default=18)
    p.add_argument("--los-long-days", type=float, default=7.0)
    p.add_argument("--readmit-days", type=int, default=30)
    p.add_argument("--mortality-horizon-days", type=int, default=365)
    p.add_argument("--limit-subjects", type=int, default=None, help="dev: cap cohort size")
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args()
    written = build(BuildConfig(
        root=a.root, out_dir=a.out_dir, min_age=a.min_age, los_long_days=a.los_long_days,
        readmit_days=a.readmit_days, mortality_horizon_days=a.mortality_horizon_days,
        limit_subjects=a.limit_subjects, threads=a.threads,
    ))
    for k, v in written.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
