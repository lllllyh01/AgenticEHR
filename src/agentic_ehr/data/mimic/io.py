"""DuckDB-backed access to the MIMIC-IV CSV files.

MIMIC-IV ships as gzipped CSVs; ``icu/chartevents`` and ``hosp/labevents`` are
hundreds of millions of rows, so we query them out-of-core with DuckDB (predicate
pushdown on ``itemid`` keeps memory bounded) instead of loading them with pandas.
"""
from __future__ import annotations

from pathlib import Path

from ...logging_utils import get_logger

logger = get_logger(__name__)

# Relative locations inside a MIMIC-IV v3.x root (the directory containing
# ``hosp/`` and ``icu/``).
TABLES = {
    "patients": "hosp/patients.csv.gz",
    "admissions": "hosp/admissions.csv.gz",
    "diagnoses_icd": "hosp/diagnoses_icd.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "prescriptions": "hosp/prescriptions.csv.gz",
    "icustays": "icu/icustays.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "inputevents": "icu/inputevents.csv.gz",
}


def resolve_root(root: str | Path) -> Path:
    """Return the MIMIC-IV root, tolerating a path that points above the
    ``physionet.org/files/mimiciv/<ver>`` mirror layout."""
    root = Path(root).expanduser()
    if (root / "hosp").is_dir() and (root / "icu").is_dir():
        return root
    # Allow pointing at a parent of the physionet mirror tree.
    for cand in sorted(root.glob("**/mimiciv/*/")):
        if (cand / "hosp").is_dir() and (cand / "icu").is_dir():
            logger.info("Resolved MIMIC root to %s", cand)
            return cand
    raise FileNotFoundError(
        f"Could not find a MIMIC-IV root (a dir containing hosp/ and icu/) under {root}"
    )


def connect(threads: int | None = None):
    import duckdb

    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={int(threads)}")
    return con


def table(root: Path, name: str) -> str:
    """Return a DuckDB ``read_csv_auto(...)`` expression for a MIMIC table.

    Use it directly in FROM clauses, e.g. ``f"SELECT ... FROM {table(root,'patients')}"``.
    """
    rel = TABLES[name]
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing MIMIC table {name!r}: {path}")
    # all_varchar=False lets DuckDB infer numeric types; we cast explicitly where needed.
    return f"read_csv_auto('{path.as_posix()}', compression='gzip')"
