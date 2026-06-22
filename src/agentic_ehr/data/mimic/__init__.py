"""MIMIC-IV cohort, feature, and label extraction (out-of-core via DuckDB).

Builds the FEMR-style ``events`` / ``labels`` tables the rest of the pipeline
consumes. Entry point: ``python -m agentic_ehr.data.mimic.build``.
"""
