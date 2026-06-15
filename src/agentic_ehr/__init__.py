"""Agentic wrapper over EHR risk models.

The package is split into decoupled layers:

* ``data``    - ingest structured longitudinal EHR, featurize, dataset abstraction.
* ``models``  - pluggable ``RiskModel`` interface + XGBoost baseline.
* ``explain`` - attributions, concept mapping, and the ``RiskProfile`` contract.
* ``agent``   - turns a ``RiskProfile`` into a patient-friendly ``PatientSummary``.
* ``eval``    - predictive metrics + pragmatic summary-quality checks.
* ``api``     - FastAPI service exposing the summary endpoint.

The predictive model and the agent communicate *only* through ``RiskProfile``,
so the model can be swapped (XGBoost -> MOTOR-T -> a foundation model) without
touching the agent.
"""

__version__ = "0.1.0"
