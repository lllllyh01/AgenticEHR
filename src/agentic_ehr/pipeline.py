"""End-to-end orchestration helpers used by the CLI, API, and tests.

These tie the decoupled layers together but add no new logic of their own, so
the boundaries stay clean: data -> model -> explain(RiskProfile) -> agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .agent.summary_agent import PatientSummary, SummaryAgent
from .config import Config
from .data.dataset import EHRDataset
from .explain.attributions import Attributor
from .explain.concept_map import ConceptMap
from .explain.risk_profile import RiskProfile, RiskProfileBuilder
from .logging_utils import get_logger
from .models.base import RiskModel
from .models.registry import build_from_config

logger = get_logger(__name__)


@dataclass
class TrainResult:
    model: RiskModel
    model_path: str
    metrics: dict


def train(cfg: Config) -> TrainResult:
    """Build the dataset, train the configured model, evaluate, and save it."""
    from .eval.model_eval import evaluate_predictions

    seed = cfg.get("seed", 42)
    dataset = EHRDataset.from_config(cfg)
    train_split, val_split, test_split = dataset.split(seed=seed) # using synthetic data for now. Will change to determined split in real EHR-shot dataset

    model = build_from_config(cfg)
    model.fit(train_split.X, train_split.y, X_val=val_split.X, y_val=val_split.y)

    test_prob = model.predict_proba(test_split.X)
    metrics = evaluate_predictions(test_split.y, test_prob)
    logger.info("Test metrics: %s", metrics.to_dict())

    model_dir = Path(cfg.get("paths.model_dir", "artifacts/models"))
    model_path = str(model_dir / f"{model.name}.joblib")
    model.save(model_path)

    return TrainResult(model=model, model_path=model_path, metrics=metrics.to_dict())


class InferenceService:
    """Holds a loaded model + dataset and produces RiskProfiles and summaries.

    The model is consumed only to produce a RiskProfile; the agent then consumes
    the RiskProfile. Swapping the model changes only ``build_from_config`` /
    the loaded artifact — not this class's interface or the agent.
    """

    def __init__(
        self,
        cfg: Config,
        model: RiskModel,
        dataset: EHRDataset,
        agent: SummaryAgent | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.concept_map = ConceptMap.from_config(cfg)
        train_split, _, _ = dataset.split(seed=cfg.get("seed", 42))
        self.attributor = Attributor(
            model, background=train_split.X, method=cfg.get("explain.method", "auto")
        )
        self.builder = RiskProfileBuilder(
            attributor=self.attributor,
            concept_map=self.concept_map,
            risk_tiers=cfg.get("agent.risk_tiers"),
        )
        # Default to the configured LLM agent; allow injecting a pre-built agent
        # (e.g. an external adapter or an offline test double) to keep this class
        # decoupled from how the agent is constructed.
        self.agent = agent if agent is not None else SummaryAgent.from_config(cfg)
        self.top_k = cfg.get("explain.top_k_contributors", 5)

    @classmethod
    def from_config(cls, cfg: Config, model_path: str | None = None) -> "InferenceService":
        dataset = EHRDataset.from_config(cfg)
        if model_path is None:
            model_dir = Path(cfg.get("paths.model_dir", "artifacts/models"))
            model_name = cfg.get("model.name", "xgboost")
            model_path = str(model_dir / f"{model_name}.joblib")
        model = _load_model(cfg, model_path)
        return cls(cfg, model, dataset)

    def risk_profile_for(self, patient_id: str) -> RiskProfile:
        x_row = self.dataset.features_for(patient_id)
        output = self.model.predict_output(x_row)[0]
        snapshot = self.dataset.snapshot(patient_id)
        return self.builder.build(
            model_output=output,
            x_row=x_row,
            snapshot=snapshot,
            task=self.dataset.task,
            top_k=self.top_k,
        )

    def summary_for(
        self, patient_id: str, use_template: bool = True
    ) -> tuple[RiskProfile, PatientSummary]:
        profile = self.risk_profile_for(patient_id)
        summary = self.agent.summarize(profile, use_template=use_template)
        return profile, summary

    def profile_from_features(self, x_row: pd.DataFrame, snapshot) -> RiskProfile:
        """Build a profile from an arbitrary feature row (e.g. an external model
        adapter or an API payload), bypassing the bundled dataset."""
        output = self.model.predict_output(x_row)[0]
        return self.builder.build(
            model_output=output, x_row=x_row, snapshot=snapshot,
            task=self.dataset.task, top_k=self.top_k,
        )


def _load_model(cfg: Config, model_path: str) -> RiskModel:
    name = cfg.get("model.name", "xgboost")
    if name == "xgboost":
        from .models.xgboost_model import XGBoostRiskModel

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `agentic-ehr train` first."
            )
        return XGBoostRiskModel.load(model_path)
    raise NotImplementedError(
        f"Loading for model '{name}' is not wired up. Add a loader in pipeline._load_model."
    )
