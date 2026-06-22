"""Command-line entry point: train / evaluate / demo."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .logging_utils import get_logger

logger = get_logger("agentic_ehr.cli")


def _cmd_train(args) -> int:
    cfg = Config.load(args.config)
    if cfg.get("data.source") == "mimic":
        from .data.mimic.multitask import train_all

        manifest = train_all(cfg, cfg.get("paths.model_dir", "artifacts/models_mimic"))
        summary = {name: {"auroc": round(i["metrics"]["auroc"], 3),
                          "auprc": round(i["metrics"]["auprc"], 3), "group": i["group"]}
                   for name, i in manifest["tasks"].items()}
        print(json.dumps({"model_dir": cfg.get("paths.model_dir"), "tasks": summary}, indent=2))
        return 0

    from .pipeline import train
    result = train(cfg)
    print(json.dumps({"model_path": result.model_path, "test_metrics": result.metrics}, indent=2))
    return 0


def _cmd_evaluate(args) -> int:
    cfg = Config.load(args.config)
    if cfg.get("data.source") == "mimic":
        from .data.mimic.multitask import MultiTaskModel

        model = MultiTaskModel.load(args.model or cfg.get("paths.model_dir", "artifacts/models_mimic"))
        metrics = {name: tm.metrics for name, tm in model.task_models.items()}
        print(json.dumps({"task_metrics": metrics}, indent=2))
        return 0

    from .eval.model_eval import evaluate_predictions
    from .eval.summary_eval import evaluate_summary
    from .pipeline import InferenceService

    svc = InferenceService.from_config(cfg, model_path=args.model)

    # Predictive metrics on the held-out test split.
    _, _, test_split = svc.dataset.split(seed=cfg.get("seed", 42))
    prob = svc.model.predict_proba(test_split.X)
    model_metrics = evaluate_predictions(test_split.y, prob).to_dict()

    # Summary-quality metrics on a sample of test patients.
    sample = test_split.records[: args.n_summaries]
    agg = {"factual_consistency": 0, "no_unsupported_claims": 0,
           "uncertainty_faithful": 0, "all_sections_present": 0, "clarity_score": 0.0, "passed": 0}
    for rec in sample:
        profile, summary = svc.summary_for(rec.patient_id)
        q = evaluate_summary(summary, profile)
        for k in ("factual_consistency", "no_unsupported_claims",
                  "uncertainty_faithful", "all_sections_present", "passed"):
            agg[k] += int(getattr(q, k) if k != "passed" else q.passed)
        agg["clarity_score"] += q.clarity_score
    n = max(1, len(sample))
    summary_metrics = {k: (v / n if isinstance(v, float) else v / n) for k, v in agg.items()}

    print(json.dumps({
        "model_metrics": model_metrics,
        "summary_metrics_over_n": {"n": len(sample), **summary_metrics},
    }, indent=2))
    return 0


def _cmd_demo(args) -> int:
    cfg = Config.load(args.config)
    if cfg.get("data.source") == "mimic":
        return _demo_mimic(args, cfg)

    from .pipeline import InferenceService

    svc = InferenceService.from_config(cfg, model_path=args.model)

    if args.patient_id:
        patient_id = args.patient_id
    else:
        # Pick a clearly higher-risk patient from the test split for a vivid demo.
        _, _, test_split = svc.dataset.split(seed=cfg.get("seed", 42))
        probs = svc.model.predict_proba(test_split.X)
        patient_id = test_split.records[int(probs.argmax())].patient_id

    # --no-template switches the LLM from the fixed five-section structured output
    # to a single free-form narrative.
    use_template = not args.no_template
    profile, summary = svc.summary_for(patient_id, use_template=use_template)

    print("=" * 72)
    print(f"PATIENT: {patient_id}   |   risk tier: {profile.risk_tier}   "
          f"|   estimate: {profile.probability_pct}%   "
          f"|   confidence: {profile.confidence_label}   "
          f"|   mode: {summary.mode}")
    print("=" * 72)
    print(summary.to_text())
    if summary.guardrail_warnings:
        print("\n[guardrail warnings]", summary.guardrail_warnings, file=sys.stderr)
    feats = svc.dataset.features_for(patient_id).iloc[0].to_dict() if args.save_response else None
    _maybe_save_response(args, patient_id, profile, summary,
                         model_info={"model_path": args.model}, features=feats)
    return 0


def _demo_mimic(args, cfg) -> int:
    """MIMIC multi-task path for `demo`: prediction panel -> health report."""
    from .agent.summary_agent import SummaryAgent
    from .data.mimic.service import MultiTaskInferenceService

    svc = MultiTaskInferenceService.from_config(cfg, model_dir=args.model)
    patient_id = args.patient_id or svc.any_patient_id()
    profile = svc.profile_for(patient_id)

    print("=" * 72)
    print(f"PATIENT {patient_id} | age {profile.demographics.get('age')} "
          f"sex {profile.demographics.get('sex')}")
    print("=" * 72)
    print("FORWARD RISK PANEL:")
    for t in profile.forward:
        if t.kind == "regression":
            print(f"  {t.label:24s} ~{t.point_estimate:.1f} days  ({t.confidence_label} confidence)")
        else:
            print(f"  {t.label:24s} {t.probability_pct:3d}%  ({t.confidence_label} confidence, AUROC {t.auroc:.2f})")
    print("CHRONIC PHENOTYPE PANEL:")
    for t in profile.chronic:
        print(f"  {t.label:48s} {t.probability_pct:3d}%")
    print("=" * 72)

    report = SummaryAgent.from_config(cfg).summarize(profile, use_template=not args.no_template)
    print(report.to_text())
    if report.guardrail_warnings:
        print("\n[guardrail warnings]", report.guardrail_warnings, file=sys.stderr)
    _maybe_save_response(args, patient_id, profile, report,
                         model_info={"model_dir": args.model or cfg.get("paths.model_dir")},
                         features=svc.features_for(patient_id) if args.save_response else None)
    return 0


def _maybe_save_response(args, patient_id, profile, summary, *, model_info, features) -> None:
    if not getattr(args, "save_response", None):
        return
    from .eval.response_log import build_record, write_record

    formats = tuple(f.strip() for f in (args.response_format or "json,md").split(","))
    record = build_record(patient_id, profile, summary, model_info=model_info, features=features)
    paths = write_record(record, args.save_response, formats=formats)
    print(f"\n[saved response] {', '.join(paths)}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-ehr", description=__doc__)
    p.add_argument("--config", default=None, help="Path to YAML config (default: config/default.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train model(s). Multi-task when data.source=mimic.")
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="Evaluate predictive (+ summary) metrics.")
    e.add_argument("--model", default=None, help="Model path (single) or model dir (mimic).")
    e.add_argument("--n-summaries", type=int, default=20)
    e.set_defaults(func=_cmd_evaluate)

    d = sub.add_parser("demo", help="Generate a patient-facing summary / health report.")
    d.add_argument("--model", default=None, help="Model path (single) or model dir (mimic).")
    d.add_argument("--patient-id", default=None)
    d.add_argument(
        "--no-template",
        action="store_true",
        help="Generate a free-form summary instead of the fixed five-section template.",
    )
    d.add_argument("--save-response", default=None, metavar="DIR",
                   help="Write the prediction metadata + report to DIR for review.")
    d.add_argument("--response-format", default="json,md",
                   help="Comma-separated formats to save: json,md,txt (default: json,md).")
    d.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        logger.error("Command failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
