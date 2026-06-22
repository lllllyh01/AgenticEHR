"""Registry of MIMIC-IV prediction tasks for the health-summary agent.

Two panels feed the LLM:
  * forward — genuinely predictive, advice-driving outcomes.
  * chronic — phenotype profile (context). For these we drop the feature columns
    that *define* the target (see ``exclude_prefixes`` / DX_DEFINING_FEATURES)
    so the task is not a no-op.

All tasks share the same input feature matrix (train == inference); only the
label vector and the per-task excluded columns differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import concepts as C


@dataclass(frozen=True)
class TaskSpec:
    name: str                       # matches labels_<name>.parquet
    kind: str                       # "binary" | "regression"
    group: str                      # "forward" | "chronic"
    label: str                      # short human label, e.g. "Diabetes"
    positive_label: str             # phrasing for the agent, e.g. "a diagnosis of diabetes"
    horizon: str
    exclude_prefixes: tuple[str, ...] = field(default=())


FORWARD_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("mortality_1y", "binary", "forward", "1-year mortality",
             "death within one year of discharge", "the next 12 months"),
    TaskSpec("readmission_30d", "binary", "forward", "30-day readmission",
             "an unplanned hospital readmission within 30 days", "the next 30 days"),
    TaskSpec("prolonged_stay", "binary", "forward", "prolonged stay",
             "a prolonged hospital stay (7 days or more)", "this hospital admission"),
    TaskSpec("los_days", "regression", "forward", "length of stay (days)",
             "the expected number of hospital days", "this hospital admission"),
)

CHRONIC_TASKS: tuple[TaskSpec, ...] = tuple(
    TaskSpec(
        name=f"dx_{g.code.lower()}",
        kind="binary",
        group="chronic",
        label=g.description,
        positive_label=f"a diagnosis of {g.description.lower()}",
        horizon="currently",
        exclude_prefixes=C.DX_DEFINING_FEATURES.get(g.code, (f"DX/{g.code}",)),
    )
    for g in C.CHRONIC_TARGETS
)

ALL_TASKS: tuple[TaskSpec, ...] = FORWARD_TASKS + CHRONIC_TASKS

# Tasks the binary multi-task trainer handles now (regression is a later milestone).
BINARY_TASKS: tuple[TaskSpec, ...] = tuple(t for t in ALL_TASKS if t.kind == "binary")


def get_task(name: str) -> TaskSpec:
    for t in ALL_TASKS:
        if t.name == name:
            return t
    raise KeyError(f"Unknown task: {name!r}")


def excluded_columns(feature_names: list[str], task: TaskSpec) -> list[str]:
    """Feature columns to drop for this task (the 'don't feed the answer' rule).

    Columns look like ``count__<code>`` / ``value__<code>``; a column is dropped
    when its code starts with any of the task's excluded prefixes.
    """
    dropped = []
    for col in feature_names:
        code = col.split("__", 1)[1] if "__" in col else col
        if any(code.startswith(p) for p in task.exclude_prefixes):
            dropped.append(col)
    return dropped
