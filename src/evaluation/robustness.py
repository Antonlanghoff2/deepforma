from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ._common import to_jsonable


@dataclass(slots=True)
class RobustnessRun:
    seed: int
    scenario: str
    metrics: dict[str, float]


@dataclass(slots=True)
class RobustnessReport:
    runs: list[RobustnessRun]
    summary: dict[str, dict[str, float]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def summarize_runs(runs: list[RobustnessRun]) -> RobustnessReport:
    grouped: dict[str, dict[str, list[float]]] = {}
    warnings: list[str] = []
    for run in runs:
        scenario = grouped.setdefault(run.scenario, {})
        for metric, value in run.metrics.items():
            scenario.setdefault(metric, []).append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for scenario, metrics in grouped.items():
        for metric, values in metrics.items():
            summary[f"{scenario}.{metric}"] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
            }
        if len(next(iter(metrics.values()), [])) > 1:
            warnings.append(f"Scénario {scenario}: plusieurs seeds agrégés.")
    return RobustnessReport(runs=runs, summary=summary, warnings=warnings)


def evaluate_robustness(
    scenarios: dict[str, list[dict[str, Any]]],
    evaluator: Callable[[list[dict[str, Any]]], dict[str, float]],
    *,
    seeds: list[int],
) -> RobustnessReport:
    runs: list[RobustnessRun] = []
    for scenario_name, rows in scenarios.items():
        for seed in seeds:
            payload = evaluator(rows)
            runs.append(RobustnessRun(seed=seed, scenario=scenario_name, metrics=payload))
    return summarize_runs(runs)
