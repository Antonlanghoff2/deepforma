from __future__ import annotations

from .binary_classification import BinaryClassMetrics, BinaryClassReport, evaluate_binary_classification
from .calibration import BinaryThresholdOptimizationResult, ThresholdOptimizationResult, optimize_binary_threshold, optimize_thresholds, save_binary_threshold_json, save_thresholds_json
from .multilabel_classification import MultilabelEvaluationReport, MultilabelLabelMetrics, evaluate_multilabel_classification
from .recommendation import RecommendationCase, RecommendationEvaluationReport, evaluate_recommendation
from .report import latest_evaluation_dir, load_evaluation_report, write_evaluation_artifacts
from .robustness import RobustnessReport, RobustnessRun, evaluate_robustness, summarize_runs
from .skill_extraction import (
    SkillExtractionDocument,
    SkillExtractionEvaluationReport,
    SkillExtractionScores,
    evaluate_skill_extraction,
    normalize_skill_text,
)

__all__ = [
    "BinaryClassMetrics",
    "BinaryClassReport",
    "BinaryThresholdOptimizationResult",
    "ThresholdOptimizationResult",
    "MultilabelEvaluationReport",
    "MultilabelLabelMetrics",
    "RecommendationCase",
    "RecommendationEvaluationReport",
    "RobustnessReport",
    "RobustnessRun",
    "SkillExtractionDocument",
    "SkillExtractionEvaluationReport",
    "SkillExtractionScores",
    "evaluate_binary_classification",
    "evaluate_multilabel_classification",
    "evaluate_recommendation",
    "evaluate_robustness",
    "evaluate_skill_extraction",
    "latest_evaluation_dir",
    "load_evaluation_report",
    "normalize_skill_text",
    "optimize_binary_threshold",
    "optimize_thresholds",
    "save_binary_threshold_json",
    "save_thresholds_json",
    "summarize_runs",
    "write_evaluation_artifacts",
]
