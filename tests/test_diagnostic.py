from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from config.thresholds import THRESHOLDS, ClassificationThresholds
from config.weights import SCORING_WEIGHTS
from models.analysis_result import (
    AnalysisResult, ClassificationInfo, SkillInfo,
    TerritorialMarketInfo, QualityInfo, Recommendation,
)


class TestThresholds:
    def test_confidence_levels_high(self):
        assert THRESHOLDS.get_confidence_level(0.85) == 'forte'

    def test_confidence_levels_medium(self):
        assert THRESHOLDS.get_confidence_level(0.70) == 'moyenne'

    def test_confidence_levels_low(self):
        assert THRESHOLDS.get_confidence_level(0.50) == 'faible'

    def test_confidence_levels_rejected(self):
        assert THRESHOLDS.get_confidence_level(0.15) == 'rejetée'

    def test_classification_fiable(self):
        state = THRESHOLDS.get_classification_state(0.85, 0.15)
        assert state['state'] == 'fiable'

    def test_classification_probable(self):
        state = THRESHOLDS.get_classification_state(0.65, 0.35)
        assert state['state'] == 'probable'

    def test_classification_incertaine(self):
        state = THRESHOLDS.get_classification_state(0.52, 0.48)
        assert state['state'] == 'incertaine'

    def test_classification_insuffisant(self):
        state = THRESHOLDS.get_classification_state(0.30, 0.30)
        assert state['state'] == 'insuffisant'


class TestScoringWeights:
    def test_total_weight_one(self):
        assert abs(SCORING_WEIGHTS.total_weight - 1.0) < 0.001

    def test_compute_global(self):
        result = SCORING_WEIGHTS.compute_global({
            'couverture_competences': 80.0,
            'pertinence_metier': 70.0,
            'adequation_territoriale': 60.0,
            'niveau_experience': 50.0,
            'employabilite': 75.0,
            'actualite_programme': 65.0,
        })
        assert 0 <= result['global_score'] <= 100
        assert 'sub_scores' in result
        assert 'explanations' in result
        assert len(result['explanations']) == 6

    def test_all_zero(self):
        result = SCORING_WEIGHTS.compute_global({
            k: 0.0 for k in ['couverture_competences', 'pertinence_metier',
                             'adequation_territoriale', 'niveau_experience',
                             'employabilite', 'actualite_programme']
        })
        assert result['global_score'] == 0.0


class TestAnalysisResult:
    def test_empty_result(self):
        result = AnalysisResult()
        d = result.to_dict()
        assert d['summary'] == {}
        assert d['classification']['state'] == 'incertaine'
        assert d['detected_skills'] == []
        assert d['quality']['model_loaded'] is False

    def test_classification_states(self):
        result = AnalysisResult(
            classification=ClassificationInfo(
                is_ia=True, predicted_class=1,
                probability_ia=0.88, probability_non_ia=0.12,
                state='fiable', state_description='Test',
                gap=0.76,
            )
        )
        d = result.to_dict()
        assert d['classification']['is_ia'] is True
        assert d['classification']['state'] == 'fiable'

    def test_skill_separation(self):
        result = AnalysisResult(
            detected_skills=[
                SkillInfo(label='Python', score_brut=0.91, niveau_confiance='forte', statut='central'),
                SkillInfo(label='ML', score_brut=0.72, niveau_confiance='moyenne', statut='secondaire'),
            ],
            low_confidence_skills=[
                SkillInfo(label='RAG', score_brut=0.42, niveau_confiance='faible', statut='a_verifier'),
            ],
            rejected_skills=[
                SkillInfo(label='NLP', score_brut=0.12, niveau_confiance='rejetee', statut='rejete'),
            ],
        )
        d = result.to_dict()
        assert len(d['detected_skills']) == 2
        assert len(d['low_confidence_skills']) == 1
        assert len(d['rejected_skills']) == 1

    def test_to_json(self):
        result = AnalysisResult()
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed['summary'] == {}

    def test_model_metadata(self):
        from models.analysis_result import ModelMetadata
        meta = ModelMetadata(
            binary_model='test', multilabel_model='test',
            binary_checkpoint='/path', multilabel_checkpoint='/path',
            device='cpu', max_length=512, num_labels=18,
            labels=['Python', 'ML'], thresholds={'multilabel': 0.35},
            inference_time_ms=42.0,
        )
        d = asdict(meta)
        assert d['device'] == 'cpu'
        assert d['num_labels'] == 18


class TestTerritorialMarket:
    def test_empty_market(self):
        tm = TerritorialMarketInfo()
        assert tm.offer_count == 0
        assert tm.statistical_robustness == 'faible'

    def test_alert_low_offers(self):
        tm = TerritorialMarketInfo(offer_count=2)
        assert tm.offer_count < 5
        tm.alert = "Nombre d'offres trop faible"

    def test_statistical_robustness(self):
        tm = TerritorialMarketInfo(offer_count=30, statistical_robustness='forte')
        assert tm.statistical_robustness == 'forte'


class TestQualityInfo:
    def test_non_discriminating_scores(self):
        q = QualityInfo(
            model_loaded=True,
            skills_discriminating=False,
            score_min=0.47, score_max=0.53,
            score_mean=0.50, score_std=0.02,
            warnings=['Scores non discriminants'],
        )
        assert q.skills_discriminating is False
        assert len(q.warnings) == 1
        assert 'Scores non discriminants' in q.warnings[0]

    def test_discriminating_scores(self):
        q = QualityInfo(
            model_loaded=True,
            skills_discriminating=True,
            score_min=0.10, score_max=0.95,
            score_mean=0.55, score_std=0.25,
        )
        assert q.skills_discriminating is True
        assert len(q.warnings) == 0


class TestRecommendations:
    def test_recommendation_creation(self):
        rec = Recommendation(
            type='competence_a_ajouter',
            skill='Python',
            justification='Test justification',
            impact_estime='eleve',
            offer_count=10,
            offer_percent=25.0,
            priorite='haute',
            niveau_confiance='forte',
        )
        assert rec.type == 'competence_a_ajouter'
        assert rec.priorite == 'haute'

    def test_empty_recommendation(self):
        rec = Recommendation()
        assert rec.type == ''
        assert rec.skill == ''


class TestUncertainClassification:
    def test_near_equal_scores(self):
        state = THRESHOLDS.get_classification_state(0.51, 0.49)
        assert state['state'] == 'incertaine'

    def test_equal_scores(self):
        state = THRESHOLDS.get_classification_state(0.50, 0.50)
        assert state['state'] == 'incertaine'

    def test_low_equal_scores_insuffisant(self):
        state = THRESHOLDS.get_classification_state(0.30, 0.30)
        assert state['state'] == 'insuffisant'

    def test_wide_gap(self):
        state = THRESHOLDS.get_classification_state(0.90, 0.10)
        assert state['state'] == 'fiable'


class TestSkillsBelowThreshold:
    def test_below_threshold_rejected(self):
        pred = 0.15
        level = THRESHOLDS.get_confidence_level(pred)
        assert level == 'rejetée'

    def test_low_confidence_skill(self):
        pred = 0.45
        level = THRESHOLDS.get_confidence_level(pred)
        assert level == 'faible'


class TestEdgeCases:
    def test_empty_text_raises(self):
        from inference.deepforma_predictor import DeepformaPredictor
        import torch
        pred = DeepformaPredictor.__new__(DeepformaPredictor)
        pred.device = torch.device('cpu')
        pred.max_length = 512
        with pytest.raises(ValueError, match='vide'):
            pred.predict_binary('')

    def test_missing_market(self):
        tm = TerritorialMarketInfo()
        assert tm.offer_count == 0


class TestModelMetadataExport:
    def test_full_export_structure(self):
        result = AnalysisResult()
        d = result.to_dict()
        required_keys = [
            'summary', 'classification', 'detected_skills', 'implicit_skills',
            'rejected_skills', 'low_confidence_skills', 'territorial_market',
            'formation_market_comparison', 'comparison_categories', 'global_score',
            'missing_skills', 'recommendations', 'quality', 'model_metadata',
        ]
        for key in required_keys:
            assert key in d, f'Clef manquante: {key}'


class TestWeightConfig:
    def test_sub_score_retrieval(self):
        sub = SCORING_WEIGHTS.get_sub_score('couverture_competences')
        assert sub is not None
        assert sub.weight == 0.25

    def test_nonexistent_sub_score(self):
        sub = SCORING_WEIGHTS.get_sub_score('inexistant')
        assert sub is None


class TestThresholdDefaults:
    def test_defaults(self):
        t = ClassificationThresholds()
        assert t.high_confidence == 0.80
        assert t.medium_confidence == 0.60
        assert t.low_confidence == 0.40
        assert t.rejection == 0.20
        assert t.uncertainty_margin == 0.10
        assert t.min_offers_for_conclusion == 5
        assert t.statistical_robustness_min == 20
