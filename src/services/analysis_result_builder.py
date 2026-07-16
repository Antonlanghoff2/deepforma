from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.text import clean_text
from config.thresholds import THRESHOLDS
from config.weights import SCORING_WEIGHTS
from models.analysis_result import (
    AnalysisResult,
    CheckpointAuditInfo,
    ClassificationInfo,
    IAClassificationInfo,
    MarketComparisonItem,
    MarketSkillInfo,
    ModelMetadata,
    QualityInfo,
    Recommendation,
    SkillExtractionInfo,
    SkillInfo,
    TerritorialMarketInfo,
)
from data_sources.ia_recommendations import load_ia_recommendations_csv
from domain.ia_recommendation_matching import match_ia_recommendations
from domain.models import IARecommendationMatch
from ai_recommendations.matcher import match_ai_recommendations
from pathlib import Path
from services.skill_normalization import normalize_skill_label


def build_analysis_result(
    analysis: dict[str, Any],
    normalized_offers: list[dict[str, Any]],
    recommendation: Any,
    territorial_stats: Any,
    departement: str,
    threshold: float,
    skill_extraction: SkillExtractionInfo | None = None,
    ia_recommendation_records: list[dict[str, Any]] | None = None,
) -> AnalysisResult:
    result = AnalysisResult()

    binary = analysis['binary']
    skills_result = analysis['skills']
    predictions = skills_result.get('predictions', [])

    class_state = THRESHOLDS.get_classification_state(
        binary['probability_ia'], binary['probability_non_ia']
    )
    result.classification = ClassificationInfo(
        is_ia=binary['is_ia'],
        predicted_class=binary['predicted_class'],
        probability_ia=binary['probability_ia'],
        probability_non_ia=binary['probability_non_ia'],
        state=class_state['state'],
        state_description=class_state['description'],
        gap=class_state['gap'],
    )

    result.ia_classification = _check_ia_classifier_quality(skills_result)

    result.skill_extraction = skill_extraction or _build_skill_extraction('')

    extracted_labels_normalized = set(
        normalize_skill_label(s.normalized_label)
        for s in result.skill_extraction.skills
    )
    extracted_tools_normalized = set(
        normalize_skill_label(s.normalized_label)
        for s in result.skill_extraction.tools
    )

    skill_extraction_ok = result.skill_extraction.status in {'success', 'partial'}
    has_extracted_skills = len(result.skill_extraction.skills) > 0 or len(result.skill_extraction.tools) > 0

    ia_detected_skills: list[SkillInfo] = []
    ia_low_confidence_skills: list[SkillInfo] = []
    ia_rejected_skills: list[SkillInfo] = []
    ia_indeterminate_skills: list[SkillInfo] = []

    discriminating = result.ia_classification.discriminating

    for p in predictions:
        prob = p['probability']
        label = p['label']
        confidence = _skill_confidence(prob)
        skill = SkillInfo(
            label=label,
            score_brut=round(prob, 4),
            niveau_confiance=confidence,
            seuil_applique=threshold,
            methode_detection='camembert_multilabel',
        )
        if not discriminating:
            skill.presence = 'indeterminate'
            skill.statut = 'indetermine'
            ia_indeterminate_skills.append(skill)
        elif prob >= threshold and confidence in ('forte', 'moyenne'):
            skill.presence = 'present'
            skill.statut = 'central' if prob >= 0.70 else 'secondaire'
            ia_detected_skills.append(skill)
        elif prob >= threshold * 0.5:
            skill.presence = 'indeterminate'
            skill.statut = 'a_verifier'
            ia_low_confidence_skills.append(skill)
        else:
            skill.presence = 'absent'
            skill.statut = 'rejete'
            ia_rejected_skills.append(skill)

    ia_detected_skills.sort(key=lambda s: s.score_brut, reverse=True)
    ia_low_confidence_skills.sort(key=lambda s: s.score_brut, reverse=True)
    ia_indeterminate_skills.sort(key=lambda s: s.score_brut, reverse=True)

    result.detected_skills = ia_detected_skills
    result.low_confidence_skills = ia_low_confidence_skills
    result.rejected_skills = ia_rejected_skills
    result.indeterminate_skills = ia_indeterminate_skills

    if not skill_extraction_ok:
        result.formation_analysis_status = 'unreliable'
        result.skills_presence = 'indeterminate'
        result.comparison_available = False
        result.recommendations_available = False
        result.blocking_reasons = ['skill_extraction_failed']
    elif not has_extracted_skills:
        result.formation_analysis_status = 'no_skills_detected'
        result.skills_presence = 'indeterminate'
        result.comparison_available = False
        result.recommendations_available = False
        result.blocking_reasons = ['no_skills_detected']
    else:
        result.formation_analysis_status = 'reliable'
        result.skills_presence = 'determinate'
        result.comparison_available = True
        result.recommendations_available = True
        result.blocking_reasons = []

    binary_model_checkpoint = str(getattr(analysis, 'binary_model_dir', 'models/binary_ia_v2/final'))
    multilabel_model_checkpoint = str(getattr(analysis, 'multilabel_model_dir', 'models/multilabel_competences_v2/final'))
    checkpoint_audit_raw = analysis.get('checkpoint_audit', {})

    result.checkpoint_audit = CheckpointAuditInfo(
        config_present=checkpoint_audit_raw.get('config_present', False),
        weights_present=checkpoint_audit_raw.get('weights_present', False),
        weights_size_bytes=checkpoint_audit_raw.get('weights_size_bytes', 0),
        architecture_declared=checkpoint_audit_raw.get('architecture_declared', ''),
        num_labels_declared=checkpoint_audit_raw.get('num_labels_declared', 0),
        num_labels_effective=checkpoint_audit_raw.get('num_labels_effective', 0),
        problem_type=checkpoint_audit_raw.get('problem_type', ''),
        id2label_count=checkpoint_audit_raw.get('id2label_count', 0),
        label2id_count=checkpoint_audit_raw.get('label2id_count', 0),
        strict_load_success=checkpoint_audit_raw.get('strict_load_success', False),
        missing_keys=checkpoint_audit_raw.get('missing_keys', []),
        unexpected_keys=checkpoint_audit_raw.get('unexpected_keys', []),
        ignored_keys=checkpoint_audit_raw.get('ignored_keys', []),
        appears_random_init=checkpoint_audit_raw.get('appears_random_init', True),
        body_params_match_base=checkpoint_audit_raw.get('body_params_match_base', True),
        parameter_errors=checkpoint_audit_raw.get('parameter_errors', []),
        classifier_params=checkpoint_audit_raw.get('classifier_params', {}),
    )

    from inference.deepforma_predictor import DEFAULT_TAXONOMY_PATH
    _taxonomy_version = ''
    _tax_path = Path(DEFAULT_TAXONOMY_PATH)
    if _tax_path.exists():
        try:
            _tax = json.loads(_tax_path.read_text(encoding='utf-8'))
            _taxonomy_version = _tax.get('version', '')
        except Exception:
            pass
    _model_name = 'Classifieur IA'
    _num_labels = len(skills_result.get('predictions', []))
    _validation_status = 'non validé'
    if result.checkpoint_audit.appears_random_init:
        _validation_status = 'non entraîné'
    elif result.checkpoint_audit.strict_load_success:
        _validation_status = 'entraîné (non validé)'
    if _taxonomy_version:
        _model_name = f'Classifieur IA v{_taxonomy_version}'

    result.model_metadata = ModelMetadata(
        binary_model='CamemBERT (CamembertForSequenceClassification)',
        multilabel_model='CamemBERT (CamembertForSequenceClassification)',
        model_name=_model_name,
        taxonomy_version=_taxonomy_version,
        validation_status=_validation_status,
        binary_checkpoint=binary_model_checkpoint,
        multilabel_checkpoint=multilabel_model_checkpoint,
        device=analysis.get('device', 'cpu'),
        max_length=512,
        num_labels=_num_labels,
        labels=[p['label'] for p in predictions] if predictions else [],
        thresholds={'multilabel': threshold, 'binary': None},
        inference_time_ms=analysis.get('inference_time_ms', 0.0),
        classifier_weight_stats={
            'appears_random_init': result.checkpoint_audit.appears_random_init,
            'out_proj': result.checkpoint_audit.classifier_params.get('classifier.out_proj.weight', {}),
            'dense': result.checkpoint_audit.classifier_params.get('classifier.dense.weight', {}),
        },
    )

    result.quality = QualityInfo(
        model_loaded=True,
        skills_discriminating=discriminating,
        score_min=result.ia_classification.score_min,
        score_max=result.ia_classification.score_max,
        score_mean=result.ia_classification.score_mean,
        score_std=result.ia_classification.score_std,
        offers_sufficient=len(normalized_offers) >= THRESHOLDS.min_offers_for_conclusion,
        warnings=result.ia_classification.warnings,
    )

    if territorial_stats:
        market_skills_sorted = sorted(
            territorial_stats.skill_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_skills = [
            MarketSkillInfo(
                label=label,
                offer_count=count,
                share_percent=round(
                    count / territorial_stats.offer_count * 100, 2
                ) if territorial_stats.offer_count else 0.0,
            )
            for label, count in market_skills_sorted[:20]
        ]
        robust = 'forte' if territorial_stats.offer_count >= THRESHOLDS.statistical_robustness_min else (
            'moyenne' if territorial_stats.offer_count >= THRESHOLDS.min_offers_for_conclusion else 'faible'
        )
        alert = ''
        if territorial_stats.offer_count < THRESHOLDS.min_offers_for_conclusion:
            alert = (
                f"Nombre d'offres trop faible ({territorial_stats.offer_count}) "
                'pour une analyse territoriale fiable.'
            )
        elif territorial_stats.offer_count < THRESHOLDS.statistical_robustness_min:
            alert = (
                f"Volume d'offres modere ({territorial_stats.offer_count}). "
                'Les tendances restent indicatives.'
            )
        result.territorial_market = TerritorialMarketInfo(
            territory=departement,
            period='Derniers mois (source: France Travail)',
            offer_count=territorial_stats.offer_count,
            exploitable_offers=len(normalized_offers),
            top_skills=top_skills,
            contract_types=getattr(territorial_stats, 'contract_types', {}),
            statistical_robustness=robust,
            alert=alert,
        )

    if recommendation and result.comparison_available:
        formation_labels = extracted_labels_normalized | extracted_tools_normalized
        market_lookup = {}
        for ms in recommendation.market_skills:
            market_lookup[normalize_skill_label(ms.label)] = ms

        comparison_items: list[MarketComparisonItem] = []
        all_compared_labels = set()

        for skill_key, ms in market_lookup.items():
            in_formation = skill_key in formation_labels
            detection_conf = 0.0
            for es in result.skill_extraction.skills:
                if normalize_skill_label(es.normalized_label) == skill_key:
                    detection_conf = es.confidence
                    break
            if detection_conf == 0.0:
                for es_t in result.skill_extraction.tools:
                    if normalize_skill_label(es_t.normalized_label) == skill_key:
                        detection_conf = es_t.confidence
                        break
            coverage = 'complete' if in_formation else 'absente'
            priority = 'haute' if ms.offer_count >= 5 else 'moyenne'
            comparison_items.append(MarketComparisonItem(
                skill=ms.label,
                in_formation=in_formation,
                detection_confidence=detection_conf,
                frequency_in_offers=ms.share_percent,
                offer_count=ms.offer_count,
                coverage_level=coverage,
                priority=priority,
            ))
            all_compared_labels.add(skill_key)

        covered = [c for c in comparison_items if c.in_formation]
        overrepresented = [
            c for c in comparison_items
            if c.in_formation and c.frequency_in_offers < 5.0
        ]
        missing = [c for c in comparison_items if not c.in_formation]

        result.formation_market_comparison = comparison_items
        result.comparison_categories = {
            'covered': covered,
            'overrepresented': overrepresented,
            'missing': missing,
        }
        result.missing_skills = [
            MarketSkillInfo(label=c.skill, offer_count=c.offer_count, share_percent=c.frequency_in_offers)
            for c in missing
        ]

        sub_score_values = {}
        if len(result.skill_extraction.skills) > 0 and len(market_lookup) > 0:
            coverage_pct = len(covered) / max(len(market_lookup), 1)
            sub_score_values['couverture_competences'] = coverage_pct * 100
            sub_score_values['pertinence_metier'] = min(100.0, coverage_pct * 120)
            sub_score_values['adequation_territoriale'] = min(100.0, coverage_pct * 100)
            sub_score_values['niveau_experience'] = 50.0
            sub_score_values['employabilite'] = min(100.0, coverage_pct * 150)
            sub_score_values['actualite_programme'] = 50.0
            result.global_score = SCORING_WEIGHTS.compute_global(sub_score_values)

        recommendations: list[Recommendation] = []
        seen_recs: set[str] = set()

        for c in missing[:5]:
            if c.skill not in seen_recs:
                seen_recs.add(c.skill)
                recommendations.append(Recommendation(
                    type='competence_a_ajouter',
                    skill=c.skill,
                    justification=(
                        f'Competence demandee dans {c.offer_count} offres locales '
                        f'({c.frequency_in_offers:.1f}%) mais absente de la formation.'
                    ),
                    impact_estime='eleve' if c.offer_count >= 5 else 'moyen',
                    offer_count=c.offer_count,
                    offer_percent=round(c.frequency_in_offers, 1),
                    priorite='haute' if c.offer_count >= 5 else 'moyenne',
                    niveau_confiance='forte' if c.offer_count >= 10 else 'moyenne',
                ))

        for es in result.skill_extraction.skills:
            skill_key = normalize_skill_label(es.normalized_label)
            if skill_key not in market_lookup and es.confidence >= 0.70:
                if es.normalized_label not in seen_recs:
                    seen_recs.add(es.normalized_label)
                    recommendations.append(Recommendation(
                        type='competence_peu_utile_localement',
                        skill=es.normalized_label,
                        justification=(
                            f"Competence '{es.normalized_label}' bien detectee dans la formation "
                            'mais peu presente dans les offres locales.'
                        ),
                        impact_estime='faible',
                        offer_count=0,
                        offer_percent=0.0,
                        priorite='basse',
                        niveau_confiance='moyenne',
                    ))

        if len(overrepresented) > 0:
            for c in overrepresented[:3]:
                if c.skill not in seen_recs:
                    seen_recs.add(c.skill)
                    recommendations.append(Recommendation(
                        type='contenu_surrepresente',
                        skill=c.skill,
                        justification=(
                            f"Competence '{c.skill}' presente dans la formation "
                            f"mais faiblement demandee localement ({c.frequency_in_offers:.1f}% des offres)."
                        ),
                        impact_estime='moyen',
                        offer_count=c.offer_count,
                        offer_percent=round(c.frequency_in_offers, 1),
                        priorite='moyenne',
                        niveau_confiance='moyenne',
                    ))

        priorities = {'haute': 0, 'moyenne': 1, 'basse': 2}
        recommendations.sort(key=lambda r: (priorities.get(r.priorite, 99), -r.offer_count))
        result.recommendations = recommendations

    result.summary = {
        'formation_analysis_status': result.formation_analysis_status,
        'skill_extraction_status': result.skill_extraction.status,
        'total_skills_extracted': len(result.skill_extraction.skills),
        'total_tools_detected': len(result.skill_extraction.tools),
        'total_ia_categories': len(result.ia_classification.categories),
        'ia_classification_status': result.ia_classification.status,
        'ia_classification_discriminating': result.ia_classification.discriminating,
        'total_skills_detected': len(ia_detected_skills),
        'total_skills_low_confidence': len(ia_low_confidence_skills),
        'total_skills_indeterminate': len(ia_indeterminate_skills),
        'total_skills_rejected': len(ia_rejected_skills),
        'total_offers_analyzed': len(normalized_offers),
        'classification_state': class_state['state'],
        'global_score': result.global_score.get('global_score') if result.global_score else None,
        'inference_time_ms': analysis.get('inference_time_ms', 0.0),
        'analyzed_at': datetime.now(timezone.utc).isoformat(),
    }

    ia_matches: list[IARecommendationMatch] = []
    if ia_recommendation_records is not None:
        skill_dicts: list[dict[str, str]] = []
        seen: set[str] = set()
        for s in skill_extraction.skills:
            key = normalize_skill_label(s.normalized_label or s.source_label)
            if key and key not in seen:
                seen.add(key)
                skill_dicts.append({"name": s.normalized_label or s.source_label, "normalized_name": key})
        for s in skill_extraction.tools:
            key = normalize_skill_label(s.normalized_label or s.source_label)
            if key and key not in seen:
                seen.add(key)
                skill_dicts.append({"name": s.normalized_label or s.source_label, "normalized_name": key})
        for s in ia_detected_skills:
            key = normalize_skill_label(s.label)
            if key and key not in seen:
                seen.add(key)
                skill_dicts.append({"name": s.label, "normalized_name": key})
        if skill_dicts:
            ia_matches = match_ia_recommendations(skill_dicts, ia_recommendation_records)
    result.ia_recommendations = ia_matches

    ai_hybrid = analysis.get('ai_recommendations_hybrid')
    if not ai_hybrid:
        ai_hybrid = match_ai_recommendations(
            referential_title=clean_text(str(analysis.get('document_title') or analysis.get('title') or '')),
            activities=[str(item) for item in (analysis.get('activities') or []) if clean_text(item)],
            official_skills=[s.label for s in result.detected_skills if clean_text(s.label)],
            subskills=[s.label for s in result.skill_extraction.skills if clean_text(s.label)],
            full_text=clean_text(str(analysis.get('full_text') or analysis.get('text') or '')),
            model_score_std=result.ia_classification.score_std,
            model_mean_score=result.ia_classification.score_mean,
            model_non_discriminant=not result.quality.skills_discriminating,
        )
    if isinstance(ai_hybrid, dict):
        category_labels = {item.get('label') for item in result.ia_classification.categories if isinstance(item, dict)}
        for category in ai_hybrid.get('detected_categories') or []:
            if not isinstance(category, dict):
                continue
            label = clean_text(category.get('label') or '')
            if label and label not in category_labels:
                result.ia_classification.categories.append(category)
                category_labels.add(label)
        for rec in ai_hybrid.get('recommendations') or []:
            if not isinstance(rec, dict):
                continue
            result.recommendations.append(Recommendation(
                type='ia_recommendation',
                skill=clean_text(rec.get('keyword') or ''),
                justification=clean_text(rec.get('recommendation') or ''),
                impact_estime='moyen',
                offer_count=0,
                offer_percent=0.0,
                priorite='moyenne',
                niveau_confiance=clean_text(rec.get('status') or 'à vérifier'),
            ))
        result.summary['ai_default_recommendation_applied'] = bool(ai_hybrid.get('default_recommendation_applied'))

    return result


def _build_skill_extraction(text: str) -> SkillExtractionInfo:
    from models.analysis_result import OpenExtractedSkill

    return SkillExtractionInfo(
        status='failed' if not text else 'partial',
        skills=[],
        tools=[],
        knowledge_items=[],
        warnings=[],
    )


def _skill_confidence(score: float) -> str:
    return THRESHOLDS.get_confidence_level(score)


def _check_ia_classifier_quality(skills_result: dict[str, Any]) -> IAClassificationInfo:
    score_std = skills_result.get('score_std', 0.0)
    score_max = skills_result.get('score_max', 0.0)
    score_mean = skills_result.get('score_mean', 0.0)
    score_min = skills_result.get('score_min', 0.0)
    discriminating = score_std > 0.05 or score_max > 0.70
    warnings: list[str] = []
    if not discriminating:
        warnings.append(
            'Le modele specialise dans les 18 categories IA ne produit pas de scores '
            'suffisamment discriminants (ecart-type={:.4f}, max={:.4f}). '
            'Ses resultats sont desactives. '
            'Cette anomalie n empeche pas l extraction directe des competences depuis le texte.'.format(
                score_std, score_max
            )
        )
    if score_max < 0.50:
        warnings.append('Aucune categorie IA ne depasse 50%% de probabilite.')
    if score_min > 0.40 and score_max < 0.60:
        warnings.append(
            'Tous les scores sont compris entre {:.2f} et {:.2f}.'.format(score_min, score_max)
        )
    predictions = skills_result.get('predictions', [])
    categories = [
        {'label': p['label'], 'probability': p['probability']}
        for p in predictions if p['probability'] >= 0.35
    ]
    families = skills_result.get('family_groups', [])
    status = 'success' if discriminating and categories else (
        'unreliable' if not discriminating else 'unavailable'
    )
    return IAClassificationInfo(
        status=status,
        categories=categories,
        families=families,
        scores=[p['probability'] for p in predictions],
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_std=score_std,
        discriminating=discriminating,
        warnings=warnings,
        threshold_applied=0.35,
    )
