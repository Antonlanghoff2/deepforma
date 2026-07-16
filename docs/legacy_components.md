# Composants legacy et incohérences

Ce document liste les éléments présents dans le dépôt mais qui ne doivent pas être confondus avec la chaîne d'inférence principale actuelle.

## Checkpoints legacy ou non utilisés directement

- [`models/ia-classifier-v2/final`](../models/ia-classifier-v2/final) et [`models/ia-classifier-v2/best`](../models/ia-classifier-v2/best): présents, évaluables, mais non utilisés par [`src/inference/deepforma_predictor.py`](../src/inference/deepforma_predictor.py).
- [`models/multilabel_v2/final`](../models/multilabel_v2/final): ancienne version multilabel référentiel.
- [`models/multilabel_competences_v2/final.untrained`](../models/multilabel_competences_v2/final.untrained): snapshot d'audit.
- [`models/cpf-recommender/checkpoints/`](../models/cpf-recommender/checkpoints/): snapshots d'entraînement intermédiaires.
- [`models/skill-extractor/checkpoint-13`](../models/skill-extractor/checkpoint-13) et [`models/skill-extractor/checkpoint-65`](../models/skill-extractor/checkpoint-65): versions intermédiaires de continual learning.

## Chemins déclarés mais absents sur disque

Les chemins suivants sont référencés par le code mais ne sont pas présents dans l'arborescence observée de `models/`:

- `models/referential-section-classifier/current` via [`src/referential_learning/pipeline.py`](../src/referential_learning/pipeline.py)
- `models/referential-skill-ner/current` via [`src/referential_learning/pipeline.py`](../src/referential_learning/pipeline.py)
- `models/referential-multilabel/current` via [`src/referential_learning/pipeline.py`](../src/referential_learning/pipeline.py)

Le pipeline référentiel retombe donc sur les heuristiques lorsqu'aucun modèle n'est déployé.

## Composants hybrides ou purement règles

- [`src/referential_import/title_extractor.py`](../src/referential_import/title_extractor.py): extracteur structuré d'intitulé fondé sur la mise en page, les labels explicites et le RNCP.
- [`src/ai_recommendations/loader.py`](../src/ai_recommendations/loader.py): import et nettoyage du CSV de règles IA.
- [`src/ai_recommendations/matcher.py`](../src/ai_recommendations/matcher.py): correspondance exacte, lexicale, sémantique et fallback par défaut.
- [`src/ai_recommendations/category_mapper.py`](../src/ai_recommendations/category_mapper.py): mapping vers la taxonomie IA.
- [`src/ai_recommendations/fusion.py`](../src/ai_recommendations/fusion.py): fusion des sources de score et neutralisation du modèle non discriminant.
- [`src/deepforma/cpf/embeddings.py`](../src/deepforma/cpf/embeddings.py): index vectoriel NumPy / FAISS optionnel.
- [`src/referentials/rncp_rome_mapper.py`](../src/referentials/rncp_rome_mapper.py): score RNCP/ROME sans modèle entraîné.
- [`src/referentials/offer_skill_enricher.py`](../src/referentials/offer_skill_enricher.py): enrichissement d'offres basé sur mappings et référentiel unifié.

## Incohérences constatées

- Le script [`scripts/evaluate_ia_multilabel_classifier.py`](../scripts/evaluate_ia_multilabel_classifier.py) cible `models/ia-classifier-v2/final`, alors que l'inférence applicative par défaut utilise `models/multilabel_competences_v2/final`.
- Le script [`scripts/train_ia_multilabel_classifier.py`](../scripts/train_ia_multilabel_classifier.py) exporte aussi vers `models/ia-classifier-v2`, ce qui entretient deux chemins pour le même concept.
- Le classifieur binaire n'expose pas de `id2label` dans son `config.json`; la convention positive = indice 1 est codée dans [`src/inference/deepforma_predictor.py`](../src/inference/deepforma_predictor.py).
- Les scripts de déploiement référentiels s'attendent à des modèles `current` qui ne sont pas versionnés dans `models/`.
- Plusieurs scripts historiques et notebooks partagent des logiques de normalisation de labels: [`scripts/prepare_ia_training_dataset.py`](../scripts/prepare_ia_training_dataset.py), [`src/services/skill_normalization.py`](../src/services/skill_normalization.py), [`src/deepforma/skills/normalizer.py`](../src/deepforma/skills/normalizer.py).
- Les modèles référentiels ML sont optionnels et ne doivent pas être présentés comme la voie d'inférence principale si les répertoires `current` manquent.

## Recommandations de consolidation prioritaires

1. Unifier le chemin de vérité pour la famille multilabel IA: conserver un seul checkpoint actif et convertir l'autre en archive explicitement legacy.
2. Documenter la convention binaire `0 -> non-IA`, `1 -> IA` au même endroit que l'entraînement et l'inférence.
3. Versionner ou supprimer les chemins `current` non déployés pour les modèles référentiels.
4. Centraliser la normalisation des labels et des seuils dans un seul module partagé.
5. Séparer clairement dans l'UI les artefacts d'évaluation, les checkpoints actifs et les snapshots historiques.
