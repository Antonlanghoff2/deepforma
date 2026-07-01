# Rapport d'evaluation du classifieur multilabel IA

- **Fichier de test** : data/processed/ia_multilabel_test.jsonl
- **Echantillons** : 117
- **Labels** : 20

## Metriques globales

| Metrique | Valeur |
|----------|--------|
| Micro-F1 | 0.5634 |
| Macro-F1 | 0.563 |
| Weighted-F1 | 0.6086 |
| Precision micro | 0.4852 |
| Precision macro | 0.5771 |
| Rappel micro | 0.6716 |
| Rappel macro | 0.704 |
| Average precision micro | 0.5776 |
| Average precision macro | 0.6902 |

## Distribution des probabilites

| | Positive | Negative |
|----------|----------|----------|
| Mean | 0.5772393345832825 | 0.4489430785179138 |
| Std | 0.06542038917541504 | 0.07735034078359604 |
| Min | 0.355716735124588 | 0.3226977586746216 |
| Max | 0.672683596611023 | 0.6583055257797241 |

## Signal d'entrainement

- Ecart-type moyen des probabilites : 0.0778
- Signal faible detecte : NON

## Statistiques de prediction

- Predictions vides : 15 (12.82%)
- Tous les labels predits : 0 (0.0%)

## Metriques par label

| Label | Support | F1 | Precision | Rappel | ROC-AUC | Seuil | Predits |
|-------|---------|----|-----------|--------|---------|-------|---------|
| Automatisation | 18 | 0.5 | 1.0 | 0.3333 | 0.8558 | 0.52 | 6 |
| Big Data | 15 | 0.549 | 0.3889 | 0.9333 | 0.9281 | 0.59 | 36 |
| Computer Vision | 2 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.59 | 4 |
| Data Engineering | 9 | 0.3051 | 0.18 | 1.0 | 0.8416 | 0.5 | 50 |
| Data Science | 23 | 0.5625 | 0.439 | 0.7826 | 0.8256 | 0.59 | 41 |
| Deep Learning | 8 | 0.2 | 0.5 | 0.125 | 0.7454 | 0.63 | 2 |
| Ethique IA & RGPD | 31 | 0.6939 | 0.9444 | 0.5484 | 0.8732 | 0.59 | 18 |
| Gestion de projet IA | 23 | 0.7895 | 1.0 | 0.6522 | 0.8747 | 0.6 | 15 |
| IA Generative | 51 | 0.7711 | 1.0 | 0.6275 | 0.9352 | 0.6 | 32 |
| LangChain / Agents RAG | 6 | 0.1176 | 0.0909 | 0.1667 | 0.7958 | 0.5 | 11 |
| Machine Learning | 17 | 0.3667 | 0.2558 | 0.6471 | 0.7288 | 0.56 | 43 |
| MLOps / Deploiement | 6 | 0.3571 | 0.2273 | 0.8333 | 0.9174 | 0.49 | 22 |
| NLP / Traitement du langage | 2 | 0.4 | 0.3333 | 0.5 | 0.6435 | 0.55 | 3 |
| No-code / Low-code | 4 | 0.6667 | 1.0 | 0.5 | 0.9779 | 0.53 | 2 |
| Prompt Engineering | 6 | 0.6667 | 0.5556 | 0.8333 | 0.991 | 0.57 | 9 |
| Python | 17 | 0.7692 | 0.6818 | 0.8824 | 0.9706 | 0.61 | 22 |
| Reinforcement Learning | 1 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.56 | 2 |
| Series temporelles | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.57 | 1 |
| SQL / Data Engineering | 14 | 0.6667 | 0.5455 | 0.8571 | 0.9071 | 0.57 | 22 |
| Visualisation | 14 | 0.5455 | 0.4 | 0.8571 | 0.9182 | 0.54 | 30 |

## Analyse d'erreurs (74 echantillons)

Voir le fichier CSV `ia_classifier_errors.csv` pour le detail.