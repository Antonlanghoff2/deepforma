# Rapport d'evaluation du classifieur multilabel IA

- **Fichier de test** : data/processed/ia_multilabel_test.jsonl
- **Echantillons** : 117
- **Labels** : 20

## Metriques globales

| Metrique | Valeur |
|----------|--------|
| Micro-F1 | 0.5579 |
| Macro-F1 | 0.542 |
| Weighted-F1 | 0.592 |
| Precision micro | 0.4716 |
| Precision macro | 0.5373 |
| Rappel micro | 0.6828 |
| Rappel macro | 0.7024 |
| Average precision micro | 0.5648 |
| Average precision macro | 0.6819 |

## Distribution des probabilites

| | Positive | Negative |
|----------|----------|----------|
| Mean | 0.5768566131591797 | 0.449182391166687 |
| Std | 0.06634161621332169 | 0.07853875309228897 |
| Min | 0.3523711860179901 | 0.3199882507324219 |
| Max | 0.6716859936714172 | 0.6591944694519043 |

## Signal d'entrainement

- Ecart-type moyen des probabilites : 0.0787
- Signal faible detecte : NON

## Statistiques de prediction

- Predictions vides : 12 (10.26%)
- Tous les labels predits : 0 (0.0%)

## Metriques par label

| Label | Support | F1 | Precision | Rappel | ROC-AUC | Seuil | Predits |
|-------|---------|----|-----------|--------|---------|-------|---------|
| Automatisation | 18 | 0.4348 | 1.0 | 0.2778 | 0.8238 | 0.52 | 5 |
| Big Data | 15 | 0.5283 | 0.3684 | 0.9333 | 0.9275 | 0.59 | 38 |
| Computer Vision | 2 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.59 | 4 |
| Data Engineering | 9 | 0.3051 | 0.18 | 1.0 | 0.8395 | 0.5 | 50 |
| Data Science | 23 | 0.5846 | 0.4524 | 0.8261 | 0.8261 | 0.59 | 42 |
| Deep Learning | 8 | 0.0 | 0.0 | 0.0 | 0.7511 | 0.64 | 0 |
| Ethique IA & RGPD | 31 | 0.6939 | 0.9444 | 0.5484 | 0.8691 | 0.59 | 18 |
| Gestion de projet IA | 23 | 0.7895 | 1.0 | 0.6522 | 0.8719 | 0.6 | 15 |
| IA Generative | 51 | 0.7955 | 0.9459 | 0.6863 | 0.9346 | 0.6 | 37 |
| LangChain / Agents RAG | 6 | 0.1667 | 0.1667 | 0.1667 | 0.7748 | 0.51 | 6 |
| Machine Learning | 17 | 0.3607 | 0.25 | 0.6471 | 0.7259 | 0.56 | 44 |
| MLOps / Deploiement | 6 | 0.3571 | 0.2273 | 0.8333 | 0.9129 | 0.49 | 22 |
| NLP / Traitement du langage | 2 | 0.4 | 0.3333 | 0.5 | 0.5913 | 0.55 | 3 |
| No-code / Low-code | 4 | 0.6667 | 1.0 | 0.5 | 0.958 | 0.53 | 2 |
| Prompt Engineering | 6 | 0.625 | 0.5 | 0.8333 | 0.9775 | 0.57 | 10 |
| Python | 17 | 0.5965 | 0.425 | 1.0 | 0.9688 | 0.55 | 40 |
| Reinforcement Learning | 1 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.56 | 2 |
| Series temporelles | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.58 | 1 |
| SQL / Data Engineering | 14 | 0.6667 | 0.5455 | 0.8571 | 0.9092 | 0.57 | 22 |
| Visualisation | 14 | 0.5366 | 0.4074 | 0.7857 | 0.9098 | 0.55 | 27 |

## Analyse d'erreurs (75 echantillons)

Voir le fichier CSV `ia_classifier_errors.csv` pour le detail.