# Rapport d'evaluation du classifieur multilabel IA

- **Fichier de test** : data/processed/ia_multilabel_test.jsonl
- **Echantillons** : 117
- **Labels** : 20

## Metriques globales

| Metrique | Valeur |
|----------|--------|
<<<<<<< HEAD
| Micro-F1 | 0.5784 |
| Macro-F1 | 0.5699 |
| Weighted-F1 | 0.6021 |
| Precision micro | 0.51 |
| Precision macro | 0.5663 |
| Rappel micro | 0.6679 |
| Rappel macro | 0.7012 |
| Average precision micro | 0.5622 |
| Average precision macro | 0.6926 |
=======
| Micro-F1 | 0.5764 |
| Macro-F1 | 0.5782 |
| Weighted-F1 | 0.6114 |
| Precision micro | 0.4986 |
| Precision macro | 0.5994 |
| Rappel micro | 0.6828 |
| Rappel macro | 0.698 |
| Average precision micro | 0.5861 |
| Average precision macro | 0.6958 |
>>>>>>> 6aa820937574aeaf7afddbc71fe87107ec5063ee

## Distribution des probabilites

| | Positive | Negative |
|----------|----------|----------|
<<<<<<< HEAD
| Mean | 0.5812138915061951 | 0.4499942362308502 |
| Std | 0.06837496906518936 | 0.08073445409536362 |
| Min | 0.35599851608276367 | 0.32481351494789124 |
| Max | 0.6845471262931824 | 0.6643111705780029 |

## Signal d'entrainement

- Ecart-type moyen des probabilites : 0.0817
=======
| Mean | 0.5792834162712097 | 0.4486272931098938 |
| Std | 0.0666670948266983 | 0.07811641693115234 |
| Min | 0.34217190742492676 | 0.3197963535785675 |
| Max | 0.6742014288902283 | 0.661303699016571 |

## Signal d'entrainement

- Ecart-type moyen des probabilites : 0.0788
>>>>>>> 6aa820937574aeaf7afddbc71fe87107ec5063ee
- Signal faible detecte : NON

## Statistiques de prediction

<<<<<<< HEAD
- Predictions vides : 11 (9.4%)
=======
- Predictions vides : 10 (8.55%)
>>>>>>> 6aa820937574aeaf7afddbc71fe87107ec5063ee
- Tous les labels predits : 0 (0.0%)

## Metriques par label

| Label | Support | F1 | Precision | Rappel | ROC-AUC | Seuil | Predits |
|-------|---------|----|-----------|--------|---------|-------|---------|
<<<<<<< HEAD
| Automatisation | 18 | 0.4348 | 1.0 | 0.2778 | 0.8081 | 0.52 | 5 |
| Big Data | 15 | 0.6087 | 0.4516 | 0.9333 | 0.9248 | 0.6 | 31 |
| Computer Vision | 2 | 0.8 | 0.6667 | 1.0 | 1.0 | 0.61 | 3 |
| Data Engineering | 9 | 0.3051 | 0.18 | 1.0 | 0.8477 | 0.5 | 50 |
| Data Science | 23 | 0.5424 | 0.4444 | 0.6957 | 0.8173 | 0.61 | 36 |
| Deep Learning | 8 | 0.2 | 0.5 | 0.125 | 0.7397 | 0.63 | 2 |
| Ethique IA & RGPD | 31 | 0.6667 | 0.9412 | 0.5161 | 0.8777 | 0.6 | 17 |
| Gestion de projet IA | 23 | 0.6667 | 0.64 | 0.6957 | 0.8682 | 0.59 | 25 |
| IA Generative | 51 | 0.8043 | 0.9024 | 0.7255 | 0.932 | 0.59 | 41 |
| LangChain / Agents RAG | 6 | 0.0 | 0.0 | 0.0 | 0.7973 | 0.53 | 1 |
| Machine Learning | 17 | 0.3667 | 0.2558 | 0.6471 | 0.7124 | 0.58 | 43 |
| MLOps / Deploiement | 6 | 0.7273 | 0.8 | 0.6667 | 0.964 | 0.52 | 5 |
| NLP / Traitement du langage | 2 | 0.4 | 0.3333 | 0.5 | 0.6826 | 0.56 | 3 |
| No-code / Low-code | 4 | 0.7273 | 0.5714 | 1.0 | 0.9934 | 0.53 | 7 |
| Prompt Engineering | 6 | 0.7692 | 0.7143 | 0.8333 | 0.9865 | 0.58 | 7 |
| Python | 17 | 0.7429 | 0.7222 | 0.7647 | 0.9688 | 0.64 | 18 |
| Reinforcement Learning | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.59 | 1 |
| Series temporelles | 1 | 0.5 | 0.3333 | 1.0 | 1.0 | 0.57 | 3 |
| SQL / Data Engineering | 14 | 0.6 | 0.4615 | 0.8571 | 0.9071 | 0.56 | 26 |
| Visualisation | 14 | 0.5366 | 0.4074 | 0.7857 | 0.9015 | 0.56 | 27 |

## Analyse d'erreurs (81 echantillons)
=======
| Automatisation | 18 | 0.5 | 1.0 | 0.3333 | 0.8272 | 0.52 | 6 |
| Big Data | 15 | 0.549 | 0.3889 | 0.9333 | 0.9229 | 0.58 | 36 |
| Computer Vision | 2 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.59 | 4 |
| Data Engineering | 9 | 0.3051 | 0.18 | 1.0 | 0.8344 | 0.5 | 50 |
| Data Science | 23 | 0.5625 | 0.439 | 0.7826 | 0.8228 | 0.59 | 41 |
| Deep Learning | 8 | 0.2 | 0.5 | 0.125 | 0.7466 | 0.63 | 2 |
| Ethique IA & RGPD | 31 | 0.6939 | 0.9444 | 0.5484 | 0.871 | 0.59 | 18 |
| Gestion de projet IA | 23 | 0.7347 | 0.6923 | 0.7826 | 0.8844 | 0.59 | 26 |
| IA Generative | 51 | 0.8046 | 0.9722 | 0.6863 | 0.9376 | 0.6 | 36 |
| LangChain / Agents RAG | 6 | 0.1538 | 0.1429 | 0.1667 | 0.8048 | 0.51 | 7 |
| Machine Learning | 17 | 0.3667 | 0.2558 | 0.6471 | 0.7259 | 0.56 | 43 |
| MLOps / Deploiement | 6 | 0.4444 | 0.3333 | 0.6667 | 0.9264 | 0.5 | 12 |
| NLP / Traitement du langage | 2 | 0.6667 | 1.0 | 0.5 | 0.6957 | 0.56 | 1 |
| No-code / Low-code | 4 | 0.6667 | 1.0 | 0.5 | 0.9912 | 0.53 | 2 |
| Prompt Engineering | 6 | 0.6667 | 0.5556 | 0.8333 | 0.9925 | 0.57 | 9 |
| Python | 17 | 0.75 | 0.6522 | 0.8824 | 0.9706 | 0.61 | 23 |
| Reinforcement Learning | 1 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.56 | 2 |
| Series temporelles | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.58 | 1 |
| SQL / Data Engineering | 14 | 0.6286 | 0.5238 | 0.7857 | 0.9036 | 0.58 | 21 |
| Visualisation | 14 | 0.5366 | 0.4074 | 0.7857 | 0.9161 | 0.55 | 27 |

## Analyse d'erreurs (77 echantillons)
>>>>>>> 6aa820937574aeaf7afddbc71fe87107ec5063ee

Voir le fichier CSV `ia_classifier_errors.csv` pour le detail.