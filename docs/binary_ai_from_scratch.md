# Binary AI from Scratch

Cette expérimentation ajoute deux modèles entraînés uniquement sur les données DeepForma pour la classification binaire `IA` / `non-IA`.

## Définition de "from scratch"

- Aucun modèle préentraîné n'est chargé.
- Aucun embedding préentraîné n'est utilisé.
- Le vocabulaire, les vecteurs et tous les poids sont appris à partir du dataset DeepForma.
- Le backend existant reste disponible et n'est pas remplacé automatiquement.

## Dataset

Source par défaut:

- `data/processed/dataset_entrainement.csv`

Champs utilisés pour construire le texte:

- `intitule`
- `description`
- `objectifs`
- `programme`
- `competences_ia`
- `competences_ia_suggerees`
- `texte_modele`
- `certification`
- `code_certification`
- `code_rncp`
- `code_rs`

Champ cible:

- `est_lie_ia` -> `is_ai`

Champ groupe:

- `formation_group_id` si disponible, sinon groupe stable dérivé du contenu

## Préparation

Commande:

```bash
make binary-ai-prepare
```

Sorties:

- `data/processed/binary_ai/dataset.parquet`
- `data/training/binary_ai/train.parquet`
- `data/training/binary_ai/validation.parquet`
- `data/training/binary_ai/test.parquet`
- `data/training/binary_ai/split_manifest.json`
- `reports/binary_ai/dataset_audit.json`
- `reports/binary_ai/dataset_duplicates.csv`
- `reports/binary_ai/dataset_conflicts.csv`

Splits reproduits:

- `70 %` train
- `15 %` validation
- `15 %` test
- seed `42`

Prévention des fuites:

- les groupes identiques ne peuvent pas se retrouver dans plusieurs splits
- les quasi-doublons normalisés sont consolidés
- les contradictions de label sont isolées dans le rapport de conflits

## Modèle ML

Fichier principal:

- `src/deepforma/training/binary_ai_ml.py`

Architecture:

- `FeatureUnion`
  - TF-IDF mots: `ngram_range=(1, 2)`, `strip_accents=unicode`, `sublinear_tf=True`
  - TF-IDF caractères: `analyzer=char_wb`, `ngram_range=(3, 5)`
- classifieur principal: `LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)`

Artefacts:

- `models/binary_ai_ml/vectorizer.joblib`
- `models/binary_ai_ml/classifier.joblib`
- `models/binary_ai_ml/pipeline.joblib`
- `models/binary_ai_ml/thresholds.json`
- `models/binary_ai_ml/metadata.json`
- `models/binary_ai_ml/ml_top_positive_features.csv`
- `models/binary_ai_ml/ml_top_negative_features.csv`

## Modèle TextCNN

Fichier principal:

- `src/deepforma/training/binary_ai_textcnn.py`

Architecture:

- vocabulaire entraîné sur le train uniquement
- token `PAD` et `UNK`
- embedding aléatoire et entraînable
- `Conv1D` avec noyaux `3`, `4`, `5`
- `128` filtres par noyau
- `ReLU`
- `global max pooling`
- dense `128`
- dropout `0.4`
- sortie binaire
- `BCEWithLogitsLoss`

Artefacts:

- `models/binary_ai_textcnn/model.pt`
- `models/binary_ai_textcnn/vocabulary.json`
- `models/binary_ai_textcnn/config.json`
- `models/binary_ai_textcnn/thresholds.json`
- `models/binary_ai_textcnn/metadata.json`
- `models/binary_ai_textcnn/training_history.json`

Preuves anti-préentraînement dans `metadata.json`:

- `pretrained_model: false`
- `pretrained_embeddings: false`
- `random_initialization: true`

## Seuils

Le seuil n'est pas figé à `0.5` par principe.

- optimisation réalisée sur validation
- mode par défaut: maximisation du `F1` IA
- le seuil retenu est sauvegardé dans `thresholds.json`

Dans l'exécution actuelle, le seuil retenu est `0.5` pour les deux modèles.

## Métriques

Les métriques partagées sont calculées dans:

- `src/deepforma/evaluation/binary_classification_metrics.py`

Métriques disponibles:

- accuracy
- balanced accuracy
- précision / rappel / F1 IA et non-IA
- macro F1
- weighted F1
- ROC-AUC
- PR-AUC
- MCC
- Cohen kappa
- log loss
- Brier score
- matrice de confusion
- courbe ROC
- courbe précision-rappel
- calibration
- analyse des seuils
- temps moyen d'inférence
- taille sur disque

## Résultats observés sur le dataset actuel

Dataset final:

- `1179` lignes
- `279` positives IA
- `900` négatives non-IA

Splits:

- train: `465`
- validation: `355`
- test: `359`

Résultats test:

- `binary_ai_ml`
  - accuracy `0.9972`
  - balanced accuracy `0.9936`
  - F1 IA `0.9935`
  - ROC-AUC `1.0`
  - PR-AUC `1.0`
- `binary_ai_textcnn`
  - accuracy `1.0`
  - balanced accuracy `1.0`
  - F1 IA `1.0`
  - ROC-AUC `1.0`
  - PR-AUC `1.0`

Le modèle recommandé par le script de comparaison courant est `binary_ai_textcnn`.

## Exécution

```bash
make binary-ai-prepare
make binary-ai-train-ml
make binary-ai-train-dl
make binary-ai-compare
make binary-ai-all
```

## Intégration DeepForma

Façade d'inférence:

- `src/inference/binary_ai_predictor.py`

Choix du backend:

- `existing`
- `ml_from_scratch`
- `textcnn_from_scratch`

La configuration par défaut conserve le comportement existant.

## Limites connues

- le `threshold` optimal peut rester à `0.5` si la séparation est très nette
- le pipeline ML peut être coûteux à entraîner à cause de la recherche d'hyperparamètres
- la suite de tests globale reste bloquée par un problème préexistant sur `data/referentials/skills.json`
