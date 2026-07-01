# DeepForma

Pipeline de nettoyage, classification et recommandation pour les formations IA et généralistes.

Trois pipelines principaux :

- **Classifieur IA v2** (20 labels) — entraînement d'un CamemBERT pour la classification multilabel de compétences IA
- **CPF généraliste** (v4) — préparation et entraînement d'un recommender Sentence-Transformers sur le catalogue CPF
- **CPF V3** (historique) — pipeline existant avec enrichissement par offres France Travail

## Fichiers sources

- `Dataset_V7_Anton_CSV - Dataset_V7_Anton_CSV.csv.csv` : dataset IA existant.
- `Dataset_Generaliste_CPF_V3.xlsx` : source CPF V3 (historique).
- `Dataset_Generaliste_CPF_V4.xlsx` : source CPF généraliste (nouveau).
- `Dataset_IA_V9_synth.xlsx` : dataset IA pour l'entraînement du classifieur 20 labels.
- `entrainement_camembert_competences_ia.ipynb` : notebook historique conservé intact.

## Installation

```bash
python -m pip install -r requirements.txt
# ou
python -m pip install -e .
```

---

## Déploiement Ubuntu

Le guide de production est disponible dans [docs/deployment_ubuntu.md](docs/deployment_ubuntu.md).
Il décrit le premier déploiement, les mises à jour, les logs, le rollback et la configuration Nginx/systemd.

## Pipeline : Classifieur IA (v2, 20 labels)

Entraîne un `CamemBERTForSequenceClassification` en multilabel sur une taxonomie de 20 compétences IA.

### 1. Préparation du dataset

```bash
python scripts/prepare_ia_training_dataset.py \
    --input data/raw/Dataset_IA_V9_synth.xlsx \
    --output-dir data/processed \
    --taxonomy config/ia_taxonomy_v2.json
```

Étapes : normalisation des labels via `config/alias_map.json` intégré, construction du texte (titre + description + objectifs + résultats), groupement par certification/organisme, `GroupShuffleSplit` anti-fuite, calcul des `pos_weights`, multi-hot encoding. Produit des fichiers JSONL (train/val/test).

### 2. Entraînement

```bash
python scripts/train_ia_multilabel_classifier.py \
    --input-dir data/processed \
    --output-dir models/ia-classifier-v2 \
    --base-model camembert-base \
    --epochs 10 --batch-size 16 --lr 2e-5
```

Options : `--fp16` / `--no-fp16`, `--gradient-checkpointing` / `--no-gradient-checkpointing`, `--device cpu` / `--device cuda`. Early stopping (patience 3), sauvegarde du meilleur modèle + `thresholds.json` (seuils optimisés par label).

### 3. Évaluation

```bash
python scripts/evaluate_ia_multilabel_classifier.py \
    --model-dir models/ia-classifier-v2/final \
    --test-file data/processed/ia_multilabel_test.jsonl \
    --output-dir reports \
    --taxonomy config/ia_taxonomy_v2.json
```

Produit : métriques globales et par label (F1, précision, rappel), matrice de cooccurrence, distribution des probabilités, erreurs d'inférence.

### 4. Inférence (module Python)

```python
from src.models.ia_classifier import IAClassifier

clf = IAClassifier("models/ia-classifier-v2/final", top_k=5)
result = clf.predict("Formation en Deep Learning avec PyTorch")
# [{"label": "Deep Learning", "probability": 0.92, "selected": True}, ...]
```

### Makefile

```bash
make ia-prepare   # préparation du dataset
make ia-train     # entraînement (dépend de ia-prepare)
make ia-evaluate  # évaluation
make ia-all       # pipeline complet
```

Variables : `IA_DATASET`, `IA_TAXONOMY`, `IA_BASE_MODEL`, `IA_MODEL_OUTPUT`, `IA_EPOCHS`, `IA_BATCH_SIZE`, `IA_LEARNING_RATE`.

---

## Pipeline : CPF généraliste (v4)

Prépare le dataset CPF généraliste et génère des paires d'entraînement (positives et négatives) pour un Sentence-Transformer.

### 1. Préparation du dataset

```bash
python scripts/prepare_general_cpf_dataset.py \
    --input data/raw/Dataset_Generaliste_CPF_V4.xlsx \
    --output-dir data/processed/cpf
```

Étapes : normalisation du texte, parsing des compétences structurées, construction des `formation_id` et `group_id` (par code certification), split anti-fuite, dédup. Produit `formations_generalistes.jsonl` + `.parquet`.

### 2. Construction des paires d'entraînement

```bash
python scripts/build_cpf_training_pairs.py \
    --input data/processed/cpf/formations_generalistes.jsonl \
    --output-dir data/processed/cpf \
    --output-pairs pairs_generalistes.jsonl
```

Types de paires :
- **Positives** : même certification (`same_certification`), ou compétences similaires (`same_skills`, Jaccard ≥ 0.3)
- **Négatives** : même secteur avec compétences différentes (`same_sector_diff_skills`), même code ROME avec compétences différentes (`same_rome_diff_skills`), titres lexicalement similaires avec compétences différentes (`similar_title_diff_skills`)

### 3. Entraînement du recommender

```bash
python scripts/train_cpf_recommender.py \
    --input-pairs data/processed/cpf/pairs_generalistes.jsonl \
    --output-dir models/cpf-recommender \
    --base-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --epochs 3 --batch-size 16
```

### Makefile

```bash
make cpf-general-prepare   # préparation du dataset
make cpf-pairs             # construction des paires (dépend de cpf-general-prepare)
make cpf-train             # entraînement du recommender (dépend de cpf-pairs)
make cpf-all               # pipeline complet
```

---

## Pipeline : CPF V3 (historique)

Pipeline existant pour le catalogue `Dataset_Generaliste_CPF_V3.xlsx` avec enrichissement par les offres France Travail.

```bash
make cpf-v3-all CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx
```

Étapes : inspection → préparation → extraction compétences → build pairs (avec offres) → train → evaluate → reindex.

Le guide détaillé est dans [docs/CPF_CATALOG_PIPELINE.md](docs/CPF_CATALOG_PIPELINE.md).

## Application web Flask

Lancement local:

```bash
source .venv/bin/activate
python -m src.web_app
```

Puis ouvrir `http://127.0.0.1:5000`.

Test API :

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"programme": "Formation Python et IA", "departement": "93"}'
```

## Tests

```bash
make test
# ou
python -m pytest tests/
```

238 tests (15 fichiers).

## Notebooks

- `entrainement_camembert_competences_ia.ipynb` : notebook historique (v1, pipeline binaire + multi-étiquette 18 labels)
- `entrainement_camembert_competences_ia_v2.ipynb` : notebook v2 (binaire + multi-étiquette sur ancienne taxonomie)

## Structure du dépôt

```
config/
  ia_taxonomy_v2.json       # Taxonomie 20 labels IA
scripts/
  prepare_ia_training_dataset.py
  train_ia_multilabel_classifier.py
  evaluate_ia_multilabel_classifier.py
  prepare_general_cpf_dataset.py
  build_cpf_training_pairs.py
  train_cpf_recommender.py
  evaluate_cpf_recommender.py
  clean_and_merge_datasets.py
  ...
src/
  models/
    ia_classifier.py        # Module d'inférence IAClassifier
  web_app.py                # Application Flask
  deepforma/                # Module principal (CPF datasets, training)
tests/
  test_prepare_ia_training_dataset.py
  test_ia_classifier.py
  test_prepare_general_cpf_dataset.py
  test_build_cpf_training_pairs.py
  ...
```
