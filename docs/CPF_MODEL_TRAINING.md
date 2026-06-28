# Entraînement du recommender CPF

## Données sources

La source principale du pipeline est `data/raw/Dataset_Generaliste_CPF_V3.xlsx`. Le pipeline peut aussi retrouver `Dataset_Generaliste_CPF_V2.xlsx` ou `Dataset_Generaliste_CPF_V1.xlsx` si la V3 n'est pas disponible localement, mais la V3 doit être privilégiée.

## Commandes exactes

```bash
cd /home/bibi/deepforma
source .venv/bin/activate
python -m pip install -e .
make cpf-inspect CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx
make cpf-prepare CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx
make cpf-enrich-skills CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx
make cpf-build-pairs CPF_FORMATIONS=data/processed/cpf/formations_with_skills.parquet
make cpf-train CPF_TRAIN=data/training/cpf_train.jsonl CPF_VALIDATION=data/training/cpf_validation.jsonl
make cpf-evaluate CPF_TEST=data/training/cpf_test.jsonl
make cpf-reindex CPF_MODEL_OUTPUT=models/cpf-recommender
make cpf-training-pipeline CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx CPF_FORMATIONS=data/processed/cpf/formations_with_skills.parquet CPF_BASE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 CPF_MODEL_OUTPUT=models/cpf-recommender
```

## Pourquoi le catalogue CPF n'est pas directement supervisé

Le catalogue Mon Compte Formation contient des formations, pas des couples explicites utilisateur/formation. Il faut donc construire un signal d'apprentissage à partir d'heuristiques contrôlées:

- extraction des compétences du texte de la formation;
- rapprochement avec les compétences demandées par les offres France Travail;
- prise en compte du métier ciblé;
- prise en compte du territoire et du mode distanciel;
- prise en compte du niveau et de la certification RNCP/RS.

Les labels heuristiques servent à amorcer un modèle, pas à définir une vérité terrain.

## Création des labels

Chaque exemple est construit autour d'une requête:

- `target_job`;
- `required_skills`;
- `missing_skills`;
- `region_code`;
- `department_code`.

Une formation est positive si:

- la couverture des compétences demandées dépasse le seuil configuré;
- la similarité sémantique est suffisante;
- la formation est compatible avec le territoire ou disponible à distance;
- le texte est assez riche.

## Positifs et négatifs

Le générateur produit:

- des positifs à partir des formations CPF compatibles avec la requête;
- des négatifs `easy` issus d'un domaine sans rapport;
- des négatifs `hard` proches en intitulé ou certification mais insuffisants sur les compétences;
- des négatifs `territorial` pertinents sur le fond mais incompatibles géographiquement.

## Prévention des fuites

Les splits sont faits par groupe afin d'éviter de mélanger:

- des sessions proches d'une même formation;
- des versions quasi identiques;
- des formations partageant la même certification et un texte quasi identique.

La répartition par défaut est:

- 70 % train;
- 15 % validation;
- 15 % test.

## Entraînement

Le modèle de base utilisé par défaut est:

```bash
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Deux pertes sont disponibles:

- `MultipleNegativesRankingLoss`;
- `TripletLoss`.

Le pipeline détecte automatiquement CUDA et utilise le CPU si nécessaire.

## Évaluation

Le script d'évaluation compare:

- le modèle de base non fine-tuné;
- le modèle CPF fine-tuné;
- une baseline TF-IDF.

Les métriques calculées incluent:

- Recall@1, Recall@5, Recall@10;
- Precision@5;
- MRR;
- NDCG@10;
- similarités positives et négatives moyennes;
- compatibilité territoriale;
- couverture des compétences.

## Utilisation GPU

Pour une RTX 4060, les valeurs de départ recommandées sont:

- batch size: 16;
- gradient accumulation: 2;
- epochs: 2 ou 3;
- max sequence length: 256;
- mixed precision FP16.

## Reprise après interruption

Le script d'entraînement accepte:

- `--resume-from-checkpoint`

Les checkpoints sont enregistrés sous:

- `models/cpf-recommender/checkpoints/`

## Limites des labels heuristiques

Les exemples construits automatiquement peuvent contenir des biais:

- requêtes trop proches du texte d'une formation;
- négatifs trop faciles;
- couverture territoriale approximative;
- différence faible entre niveau et certification.

Une révision humaine reste nécessaire pour consolider le dataset.

## Annotation humaine et retour utilisateur futur

Le fichier `cpf_pairs_review.csv` est prévu pour la révision humaine. Il contient:

- `reviewer_label`;
- `reviewer_comment`;
- `validated_at`.

Il pourra aussi accueillir du retour utilisateur plus tard.

## Commandes

Extraction des compétences:

```bash
python scripts/extract_cpf_skills.py --input data/processed/cpf/formations_normalized.parquet --output data/processed/cpf/formations_with_skills.parquet
```

Construction des paires:

```bash
python scripts/build_cpf_training_pairs.py --formations data/processed/cpf/formations_with_skills.parquet --offers-dir data/france_travail/normalized
```

Entraînement:

```bash
python scripts/train_cpf_recommender.py   --train data/training/cpf_train.jsonl   --validation data/training/cpf_validation.jsonl   --base-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   --output-dir models/cpf-recommender
```

Évaluation:

```bash
python scripts/evaluate_cpf_recommender.py --test data/training/cpf_test.jsonl --formations data/processed/cpf/formations_with_skills.parquet
```

Ré-indexation:

```bash
python scripts/build_cpf_embeddings.py   --input data/processed/cpf/formations_with_skills.parquet   --model models/cpf-recommender/final
```
