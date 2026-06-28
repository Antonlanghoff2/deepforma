# Pipeline CPF Deepforma

Ce document décrit l'intégration du catalogue Mon Compte Formation comme un corpus documentaire de formations, et non comme un dataset de classification supervisée.

## Rôle du catalogue CPF

Le catalogue CPF sert de base documentaire pour:

- retrouver des formations pertinentes par territoire;
- couvrir des compétences manquantes;
- relier métier cible, compétences requises et offre de formation;
- alimenter un index sémantique filtrable.

Il ne doit pas être traité comme une liste de classes à prédire.

## Différence corpus / dataset supervisé

- Corpus documentaire: ensemble de formations nettoyées, enrichies et indexées.
- Dataset supervisé: paires ou triplets query/positive/negative destinés à l'apprentissage d'un reranker ou à l'annotation humaine.

Le catalogue CPF entre d'abord dans la première catégorie.

## Téléchargement

Le téléchargement est géré par `scripts/download_cpf_catalog.py`.

Fonctionnalités:

- URL locale ou distante;
- tentative d'export OpenDataSoft quand l'URL pointe vers la page de dataset;
- téléchargement via fichier temporaire;
- checksum SHA-256;
- date de téléchargement;
- reprise propre en cas d'échec;
- logs en français.

Exemple:

```bash
python scripts/download_cpf_catalog.py \
  --output-dir data/raw/cpf \
  --source-url "https://opendata.caissedesdepots.fr/explore/dataset/moncompteformation_catalogueformation/"
```

## Inspection du schéma

`scripts/inspect_cpf_catalog.py` détecte:

- encodage;
- séparateur;
- nombre de colonnes;
- noms de colonnes;
- types estimés;
- taux de valeurs manquantes;
- alias probables pour les champs canoniques.

Le rapport est écrit dans `data/reports/cpf_schema_report.json`.

## Configuration des colonnes

Les alias sont définis dans `config/cpf_columns.yaml`.

La détection est tolérante:

- casse;
- accents;
- espaces;
- tirets;
- underscores.

## Nettoyage et normalisation

`scripts/prepare_cpf_catalog.py` applique:

- nettoyage HTML;
- décodage des entités HTML;
- normalisation Unicode NFKC;
- suppression des caractères de contrôle;
- normalisation des espaces;
- conversion des valeurs vides en `null`;
- normalisation des SIRET, codes département et région;
- distinction RNCP / RS;
- déduplication en deux niveaux;
- construction de `search_text`;
- génération d'un `formation_uid` stable.

Le texte final conserve les accents.

## Déduplication

Deux niveaux:

1. doublon exact sur `formation_uid`;
2. doublon proche sur certification, organisme, territoire et similarité d'intitulé.

Deux formations identiques mais dans deux territoires différents ne doivent pas être fusionnées.

## Extraction de compétences

Le composant `src/deepforma/cpf/skill_extractor.py` combine:

- correspondance exacte normalisée;
- alias et synonymes;
- expressions régulières;
- correspondance sémantique heuristique;
- modèle NER si disponible plus tard.

Sorties par formation:

- `skills_explicit`;
- `skills_inferred`;
- `skills_normalized`;
- `skills_confidence`;
- `skills_evidence`.

Les preuves gardent le passage de texte source.

## Taxonomie commune

`src/deepforma/skills/normalizer.py` sert de point commun entre:

- compétences extraites des formations CPF;
- compétences extraites des offres France Travail.

Le normaliseur conserve:

- `canonical_id`;
- `canonical_label`;
- `original_label`;
- `aliases`;
- `extraction_source`;
- `confidence`.

Des garde-fous empêchent les rapprochements abusifs, par exemple entre `Java` et `JavaScript`.

## Embeddings et index

`scripts/build_cpf_embeddings.py`:

- lit `formations.parquet`;
- encode `search_text`;
- normalise les vecteurs;
- utilise CUDA si disponible;
- bascule sur CPU sinon;
- enregistre le modèle, la dimension, la date et le hash du corpus;
- expose un backend d'index interchangeable.

Stockage:

- `data/indexes/cpf/faiss.index`;
- `data/indexes/cpf/metadata.parquet`;
- `data/indexes/cpf/index_manifest.json`.

## Recherche et scoring

Le moteur `src/deepforma/recommendation/training_recommender.py`:

- construit la requête à partir du métier et des compétences;
- interroge l'index vectoriel;
- filtre territorialement;
- calcule un score explicable;
- élimine les résultats trop similaires;
- retourne les formations les plus pertinentes.

Le score est dominé par la couverture des compétences manquantes.

## Génération de paires d'entraînement

`scripts/build_training_pairs.py` produit:

- `query`;
- `positive`;
- `negative`;
- `label_source`.

Les paires sont heuristiques tant qu'aucune annotation humaine n'a validé les triplets.

## Mise à jour incrémentale

`python scripts/update_cpf_catalog.py` doit:

- télécharger la nouvelle version;
- comparer le checksum;
- détecter les ajouts, modifications et suppressions;
- ne recalculer que le nécessaire;
- reconstruire l'index;
- produire `data/reports/cpf_update_report.json`.

## Limites

- Les regroupements sémantiques restent heuristiques tant que le dataset annoté n'existe pas.
- Les embeddings et l'index nécessitent les dépendances optionnelles du runtime cible.
- Le catalogue brut ne doit pas être versionné.

## Commandes

```bash
make cpf-download ARGS='--output-dir data/raw/cpf --source-url ...'
make cpf-inspect ARGS='--input data/raw/cpf/cpf_catalog.csv'
make cpf-prepare ARGS='--input data/raw/cpf/cpf_catalog.csv --output-dir data'
make cpf-embed ARGS='--input data/processed/cpf/formations.parquet'
make cpf-test
make cpf-all
```

## Dépannage

- Si `pyarrow` manque, l'écriture parquet échoue avec une erreur explicite.
- Si `sentence-transformers` manque, la génération d'embeddings doit être lancée dans l'environnement cible.
- Si le schéma du CSV change, relancer d'abord l'inspection puis ajuster `config/cpf_columns.yaml`.
