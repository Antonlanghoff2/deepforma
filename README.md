# DeepForma

DeepForma est une application de veille et d'aide à la décision autour des compétences, des formations et des referentiels métiers.
Le dépôt regroupe des pipelines pour la préparation de jeux de données, l'entrainement de modèles ML/DL, l'extraction de compétences, l'intégration France Travail et l'import de referentiels officiels.

## 1. Présentation du projet

Le projet combine plusieurs flux de traitement:

- préparation de datasets CPF généralistes;
- préparation d'un jeu de données IA multilabel;
- classification binaire IA / non-IA;
- extraction et normalisation de compétences;
- import et croisement de referentiels (France Compétences, ROME, IA);
- interrogation et enrichissement des offres France Travail;
- exposition web pour l'analyse et la recommandation.

## 2. Objectif métier

L'objectif est d'identifier les compétences, d'évaluer leur couverture dans le marché et de recommander des formations ou des correspondances métiers exploitables dans DeepForma.

## 3. Architecture générale

- `src/` contient la logique métier, les services, les connecteurs et l'application web.
- `scripts/` contient les CLI de préparation, d'entraînement, d'évaluation et d'import.
- `data/` contient les sources brutes, les jeux préparés et les exports intermédiaires.
- `models/` contient les artefacts de modèles locaux.
- `reports/` contient les audits, métriques et exports.
- `docs/` documente les pipelines et les choix d'architecture.
- `deploy/` contient les exemples de déploiement systemd, Apache et Nginx.

## 4. Prérequis système

- Python 3.10+.
- `venv` disponible.
- GNU Make.
- Dépendances Python installées via `requirements.txt`.
- Accès réseau seulement pour les pipelines qui interrogent France Travail ou téléchargent des ressources.

## 5. Installation

Sur une machine neuve, utiliser de préférence:

```bash
make setup
```

`make setup` crée le venv si nécessaire puis installe les dépendances.

## 6. Création du venv

Si le venv n'existe pas encore, `make setup` s'en charge. Le dépôt attend un interpréteur dans `.venv/bin/python`.

## 7. Installation des dépendances

Pour réinstaller les dépendances dans un venv déjà présent:

```bash
make install
```

## 8. Variables d'environnement

Les secrets ne doivent pas être committés. Le fichier `.env.example` ne contient que des noms de variables et des valeurs factices.

Variables principales:

- `FRANCE_TRAVAIL_CLIENT_ID`
- `FRANCE_TRAVAIL_CLIENT_SECRET`
- `FRANCE_TRAVAIL_SCOPE`
- `FRANCE_TRAVAIL_TOKEN_URL`
- `FRANCE_TRAVAIL_API_BASE_URL`

Variables utiles en ligne de commande Make:

- `CPF_GENERAL_INPUT`
- `CPF_GENERAL_OUTPUT_DIR`
- `IA_DATASET`
- `IA_TAXONOMY`
- `IA_PROCESSED_DIR`
- `FRANCE_TRAVAIL_COLLECT_ARGS`
- `FRANCE_TRAVAIL_OUTPUT`
- `SMOKE_TEST_TASK`
- `SMOKE_TEST_SAMPLES`
- `MODEL_CHECK_DIR`

Exemple:

```bash
make cpf-general-prepare   CPF_GENERAL_INPUT="data/raw/Dataset_Generaliste_CPF_V4.xlsx"   CPF_GENERAL_OUTPUT_DIR="data/processed/cpf"
```

## 9. Arborescence utile

- `data/raw/` sources d'entrée.
- `data/processed/` jeux transformés et splits.
- `data/training/` sorties intermédiaires d'entrainement.
- `data/france_travail/` offres normalisées et rapports.
- `data/referentials/` taxonomies et référentiels locaux.
- `models/` checkpoints et modèles entraînés.
- `reports/` audits et métriques.

## 10. Datasets requis

| Dataset | Format | Emplacement | Usage | Statut |
|---|---|---|---|---|
| Dataset généraliste CPF V4 | Excel | `data/raw/Dataset_Generaliste_CPF_V4.xlsx` | Préparation CPF généraliste | requis |
| Dataset IA V9 synth | Excel | `data/raw/Dataset_IA_V9_synth.xlsx` | Préparation IA multilabel | requis |
| Taxonomie IA | JSON | `config/ia_taxonomy_v2.json` | Préparation IA et entraînement | requis |
| Dataset binaire IA / non-IA annoté | Excel | `data/raw/dataset_competences_IA_annotees.xlsx` | Audits et analyses binaires | requis pour l'audit |
| Dataset recommandations IA | CSV | `data/raw/recommandations_IA_consolide.csv` | Module recommandations IA | optionnel |
| Référentiels France Compétences | ZIP / manifest | `data/raw/france_competences/` | Import des référentiels | optionnel selon pipeline |
| Référentiel PDF | PDF | `data/raw/referentiel/*.pdf` | Extraction des compétences référentielles | optionnel |
| Catalogue CPF préparé | CSV | `data/raw/cpf/cpf_catalog.csv` | Inspection et normalisation CPF | généré |

## 11. Préparation des données CPF

La chaîne CPF généraliste utilise les cibles suivantes:

```bash
make cpf-general-check
make cpf-general-prepare
make cpf-pairs
make cpf-train
make cpf-general-all
make cpf-all
```

Exemple avec chemins surchargés:

```bash
make cpf-general-prepare   CPF_GENERAL_INPUT="data/raw/Dataset_Generaliste_CPF_V4.xlsx"   CPF_GENERAL_OUTPUT_DIR="data/processed/cpf"
```

Sorties principales:

- `data/processed/cpf/formations_generalistes.jsonl`
- `data/processed/cpf/pairs_generalistes.jsonl`
- `models/cpf-recommender/`

## 12. Préparation des données IA

La chaîne IA multilabel utilise:

```bash
make ia-check
make ia-prepare
make ia-train
make ia-evaluate
make ia-all
```

Exemple:

```bash
make ia-prepare   IA_DATASET="data/raw/Dataset_IA_V9_synth.xlsx"   IA_TAXONOMY="config/ia_taxonomy_v2.json"   IA_PROCESSED_DIR="data/processed"
```

Sorties principales:

- `data/processed/ia_multilabel_train.jsonl`
- `data/processed/ia_multilabel_validation.jsonl`
- `data/processed/ia_multilabel_test.jsonl`
- `models/ia-classifier-v2/final/`
- `reports/ia_multilabel/evaluation_report.json`

## 13. Binary AI from scratch

La pipeline binaire IA / non-IA reste séparée de la pipeline multilabel:

```bash
make binary-ai-prepare
make binary-ai-train-ml
make binary-ai-train-dl
make binary-ai-compare
make binary-ai-all
```

Cette chaîne produit les artefacts de comparaison ML vs DL dans `reports/binary_ai/` et les modèles dans `models/binary_ai_ml/` et `models/binary_ai_textcnn/`.

## 14. Entraînement Machine Learning

L'entraînement ML principal du dépôt correspond surtout à la partie binaire IA / non-IA et à certains modèles de recommandation.
Pour une vérification rapide avant entrainement, utiliser `make smoke-test`.

## 15. Entraînement Deep Learning

La partie DL principale du dépôt est la variante TextCNN de la classification binaire:

```bash
make binary-ai-train-dl
```

## 16. Évaluation des modèles

Commandes utiles:

```bash
make ia-evaluate
make binary-ai-compare
make model-check
```

Le checkpoint IA multilabel final est vérifié par `make model-check`.

## 17. Extraction des compétences

Commandes principales:

```bash
make build-referential-ner-candidates
make build-referential-multilabel-candidates
make export-referential-training-data
make train-referential-ner
make train-referential-multilabel
make evaluate-referential-models
make test-referential-ml-dl
```

## 18. Utilisation de l'API France Travail

La configuration est validée par:

```bash
make france-travail-check
```

La collecte et l'enrichissement des offres s'exécutent via:

```bash
make france-travail-collect   FRANCE_TRAVAIL_COLLECT_ARGS='--departement 75 --keywords "intelligence artificielle" --max-pages 1 --run-model --overwrite'   FRANCE_TRAVAIL_OUTPUT='data/france_travail/normalized/offers_75_ia.jsonl'
```

## 19. Codes ROME

Les codes ROME sont utilisés dans plusieurs pipelines d'import et de comparaison:

- `import-rome-referential`
- `map-rncp-to-rome`
- `build-unified-skill-referential`
- `build-rome-rncp-training-dataset`
- `train-skill-extractor`

## 20. Import des référentiels

Commandes disponibles:

```bash
make import-france-competences
make import-rome-referential
make map-rncp-to-rome
make build-unified-skill-referential
make build-rome-rncp-training-dataset
make train-skill-extractor
```

## 21. Lancement de l'application

```bash
make run
```

`make dev` est un alias de `make run`.

## 22. Tests

```bash
make test
make smoke-test
```

`make smoke-test` réalise un contrôle rapide d'entrainement sur un petit échantillon.

## 23. Déploiement

Les cibles de déploiement présentes dans le Makefile sont:

- `deploy-check`
- `deploy-install`
- `deploy-update`
- `deploy-restart`
- `deploy-status`
- `deploy-logs`
- `deploy-apache-test`
- `deploy-nginx-test`

Les exemples de service et de proxy se trouvent dans `deploy/`.

## 24. Commandes Makefile

| Commande | Description | Prérequis | Variables configurables | Sorties produites |
|---|---|---|---|---|
| `make help` | Liste les commandes documentées | aucun | aucune | affichage |
| `make install` | Installe les dépendances dans le venv | `.venv` ou `python3` disponible | aucune | environnement Python prêt |
| `make setup` | Crée le venv si besoin puis installe les dépendances | `python3` disponible | aucune | `.venv` + dépendances |
| `make run` | Lance l'application Flask | dépendances installées | aucune | serveur web |
| `make dev` | Alias de `make run` | dépendances installées | aucune | serveur web |
| `make clean` | Nettoie les caches locaux | aucun | aucune | aucun |
| `make test` | Lance pytest | dépendances installées | aucune | sortie pytest |
| `make smoke-test` | Test rapide d'entrainement | dataset de smoke test | `SMOKE_TEST_*` | `reports/smoke_test.json` |
| `make cpf-general-check` | Vérifie le dataset CPF généraliste | fichier Excel CPF | `CPF_GENERAL_INPUT`, `CPF_GENERAL_SHEET` | rapport d'inspection |
| `make cpf-general-prepare` | Prépare le dataset CPF généraliste | fichier Excel CPF | `CPF_GENERAL_INPUT`, `CPF_GENERAL_OUTPUT_DIR`, `CPF_GENERAL_SHEET` | JSONL CPF |
| `make cpf-pairs` | Construit les paires CPF | `cpf-general-prepare` | `CPF_SEED`, `CPF_MAX_PAIRS_PER_FORMATION` | paires d'entraînement |
| `make cpf-train` | Entraîne le recommender CPF | paires CPF | `CPF_BASE_MODEL`, `CPF_EPOCHS`, `CPF_BATCH_SIZE`, `CPF_MODEL_OUTPUT` | modèle CPF |
| `make cpf-general-all` | Chaîne CPF complète | dataset CPF | variables CPF | modèle + données |
| `make cpf-all` | Alias de `cpf-general-all` | idem | idem | idem |
| `make ia-check` | Vérifie la taxonomie IA | dataset IA + taxonomie | `IA_DATASET`, `IA_TAXONOMY` | validation console |
| `make ia-prepare` | Prépare les splits IA multilabel | dataset IA + taxonomie | `IA_DATASET`, `IA_TAXONOMY`, `IA_PROCESSED_DIR` | splits JSONL |
| `make ia-train` | Entraîne le classifieur IA multilabel | splits IA préparés | `IA_BASE_MODEL`, `IA_EPOCHS`, `IA_BATCH_SIZE`, `IA_MODEL_OUTPUT` | checkpoint IA |
| `make ia-evaluate` | Évalue le checkpoint IA | checkpoint IA + test | `IA_MODEL_OUTPUT`, `IA_EVALUATION_DIR` | rapport d'évaluation |
| `make ia-all` | Chaîne IA complète | dataset IA | variables IA | modèle + rapport |
| `make binary-ai-prepare` | Prépare le dataset binaire IA / non-IA | dataset binaire source | `BINARY_AI_*` | dataset + splits |
| `make binary-ai-train-ml` | Entraîne le modèle ML binaire | splits binaires | `BINARY_AI_ML_MODEL_OUTPUT` | modèle ML |
| `make binary-ai-train-dl` | Entraîne le modèle TextCNN binaire | splits binaires | `BINARY_AI_TEXTCNN_MODEL_OUTPUT` | modèle DL |
| `make binary-ai-compare` | Compare les modèles binaires | modèles entraînés | `BINARY_AI_REPORT_DIR` | CSV/JSON de comparaison |
| `make binary-ai-all` | Chaîne binaire complète | dataset binaire | variables binaires | modèles + comparaison |
| `make france-travail-check` | Vérifie la config France Travail | variables d'environnement | `FRANCE_TRAVAIL_*` | validation console |
| `make france-travail-collect` | Collecte et enrichit des offres France Travail | credentials + arguments de collecte | `FRANCE_TRAVAIL_COLLECT_ARGS`, `FRANCE_TRAVAIL_OUTPUT` | JSONL + rapports |
| `make model-check` | Vérifie le checkpoint IA final | checkpoint IA présent | `MODEL_CHECK_DIR`, `MODEL_CHECK_REPORT` | rapport d'audit checkpoint |
| `make import-france-competences` | Importe les référentiels France Compétences | archives téléchargées | `FRANCE_COMPETENCES_*` | référentiels importés |
| `make import-rome-referential` | Importe le ROME | fichier source ROME | chemins ROME | référentiel ROME |
| `make build-unified-skill-referential` | Construit le référentiel unifié | référentiels importés | chemins de référentiels | JSONL unifié |
| `make train-skill-extractor` | Entraîne l'extracteur de compétences | données d'entraînement | `CPF_*`, `IA_*` et chemins associés | modèle extraction |
| `make build-referential-ner-candidates` | Génère les candidats NER | PDF référentiels | `REFERENTIAL_PDF_DIR` | JSONL candidats |
| `make build-referential-multilabel-candidates` | Génère les candidats multilabel | PDF référentiels | `REFERENTIAL_PDF_DIR` | JSONL candidats |
| `make export-referential-training-data` | Exporte les données d'entraînement référentielles | candidats générés | `REFERENTIAL_TRAIN_DIR` | JSONL de splits |
| `make train-referential-ner` | Entraîne le modèle NER référentiel | splits exportés | `REFERENTIAL_NER_*` | modèle NER |
| `make train-referential-multilabel` | Entraîne le modèle multilabel référentiel | splits exportés | `REFERENTIAL_MULTILABEL_*` | modèle multilabel |
| `make evaluate-referential-models` | Évalue les modèles référentiels | modèles entraînés | chemins de modèles | rapports |
| `make test-referential-ml-dl` | Lance les tests référentiels ML/DL | modèles et données préparés | aucune | sortie pytest |
| `make deploy-check` | Vérifie la chaîne de déploiement | environnement de déploiement | `DEPLOY_*` | diagnostic |
| `make deploy-install` | Installe le service | accès serveur | `DEPLOY_*` | service installé |
| `make deploy-update` | Met à jour le service | accès serveur | `DEPLOY_*` | service mis à jour |
| `make deploy-restart` | Redémarre le service | service installé | `DEPLOY_SERVICE` | service redémarré |
| `make deploy-status` | Affiche l'état du service | service installé | `DEPLOY_SERVICE` | état du service |
| `make deploy-logs` | Affiche les logs du service | service installé | `DEPLOY_SERVICE` | logs |

## 25. Limites connues

- La classification binaire IA / non-IA peut rester peu discriminante sur certains jeux de données et doit être interprétée avec prudence.
- Les scripts France Travail nécessitent des credentials valides et peuvent être limités par le quota ou la latence réseau.
- Plusieurs pipelines sont hérités et coexistent pour compatibilité; `make help` reste la source de vérité opérationnelle.
- Les analyses de domaine doivent être distinguées entre données courtes de compétences et descriptions longues de formations.

## 26. Sécurité et données sensibles

- Ne pas committer de secrets, jetons OAuth, clés API ou credentials.
- Utiliser `.env` local ou des variables d'environnement pour France Travail.
- Ne pas publier de données brutes sensibles dans les rapports ou notebooks.
- Les chemins personnels ou machine-specific ne doivent pas être figés dans la documentation.
- Le fichier `.env.example` doit rester factice et sans secret réel.

## Commandes recommandées

- Installer le projet: `make setup`
- Préparer les données CPF: `make cpf-general-prepare CPF_GENERAL_INPUT=... CPF_GENERAL_OUTPUT_DIR=...`
- Entraîner le modèle ML IA: `make ia-train`
- Entraîner le modèle DL binaire: `make binary-ai-train-dl`
- Évaluer les modèles: `make ia-evaluate` ou `make binary-ai-compare`
- Lancer l'application: `make run`
- Exécuter les tests: `make test`

## Documentation complémentaire

- `docs/architecture.md`
- `docs/data_flow.md`
- `docs/deployment.md`
- `docs/france_travail_integration.md`
- `docs/france_competences_pipeline.md`
- `docs/france_competences_rome_training_pipeline.md`
