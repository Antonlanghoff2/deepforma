# Apprentissage continu supervisé

Ce dépôt met en place un apprentissage continu, mais pas une auto-amélioration aveugle.
Le modèle ne doit jamais réentraîner ses propres prédictions comme si elles étaient des vérités.
Les nouvelles données passent d'abord par une validation humaine, une source fiable, ou un pseudo-étiquetage contrôlé et journalisé séparément.

## Pourquoi ce garde-fou

Une prédiction de modèle n'est pas un label.
Si on transforme automatiquement les sorties du modèle en vérité terrain, on recycle ses erreurs, on amplifie les biais et on perd le signal de correction humaine.
Le système doit donc distinguer explicitement:

- `human_review`
- `france_travail_api`
- `exact_reference_match`
- `semantic_match`
- `model_prediction`
- `imported_gold_dataset`

Par défaut, `model_prediction` est exclu de l'entraînement.

## Flux cible

1. Collecte des offres France Travail.
2. Exécution du modèle actuellement déployé.
3. Enregistrement des prédictions, des offsets et des incertitudes.
4. Sélection active des exemples à vérifier.
5. Validation/correction humaine.
6. Constitution d'un dataset incrémental approuvé.
7. Entraînement périodique sur GPU.
8. Évaluation contre le modèle de production.
9. Promotion uniquement si les critères d'amélioration sont respectés.
10. Déploiement versionné avec rollback possible.

## Stockage

Le stockage persistant est basé sur SQLite dans `data/continual_learning/continual_learning.sqlite3`.
Il conserve:

- `offer_id`
- `content_version`
- titre et description brute
- date de collecte
- localisation / territoire
- compétences structurées France Travail
- compétences prédites
- formes textuelles détectées
- offsets `start/end`
- confiance
- source et provenance de chaque label
- version du modèle
- statut de validation
- corrections humaines
- compétences rejetées ou ajoutées
- date et auteur logique de validation

Les statuts gérés sont:

- `pending`
- `approved`
- `corrected`
- `rejected`
- `excluded`
- `used_for_training`

## Compétences structurées France Travail

Les compétences structurées fournies par l'API sont stockées séparément.
Elles conservent leur code et leur libellé d'origine, puis sont normalisées contre le référentiel.
Elles reçoivent la provenance `france_travail_api`.

Elles ne sont pas supposées exhaustives.
Un désaccord entre les compétences structurées et les compétences extraites de la description peut être envoyé en revue humaine.

## Active learning

La file de revue utilise plusieurs signaux:

- faible confiance
- faible marge entre les deux meilleures prédictions
- compétence inconnue du référentiel
- désaccord modèle / France Travail
- compétence ambiguë
- texte éloigné des données d'entraînement
- nouvelle compétence fréquente
- nouvelle famille métier ou territoriale
- échantillon de contrôle à haute confiance

La sélection maintient une diversité de métiers, territoires et types de compétences.

## Export du dataset

Le script `scripts/export_continual_training_dataset.py` produit un JSONL compatible NER.

Les annotations sans offsets fiables sont exportées séparément dans `document_skills`.
Elles ne sont pas converties artificiellement en entités NER.

## Entraînement GPU

Le script `scripts/train_continual_skill_extractor.py` charge:

- le dataset historique approuvé
- le dataset incrémental approuvé
- un échantillon de répétition des anciennes données
- la validation
- le test fixe

Les exemples sont dédupliqués par hash du texte normalisé pour éviter la fuite entre splits.

Le mode `--fp16` peut être activé sur GPU.

## Évaluation

Le script `scripts/compare_model_versions.py` compare le candidat au modèle de production sur le test fixe.
Il mesure:

- précision
- rappel
- F1 exact
- F1 normalisé
- faux positifs
- faux négatifs
- performances par compétence
- performances par famille métier
- performances par territoire
- performances sur compétences nouvelles
- performances sur compétences anciennes
- taux d'entités sans justification textuelle

Règle initiale de promotion:

- F1 exact +1 point minimum
- perte de précision inférieure à 2 points
- aucune régression critique configurée
- aucune hausse significative des compétences inventées

## Registre des modèles

Les modèles sont stockés sous `models/skill-extractor/`:

- `production`
- `candidates/`
- `versions/`
- `registry.json`

Chaque version enregistre la version, la date, le hash Git, le modèle de base, les hashes de datasets, le nombre d'exemples, les métriques, la taxonomie, le référentiel, l'état, et la version précédente.

## Déploiement et rollback

Le VPS ne fait que l'inférence.
L'entraînement reste sur la machine locale GPU.

Le déploiement suit une logique atomique:

- transfert vers un dossier cible
- vérification du chargement du modèle
- mise à jour atomique du lien `production`
- redémarrage du service
- health check
- rollback automatique si l'étape finale échoue

Le rollback restaure la version précédente du lien de production.

## Dérive

La dérive ne déclenche pas automatiquement une promotion.
Elle alimente la priorité de revue.
Le rapport de dérive suit notamment:

- nouvelles expressions fréquentes
- compétences inconnues
- évolution des distributions
- nouveaux métiers
- nouveaux territoires
- distance moyenne des embeddings
- évolution du taux de faible confiance

## Conservation et confidentialité

- Aucun secret API ne doit être stocké dans la base ou les rapports.
- Les jetons France Travail ne doivent pas être journalisés.
- Les données personnelles inutiles ne doivent pas être exportées.
- Les mécanismes existants de suppression doivent être respectés.
- La durée de conservation des offres et annotations doit être paramétrée selon les règles internes de rétention.

## Sauvegarde

Sauvegarder régulièrement:

- `data/continual_learning/continual_learning.sqlite3`
- `data/continual_learning/`
- `models/skill-extractor/registry.json`
- le répertoire `models/skill-extractor/versions/`

Un backup cohérent doit être fait avant promotion ou rollback majeur.
