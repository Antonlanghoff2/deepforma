# Intégration France Travail

## Vue d'ensemble

Le pipeline Deepforma interroge l'API France Travail, normalise les offres, extrait les compétences structurées, applique le modèle CamemBERT multi-étiquette sur le titre et la description, puis fusionne les compétences obtenues pour alimenter l'analyse territoriale.

## Authentification

Le client utilise OAuth2 Client Credentials. Les secrets sont lus depuis `.env` via `python-dotenv`.

Variables attendues:

- `FRANCE_TRAVAIL_CLIENT_ID`
- `FRANCE_TRAVAIL_CLIENT_SECRET`
- `FRANCE_TRAVAIL_SCOPE`
- `FRANCE_TRAVAIL_TOKEN_URL`
- `FRANCE_TRAVAIL_API_BASE_URL`

## Commandes

Collecte:

```bash
python -m src.jobs.collect_france_travail_offers \
  --departement 93 \
  --keywords "intelligence artificielle" \
  --max-pages 5 \
  --output data/france_travail/normalized/offers_93_ia.jsonl \
  --keep-raw \
  --run-model
```

Tests:

```bash
python -m pytest -q
```

## Schéma des données

Les offres normalisées contiennent notamment:

- `offer_id`
- `title`
- `description`
- `offer_text`
- `structured_skills`
- `model_skills`
- `normalized_skills`
- `raw_offer`

## Pagination

La récupération s'effectue par plages via l'en-tête `Range`. Le client arrête la pagination si:

- la réponse est vide;
- le nombre d'offres est inférieur à la taille de page;
- la limite totale d'offres est atteinte.

## Limites

- Les paramètres exacts de recherche peuvent nécessiter un ajustement si l'API évolue.
- Le modèle Deepforma est un extracteur spécialisé sur les compétences vues à l'entraînement.
- Les compétences structurées France Travail restent prioritaires.

## Sécurité

- Les secrets ne doivent jamais être commit.
- Le client n'affiche jamais le secret dans les logs.
- `.env` est ignoré par Git.

## Ajout d'une compétence

1. Ajouter l'entrée dans `data/referentials/skills.json`.
2. Recharger le normaliseur.
3. Ajouter si nécessaire des alias et une catégorie.

