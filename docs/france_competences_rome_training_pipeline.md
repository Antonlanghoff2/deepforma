# France Compétences / ROME Training Pipeline

## Sources
- France Compétences RNCP/RS via `FRANCE_COMPETENCES_SOURCE=open_data` or API documentée si configurée.
- France Travail ROME via les exports/inputs officiels déjà présents dans le dépôt.
- Offres France Travail normalisées par le pipeline existant.

## Schémas
- `data/referentials/france_competences/certifications.jsonl`
- `data/referentials/france_competences/blocks.jsonl`
- `data/referentials/france_competences/skills.jsonl`
- `data/referentials/rome/jobs.jsonl`
- `data/referentials/rome/job_titles.jsonl`
- `data/referentials/rome/skills.jsonl`
- `data/referentials/mappings/rncp_rome_links.jsonl`
- `data/referentials/unified/skills.jsonl`
- `data/training/skill_extraction/train.jsonl`
- `data/training/skill_extraction/validation.jsonl`
- `data/training/skill_extraction/test.jsonl`

## Règles
- Le texte de l’offre reste la source principale.
- ROME sert de contexte métier, pas de vérité à reproduire.
- RNCP et ROME ne sont fusionnés qu’avec un mapping explicite ou une similarité suffisamment étayée.
- Aucun téléchargement de modèle n’est effectué sans configuration explicite.

## Seuils
- `RNCP_ROME_AUTO_VALIDATE_THRESHOLD=0.82`
- `RNCP_ROME_REVIEW_THRESHOLD=0.65`
- Les nouveaux liens entre RNCP et ROME restent en révision sous le seuil d’auto-validation.

## Commandes
- `python scripts/import_france_competences.py --active-only --dry-run`
- `python scripts/import_france_competences.py --active-only --write`
- `python scripts/import_rome_referential.py --dry-run`
- `python scripts/import_rome_referential.py --write`
- `python scripts/map_rncp_to_rome.py --dry-run`
- `python scripts/map_rncp_to_rome.py --write`
- `python scripts/build_unified_skill_referential.py --dry-run`
- `python scripts/build_unified_skill_referential.py --write`
- `python scripts/enrich_offers_with_rome_rncp.py --input <source> --dry-run`
- `python scripts/build_rome_rncp_training_dataset.py`
- `python scripts/train_skill_extractor.py --train data/training/skill_extraction/train.jsonl --validation data/training/skill_extraction/validation.jsonl --test data/training/skill_extraction/test.jsonl`

## Limites
- Le référentiel est dominé par l’axe ingénieur IA et par les familles présentes dans les offres étudiées.
- Les correspondances sémantiques restent des propositions à valider.
- Les compétences sans preuve textuelle ne doivent pas être injectées comme labels positifs.
