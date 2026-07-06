# Extraction des compétences IA du référentiel de certification

## Source

Le référentiel est construit à partir du PDF :

`data/raw/Referentiel de certification - Ingenieur en intelligence artificelle janvier 2025.pdf`

Le script de construction produit :

- `data/referentials/ai_engineer_certification_2025.json`
- `data/referentials/ai_engineer_certification_2025.csv`
- `data/referentials/ai_engineer_certification_2025.metadata.json`

## Distinction activité / compétence / critère

- `BLOC X` : famille de bloc du référentiel.
- `A1`, `A2`, ... : activité rattachée à un bloc.
- `A1-C1`, `A2-C3`, ... : compétence officielle.
- `A1-C1-E1`, ... : critère d'évaluation, toujours exclu de l'extraction métier.

Le chargeur conserve uniquement les compétences du référentiel de compétences.

## Construction du JSON

Le référentiel JSON contient :

- `referential_id`
- `title`
- `version`
- `skills`
- `metadata`

Chaque compétence contient :

- `id`
- `block`
- `activity`
- `code`
- `label`
- `official_description`
- `normalized_label`
- `aliases`
- `source_page`
- `active`

## Règles d'extraction

- utiliser le titre fourni s'il existe ;
- sinon le chercher dans les premières lignes ;
- sinon retourner `null` ;
- ne jamais inventer un titre ;
- extraire uniquement les compétences explicitement présentes, leurs alias prudents ou une formulation implicite forte ;
- fournir une preuve textuelle `evidence` ;
- ne pas retourner les modalités, critères, mises en situation, études de cas, jeux de rôle, conditions pratiques ou textes marketing.

## Activation

Variables d'environnement :

- `AI_CERTIFICATION_EXTRACTION_ENABLED=true`
- `AI_CERTIFICATION_REFERENTIAL_PATH=data/referentials/ai_engineer_certification_2025.json`
- `AI_CERTIFICATION_EXTRACTION_DRY_RUN=true`

Quand le mode est activé et que le dry-run est désactivé, le pipeline met à jour uniquement :

- `title`
- `competences`

Aucune autre colonne d'offre n'est modifiée.

## Colonne `competences`

La sortie stockée pour une offre contient uniquement :

- `title`
- `competences`

`competences` est un tableau JSON d'objets :

```json
[
  {
    "referential_id": "B2-A2-C3",
    "code": "A2-C3",
    "libelle": "Préparer le texte pour l’apprentissage",
    "libelle_officiel": "...",
    "evidence": "préparer le texte...",
    "confidence": 0.91,
    "match_type": "semantic"
  }
]
```

Si rien n'est détecté, la colonne vaut `[]`.

## Seuils

- `AI_CERT_SKILL_EXACT_THRESHOLD=1.0`
- `AI_CERT_SKILL_ALIAS_THRESHOLD=0.92`
- `AI_CERT_SKILL_SEMANTIC_THRESHOLD=0.72`
- `AI_CERT_SKILL_IMPLICIT_THRESHOLD=0.80`

## Dry-run

```bash
python scripts/extract_ai_certification_skills.py   --input data/continual_learning/continual_learning.sqlite3   --referential data/referentials/ai_engineer_certification_2025.json   --batch-size 100   --dry-run
```

Le mode dry-run affiche :

- le nombre d'offres analysées ;
- le nombre d'offres avec compétences ;
- la moyenne de compétences par offre ;
- les compétences les plus détectées ;
- des exemples avec preuve ;
- le nombre d'offres sans compétence ;
- aucune modification en base.

## Écriture

```bash
python scripts/extract_ai_certification_skills.py   --input data/continual_learning/continual_learning.sqlite3   --referential data/referentials/ai_engineer_certification_2025.json   --batch-size 100   --write
```

## Limites

Le référentiel couvre principalement le métier d'ingénieur en intelligence artificielle et les compétences de ses blocs de certification. Il ne remplace pas une taxonomie métier générale.
