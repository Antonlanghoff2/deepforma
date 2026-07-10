# Gestion des référentiels Deepforma

## Flux de travail

### 1. Import d'un référentiel

L'import se fait via l'interface admin :
- **URL** : `/admin/referential-import`
- **Action** : Uploader un PDF de référentiel de formation
- **Résultat** : 
  - Le PDF est analysé et les compétences sont extraites
  - Un fichier JSON est créé dans `data/referentials/imported/`
  - Les candidats d'annotation sont automatiquement générés
  - Le référentiel est disponible pour la comparaison de marché

### 2. Annotation des compétences

- **URL** : `/admin/referential-annotation`
- **Action** : Valider/rejeter les compétences extraites
- **Résultat** : Les compétences validées sont utilisées pour améliorer l'extraction

### 3. Comparaison de marché

- **URL** : `/admin/ai-certification-market-comparison`
- **Action** : Comparer le référentiel avec les offres d'emploi
- **Résultat** : Analyse de couverture du marché

## Commandes de maintenance

### Reset complet des référentiels

Efface tous les imports, annotations et la base de données :

```bash
# Mode simulation (recommandé avant exécution)
python3 scripts/reset_referentials.py --dry-run

# Reset complet
python3 scripts/reset_referentials.py --yes

# Reset en gardant les référentiels de base
python3 scripts/reset_referentials.py --yes --keep-base
```

### Générer les candidats d'annotation

Génère les candidats d'annotation à partir des référentiels importés :

```bash
python3 scripts/generate_annotation_candidates_from_imported.py
```

### Via l'interface admin

Deux boutons sont disponibles dans la page de comparaison de marché :
- **Générer candidats d'annotation** : Génère les candidats à partir des imports
- **Réinitialiser tous les référentiels** : Efface tout (avec option pour garder les bases)

## Structure des données

```
data/
├── referentials/
│   ├── imported/              # Référentiels importés (JSON)
│   ├── referential_imports.sqlite3  # Base de données des imports
│   ├── ai_engineer_certification_2025.json  # Référentiel de base
│   └── ...
├── annotation/
│   ├── referential_candidates.jsonl  # Candidats pour annotation
│   ├── referential_ner_candidates.jsonl
│   └── referential_multilabel_candidates.jsonl
└── ...
```

## Dépannage

### Problème : "0 compétence" affichée

Si un référentiel affiche "0 compétence" alors qu'il contient des sous-compétences :
1. Vérifier que le fichier JSON contient bien les clés `skills` ou `derived_skills`
2. Régénérer les candidats : `python3 scripts/generate_annotation_candidates_from_imported.py`
3. Redémarrer l'application

### Problème : Candidats d'annotation manquants

Si la page d'annotation est vide :
1. Vérifier que des référentiels sont importés
2. Générer les candidats : `python3 scripts/generate_annotation_candidates_from_imported.py`

### Problème : Base de données corrompue

Si la base de données est corrompue :
1. Reset complet : `python3 scripts/reset_referentials.py --yes --keep-base`
2. Réimporter les référentiels nécessaires

## Extraction de compétences

L'extracteur de compétences utilise maintenant un système amélioré :
- **Détection des sections** : OBJECTIFS, PROGRAMME, COMPÉTENCES VISÉES, PRÉREQUIS
- **Extraction des listes** : Puces, virgules, numéros
- **Lexique Data/IA** : Reconnaissance de 30+ compétences spécifiques
- **Normalisation** : Contextuelle (ex: "jointures" → "Jointures SQL")

Pour tester l'extraction :
```bash
python3 tests/test_training_skill_extractor.py
```
