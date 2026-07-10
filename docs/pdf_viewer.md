# Visualisation des PDF des référentiels

## Description

Cette fonctionnalité permet de visualiser les PDF originaux des référentiels importés directement dans l'interface d'annotation.

## Fonctionnalités

- **Visualisation inline** : Affichage du PDF dans le navigateur avec PDF.js
- **Navigation** : Boutons précédent/suivant pour naviguer entre les pages
- **Zoom** : Contrôles de zoom (+/-) pour ajuster l'affichage
- **Navigation clavier** : 
  - Flèches gauche/droite pour changer de page
  - +/- pour zoomer/dézoomer
- **Mapping automatique** : Association automatique entre les référentiels importés et les PDF originaux via le hash SHA256

## Utilisation

### Depuis la page d'annotation

1. Accédez à `/admin/referential-annotation`
2. Dans la liste des référentiels, cliquez sur le bouton "📄 Voir le PDF" à côté du référentiel souhaité
3. Le PDF s'ouvre dans une nouvelle page avec le visualiseur

### Depuis l'URL directe

Accédez directement à :
```
/admin/referential-viewer?referential_id=<document_id>
```

## Architecture

### Routes Flask

- `GET /admin/referential-viewer` : Page de visualisation
- `GET /admin/referential-pdf/<filename>` : Sert les fichiers PDF

### Fichiers

- `src/web_app.py` : Routes Flask
- `templates/admin_referential_viewer.html` : Template du visualiseur
- `scripts/map_referentials_to_pdfs.py` : Script de mapping
- `tests/test_pdf_viewer.py` : Tests unitaires

### Mapping PDF

Le mapping entre les référentiels importés et les PDF originaux se fait via le hash SHA256 :

1. Lors de l'import d'un PDF, son hash SHA256 est calculé et stocké dans le JSON
2. Pour trouver le PDF original, on calcule le hash de tous les PDF dans `data/raw/referentiel/`
3. On compare les hashes pour trouver le PDF correspondant

## Structure des fichiers

```
data/
├── raw/
│   └── referentiel/          # PDF originaux
│       ├── 01_DataScientest_...pdf
│       ├── 02_DataScientest_...pdf
│       └── ...
├── referentials/
│   └── imported/             # JSON des référentiels importés
│       ├── tmp1234.pdf.json
│       └── ...
└── annotation/
    └── referential_candidates.jsonl

src/
├── web_app.py                # Routes Flask
└── referential_import/
    └── import_service.py     # Service d'import

templates/
└── admin_referential_viewer.html  # Template du visualiseur

scripts/
└── map_referentials_to_pdfs.py    # Script de mapping

tests/
└── test_pdf_viewer.py             # Tests
```

## Commandes utiles

### Lister tous les mappings

```bash
python3 scripts/map_referentials_to_pdfs.py
```

### Exécuter les tests

```bash
.venv/bin/python -m pytest tests/test_pdf_viewer.py -v
```

### Démarrer l'application

```bash
.venv/bin/python -m src.web_app
```

## Dépendances

- **PDF.js** (CDN) : Bibliothèque JavaScript pour l'affichage des PDF dans le navigateur
  - Version : 3.11.174
  - URL : https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js

## Limitations

- Les PDF doivent être présents dans `data/raw/referentiel/`
- Le mapping se fait par hash SHA256, donc le PDF original ne doit pas avoir été modifié
- La visualisation nécessite une connexion Internet pour charger PDF.js depuis le CDN

## Améliorations futures

- [ ] Support du mode hors-ligne (PDF.js en local)
- [ ] Recherche de texte dans le PDF
- [ ] Annotations directes sur le PDF
- [ ] Surlignage des compétences détectées dans le PDF
- [ ] Export des annotations

## Dépannage

### Le PDF ne s'affiche pas

1. Vérifier que le PDF existe dans `data/raw/referentiel/`
2. Vérifier que le hash SHA256 correspond (utiliser `scripts/map_referentials_to_pdfs.py`)
3. Vérifier les logs du serveur Flask pour les erreurs

### Erreur "PDF introuvable"

Cela signifie que le PDF original n'a pas été trouvé dans `data/raw/referentiel/`. 
Solutions :
- Copier le PDF dans le répertoire `data/raw/referentiel/`
- Vérifier que le nom du fichier n'a pas été modifié

### Le visualiseur ne charge pas

Vérifier que vous avez une connexion Internet pour charger PDF.js depuis le CDN.
