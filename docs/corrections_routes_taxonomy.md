# Corrections Deepforma - Routes et Taxonomie IA

## Problèmes résolus

### 1. Route `/referential/import` retournait 405 (Method Not Allowed)

**Cause** : La route `/referential/import` n'acceptait que POST, pas GET.

**Solution** : Ajout d'une route GET qui redirige vers `/admin/referential-import`.

**Fichier modifié** : `src/web_app.py`

```python
@app.get('/referential/import')
def referential_import_get():
    """Redirige vers la page admin d'import de référentiel."""
    return redirect(url_for('admin_referential_import'))
```

**Résultat** : 
- GET `/referential/import` → redirige vers `/admin/referential-import`
- POST `/referential/import` → traite l'upload PDF (comportement inchangé)

---

### 2. Warning "Taxonomie non trouvée" au démarrage

**Cause** : Le fichier `data/referentials/ai_skill_taxonomy.json` n'existait pas.

**Solution** : 
1. Création du fichier de taxonomie avec les 18 labels IA
2. Ajout d'un fallback automatique si le fichier est absent
3. Support de deux formats de taxonomie (ancien et nouveau)

**Fichiers créés/modifiés** :
- `data/referentials/ai_skill_taxonomy.json` (créé)
- `src/inference/deepforma_predictor.py` (modifié)

**Résultat** :
- Plus de warning bloquant
- Fallback propre si taxonomie absente
- Support backward-compatible

---

### 3. Outils de validation manquants

**Solution** : Ajout de scripts et cibles Makefile pour valider la configuration.

**Fichiers créés** :
- `scripts/list_routes.py` : Liste toutes les routes Flask
- `scripts/check_ai_taxonomy.py` : Vérifie la taxonomie IA

**Cibles Makefile ajoutées** :
- `make list-routes` : Affiche toutes les routes
- `make check-ai-taxonomy` : Vérifie la taxonomie

---

## Taxonomie IA

### Structure du fichier

```json
{
  "schema_version": "1.0",
  "taxonomy_id": "ai_skill_taxonomy",
  "title": "Taxonomie IA Deepforma",
  "labels": [
    {
      "id": "machine_learning",
      "label": "Machine Learning",
      "description": "...",
      "aliases": [...],
      "parent_id": null
    }
  ]
}
```

### Les 18 labels

1. Automatisation
2. Big Data
3. Computer Vision
4. Data Engineering
5. Data Science
6. Deep Learning
7. Gestion de projet IA
8. IA générative
9. Machine Learning
10. NLP
11. No-code / Low-code
12. Prompt Engineering
13. Python pour l'IA
14. RAG
15. Reinforcement Learning
16. Séries temporelles
17. Visualisation
18. Éthique de l'IA

---

## Routes Flask

### Routes principales

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Page d'accueil |
| GET | `/referential/import` | Redirige vers admin |
| POST | `/referential/import` | Upload PDF |
| GET | `/admin/referential-import` | Page admin import |
| POST | `/admin/referential-import` | Traite l'import |
| GET | `/admin/ai-certification-market-comparison` | Comparaison marché |
| GET | `/admin/referential-annotation` | Annotation |
| GET | `/health` | Health check |

### Commandes utiles

```bash
# Lister toutes les routes
make list-routes

# Vérifier la taxonomie
make check-ai-taxonomy

# Démarrer l'application
python -m src.web_app

# Tester les routes
curl http://127.0.0.1:5000/
curl http://127.0.0.1:5000/referential/import  # Redirige
curl http://127.0.0.1:5000/admin/referential-import  # Page admin
```

---

## Tests

### Tests ajoutés

**Fichier** : `tests/test_taxonomy_and_routes.py`

- `test_taxonomy_file_exists` : Vérifie l'existence du fichier
- `test_taxonomy_valid_json` : Vérifie le format JSON
- `test_taxonomy_has_required_fields` : Vérifie la structure
- `test_taxonomy_has_18_labels` : Vérifie le nombre de labels
- `test_taxonomy_expected_labels` : Vérifie les labels attendus
- `test_taxonomy_no_duplicates` : Vérifie l'absence de doublons
- `test_get_referential_import_redirects` : Vérifie la redirection
- `test_get_admin_referential_import_accessible` : Vérifie l'accès admin
- `test_post_referential_import_accepts_pdf` : Vérifie POST
- `test_routes_list_available` : Vérifie l'enregistrement des routes
- `test_fallback_taxonomy_structure` : Vérifie le fallback

**Résultat** : 11 tests passent ✅

---

## Fallback de taxonomie

Si le fichier `ai_skill_taxonomy.json` est absent, le système génère automatiquement une taxonomie minimale à partir des labels du modèle :

```python
def _build_fallback_taxonomy(self) -> dict:
    """Construit une taxonomie minimale à partir des labels du modèle."""
    skills = []
    for label in self.labels:
        skills.append({
            'id': label.lower().replace(' ', '_'),
            'label': label,
        })
    
    return {
        'schema_version': '1.0',
        'taxonomy_id': 'fallback_from_checkpoint',
        'title': 'Taxonomie IA Deepforma (fallback)',
        'families': [{
            'id': 'ia',
            'label': 'Intelligence Artificielle',
            'skills': skills,
        }]
    }
```

**Avantages** :
- L'application démarre même sans fichier de taxonomie
- Pas de dépendance bloquante
- Message de log informatif

---

## Validation

### Commandes à exécuter

```bash
# 1. Compiler les fichiers modifiés
python -m py_compile src/web_app.py
python -m py_compile src/inference/deepforma_predictor.py

# 2. Vérifier la taxonomie
make check-ai-taxonomy
# ou
python scripts/check_ai_taxonomy.py

# 3. Lister les routes
make list-routes
# ou
python scripts/list_routes.py

# 4. Exécuter les tests
pytest tests/test_taxonomy_and_routes.py -v

# 5. Démarrer l'application
python -m src.web_app

# 6. Tester les routes
curl -i http://127.0.0.1:5000/
curl -i http://127.0.0.1:5000/referential/import  # Doit rediriger
curl -i http://127.0.0.1:5000/admin/referential-import  # Page admin
```

### Critères d'acceptation

✅ Plus de warning bloquant sur ai_skill_taxonomy.json  
✅ Fallback propre si taxonomie absente  
✅ Les routes d'import sont claires  
✅ GET /referential/import ne renvoie plus un 405 incompréhensible  
✅ L'import PDF utilise le schéma canonique  
✅ Les tests passent  

---

## Fichiers modifiés/créés

### Créés
- `data/referentials/ai_skill_taxonomy.json`
- `scripts/list_routes.py`
- `scripts/check_ai_taxonomy.py`
- `tests/test_taxonomy_and_routes.py`

### Modifiés
- `src/web_app.py` (ajout route GET /referential/import)
- `src/inference/deepforma_predictor.py` (fallback taxonomie)
- `Makefile` (ajout cibles check-ai-taxonomy et list-routes)

---

## Prochaines étapes

1. **Intégration continue** : Ajouter `make check-ai-taxonomy` dans le pipeline CI
2. **Documentation** : Mettre à jour la documentation utilisateur avec les nouvelles routes
3. **Monitoring** : Ajouter des métriques sur l'utilisation des routes
4. **Tests E2E** : Ajouter des tests de bout en bout pour l'import PDF

---

## Notes

- Le fichier `ai_skill_taxonomy.json` utilise le format "labels" (nouveau format)
- Le code supporte aussi l'ancien format "families/skills" pour backward compatibility
- Le fallback génère une taxonomie avec une seule famille "IA" contenant tous les labels
- Les routes admin nécessitent une authentification (non testée dans les tests unitaires)
