# Améliorations de la page d'annotation des référentiels

## Vue d'ensemble

La page d'annotation (`/admin/referential-annotation`) a été améliorée pour permettre l'ajout et la suppression de compétences/entités, offrant ainsi un contrôle total sur le processus d'annotation.

## Fonctionnalités ajoutées

### 1. Ajout de compétences/entités

#### Pour les référentiels de type "candidates"
- **Formulaire d'ajout** en bas de la liste des compétences extraites
- Champs disponibles :
  - Libellé (obligatoire)
  - Catégorie (SKILL, TOOL, METHOD, DOMAIN, SOFT_SKILL, OTHER)
  - Nom canonique (optionnel)
- Les compétences ajoutées manuellement sont automatiquement marquées comme "approved"
- Le compteur `skills_count` est mis à jour automatiquement

#### Pour les référentiels de type "ner"
- **Formulaire d'ajout** existant (déjà présent)
- Champs disponibles :
  - Texte (obligatoire)
  - Type (SKILL, TOOL, METHOD, DOMAIN, SOFT_SKILL, OTHER)
  - Position début/fin
  - Nom canonique
  - Référentiel canonique
- Page (optionnel)

### 2. Suppression de compétences/entités

#### Bouton "🗑️ Supprimer"
- Ajouté pour chaque compétence/entité
- Confirmation avant suppression (dialogue JavaScript)
- Suppression définitive de l'enregistrement
- Mise à jour automatique des compteurs

### 3. Actions en masse

Les actions suivantes sont disponibles :
- **Approuver tout** : Approuve toutes les compétences/entités
- **Rejeter tout** : Rejette toutes les compétences/entités
- **Valider le document** : Marque le document comme validé

## Modifications techniques

### Backend (`src/web_app.py`)

#### Fonction `_update_annotation_status`
Refactorisée pour gérer les trois types de référentiels :

```python
def _update_annotation_status(record: dict[str, Any], *, action: str, form: Any) -> None:
    kind = clean_text(record.get('kind') or 'ner')
    
    # Gestion des compétences candidates (kind == 'candidates')
    if kind == 'candidates':
        # Actions : approve_entity, reject_entity, delete_entity, add_entity, validate_document
        ...
    
    # Gestion des entités NER (kind == 'ner')
    elif kind == 'ner':
        # Actions : approve_entity, reject_entity, delete_entity, add_entity, validate_document
        ...
    
    # Gestion multilabel (kind == 'multilabel')
    else:
        # Actions : approve_multilabel, reject_multilabel, validate_document
        ...
```

#### Nouvelles actions supportées

**Pour "candidates" et "ner" :**
- `add_entity` : Ajoute une compétence/entité
- `delete_entity` : Supprime une compétence/entité
- `approve_entity` : Approuve une compétence/entité
- `reject_entity` : Rejette une compétence/entité
- `approve_all_entities` : Approuve tout
- `reject_all_entities` : Rejette tout
- `validate_document` : Valide le document

### Frontend (`templates/admin_referential_annotation.html`)

#### Ajout du bouton "Supprimer"
```html
<button type="submit" name="action" value="delete_entity" class="danger" 
        onclick="return confirm('Êtes-vous sûr de vouloir supprimer cette compétence ?')">
    🗑️ Supprimer
</button>
```

#### Ajout du formulaire pour "candidates"
```html
<form method="post" action="{{ url_for('admin_referential_annotation_action') }}" class="panel">
    <h4>Ajouter une compétence</h4>
    <div class="grid cols-2">
        <label>Libellé <input type="text" name="text" required></label>
        <label>Catégorie <select name="entity_label">...</select></label>
    </div>
    <div class="grid cols-2">
        <label>Nom canonique <input type="text" name="canonical_name"></label>
    </div>
    <div class="actions">
        <button type="submit" name="action" value="add_entity" class="success">
            ➕ Ajouter
        </button>
    </div>
</form>
```

## Tests

### Fichier de tests
`tests/test_annotation_add_delete.py`

### Couverture
8 tests unitaires couvrant :
1. ✅ Ajout de compétence à un candidat
2. ✅ Suppression de compétence d'un candidat
3. ✅ Ajout d'entité NER
4. ✅ Suppression d'entité NER
5. ✅ Approbation d'une compétence
6. ✅ Rejet d'une compétence
7. ✅ Approbation de toutes les compétences
8. ✅ Rejet de toutes les compétences

### Exécution
```bash
.venv/bin/python -m pytest tests/test_annotation_add_delete.py -v
```

## Structure des données

### Compétences candidates
```json
{
  "record_id": "xxx",
  "kind": "candidates",
  "skills": [
    {
      "skill_id": "skill-001",
      "label": "Python",
      "category": "tool",
      "canonical_label": "Python Programming",
      "confidence": 0.9,
      "source": "extraction",
      "status": "approved"
    }
  ],
  "skills_count": 1
}
```

### Entités NER
```json
{
  "record_id": "xxx",
  "kind": "ner",
  "entities": [
    {
      "entity_id": "entity-001",
      "text": "Python",
      "start": 0,
      "end": 6,
      "predicted_label": "TOOL",
      "approved_label": "TOOL",
      "canonical_name": "Python",
      "status": "approved"
    }
  ]
}
```

## Workflow d'utilisation

### 1. Accéder à la page d'annotation
```
http://127.0.0.1:5000/admin/referential-annotation
```

### 2. Sélectionner un référentiel
- Choisir dans la liste latérale
- Le référentiel s'affiche dans la zone principale

### 3. Annoter les compétences
- **Approuver** : Cliquer sur "Accepter" pour une compétence
- **Rejeter** : Cliquer sur "Rejeter" pour une compétence
- **Modifier** : Changer la catégorie ou le nom canonique
- **Supprimer** : Cliquer sur "🗑️ Supprimer" (avec confirmation)
- **Ajouter** : Utiliser le formulaire en bas de page

### 4. Actions en masse
- **Approuver tout** : Approuve toutes les compétences d'un coup
- **Rejeter tout** : Rejette toutes les compétences d'un coup
- **Valider le document** : Marque le document comme validé

### 5. Navigation
- **Flèche gauche** : Référentiel précédent
- **Flèche droite** : Référentiel suivant

## Améliorations UX

### Confirmation avant suppression
Évite les suppressions accidentelles avec un dialogue de confirmation.

### Mise à jour automatique des compteurs
Le nombre de compétences est mis à jour automatiquement après ajout/suppression.

### Feedback visuel
- Boutons colorés (vert pour approuver, rouge pour rejeter/supprimer)
- Badges de statut (pending, approved, rejected)
- Emojis pour une meilleure identification (🗑️, ➕)

### Formulaire intuitif
- Champs obligatoires marqués
- Placeholders explicites
- Layout en grille responsive

## Compatibilité

### Types de référentiels supportés
- ✅ `candidates` : Compétences extraites de PDF
- ✅ `ner` : Entités nommées détectées
- ✅ `multilabel` : Classification multi-labels

### Rétrocompatibilité
- Les anciens enregistrements continuent de fonctionner
- Pas de migration de données nécessaire
- Les nouvelles actions sont optionnelles

## Limitations connues

### Pas d'annulation
- La suppression est définitive
- Pas de fonction "undo" pour le moment
- Recommandation : exporter avant modification massive

### Pas d'édition inline
- Impossible de modifier le libellé d'une compétence existante
- Workaround : supprimer et recréer

### Pas de recherche/filtre
- Pas de recherche dans la liste des compétences
- Pas de filtrage par statut/catégorie

## Améliorations futures

### Priorité haute
- [ ] Édition inline des compétences
- [ ] Fonction d'annulation (undo)
- [ ] Recherche et filtrage dans la liste

### Priorité moyenne
- [ ] Import/export en masse
- [ ] Historique des modifications
- [ ] Collaboration multi-utilisateurs

### Priorité basse
- [ ] Suggestions automatiques
- [ ] Détection de doublons
- [ ] Statistiques d'annotation

## Dépannage

### Le bouton "Supprimer" ne fonctionne pas
- Vérifier que JavaScript est activé
- Vérifier les logs du navigateur pour les erreurs
- Vérifier que le CSRF token est présent

### La compétence ajoutée n'apparaît pas
- Recharger la page (F5)
- Vérifier les logs serveur
- Vérifier que le formulaire est bien soumis

### Erreur 401 Unauthorized
- Se connecter en tant qu'admin
- Vérifier les variables d'environnement :
  - `DEEPFORMA_ADMIN_USER`
  - `DEEPFORMA_ADMIN_PASSWORD`

## Métriques

### Avant
- ❌ Pas d'ajout de compétences
- ❌ Pas de suppression
- ❌ Annotation limitée aux compétences extraites

### Après
- ✅ Ajout illimité de compétences
- ✅ Suppression avec confirmation
- ✅ Contrôle total sur l'annotation
- ✅ 8 tests unitaires
- ✅ Documentation complète

## Conclusion

Les améliorations apportées à la page d'annotation offrent un contrôle total sur le processus d'annotation des référentiels. Les utilisateurs peuvent maintenant :
- Ajouter des compétences manquantes
- Supprimer les faux positifs
- Approuver/rejeter en masse
- Valider les documents

Ces fonctionnalités sont essentielles pour garantir la qualité des référentiels et accélérer le processus d'annotation.
