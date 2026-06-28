# DeepForma

Pipeline de nettoyage et d'entraînement pour les formations IA et généralistes.

## Fichiers sources

- `Dataset_V7_Anton_CSV - Dataset_V7_Anton_CSV.csv.csv` : dataset IA existant.
- `Dataset_Generaliste_CPF_V1.xlsx` : nouveau dataset généraliste.
- `entrainement_camembert_competences_ia.ipynb` : notebook historique conservé intact.

## Nettoyage effectué

Le script `scripts/clean_and_merge_datasets.py` applique les règles suivantes :

- suppression des lignes entièrement vides dans les sources exploitées ;
- normalisation Unicode et espaces ;
- normalisation des retours ligne ;
- correction des valeurs nulles textuelles ;
- normalisation des intitulés, niveaux, modalités, durées et codes ROME ;
- suppression des doublons exacts ;
- détection heuristique des doublons proches via groupes de titres normalisés ;
- construction d'un `formation_id` stable et d'un `formation_group_id` pour les séparations sans fuite ;
- conservation des textes utiles dans `texte_modele` ;
- ajout du fichier source et du numéro de ligne source ;
- classification conservatrice en `ia_confirmee`, `non_ia_confirmee`, `a_verifier`.

## Schéma final

Colonnes principales générées :

- `formation_id`
- `intitule`
- `description`
- `objectifs`
- `programme`
- `public_cible`
- `prerequis`
- `niveau`
- `modalite`
- `duree`
- `certification`
- `codes_rome`
- `organisme`
- `source_dataset`
- `texte_modele`
- `competences_ia`
- `competences_ia_suggerees`
- `est_lie_ia`
- `statut_annotation`
- `source_file`
- `source_row`
- `formation_group_id`

Des colonnes techniques supplémentaires sont conservées pour la traçabilité.

## Catégories

- `ia_confirmee` : formation clairement liée à l'IA avec compétences annotées.
- `non_ia_confirmee` : formation clairement hors IA utilisable comme négatif.
- `a_verifier` : formation ambiguë ou potentiellement liée à l'IA sans annotation fiable.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Nettoyage et fusion

```bash
python scripts/clean_and_merge_datasets.py
```

Sorties produites dans `data/processed` :

- `dataset_formations_nettoye.csv`
- `dataset_entrainement.csv`
- `dataset_a_verifier.csv`
- `dataset_formations_nettoye.xlsx`

## Notebook v2

Ouvrir et exécuter `entrainement_camembert_competences_ia_v2.ipynb`.

Architecture :

1. modèle binaire `IA / non-IA` sur `texte_modele` ;
2. modèle multi-étiquette sur `competences_ia`, entraîné uniquement sur les formations IA confirmées.

Modèle de base recommandé : `camembert-base`.

## Fonction de prédiction

Le notebook v2 expose `predict_formation(formation_data)` qui :

- calcule d'abord la probabilité IA ;
- ne lance le modèle de compétences que si la probabilité dépasse le seuil retenu ;
- renvoie une structure JSON-like avec la probabilité IA et les compétences prédites.

## Limites

- Les formations ambiguës sont conservativement envoyées dans `a_verifier`.
- L'absence d'annotation n'est jamais interprétée automatiquement comme un négatif.
- La détection de doublons proches reste heuristique.
- Le notebook n'entraîne pas un modèle depuis zéro : il affine un modèle français préentraîné.

## Contrôles qualité

Le pipeline vérifie notamment :

- absence de doublons exacts dans `dataset_entrainement.csv` ;
- absence d'entrée vide dans `texte_modele` ;
- absence de fuite des cibles dans `texte_modele` ;
- absence de compétences IA vides pour les exemples IA confirmés ;
- absence de compétences IA sur les exemples non-IA confirmés.

## Résultats observés sur l'exécution courante

- Lignes avant nettoyage : 1260
- Lignes après nettoyage : 1208
- Doublons exacts supprimés : 52
- IA confirmées : 334
- Non-IA confirmées : 441
- À vérifier : 433
- Compétences distinctes observées : 18

## Application web Flask

Lancement local:

```bash
cd /home/bibi/deepforma
source .venv/bin/activate
pip install flask
python -m src.web_app
```

Puis ouvrir:

```text
http://127.0.0.1:5000
```

Test API avec `curl`:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze   -H 'Content-Type: application/json'   -d '{
    "programme": "Programme de formation en Python, data et IA",
    "departement": "93",
    "keywords": "python,data",
    "threshold": 0.35,
    "model_only": true
  }'
```

Avertissement:

> Résultat expérimental. Le modèle doit encore être validé avant utilisation opérationnelle.
