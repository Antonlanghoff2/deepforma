#!/usr/bin/env python3
"""
Vérifie la cohérence du fichier ai_skill_taxonomy.json avec les labels du modèle.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "referentials" / "ai_skill_taxonomy.json"
MODEL_LABELS_PATH = PROJECT_ROOT / "models" / "multilabel_competences_v2" / "label_classes.json"

EXPECTED_LABELS = [
    "Automatisation",
    "Big Data",
    "Computer Vision",
    "Data Engineering",
    "Data Science",
    "Deep Learning",
    "Gestion de projet IA",
    "IA générative",
    "Machine Learning",
    "NLP",
    "No-code / Low-code",
    "Prompt Engineering",
    "Python pour l'IA",
    "RAG",
    "Reinforcement Learning",
    "Séries temporelles",
    "Visualisation",
    "Éthique de l'IA",
]


def check_taxonomy_exists():
    """Vérifie que le fichier de taxonomie existe."""
    if not TAXONOMY_PATH.exists():
        print(f"❌ Fichier de taxonomie introuvable: {TAXONOMY_PATH}")
        return False
    print(f"✅ Fichier de taxonomie trouvé: {TAXONOMY_PATH}")
    return True


def check_taxonomy_valid_json():
    """Vérifie que le fichier est un JSON valide."""
    try:
        with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON valide")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON invalide: {e}")
        return None


def check_taxonomy_structure(data):
    """Vérifie la structure du fichier de taxonomie."""
    required_fields = ['schema_version', 'taxonomy_id', 'title', 'labels']
    missing = [f for f in required_fields if f not in data]
    
    if missing:
        print(f"❌ Champs manquants: {', '.join(missing)}")
        return False
    
    print(f"✅ Structure valide (version: {data.get('schema_version')})")
    return True


def check_expected_labels(data):
    """Vérifie que les 18 labels attendus sont présents."""
    labels = data.get('labels', [])
    label_names = {label['label'] for label in labels}
    
    missing = [label for label in EXPECTED_LABELS if label not in label_names]
    
    if missing:
        print(f"❌ Labels manquants ({len(missing)}):")
        for label in missing:
            print(f"   - {label}")
        return False
    
    print(f"✅ Les {len(EXPECTED_LABELS)} labels attendus sont présents")
    return True


def check_no_duplicates(data):
    """Vérifie qu'il n'y a pas de doublons."""
    labels = data.get('labels', [])
    label_ids = [label['id'] for label in labels]
    label_names = [label['label'] for label in labels]
    
    duplicate_ids = [id for id in label_ids if label_ids.count(id) > 1]
    duplicate_names = [name for name in label_names if label_names.count(name) > 1]
    
    if duplicate_ids or duplicate_names:
        print(f"❌ Doublons détectés:")
        if duplicate_ids:
            print(f"   IDs: {set(duplicate_ids)}")
        if duplicate_names:
            print(f"   Noms: {set(duplicate_names)}")
        return False
    
    print(f"✅ Aucun doublon")
    return True


def check_consistency_with_model():
    """Vérifie la cohérence avec les labels du modèle."""
    if not MODEL_LABELS_PATH.exists():
        print(f"⚠️  Fichier de labels du modèle introuvable: {MODEL_LABELS_PATH}")
        print(f"   (Cette vérification est optionnelle)")
        return True
    
    try:
        with open(MODEL_LABELS_PATH, 'r', encoding='utf-8') as f:
            model_labels = json.load(f)
        
        with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
            taxonomy_data = json.load(f)
        
        taxonomy_labels = {label['label'] for label in taxonomy_data.get('labels', [])}
        model_label_set = set(model_labels)
        
        # Vérifier que tous les labels du modèle sont dans la taxonomie
        missing_in_taxonomy = model_label_set - taxonomy_labels
        
        if missing_in_taxonomy:
            print(f"⚠️  Labels du modèle absents de la taxonomie ({len(missing_in_taxonomy)}):")
            for label in sorted(missing_in_taxonomy):
                print(f"   - {label}")
            print(f"   (Ces labels peuvent être ajoutés ultérieurement)")
            return True  # Warning, pas d'erreur
        
        print(f"✅ Cohérence avec les {len(model_labels)} labels du modèle")
        return True
    
    except Exception as e:
        print(f"⚠️  Erreur lors de la vérification de cohérence: {e}")
        return True  # Warning, pas d'erreur


def main():
    print("=" * 80)
    print("VÉRIFICATION DE LA TAXONOMIE IA")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Vérification 1: existence
    if not check_taxonomy_exists():
        all_ok = False
        print()
        print("❌ Vérification échouée")
        sys.exit(1)
    
    # Vérification 2: JSON valide
    data = check_taxonomy_valid_json()
    if data is None:
        all_ok = False
        print()
        print("❌ Vérification échouée")
        sys.exit(1)
    
    print()
    
    # Vérification 3: structure
    if not check_taxonomy_structure(data):
        all_ok = False
    
    print()
    
    # Vérification 4: labels attendus
    if not check_expected_labels(data):
        all_ok = False
    
    print()
    
    # Vérification 5: pas de doublons
    if not check_no_duplicates(data):
        all_ok = False
    
    print()
    
    # Vérification 6: cohérence avec le modèle
    if not check_consistency_with_model():
        all_ok = False
    
    print()
    print("=" * 80)
    
    if all_ok:
        print("✅ Toutes les vérifications ont réussi")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ Certaines vérifications ont échoué")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
