#!/usr/bin/env python3
"""
Script de reset des référentiels.
Efface tous les imports et annotations pour repartir de zéro.
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def reset_referentials(dry_run: bool = False, keep_base: bool = False) -> dict[str, int]:
    """
    Efface les référentiels importés et les annotations.
    
    Args:
        dry_run: Si True, affiche ce qui serait supprimé sans supprimer
        keep_base: Si True, garde les référentiels de base (ai_engineer_certification_2025.json, etc.)
    
    Returns:
        Dict avec le nombre de fichiers supprimés par catégorie
    """
    stats = {
        "imported_files": 0,
        "annotation_files": 0,
        "database": 0,
    }
    
    # 1. Effacer les fichiers importés
    imported_dir = PROJECT_ROOT / "data" / "referentials" / "imported"
    if imported_dir.exists():
        for file_path in imported_dir.glob("*.json"):
            if dry_run:
                print(f"[DRY RUN] Supprimerait: {file_path}")
            else:
                file_path.unlink()
                print(f"Supprimé: {file_path}")
            stats["imported_files"] += 1
    
    # 2. Effacer les fichiers d'annotation
    annotation_dir = PROJECT_ROOT / "data" / "annotation"
    if annotation_dir.exists():
        for file_path in annotation_dir.glob("*.jsonl"):
            if dry_run:
                print(f"[DRY RUN] Supprimerait: {file_path}")
            else:
                file_path.unlink()
                print(f"Supprimé: {file_path}")
            stats["annotation_files"] += 1
    
    # 3. Effacer la base de données des imports
    db_path = PROJECT_ROOT / "data" / "referentials" / "referential_imports.sqlite3"
    if db_path.exists():
        if dry_run:
            print(f"[DRY RUN] Supprimerait: {db_path}")
        else:
            db_path.unlink()
            print(f"Supprimé: {db_path}")
        stats["database"] = 1
    
    # 4. Optionnel : effacer les fichiers de base
    if not keep_base:
        base_files = [
            PROJECT_ROOT / "data" / "referentials" / "ai_engineer_certification_2025.json",
            PROJECT_ROOT / "data" / "referentials" / "ai_engineer_certification_2025.csv",
            PROJECT_ROOT / "data" / "referentials" / "ai_engineer_certification_2025.metadata.json",
        ]
        for file_path in base_files:
            if file_path.exists():
                if dry_run:
                    print(f"[DRY RUN] Supprimerait: {file_path}")
                else:
                    file_path.unlink()
                    print(f"Supprimé: {file_path}")
                stats["imported_files"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Reset des référentiels Deepforma")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait supprimé sans supprimer",
    )
    parser.add_argument(
        "--keep-base",
        action="store_true",
        help="Garde les référentiels de base (ai_engineer_certification_2025.json, etc.)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirme la suppression sans demander",
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.yes:
        print("⚠️  ATTENTION : Cette opération va effacer tous les référentiels importés et les annotations.")
        print("   Utilisez --dry-run pour voir ce qui serait supprimé.")
        response = input("Êtes-vous sûr de vouloir continuer ? (oui/non): ")
        if response.lower() not in ["oui", "yes", "y"]:
            print("Annulé.")
            sys.exit(0)
    
    print("\n🔄 Reset des référentiels...")
    if args.dry_run:
        print("   Mode DRY RUN (aucune suppression effective)\n")
    
    stats = reset_referentials(dry_run=args.dry_run, keep_base=args.keep_base)
    
    print("\n📊 Résumé :")
    print(f"   Fichiers importés supprimés : {stats['imported_files']}")
    print(f"   Fichiers d'annotation supprimés : {stats['annotation_files']}")
    print(f"   Base de données supprimée : {stats['database']}")
    
    if args.dry_run:
        print("\n💡 Pour effectuer la suppression réelle, relancez sans --dry-run")
    else:
        print("\n✅ Reset terminé. Vous pouvez maintenant importer de nouveaux référentiels.")


if __name__ == "__main__":
    main()
