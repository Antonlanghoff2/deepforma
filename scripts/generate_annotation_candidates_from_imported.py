#!/usr/bin/env python3
"""
Génère les candidats d'annotation à partir des référentiels importés.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from referential_learning.store import AnnotationStore


def generate_annotation_candidates_from_imported(
    imported_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, int]:
    """
    Génère les candidats d'annotation à partir des JSON importés.
    
    Args:
        imported_dir: Répertoire des JSON importés (par défaut: data/referentials/imported)
        output_path: Chemin de sortie (par défaut: data/annotation/referential_candidates.jsonl)
    
    Returns:
        Stats de génération
    """
    if imported_dir is None:
        imported_dir = PROJECT_ROOT / "data" / "referentials" / "imported"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "annotation" / "referential_candidates.jsonl"
    
    stats = {
        "files_processed": 0,
        "candidates_generated": 0,
    }
    
    if not imported_dir.exists():
        print(f"⚠️  Répertoire introuvable: {imported_dir}")
        return stats
    
    # Charger les candidats existants
    store = AnnotationStore(output_path)
    existing_records = store.load()
    existing_ids = {r.get("document_id") for r in existing_records if r.get("document_id")}
    
    new_records = []
    
    # Parcourir les JSON importés
    for json_path in sorted(imported_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            document_id = data.get("document_id") or data.get("document", {}).get("id")
            if not document_id:
                print(f"⚠️  Pas de document_id dans {json_path.name}")
                continue
            
            # Vérifier si déjà présent
            if document_id in existing_ids:
                print(f"⏭️  Déjà présent: {json_path.name}")
                continue
            
            # Extraire les informations du document
            document = data.get("document", {})
            competencies = data.get("competencies", [])
            derived_skills = data.get("derived_skills", [])
            blocks = data.get("blocks", [])
            
            # Créer un candidat d'annotation
            candidate = {
                "document_id": document_id,
                "source_file": document.get("file_name", json_path.name),
                "sha256": document.get("sha256", ""),
                "page_count": document.get("page_count", 0),
                "title": document.get("title", ""),
                "provider": document.get("provider", ""),
                "reference": document.get("reference", ""),
                "duration_hours": document.get("duration_hours"),
                "cpf_eligible": document.get("cpf_eligible"),
                "source_type": document.get("source_type", "pdf"),
                "text_extraction_method": document.get("text_extraction_method", ""),
                "review_status": document.get("review_status", "pending"),
                "collected_at": document.get("collected_at", ""),
                "validated_at": document.get("validated_at", ""),
                "validated_by": document.get("validated_by", ""),
                "notes": document.get("notes", ""),
                "competencies_count": len(competencies),
                "derived_skills_count": len(derived_skills),
                "blocks_count": len(blocks),
                "status": "pending",
            }
            
            # Extraire les compétences et derived_skills pour annotation
            candidate["competencies"] = [
                {
                    "code": c.get("code", ""),
                    "label": c.get("official_label", ""),
                    "block_code": c.get("block_code", ""),
                    "activity_code": c.get("activity_code", ""),
                    "page_start": c.get("page_start", 0),
                    "page_end": c.get("page_end", 0),
                    "confidence": c.get("confidence", 0.0),
                    "derived_skills": [
                        {
                            "label": ds.get("label", ""),
                            "canonical_label": ds.get("canonical_label", ""),
                            "category": ds.get("category", ""),
                            "confidence": ds.get("confidence", 0.0),
                        }
                        for ds in c.get("derived_skills", [])
                    ],
                }
                for c in competencies
            ]
            
            candidate["derived_skills"] = [
                {
                    "label": ds.get("label", ""),
                    "canonical_label": ds.get("canonical_label", ""),
                    "category": ds.get("category", ""),
                    "confidence": ds.get("confidence", 0.0),
                    "source_code": ds.get("source_code", ""),
                }
                for ds in derived_skills
            ]
            
            # Combiner toutes les compétences dans un seul champ 'skills'
            all_skills = []
            seen_labels = set()
            
            def normalize_label(label):
                """Normaliser un label pour la comparaison."""
                if not label:
                    return ""
                # Supprimer les espaces en début/fin
                label = label.strip()
                # Mettre en minuscules
                label = label.lower()
                # Supprimer les caractères spéciaux
                label = ''.join(c for c in label if c.isalnum() or c.isspace())
                # Normaliser les espaces multiples
                label = ' '.join(label.split())
                return label
            
            def is_valid_skill_label(label):
                """Vérifier si un label de compétence est valide."""
                if not label:
                    return False
                # Rejeter les labels trop courts
                if len(label) < 3:
                    return False
                # Rejeter les labels trop longs
                if len(label) > 100:
                    return False
                # Rejeter les labels avec des caractères suspects
                if '→' in label or '←' in label:
                    return False
                # Rejeter les labels qui commencent par des mots-clés suspects
                label_lower = label.lower()
                if label_lower.startswith(('les ', 'le ', 'la ', 'l\'', 'un ', 'une ')):
                    # Vérifier si c'est juste un article + compétence
                    words = label.split()
                    if len(words) <= 2:
                        return False
                # Rejeter les labels qui contiennent des mots-clés de section
                if any(keyword in label_lower for keyword in ['module', 'section', 'chapitre', 'partie']):
                    return False
                # Rejeter les labels avec des parenthèses non fermées
                if label.count('(') != label.count(')'):
                    return False
                return True
            
            # Ajouter les compétences du champ 'skills' (generated_group avec children)
            skills_field = data.get("skills", [])
            for skill_group in skills_field:
                if skill_group.get("type") == "generated_group":
                    # Extraire les children du generated_group
                    for child in skill_group.get("children", []):
                        label = child.get("label", "").strip()
                        normalized = normalize_label(label)
                        if label and normalized and normalized not in seen_labels and is_valid_skill_label(label):
                            all_skills.append({
                                "skill_id": child.get("id", ""),
                                "label": label,
                                "category": child.get("category", ""),
                                "confidence": child.get("confidence", 0.0),
                                "source": "generated_group",
                                "status": child.get("status", "pending"),
                            })
                            seen_labels.add(normalized)
                else:
                    # C'est une compétence directe
                    label = skill_group.get("label", "").strip()
                    normalized = normalize_label(label)
                    if label and normalized and normalized not in seen_labels and is_valid_skill_label(label):
                        all_skills.append({
                            "skill_id": skill_group.get("id", ""),
                            "label": label,
                            "category": skill_group.get("category", ""),
                            "confidence": skill_group.get("confidence", 0.0),
                            "source": "skill",
                            "status": skill_group.get("status", "pending"),
                        })
                        seen_labels.add(normalized)
            
            # Ajouter les derived_skills qui ne sont pas déjà dans skills
            for ds in derived_skills:
                label = ds.get("label", "").strip()
                normalized = normalize_label(label)
                if label and normalized and normalized not in seen_labels and is_valid_skill_label(label):
                    all_skills.append({
                        "skill_id": ds.get("source_code", f"derived_{len(all_skills)}"),
                        "label": label,
                        "category": ds.get("category", ""),
                        "confidence": ds.get("confidence", 0.0),
                        "source": "derived_skill",
                        "status": "pending",
                    })
                    seen_labels.add(normalized)
            
            # Ajouter les competencies qui ne sont pas déjà dans skills
            for comp in competencies:
                label = (comp.get("official_label", "") or comp.get("label", "")).strip()
                normalized = normalize_label(label)
                if label and normalized and normalized not in seen_labels and is_valid_skill_label(label):
                    all_skills.append({
                        "skill_id": comp.get("code", f"comp_{len(all_skills)}"),
                        "label": label,
                        "category": "competency",
                        "confidence": comp.get("confidence", 0.0),
                        "source": "competency",
                        "status": "pending",
                    })
                    seen_labels.add(normalized)
            
            candidate["skills"] = all_skills
            candidate["skills_count"] = len(all_skills)
            
            new_records.append(candidate)
            stats["files_processed"] += 1
            stats["candidates_generated"] += 1
            print(f"✅ Généré: {json_path.name} ({len(competencies)} compétences, {len(derived_skills)} derived)")
            
        except Exception as e:
            print(f"❌ Erreur avec {json_path.name}: {e}")
            continue
    
    # Sauvegarder les nouveaux candidats
    if new_records:
        all_records = existing_records + new_records
        store.save(all_records)
        print(f"\n💾 {len(new_records)} candidats sauvegardés dans {output_path}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Génère les candidats d'annotation à partir des JSON importés")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "referentials" / "imported",
        help="Répertoire des JSON importés",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotation" / "referential_candidates.jsonl",
        help="Chemin de sortie",
    )
    
    args = parser.parse_args()
    
    print("🔄 Génération des candidats d'annotation...")
    stats = generate_annotation_candidates_from_imported(args.input_dir, args.output)
    
    print("\n📊 Résumé :")
    print(f"   Fichiers traités : {stats['files_processed']}")
    print(f"   Candidats générés : {stats['candidates_generated']}")


if __name__ == "__main__":
    main()
