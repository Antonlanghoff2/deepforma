#!/usr/bin/env python3
"""
Génère les candidats d'annotation à partir des référentiels importés.
"""
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from referential_learning.store import AnnotationStore


def extract_skills_from_referential(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrait les compétences d'un référentiel importé."""
    skills = []
    
    # Extraire depuis 'skills'
    for skill in payload.get("skills", []):
        if isinstance(skill, dict):
            label = skill.get("label") or skill.get("raw_label") or ""
            if label:
                skills.append({
                    "text": label,
                    "label": label,
                    "category": skill.get("category", "skill"),
                    "confidence": skill.get("confidence", 0.8),
                    "source": "referential_skill",
                    "skill_id": skill.get("id"),
                })
    
    # Extraire depuis 'derived_skills'
    for skill in payload.get("derived_skills", []):
        if isinstance(skill, dict):
            label = skill.get("label") or skill.get("canonical_label") or ""
            if label:
                skills.append({
                    "text": label,
                    "label": label,
                    "category": skill.get("category", "skill"),
                    "confidence": skill.get("confidence", 0.7),
                    "source": "derived_skill",
                    "skill_id": skill.get("id"),
                })
    
    # Extraire depuis 'competencies'
    for comp in payload.get("competencies", []):
        if isinstance(comp, dict):
            label = comp.get("label") or comp.get("official_label") or ""
            if label:
                skills.append({
                    "text": label,
                    "label": label,
                    "category": "competency",
                    "confidence": 0.9,
                    "source": "competency",
                    "skill_id": comp.get("id") or comp.get("code"),
                })
    
    return skills


def generate_annotation_candidates():
    """Génère les candidats d'annotation depuis les référentiels importés."""
    imported_dir = PROJECT_ROOT / "data" / "referentials" / "imported"
    output_path = PROJECT_ROOT / "data" / "annotation" / "referential_candidates.jsonl"
    
    if not imported_dir.exists():
        print(f"❌ Répertoire introuvable: {imported_dir}")
        return
    
    store = AnnotationStore(output_path)
    existing_records = {r.get("document_id"): r for r in store.load()}
    
    new_count = 0
    updated_count = 0
    
    for json_file in imported_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            
            # Extraire l'ID du document
            document_id = (
                payload.get("referential_id") or
                payload.get("document", {}).get("id") or
                payload.get("document_id") or
                json_file.stem
            )
            
            # Extraire les compétences
            skills = extract_skills_from_referential(payload)
            
            if not skills:
                print(f"⚠️  Aucune compétence trouvée dans {json_file.name}")
                continue
            
            # Créer ou mettre à jour le candidat
            candidate = {
                "document_id": document_id,
                "source_file": json_file.name,
                "title": payload.get("title") or payload.get("document", {}).get("title", ""),
                "skills": skills,
                "skills_count": len(skills),
                "status": "pending",
                "kind": "candidates",
            }
            
            if document_id in existing_records:
                # Mettre à jour si le nombre de compétences a changé
                existing = existing_records[document_id]
                if existing.get("skills_count", 0) != len(skills):
                    existing.update(candidate)
                    updated_count += 1
                    print(f"🔄 Mis à jour: {json_file.name} ({len(skills)} compétences)")
                else:
                    print(f"⏭️  Déjà à jour: {json_file.name}")
            else:
                # Nouveau candidat
                existing_records[document_id] = candidate
                new_count += 1
                print(f"✅ Ajouté: {json_file.name} ({len(skills)} compétences)")
        
        except Exception as e:
            print(f"❌ Erreur avec {json_file.name}: {e}")
    
    # Sauvegarder tous les candidats
    store.save(list(existing_records.values()))
    
    print(f"\n📊 Résumé: {new_count} ajoutés, {updated_count} mis à jour")
    print(f"💾 Fichier de sortie: {output_path}")


if __name__ == "__main__":
    generate_annotation_candidates()
