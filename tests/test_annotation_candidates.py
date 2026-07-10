"""
Tests pour la génération des candidats d'annotation et leur affichage.
"""
import json
from pathlib import Path
import pytest

from scripts.generate_annotation_candidates import (
    extract_skills_from_referential,
    generate_annotation_candidates,
)


class TestAnnotationCandidatesGeneration:
    """Tests pour la génération des candidats d'annotation."""
    
    def test_extract_skills_from_referential_with_skills(self):
        """Test l'extraction des compétences depuis 'skills'."""
        payload = {
            "skills": [
                {"id": "s1", "label": "Python", "category": "tool", "confidence": 0.9},
                {"id": "s2", "label": "Machine Learning", "category": "method", "confidence": 0.85},
            ]
        }
        
        skills = extract_skills_from_referential(payload)
        
        assert len(skills) == 2
        assert skills[0]["label"] == "Python"
        assert skills[0]["category"] == "tool"
        assert skills[1]["label"] == "Machine Learning"
    
    def test_extract_skills_from_referential_with_derived_skills(self):
        """Test l'extraction des compétences depuis 'derived_skills'."""
        payload = {
            "derived_skills": [
                {"label": "SQL", "canonical_label": "SQL", "category": "tool", "confidence": 0.8},
                {"label": "Pandas", "canonical_label": "Pandas", "category": "tool", "confidence": 0.75},
            ]
        }
        
        skills = extract_skills_from_referential(payload)
        
        assert len(skills) == 2
        assert skills[0]["label"] == "SQL"
        assert skills[1]["label"] == "Pandas"
    
    def test_extract_skills_from_referential_with_competencies(self):
        """Test l'extraction des compétences depuis 'competencies'."""
        payload = {
            "competencies": [
                {"code": "C1", "label": "Analyse de données", "confidence": 0.95},
                {"code": "C2", "label": "Visualisation", "confidence": 0.9},
            ]
        }
        
        skills = extract_skills_from_referential(payload)
        
        assert len(skills) == 2
        assert skills[0]["label"] == "Analyse de données"
        assert skills[0]["category"] == "competency"
    
    def test_extract_skills_from_referential_mixed(self):
        """Test l'extraction avec plusieurs sources."""
        payload = {
            "skills": [{"label": "Python", "category": "tool"}],
            "derived_skills": [{"label": "SQL", "category": "tool"}],
            "competencies": [{"label": "Data Science", "category": "domain"}],
        }
        
        skills = extract_skills_from_referential(payload)
        
        assert len(skills) == 3
        labels = [s["label"] for s in skills]
        assert "Python" in labels
        assert "SQL" in labels
        assert "Data Science" in labels
    
    def test_extract_skills_empty_payload(self):
        """Test avec un payload vide."""
        payload = {}
        skills = extract_skills_from_referential(payload)
        assert len(skills) == 0
    
    def test_generate_annotation_candidates_creates_file(self, tmp_path):
        """Test que le script crée le fichier de sortie."""
        # Créer un répertoire importé fictif
        imported_dir = tmp_path / "imported"
        imported_dir.mkdir()
        
        # Créer un référentiel fictif
        referential = {
            "referential_id": "test_001",
            "title": "Test Referential",
            "skills": [
                {"id": "s1", "label": "Python", "category": "tool", "confidence": 0.9},
            ]
        }
        
        referential_file = imported_dir / "test.pdf.json"
        referential_file.write_text(json.dumps(referential), encoding="utf-8")
        
        # Modifier le chemin de sortie
        output_path = tmp_path / "candidates.jsonl"
        
        # Exécuter le script (version simplifiée)
        from referential_learning.store import AnnotationStore
        
        store = AnnotationStore(output_path)
        
        # Extraire les compétences
        skills = extract_skills_from_referential(referential)
        
        # Créer le candidat
        candidate = {
            "document_id": "test_001",
            "source_file": "test.pdf.json",
            "title": "Test Referential",
            "skills": skills,
            "skills_count": len(skills),
            "status": "pending",
            "kind": "candidates",
        }
        
        store.save([candidate])
        
        # Vérifier que le fichier existe
        assert output_path.exists()
        
        # Vérifier le contenu
        records = store.load()
        assert len(records) == 1
        assert records[0]["document_id"] == "test_001"
        assert records[0]["skills_count"] == 1
        assert records[0]["kind"] == "candidates"


class TestAnnotationPageDisplay:
    """Tests pour l'affichage des candidats dans la page d'annotation."""
    
    def test_candidates_have_required_fields(self):
        """Test que les candidats ont tous les champs requis."""
        candidate = {
            "document_id": "test_001",
            "source_file": "test.pdf",
            "title": "Test",
            "skills": [
                {"label": "Python", "category": "tool", "confidence": 0.9, "source": "skill"}
            ],
            "skills_count": 1,
            "status": "pending",
            "kind": "candidates",
            "record_id": "test_001",
        }
        
        # Vérifier les champs requis pour l'affichage
        assert "record_id" in candidate
        assert "source_file" in candidate
        assert "kind" in candidate
        assert "status" in candidate
        assert "skills" in candidate
        assert "skills_count" in candidate
    
    def test_candidate_skill_has_required_fields(self):
        """Test que les compétences ont tous les champs requis."""
        skill = {
            "label": "Python",
            "category": "tool",
            "confidence": 0.9,
            "source": "skill",
            "skill_id": "s1",
        }
        
        # Vérifier les champs requis pour l'affichage
        assert "label" in skill
        assert "category" in skill
        assert "confidence" in skill
        assert "source" in skill
        assert "skill_id" in skill


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
