"""
Tests pour l'extracteur ouvert de compétences.
"""
import pytest

from skill_extraction.open_skill_extractor import (
    SkillCandidate,
    extract_open_skills,
    detect_referential_sections,
    split_skill_list,
    extract_by_action_verbs,
    match_candidate_to_known_skills,
)


class TestSkillCandidate:
    """Tests pour le modèle SkillCandidate."""
    
    def test_skill_candidate_creation(self):
        """Test la création d'un candidat compétence."""
        candidate = SkillCandidate(
            id="test_001",
            label="Python",
            raw_label="Python",
            source_text="Maîtrise de Python",
            confidence=0.95,
        )
        
        assert candidate.id == "test_001"
        assert candidate.label == "Python"
        assert candidate.is_new is True
        assert candidate.status == "pending"
        assert candidate.matched_existing_skill_id is None
    
    def test_skill_candidate_to_dict(self):
        """Test la conversion en dictionnaire."""
        candidate = SkillCandidate(
            id="test_002",
            label="SQL",
            raw_label="SQL",
            source_text="Requêtes SQL",
            confidence=0.90,
            is_new=False,
            matched_existing_skill_id="sql_001",
            matched_existing_label="SQL Avancé",
        )
        
        data = candidate.to_dict()
        
        assert data["id"] == "test_002"
        assert data["label"] == "SQL"
        assert data["is_new"] is False
        assert data["matched_existing_skill_id"] == "sql_001"
        assert data["status"] == "pending"


class TestSectionDetection:
    """Tests pour la détection des sections."""
    
    def test_detect_competences_visees(self):
        """Test la détection de la section COMPÉTENCES VISÉES."""
        text = """
        COMPÉTENCES VISÉES
        • Python
        • SQL
        • Machine Learning
        """
        
        sections = detect_referential_sections(text)
        
        assert len(sections) >= 1
        assert any(s["name"] == "COMPÉTENCES VISÉES" for s in sections)
    
    def test_detect_programme(self):
        """Test la détection de la section PROGRAMME."""
        text = """
        PROGRAMME DÉTAILLÉ
        Module 1 : Python
        Module 2 : SQL
        """
        
        sections = detect_referential_sections(text)
        
        assert len(sections) >= 1
        assert any(s["name"] == "PROGRAMME" for s in sections)
    
    def test_detect_objectifs(self):
        """Test la détection de la section OBJECTIFS."""
        text = """
        OBJECTIFS PÉDAGOGIQUES
        1. Maîtriser Python
        2. Comprendre SQL
        """
        
        sections = detect_referential_sections(text)
        
        assert len(sections) >= 1
        assert any(s["name"] == "OBJECTIFS PÉDAGOGIQUES" for s in sections)


class TestSplitSkillList:
    """Tests pour le découpage des listes."""
    
    def test_split_by_comma(self):
        """Test le découpage par virgules."""
        text = "Python, SQL, Machine Learning"
        items = split_skill_list(text)
        
        assert len(items) == 3
        assert "Python" in items
        assert "SQL" in items
        assert "Machine Learning" in items
    
    def test_split_by_bullet(self):
        """Test le découpage par puces."""
        text = "• Python • SQL • Machine Learning"
        items = split_skill_list(text)
        
        assert len(items) >= 3
    
    def test_split_with_parentheses(self):
        """Test le découpage avec parenthèses."""
        text = "Visualisation (Power BI, Tableau)"
        items = split_skill_list(text)
        
        assert len(items) >= 2
        assert "Visualisation" in items
        assert "Power BI" in items
        assert "Tableau" in items


class TestActionVerbExtraction:
    """Tests pour l'extraction par verbes d'action."""
    
    def test_extract_automatiser(self):
        """Test l'extraction avec le verbe 'automatiser'."""
        text = """
        Automatiser la consolidation de sources hétérogènes avec Python et SQL
        """
        
        candidates = extract_by_action_verbs(text)
        
        assert len(candidates) >= 1
        assert any("automatiser" in c.label.lower() for c in candidates)
    
    def test_extract_rediger(self):
        """Test l'extraction avec le verbe 'rédiger'."""
        text = """
        Rédiger des rapports d'analyse à destination des décideurs
        """
        
        candidates = extract_by_action_verbs(text)
        
        assert len(candidates) >= 1
        assert any("rédiger" in c.label.lower() for c in candidates)
    
    def test_extract_construire(self):
        """Test l'extraction avec le verbe 'construire'."""
        text = """
        Construire des tableaux de bord interactifs
        """
        
        candidates = extract_by_action_verbs(text)
        
        assert len(candidates) >= 1
        assert any("construire" in c.label.lower() for c in candidates)


class TestMatching:
    """Tests pour le matching avec les compétences connues."""
    
    def test_match_strong(self):
        """Test un match fort."""
        candidate = SkillCandidate(
            id="test_001",
            label="Python",
            raw_label="Python",
            source_text="Maîtrise de Python",
            confidence=0.95,
        )
        
        known_skills = [
            {"id": "sql_001", "label": "SQL", "category": "Data"},
            {"id": "python_001", "label": "Python", "category": "Data"},
        ]
        
        matched = match_candidate_to_known_skills(candidate, known_skills)
        
        assert matched.is_new is False
        assert matched.matched_existing_skill_id == "python_001"
        assert matched.semantic_score >= 0.88
    
    def test_match_weak(self):
        """Test un match faible."""
        candidate = SkillCandidate(
            id="test_002",
            label="Python Avancé",
            raw_label="Python Avancé",
            source_text="Python avancé",
            confidence=0.90,
        )
        
        known_skills = [
            {"id": "python_001", "label": "Python", "category": "Data"},
        ]
        
        matched = match_candidate_to_known_skills(candidate, known_skills)
        
        # Score entre 0.60 et 0.85
        assert 0.60 <= matched.semantic_score < 0.85
        assert matched.is_new is True
        assert matched.status == "pending"
    
    def test_no_match(self):
        """Test sans match."""
        candidate = SkillCandidate(
            id="test_003",
            label="Compétence Totalement Nouvelle",
            raw_label="Compétence Totalement Nouvelle",
            source_text="Nouvelle compétence",
            confidence=0.85,
        )
        
        known_skills = [
            {"id": "python_001", "label": "Python", "category": "Data"},
        ]
        
        matched = match_candidate_to_known_skills(candidate, known_skills)
        
        assert matched.is_new is True
        assert matched.matched_existing_skill_id is None


class TestOpenExtraction:
    """Tests pour l'extraction ouverte complète."""
    
    def test_extract_unknown_skill(self):
        """Test l'extraction d'une compétence inconnue."""
        text = """
        COMPÉTENCES VISÉES
        • Automatiser la consolidation de sources hétérogènes avec Python et SQL
        """
        
        candidates = extract_open_skills(text, known_skills=[])
        
        assert len(candidates) >= 1
        # La compétence doit être extraite même si elle n'existe pas
        assert any("automatiser" in c.label.lower() for c in candidates)
        # Elle doit être marquée comme nouvelle
        assert all(c.is_new for c in candidates)
    
    def test_extract_data_analyst_skills(self):
        """Test l'extraction des compétences Data Analyst."""
        text = """
        Data Analyst — Analyse & Visualisation de données
        
        OBJECTIFS PÉDAGOGIQUES
        1. Extraire, transformer et analyser de grands volumes de données
        2. Construire des tableaux de bord interactifs (Power BI, Tableau)
        3. Rédiger des rapports d'analyse à destination des décideurs
        4. Automatiser des traitements avec Python et SQL
        
        PROGRAMME DÉTAILLÉ
        SQL & Bases de données
        Requêtes SQL avancées, jointures, sous-requêtes, agrégations, optimisation
        
        Python pour l'analyse
        Pandas, NumPy, nettoyage de données, manipulation de dataframes
        
        Statistiques appliquées
        Statistiques descriptives, corrélations, tests statistiques, régression
        
        Visualisation
        Matplotlib, Seaborn, Plotly, Power BI, Tableau
        
        Introduction au Machine Learning
        Régression linéaire, classification, clustering
        
        COMPÉTENCES VISÉES
        • SQL
        • Python
        • Power BI
        • Tableau
        • Statistiques
        • Data storytelling
        """
        
        candidates = extract_open_skills(text, known_skills=[])
        
        # Doit extraire au moins 20 compétences
        assert len(candidates) >= 20
        
        # Vérifier les compétences obligatoires
        labels = [c.label.lower() for c in candidates]
        required = [
            "sql", "python", "power bi", "tableau", "pandas", "numpy",
            "nettoyage de données", "manipulation de dataframes",
            "matplotlib", "seaborn", "plotly", "machine learning",
            "régression linéaire", "classification", "clustering",
            "tests statistiques", "tableaux de bord", "data storytelling"
        ]
        
        for req in required:
            assert any(req in label for label in labels), f"Compétence manquante: {req}"
    
    def test_extract_with_known_skills(self):
        """Test l'extraction avec des compétences connues."""
        text = """COMPÉTENCES VISÉES
• Python
• SQL
• Nouvelle Compétence Inconnue"""
        
        known_skills = [
            {"id": "python_001", "label": "Python", "category": "Data"},
            {"id": "sql_001", "label": "SQL", "category": "Data"},
        ]
        
        candidates = extract_open_skills(text, known_skills=known_skills)
        
        # Python et SQL doivent être matchés
        python_candidates = [c for c in candidates if "python" in c.label.lower()]
        sql_candidates = [c for c in candidates if "sql" in c.label.lower()]
        
        assert len(python_candidates) >= 1
        assert any(not c.is_new for c in python_candidates)
        
        assert len(sql_candidates) >= 1
        assert any(not c.is_new for c in sql_candidates)
        
        # La nouvelle compétence doit être extraite
        new_candidates = [c for c in candidates if c.is_new]
        assert len(new_candidates) >= 1
    
    def test_no_filtering_unknown_skills(self):
        """Test que les compétences inconnues ne sont pas filtrées."""
        text = """
        COMPÉTENCES VISÉES
        • Compétence Totalement Nouvelle et Inconnue
        • Autre Compétence Jamais Vue
        """
        
        candidates = extract_open_skills(text, known_skills=[])
        
        # Les deux compétences doivent être extraites
        assert len(candidates) >= 2
        
        # Toutes doivent être marquées comme nouvelles
        assert all(c.is_new for c in candidates)
        
        # Aucune ne doit être rejetée
        assert all(c.status == "pending" for c in candidates)


class TestEdgeCases:
    """Tests pour les cas limites."""
    
    def test_empty_text(self):
        """Test avec un texte vide."""
        candidates = extract_open_skills("", known_skills=[])
        assert len(candidates) == 0
    
    def test_whitespace_only(self):
        """Test avec un texte contenant seulement des espaces."""
        candidates = extract_open_skills("   \n\n   ", known_skills=[])
        assert len(candidates) == 0
    
    def test_no_skills_in_text(self):
        """Test avec un texte sans compétences."""
        text = "Ceci est un texte sans aucune compétence technique."
        candidates = extract_open_skills(text, known_skills=[])
        # Peut extraire zéro ou quelques candidats, mais ne doit pas planter
        assert isinstance(candidates, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
