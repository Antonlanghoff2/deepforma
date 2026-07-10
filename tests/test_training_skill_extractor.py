"""Test pour l'extracteur de compétences de formation Data Analyst."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skill_extraction.training_skill_extractor import extract_training_skills


def test_data_analyst_extraction():
    """Test que l'extracteur détecte au moins 20 compétences dans le texte Data Analyst."""
    text = """Data Analyst — Analyse & Visualisation de données
DataScientest
Code formation DST-DA
Niveau Bac+3/4 (Niveau 6)
Durée 400 heures
Format Bootcamp / Online
Prix TTC 5 500 € TTC
Éligible CPF Oui
Public cible Reconversions professionnelles, chargés d'études, contrôleurs de gestion
Prérequis Bac+2, aisance avec Excel, notions mathématiques
Certification Titre RNCP Niveau 6 — Data Analyst (co-certifié La Sorbonne)

OBJECTIFS PÉDAGOGIQUES
1. Extraire, transformer et analyser de grands volumes de données
2. Construire des tableaux de bord interactifs (Power BI, Tableau)
3. Rédiger des rapports d'analyse à destination des décideurs
4. Automatiser des traitements avec Python et SQL

PROGRAMME DÉTAILLÉ
Module Contenu
SQL & Bases de données
Requêtes SQL avancées, jointures, sous-requêtes, agrégations, optimisation

Python pour l'analyse
Pandas, NumPy, nettoyage de données, manipulation de dataframes

Statistiques appliquées
Statistiques descriptives, corrélations, tests statistiques, régression

Visualisation
Matplotlib, Seaborn, Plotly, Power BI, Tableau

Introduction au Machine Learning
Régression linéaire, classification, clustering — notions pratiques

Projet analytique
Étude de cas réelle, présentation des insights

COMPÉTENCES VISÉES
• SQL
• Python
• Power BI
• Tableau
• Statistiques
• Data storytelling"""

    skills = extract_training_skills(text)
    
    print(f"\nNombre de compétences détectées: {len(skills)}")
    print("\nCompétences détectées:")
    for skill in skills:
        print(f"  - {skill['label']} (confiance: {skill['confidence']:.2f}, section: {skill['source_section']}, méthode: {skill['extraction_method']})")
    
    assert len(skills) >= 20, f"Attendu au moins 20 compétences, obtenu {len(skills)}"
    
    required_skills = [
        "SQL",
        "Python",
        "Power BI",
        "Tableau",
        "Statistiques",
        "Data storytelling",
        "Pandas",
        "NumPy",
        "Nettoyage de données",
        "Manipulation de dataframes",
        "Matplotlib",
        "Seaborn",
        "Plotly",
        "Machine Learning",
        "Régression linéaire",
        "Classification",
        "Clustering",
        "Tests statistiques",
        "Tableaux de bord",
        "Rédaction de rapports d'analyse",
    ]
    
    found_labels = {skill['label'].lower() for skill in skills}
    found_labels.update({skill['raw_label'].lower() for skill in skills})
    
    missing = []
    for required in required_skills:
        required_lower = required.lower()
        if not any(required_lower in found or found in required_lower for found in found_labels):
            missing.append(required)
    
    if missing:
        print(f"\nCompétences manquantes: {missing}")
        assert False, f"Compétences obligatoires manquantes: {missing}"
    
    print("\n✓ Test réussi: toutes les compétences obligatoires sont détectées")


if __name__ == "__main__":
    test_data_analyst_extraction()
