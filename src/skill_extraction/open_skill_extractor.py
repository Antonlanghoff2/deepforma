"""
Extracteur ouvert de compétences.
Détecte les compétences même si elles n'existent pas dans skills.json.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from common.text import clean_text, normalize_for_match


@dataclass
class SkillCandidate:
    """Candidat compétence extrait d'un texte."""
    id: str
    label: str
    raw_label: str
    source_text: str
    source_section: str | None = None
    source_page: int | None = None
    extraction_method: str = "unknown"
    confidence: float = 0.0
    status: str = "pending"  # pending, validated, rejected, merged, deleted
    is_new: bool = True
    matched_existing_skill_id: str | None = None
    matched_existing_label: str | None = None
    matched_category: str | None = None
    semantic_score: float | None = None
    category: str = "technical_skill"
    aliases: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convertit le candidat en dictionnaire."""
        return {
            "id": self.id,
            "label": self.label,
            "raw_label": self.raw_label,
            "description": "",
            "aliases": self.aliases,
            "category": self.category,
            "block": self.source_section,
            "source_page": self.source_page,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "status": self.status,
            "type": "subskill",
            "is_new": self.is_new,
            "matched_existing_skill_id": self.matched_existing_skill_id,
            "matched_existing_label": self.matched_existing_label,
            "semantic_score": self.semantic_score,
            "children": [],
        }


# Verbes d'action pour l'extraction par patterns linguistiques
ACTION_VERBS = [
    "analyser", "concevoir", "construire", "produire", "rédiger", "présenter",
    "piloter", "mettre en œuvre", "exploiter", "automatiser", "configurer",
    "développer", "déployer", "superviser", "évaluer", "optimiser", "maintenir",
    "sécuriser", "interpréter", "modéliser", "extraire", "transformer",
    "visualiser", "communiquer", "implémenter", "gérer", "administrer",
    "coordonner", "organiser", "animer", "encadrer", "participer", "contribuer",
    "assurer", "garantir", "collaborer", "travailler", "utiliser", "manipuler",
    "installer", "réaliser", "effectuer", "appliquer", "maîtriser", "acquérir",
    "apprendre", "comprendre", "appréhender"
]

# Patterns pour détecter les sections
SECTION_PATTERNS = [
    ("COMPÉTENCES VISÉES", r"(?i)(?:^|\n)\s*(?:COMPÉTENCES VISÉES|COMPETENCES VISEES|COMPÉTENCES ACQUISES|COMPETENCES ACQUISES|COMPÉTENCES ATTESTÉES)\s*\n"),
    ("BLOCS DE COMPÉTENCES", r"(?i)(?:^|\n)\s*(?:BLOCS DE COMPÉTENCES|BLOCS DE COMPETENCES|BLOC DE COMPÉTENCES)\s*\n"),
    ("ACTIVITÉS VISÉES", r"(?i)(?:^|\n)\s*(?:ACTIVITÉS VISÉES|ACTIVITES VISEES|ACTIVITÉS)\s*\n"),
    ("OBJECTIFS PÉDAGOGIQUES", r"(?i)(?:^|\n)\s*(?:OBJECTIFS PÉDAGOGIQUES|OBJECTIFS PEDAGOGIQUES|OBJECTIFS DE LA FORMATION|OBJECTIFS)\s*\n"),
    ("PROGRAMME", r"(?i)(?:^|\n)\s*(?:PROGRAMME DÉTAILLÉ|PROGRAMME DETAILLE|PROGRAMME|CONTENU DE LA FORMATION|CONTENU|MODULES)\s*\n"),
    ("MODALITÉS D'ÉVALUATION", r"(?i)(?:^|\n)\s*(?:MODALITÉS D'ÉVALUATION|MODALITES D'EVALUATION|MODALITÉS D ÉVALUATION|ÉVALUATION)\s*\n"),
    ("PRÉREQUIS", r"(?i)(?:^|\n)\s*(?:PRÉREQUIS|PRE-REQUIS|PREREQUIS|CONDITIONS D'ACCÈS)\s*\n"),
    ("PUBLIC CIBLE", r"(?i)(?:^|\n)\s*(?:PUBLIC CIBLE|PUBLIC|DESTINATAIRES)\s*\n"),
]

# Patterns pour extraire les listes
BULLET_PATTERN = r"(?:^|\n)[•\-\*·]\s*(.+?)(?=(?:\n[•\-\*·])|(?:\n\n)|(?:\n[A-ZÀ-Ü]{3,})|$)"
NUMBERED_PATTERN = r"(?:^|\n)\d+\.\s*(.+?)(?=(?:\n\d+\.)|(?:\n\n)|(?:\n[A-ZÀ-Ü]{3,})|$)"

# Lexique étendu pour les compétences Data/IA
EXTENDED_LEXICON = {
    "sql": "SQL",
    "requêtes sql": "Requêtes SQL",
    "jointures": "Jointures SQL",
    "sous-requêtes": "Sous-requêtes SQL",
    "agrégations": "Agrégations SQL",
    "optimisation sql": "Optimisation SQL",
    "python": "Python",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "dataframes": "Manipulation de dataframes",
    "nettoyage de données": "Nettoyage de données",
    "statistiques descriptives": "Statistiques descriptives",
    "corrélations": "Corrélations",
    "tests statistiques": "Tests statistiques",
    "régression": "Régression statistique",
    "régression linéaire": "Régression linéaire",
    "classification": "Classification",
    "clustering": "Clustering",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "visualisation": "Visualisation de données",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "plotly": "Plotly",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "dashboard": "Tableaux de bord",
    "tableaux de bord": "Tableaux de bord",
    "data storytelling": "Data storytelling",
    "rapports d'analyse": "Rédaction de rapports d'analyse",
    "automatisation": "Automatisation de traitements",
    "etl": "ETL",
    "analyse de données": "Analyse de données",
    "grands volumes de données": "Analyse de grands volumes de données",
    "api": "API",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git": "Git",
    "rag": "RAG",
    "nlp": "NLP",
    "computer vision": "Computer Vision",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "Scikit-learn",
    "spark": "Spark",
    "hadoop": "Hadoop",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
}


def detect_referential_sections(text: str) -> list[dict[str, Any]]:
    """
    Détecte les sections structurées dans un texte de référentiel.
    
    Returns:
        Liste de dicts avec 'name', 'start', 'end', 'content'
    """
    sections = []
    
    # Trouver toutes les positions de sections
    section_positions = []
    for section_name, pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            section_positions.append({
                "name": section_name,
                "start": match.start(),
                "content_start": match.end(),
            })
    
    # Trier par position
    section_positions.sort(key=lambda x: x["start"])
    
    # Extraire le contenu de chaque section
    for i, section_info in enumerate(section_positions):
        # La section se termine au début de la prochaine section ou à la fin du texte
        if i + 1 < len(section_positions):
            end = section_positions[i + 1]["start"]
        else:
            end = len(text)
        
        content = text[section_info["content_start"]:end].strip()
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        if content:
            sections.append({
                "name": section_info["name"],
                "start": section_info["start"],
                "end": end,
                "content": content,
            })
    
    return sections


def split_skill_list(text: str) -> list[str]:
    """Découpe une liste de compétences (virgules, puces, parenthèses, etc.)."""
    text = clean_text(text)
    if not text:
        return []
    
    # Supprimer les puces en début de ligne
    text = re.sub(r'^[•\-\*·]\s*', '', text)
    
    # Remplacer les puces inline par des virgules
    text = re.sub(r'\s*[•\-\*·]\s*', ', ', text)
    
    # Gérer les parenthèses : extraire le contenu et le séparer
    # Pattern pour trouver "text (item1, item2)"
    paren_pattern = r'([^()]+?)\s*\(([^()]+)\)'
    
    def replace_parens(match):
        main = clean_text(match.group(1))
        items = match.group(2)
        # Séparer les items par virgule
        item_list = [clean_text(item) for item in items.split(',') if clean_text(item)]
        # Retourner tout séparé par virgules
        if main and item_list:
            return main + ', ' + ', '.join(item_list)
        elif main:
            return main
        elif item_list:
            return ', '.join(item_list)
        return ''
    
    # Appliquer le remplacement des parenthèses
    text = re.sub(paren_pattern, replace_parens, text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Séparer par virgules, points-virgules, "et"
    parts = re.split(r'[,;]\s*|\s+et\s+', text)
    
    items = []
    for part in parts:
        part = clean_text(part)
        if not part or len(part) < 2:
            continue
        
        if len(part) < 100:
            items.append(part)
    
    return items


def extract_by_action_verbs(text: str, section_name: str | None = None) -> list[SkillCandidate]:
    """Extrait les compétences par verbes d'action."""
    candidates = []
    seen = set()
    
    # Pattern pour détecter les phrases avec verbes d'action
    for verb in ACTION_VERBS:
        # Pattern : verbe + complément (jusqu'à point, virgule, ou fin de ligne)
        pattern = rf"(?:^|\n)\s*{verb}\b\s+([^.,\n]{{5,150}})"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            complement = clean_text(match.group(1))
            if not complement:
                continue
            
            # Construire la compétence complète
            full_skill = f"{verb.capitalize()} {complement}"
            key = normalize_for_match(full_skill)
            
            if key in seen:
                continue
            seen.add(key)
            
            candidates.append(SkillCandidate(
                id=f"skill_{uuid.uuid4().hex[:12]}",
                label=full_skill,
                raw_label=full_skill,
                source_text=match.group(0).strip(),
                source_section=section_name,
                extraction_method="action_verb",
                confidence=0.85,
                is_new=True,
                status="pending",
            ))
    
    return candidates


def extract_by_bullets(text: str, section_name: str | None = None) -> list[SkillCandidate]:
    """Extrait les compétences depuis les listes à puces."""
    candidates = []
    
    bullets = re.findall(BULLET_PATTERN, text, re.MULTILINE)
    for bullet in bullets:
        bullet = clean_text(bullet)
        if not bullet or len(bullet) > 200:
            continue
        
        items = split_skill_list(bullet)
        for item in items:
            if is_valid_skill(item):
                candidates.append(SkillCandidate(
                    id=f"skill_{uuid.uuid4().hex[:12]}",
                    label=item,
                    raw_label=item,
                    source_text=bullet,
                    source_section=section_name,
                    extraction_method="bullet_list",
                    confidence=0.95 if section_name and "COMPÉTENCES" in section_name else 0.85,
                    is_new=True,
                    status="pending",
                ))
    
    return candidates


def extract_by_numbered_list(text: str, section_name: str | None = None) -> list[SkillCandidate]:
    """Extrait les compétences depuis les listes numérotées."""
    candidates = []
    
    numbered = re.findall(NUMBERED_PATTERN, text, re.MULTILINE)
    for num in numbered:
        num = clean_text(num)
        if not num or len(num) > 200:
            continue
        
        if is_valid_skill(num):
            candidates.append(SkillCandidate(
                id=f"skill_{uuid.uuid4().hex[:12]}",
                label=num,
                raw_label=num,
                source_text=num,
                source_section=section_name,
                extraction_method="numbered_list",
                confidence=0.90 if section_name and "OBJECTIFS" in section_name else 0.85,
                is_new=True,
                status="pending",
            ))
    
    return candidates


def extract_by_lexicon(text: str, section_name: str | None = None) -> list[SkillCandidate]:
    """Extrait les compétences depuis le lexique étendu."""
    candidates = []
    text_lower = text.lower()
    seen = set()
    
    for key, normalized in EXTENDED_LEXICON.items():
        if key in text_lower:
            key_norm = normalize_for_match(normalized)
            if key_norm in seen:
                continue
            seen.add(key_norm)
            
            candidates.append(SkillCandidate(
                id=f"skill_{uuid.uuid4().hex[:12]}",
                label=normalized,
                raw_label=key,
                source_text=text[:200],
                source_section=section_name,
                extraction_method="lexicon",
                confidence=0.90,
                is_new=True,
                status="pending",
            ))
    
    return candidates


def extract_by_comma_lists(text: str, section_name: str | None = None) -> list[SkillCandidate]:
    """Extrait les compétences depuis les listes à virgules."""
    candidates = []
    
    lines = text.split('\n')
    for line in lines:
        line = clean_text(line)
        if not line or len(line) > 200:
            continue
        
        # Ignorer les lignes qui commencent par une puce ou un numéro
        if re.match(r'^[•\-\*\d]', line):
            continue
        
        if ',' in line or ';' in line:
            items = split_skill_list(line)
            if len(items) > 1:
                for item in items:
                    if is_valid_skill(item):
                        candidates.append(SkillCandidate(
                            id=f"skill_{uuid.uuid4().hex[:12]}",
                            label=item,
                            raw_label=item,
                            source_text=line,
                            source_section=section_name,
                            extraction_method="comma_list",
                            confidence=0.85,
                            is_new=True,
                            status="pending",
                        ))
    
    return candidates


def is_valid_skill(text: str) -> bool:
    """Vérifie si le texte est une compétence valide."""
    text = clean_text(text)
    if not text or len(text) < 2:
        return False
    
    if len(text) > 100:
        return False
    
    # Rejeter les textes qui contiennent des mots-clés parasites
    invalid_keywords = [
        "certifié", "co-certifié", "certification", "diplôme", "diplome",
        "niveau", "durée", "duree", "prix", "format", "éligible", "eligible",
        "public cible", "code formation", "titre",
    ]
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in invalid_keywords):
        return False
    
    # Rejeter les textes qui commencent ou finissent par des caractères bizarres
    if text.startswith(('(', '•', '-', '*', '·')):
        return False
    if text.endswith((')', '(', '•', '-', '*', '·')):
        return False
    
    return True


def match_candidate_to_known_skills(
    candidate: SkillCandidate,
    known_skills: list[dict[str, Any]],
    threshold_strong: float = 0.85,
    threshold_weak: float = 0.60,
) -> SkillCandidate:
    """
    Compare un candidat avec les compétences connues.
    
    Args:
        candidate: Le candidat à comparer
        known_skills: Liste des compétences connues (avec 'id', 'label', 'category')
        threshold_strong: Seuil pour un match fort
        threshold_weak: Seuil pour un match faible
    
    Returns:
        Le candidat mis à jour avec les informations de match
    """
    if not known_skills:
        return candidate
    
    candidate_norm = normalize_for_match(candidate.label)
    best_match = None
    best_score = 0.0
    
    for known in known_skills:
        known_label = known.get('label', '')
        known_norm = normalize_for_match(known_label)
        
        # Calculer la similarité
        score = SequenceMatcher(None, candidate_norm, known_norm).ratio()
        
        if score > best_score:
            best_score = score
            best_match = known
    
    if best_match and best_score >= threshold_strong:
        # Match fort : même compétence probable
        candidate.is_new = False
        candidate.matched_existing_skill_id = best_match.get('id')
        candidate.matched_existing_label = best_match.get('label')
        candidate.matched_category = best_match.get('category')
        candidate.semantic_score = best_score
    elif best_match and best_score >= threshold_weak:
        # Match faible : compétence proche, à vérifier
        candidate.is_new = True
        candidate.matched_existing_skill_id = best_match.get('id')
        candidate.matched_existing_label = best_match.get('label')
        candidate.matched_category = best_match.get('category')
        candidate.semantic_score = best_score
        candidate.status = "pending"  # À vérifier
    else:
        # Pas de match : nouvelle compétence
        candidate.is_new = True
        candidate.semantic_score = best_score if best_match else None
    
    return candidate


def extract_open_skills(
    text: str,
    known_skills: list[dict[str, Any]] | None = None,
) -> list[SkillCandidate]:
    """
    Extrait les compétences de manière ouverte depuis un texte.
    
    Args:
        text: Le texte à analyser
        known_skills: Liste optionnelle des compétences connues pour le matching
    
    Returns:
        Liste de candidats compétences
    """
    if not text or not text.strip():
        return []
    
    # Normaliser le texte
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    # Détecter les sections
    sections = detect_referential_sections(text)
    
    all_candidates = []
    
    if sections:
        # Extraire par section
        for section in sections:
            section_text = section["content"]
            section_name = section["name"]
            
            all_candidates.extend(extract_by_bullets(section_text, section_name))
            all_candidates.extend(extract_by_numbered_list(section_text, section_name))
            all_candidates.extend(extract_by_comma_lists(section_text, section_name))
            all_candidates.extend(extract_by_action_verbs(section_text, section_name))
            all_candidates.extend(extract_by_lexicon(section_text, section_name))
    else:
        # Pas de sections détectées, extraire sur tout le texte
        all_candidates.extend(extract_by_bullets(text))
        all_candidates.extend(extract_by_numbered_list(text))
        all_candidates.extend(extract_by_comma_lists(text))
        all_candidates.extend(extract_by_action_verbs(text))
        all_candidates.extend(extract_by_lexicon(text))
    
    # Dédupliquer
    deduplicated = deduplicate_candidates(all_candidates)
    
    # Matching avec les compétences connues
    if known_skills:
        deduplicated = [
            match_candidate_to_known_skills(candidate, known_skills)
            for candidate in deduplicated
        ]
    
    # Trier par confiance décroissante
    deduplicated.sort(key=lambda x: (-x.confidence, x.label))
    
    return deduplicated


def deduplicate_candidates(candidates: list[SkillCandidate]) -> list[SkillCandidate]:
    """Déduplique les candidats en gardant le meilleur."""
    seen = {}
    
    for candidate in candidates:
        # Normaliser le label pour la déduplication
        label = normalize_for_match(candidate.label)
        label = re.sub(r'\s+', ' ', label)
        label = re.sub(r'[^\w\s]', '', label)
        key = label.strip()
        
        if key not in seen:
            seen[key] = candidate
        else:
            existing = seen[key]
            # Garder celui avec la confiance la plus élevée
            if candidate.confidence > existing.confidence:
                seen[key] = candidate
            elif candidate.confidence == existing.confidence:
                # Même confiance : garder le plus spécifique (plus long)
                if len(candidate.label) > len(existing.label):
                    seen[key] = candidate
    
    return list(seen.values())


def save_candidates_to_json(
    candidates: list[SkillCandidate],
    output_path: str | Path,
) -> None:
    """Sauvegarde les candidats au format JSON canonique."""
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "schema_version": "1.0",
        "candidates": [c.to_dict() for c in candidates],
        "metadata": {
            "total_count": len(candidates),
            "new_count": sum(1 for c in candidates if c.is_new),
            "matched_count": sum(1 for c in candidates if not c.is_new),
        }
    }
    
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
