from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text, normalize_for_match


@dataclass
class SkillCandidate:
    label: str
    raw_label: str
    source_section: str
    source_sentence: str
    confidence: float
    extraction_method: str
    category: str = "technical_skill"
    aliases: list[str] = field(default_factory=list)


SECTION_PATTERNS = {
    "COMPÉTENCES VISÉES": r"(?:COMPÉTENCES VISÉES|COMPÉTENCES ACQUISES|COMPETENCES VISEES|COMPETENCES ACQUISES)",
    "PROGRAMME": r"(?:PROGRAMME DÉTAILLÉ|PROGRAMME|CONTENU|MODULES)",
    "OBJECTIFS": r"(?:OBJECTIFS PÉDAGOGIQUES|OBJECTIFS)",
    "PRÉREQUIS": r"(?:PRÉREQUIS|PRE-REQUIS|PREREQUIS)",
}

BULLET_PATTERN = r"(?:^|\n)[•\-\*·]\s*(.+?)(?=(?:\n[•\-\*·])|(?:\n\n)|(?:\n[A-ZÀ-Ü]{3,})|$)"
NUMBERED_PATTERN = r"(?:^|\n)\d+\.\s*(.+?)(?=(?:\n\d+\.)|(?:\n\n)|(?:\n[A-ZÀ-Ü]{3,})|$)"

DATA_IA_LEXICON = {
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
}


def detect_training_sections(text: str) -> dict[str, str]:
    """Détecte les sections structurées dans le texte de formation."""
    sections = {}
    
    # Patterns pour détecter les débuts de sections
    section_markers = [
        ("COMPÉTENCES VISÉES", r"(?i)(?:^|\n)(?:COMPÉTENCES VISÉES|COMPETENCES VISEES|COMPÉTENCES ACQUISES|COMPETENCES ACQUISES)\s*\n"),
        ("PROGRAMME", r"(?i)(?:^|\n)(?:PROGRAMME DÉTAILLÉ|PROGRAMME DETAILLE|PROGRAMME|CONTENU DE LA FORMATION|CONTENU)\s*\n"),
        ("OBJECTIFS", r"(?i)(?:^|\n)(?:OBJECTIFS PÉDAGOGIQUES|OBJECTIFS PEDAGOGIQUES|OBJECTIFS DE LA FORMATION|OBJECTIFS)\s*\n"),
        ("PRÉREQUIS", r"(?i)(?:^|\n)(?:PRÉREQUIS|PRE-REQUIS|PREREQUIS|CONDITIONS D'ACCÈS)\s*\n"),
    ]
    
    # Trouver toutes les positions de sections
    section_positions = []
    for section_name, pattern in section_markers:
        for match in re.finditer(pattern, text):
            section_positions.append((match.start(), match.end(), section_name))
    
    # Trier par position
    section_positions.sort()
    
    # Extraire le contenu de chaque section
    for i, (start, content_start, section_name) in enumerate(section_positions):
        # La section se termine au début de la prochaine section ou à la fin du texte
        if i + 1 < len(section_positions):
            end = section_positions[i + 1][0]
        else:
            end = len(text)
        
        section_text = text[content_start:end].strip()
        
        # Nettoyer le texte de la section
        section_text = re.sub(r'\n{3,}', '\n\n', section_text)
        
        if section_text:
            sections[section_name] = section_text
    
    return sections


def extract_candidates_from_section(section_text: str, section_name: str) -> list[SkillCandidate]:
    candidates = []
    
    # Extraire les puces
    bullets = re.findall(BULLET_PATTERN, section_text, re.MULTILINE)
    for bullet in bullets:
        bullet = clean_text(bullet)
        if bullet and len(bullet) < 200:
            # Vérifier si la puce contient plusieurs éléments séparés par des virgules
            items = split_skill_list(bullet)
            for item in items:
                if is_valid_skill(item, section_name):
                    candidates.append(SkillCandidate(
                        label=item,
                        raw_label=item,
                        source_section=section_name,
                        source_sentence=bullet,
                        confidence=0.95 if section_name == "COMPÉTENCES VISÉES" else 0.85,
                        extraction_method="explicit_bullet",
                    ))
    
    # Extraire les éléments numérotés
    numbered = re.findall(NUMBERED_PATTERN, section_text, re.MULTILINE)
    for num in numbered:
        num = clean_text(num)
        if num and len(num) < 200:
            if is_valid_skill(num, section_name):
                candidates.append(SkillCandidate(
                    label=num,
                    raw_label=num,
                    source_section=section_name,
                    source_sentence=num,
                    confidence=0.80 if section_name == "OBJECTIFS" else 0.85,
                    extraction_method="objective" if section_name == "OBJECTIFS" else "program_line",
                ))
    
    # Extraire les lignes avec virgules ou points-virgules
    lines = section_text.split('\n')
    for line in lines:
        line = clean_text(line)
        if not line or len(line) > 200:
            continue
        
        # Ignorer les lignes qui commencent par une puce ou un numéro
        if re.match(r'^[•\-\*\d]', line):
            continue
        
        # Ignorer les lignes qui sont des titres de module
        if is_module_title(line):
            continue
        
        if ',' in line or ';' in line:
            items = split_skill_list(line)
            if len(items) > 1:
                for item in items:
                    if is_valid_skill(item, section_name):
                        candidates.append(SkillCandidate(
                            label=item,
                            raw_label=item,
                            source_section=section_name,
                            source_sentence=line,
                            confidence=0.90 if section_name == "PROGRAMME" else 0.80,
                            extraction_method="program_line",
                        ))
    
    # Extraire les compétences du lexique
    lexicon_matches = extract_from_lexicon(section_text, section_name)
    candidates.extend(lexicon_matches)
    
    return candidates


def is_valid_skill(text: str, section_name: str = "") -> bool:
    """Vérifie si le texte est une compétence valide."""
    text = clean_text(text)
    if not text or len(text) < 2:
        return False
    
    # Rejeter les textes trop longs
    if len(text) > 100:
        return False
    
    # Pour la section PRÉREQUIS, être plus permissif
    if section_name == "PRÉREQUIS":
        # Rejeter seulement les éléments clairement non-compétences
        invalid_keywords = [
            "certification", "diplôme", "diplome", "titre",
        ]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in invalid_keywords):
            return False
        return True
    
    # Pour les autres sections, rejeter les textes qui contiennent des mots-clés parasites
    invalid_keywords = [
        "certifié", "co-certifié", "certification", "diplôme", "diplome",
        "niveau", "durée", "duree", "prix", "format", "éligible", "eligible",
        "public cible", "code formation",
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


def is_module_title(line: str) -> bool:
    """Vérifie si la ligne est un titre de module."""
    line = clean_text(line)
    # Les titres de module sont généralement courts et ne contiennent pas de virgules
    if len(line) > 60 or ',' in line or ';' in line:
        return False
    
    # Vérifier si c'est un titre de module typique
    module_keywords = [
        "sql", "python", "statistiques", "visualisation", "machine learning",
        "projet analytique", "module", "introduction"
    ]
    line_lower = line.lower()
    
    # Si la ligne contient un mot-clé de module et est courte, c'est probablement un titre
    if any(keyword in line_lower for keyword in module_keywords):
        # Mais si elle contient des verbes d'action, c'est plutôt une compétence
        action_verbs = ["analyser", "construire", "rédiger", "automatiser", "extraire", "transformer"]
        if not any(verb in line_lower for verb in action_verbs):
            return True
    return False


def split_skill_list(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    
    parts = re.split(r'[,;]\s*|\s+et\s+', text)
    
    items = []
    for part in parts:
        part = clean_text(part)
        if not part or len(part) < 2:
            continue
        
        paren_match = re.match(r'^(.+?)\s*\((.+?)\)$', part)
        if paren_match:
            main_item = clean_text(paren_match.group(1))
            paren_items = split_skill_list(paren_match.group(2))
            if main_item:
                items.append(main_item)
            items.extend(paren_items)
        else:
            if len(part) < 100:
                items.append(part)
    
    return items


def extract_from_lexicon(text: str, section_name: str) -> list[SkillCandidate]:
    candidates = []
    text_lower = text.lower()
    
    for key, normalized in DATA_IA_LEXICON.items():
        if key in text_lower:
            confidence = 0.90 if section_name == "COMPÉTENCES VISÉES" else 0.85
            candidates.append(SkillCandidate(
                label=normalized,
                raw_label=key,
                source_section=section_name,
                source_sentence=text,
                confidence=confidence,
                extraction_method="lexicon",
            ))
    
    return candidates


def normalize_skill_label(candidate: SkillCandidate, context: str = "") -> str:
    label = candidate.label
    
    # Nettoyer les suffixes parasites
    label = re.sub(r'\s*[—\-]\s*(notions pratiques|notions de base|bases|intro).*$', '', label, flags=re.IGNORECASE)
    label = label.strip()
    
    # Capitaliser la première lettre
    if label:
        label = label[0].upper() + label[1:]
    
    context_lower = context.lower()
    
    # Normalisations contextuelles
    if label.lower() == "jointures" and "sql" in context_lower:
        return "Jointures SQL"
    if label.lower() == "sous-requêtes" and "sql" in context_lower:
        return "Sous-requêtes SQL"
    if label.lower() == "agrégations" and "sql" in context_lower:
        return "Agrégations SQL"
    if label.lower() == "dataframes":
        return "Manipulation de dataframes"
    if label.lower() == "régression" and "linéaire" not in context_lower:
        return "Régression statistique"
    if label.lower() == "tableau" and ("bi" in context_lower or "visualisation" in context_lower or "power" in context_lower):
        return "Tableau"
    
    # Capitalisation des acronymes et termes techniques
    label = re.sub(r'\bsql\b', 'SQL', label, flags=re.IGNORECASE)
    label = re.sub(r'\betl\b', 'ETL', label, flags=re.IGNORECASE)
    label = re.sub(r'\bml\b', 'ML', label, flags=re.IGNORECASE)
    label = re.sub(r'\bai\b', 'IA', label, flags=re.IGNORECASE)
    
    return label


def deduplicate_skill_candidates(candidates: list[SkillCandidate]) -> list[SkillCandidate]:
    seen = {}
    
    for candidate in candidates:
        # Normaliser le label avant la déduplication
        normalized = normalize_skill_label(candidate, "")
        
        # Créer une clé de déduplication robuste
        label = normalized.lower().strip()
        # Normaliser les espaces et caractères spéciaux
        label = re.sub(r'\s+', ' ', label)
        label = re.sub(r'[^\w\s]', '', label)
        key = label.strip()
        
        if key not in seen:
            seen[key] = candidate
        else:
            existing = seen[key]
            # Garder l'entrée avec la confiance la plus élevée
            if candidate.confidence > existing.confidence:
                seen[key] = candidate
            elif candidate.confidence == existing.confidence:
                # Même confiance : garder le plus spécifique (plus long)
                if len(candidate.label) > len(existing.label):
                    seen[key] = candidate
    
    return list(seen.values())


def extract_training_skills(text: str) -> list[dict[str, Any]]:
    # Ne pas utiliser clean_text ici car il supprime les retours à la ligne
    # qui sont essentiels pour la détection des sections et des puces
    if not text or not text.strip():
        return []
    
    # Normaliser les espaces multiples mais garder les retours à la ligne
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    sections = detect_training_sections(text)
    
    all_candidates = []
    for section_name, section_text in sections.items():
        candidates = extract_candidates_from_section(section_text, section_name)
        all_candidates.extend(candidates)
    
    # Dédupliquer avant normalisation finale
    deduplicated = deduplicate_skill_candidates(all_candidates)
    
    # Normaliser les labels après déduplication
    results = []
    for i, candidate in enumerate(deduplicated):
        normalized_label = normalize_skill_label(candidate, text)
        
        results.append({
            "id": f"skill_{i+1}",
            "label": normalized_label,
            "raw_label": candidate.raw_label,
            "description": "",
            "aliases": candidate.aliases,
            "category": candidate.category,
            "source_section": candidate.source_section,
            "source_sentence": candidate.source_sentence,
            "confidence": candidate.confidence,
            "status": "pending",
            "type": "subskill",
            "children": [],
            "extraction_method": candidate.extraction_method,
        })
    
    # Dédupliquer à nouveau après normalisation finale
    final_results = []
    seen_labels = {}
    for result in results:
        label_key = result["label"].lower().strip()
        label_key = re.sub(r'\s+', ' ', label_key)
        label_key = re.sub(r'[^\w\s]', '', label_key)
        
        if label_key not in seen_labels:
            seen_labels[label_key] = result
            final_results.append(result)
        else:
            # Garder celui avec la confiance la plus élevée
            existing = seen_labels[label_key]
            if result["confidence"] > existing["confidence"]:
                # Remplacer dans final_results
                final_results = [r for r in final_results if r["id"] != existing["id"]]
                final_results.append(result)
                seen_labels[label_key] = result
    
    final_results.sort(key=lambda x: (-x["confidence"], x["label"]))
    
    return final_results
