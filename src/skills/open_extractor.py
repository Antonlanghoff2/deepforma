from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text


@dataclass
class ExtractedSkill:
    source_label: str = ""
    normalized_label: str = ""
    type: str = "technical_skill"  # technical_skill|soft_skill|tool|knowledge|certification
    source_text: str = ""
    start: int = 0
    end: int = 0
    confidence: float = 0.0
    method: str = "rule"
    referential_id: str | None = None
    referential_source: str | None = None


VERB_PATTERNS: list[tuple[str, str, float]] = [
    # Structures avec "être capable de"
    (r"être\s+capable\s+de\s+(.+?)(?:\.|,|;|$)", "action", 0.90),
    (r"capacité\s+(?:à|de)\s+(.+?)(?:\.|,|;|$)", "action", 0.88),
    (r"aptitude\s+(?:à|de)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),

    # Savoir / savoir-faire
    (r"savoir\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"savoir[- ]faire\s+(?:suivant|en|:)??\s*(.+?)(?:\.|,|;|$)", "action", 0.80),

    # Maîtrise
    (r"maîtris[erise]\s+(.+?)(?:\.|,|;|$)", "action", 0.88),
    (r"maitris[erise]\s+(.+?)(?:\.|,|;|$)", "action", 0.88),

    # Verbes d'action directs
    (r"(?:concevoir|conception\s+(?:et|,|d['e])?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:développer|développement\s+(?:d'|de|et)?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:analyser|analyse\s+(?:d'|de|et)?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:implémenter|implémentation\s+(?:d'|de|et)?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:programmer|programmation\s+(?:d'|de|et)?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:administrer|administration\s+(?:d'|de|et)?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:configurer|configuration\s+(?:d'|de|et)?)\s+(.+?)(?:\.|,|;|$)", "action", 0.85),

    # "permettre de" tournure
    (r"permett(?:re|ant)\s+de\s+(.+?)(?:\.|,|;|$)", "action", 0.80),

    # Être en mesure de
    (r"(?:être|étant)\s+en\s+mesure\s+de\s+(.+?)(?:\.|,|;|$)", "action", 0.85),

    # Acquérir / apprendre
    (r"(?:acquérir|acquisition\s+(?:de|des|d'))\s+(.+?)(?:\.|,|;|$)", "action", 0.80),
    (r"(?:apprendre|apprentissage\s+(?:de|des|d'))\s+(.+?)(?:\.|,|;|$)", "action", 0.80),

    # Mettre en œuvre / mettre en place
    (r"(?:mettre\s+en\s+(?:œuvre|place|œuvre)|mise\s+en\s+(?:œuvre|place))\s+(?:de|d'|des)?\s*(.+?)(?:\.|,|;|$)", "action", 0.88),

    # Comprendre / appréhender
    (r"(?:comprendre|compréhension\s+(?:de|des|d'))\s+(.+?)(?:\.|,|;|$)", "knowledge", 0.80),
    (r"(?:appréhender|appréhension\s+(?:de|des|d'))\s+(.+?)(?:\.|,|;|$)", "knowledge", 0.75),

    # Utiliser / manipuler
    (r"(?:utiliser|utilisation\s+(?:de|d'|des)?)\s*(.+?)(?:\.|,|;|$)", "tool", 0.82),
    (r"(?:manipuler|manipulation\s+(?:de|d'|des)?)\s*(.+?)(?:\.|,|;|$)", "tool", 0.80),

    # Gérer / piloter
    (r"(?:gérer|gestion\s+(?:de|des|d'|du)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:piloter|pilotage\s+(?:de|des|d'|du)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.85),

    # Installer / déployer
    (r"(?:installer|installation\s+(?:d'|de|des)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.85),
    (r"(?:déployer|déploiement\s+(?:d'|de|des)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.88),

    # Réaliser / effectuer
    (r"(?:réaliser|réalisation\s+(?:d'|de|des)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.82),
    (r"(?:effectuer|effectuation\s+(?:d'|de|des)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.80),

    # Appliquer / application
    (r"(?:appliquer|application\s+(?:de|des|d')?)\s*(.+?)(?:\.|,|;|$)", "action", 0.82),

    # Coordonner / organiser
    (r"(?:coordonner|coordination\s+(?:de|des|d')?)\s*(.+?)(?:\.|,|;|$)", "soft_skill", 0.80),
    (r"(?:organiser|organisation\s+(?:de|des|d')?)\s*(.+?)(?:\.|,|;|$)", "soft_skill", 0.80),

    # Animer / encadrer
    (r"(?:animer|animation\s+(?:de|des|d')?)\s*(.+?)(?:\.|,|;|$)", "soft_skill", 0.80),
    (r"(?:encadrer|encadrement\s+(?:de|des|d')?)\s*(.+?)(?:\.|,|;|$)", "soft_skill", 0.80),

    # Participer à / contribuer à
    (r"(?:participer|participation)\s+à\s+(.+?)(?:\.|,|;|$)", "action", 0.75),
    (r"(?:contribuer|contribution)\s+à\s+(.+?)(?:\.|,|;|$)", "action", 0.75),

    # Assurer / garantir
    (r"(?:assurer?\s+(?:la|le|l'|les)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.78),
    (r"(?:garantir?\s+(?:la|le|l'|les)?)\s*(.+?)(?:\.|,|;|$)", "action", 0.78),

    # Collaborer / travailler en équipe
    (r"(?:collaborer|collaboration)\s+(?:avec|au sein)?\s*(.+?)(?:\.|,|;|$)", "soft_skill", 0.80),
    (r"(travailler\s+en\s+équipe)", "soft_skill", 0.85),
    (r"(travail\s+d'équipe)", "soft_skill", 0.85),
]


TOOL_PATTERNS: list[tuple[str, float]] = [
    # Technologies standalone (context-free, lower confidence)
    (r"\b(Python|R\b|Julia|MATLAB|SAS)\b", 0.60),
    (r"\b(SQL|PostgreSQL|MySQL|MongoDB|NoSQL|Redis|Elasticsearch)\b", 0.60),
    (r"\b(TensorFlow|PyTorch|Keras|Scikit-learn|scikit-learn|JAX|Spark MLlib)\b", 0.60),
    (r"\b(Docker|Kubernetes|K8s|Jenkins|GitLab CI|GitHub Actions|CircleCI)\b", 0.60),
    (r"\b(AWS|Azure|GCP|Google Cloud|Amazon Web Services|Cloud|Heroku)\b", 0.55),
    (r"\b(Spark|Kafka|Airbyte|Kestra|Airflow|Hadoop|Flink|Storm)\b", 0.60),
    (r"\b(FastAPI|Flask|Django|Express|Spring|Node\.js|React|Vue|Angular)\b", 0.60),
    (r"\b(Jira|Confluence|Trello|Notion|Slack|Teams)\b", 0.50),
    (r"\b(Tableau|Power BI|Looker|Qlik|Metabase|Superset|Grafana)\b", 0.55),
    (r"\b(Git|GitHub|GitLab|Bitbucket|SVN)\b", 0.55),
    (r"\b(REST|GraphQL|gRPC|SOAP|OData)\b", 0.55),
    (r"\b(HTML5?|CSS3?|JavaScript|TypeScript|PHP|Ruby|Go|Rust|Swift|Kotlin)\b", 0.55),
    (r"\b(Linux|Unix|Windows Server|Bash|PowerShell)\b", 0.55),
    (r"\b(Oracle|SAP|Salesforce|ServiceNow|Splunk)\b", 0.50),
]

KNOWLEDGE_PATTERNS: list[tuple[str, float]] = [
    (r"(?:connaissances?\s+(?:en|des|de|d'|sur)?)\s*(.+?)(?:\.|,|;|$)", 0.75),
    (r"(?:notions?\s+(?:en|de|d'|sur)?)\s*(.+?)(?:\.|,|;|$)", 0.65),
    (r"(?:culture\s+(?:de|des|en|d'|du)?)\s*(.+?)(?:\.|,|;|$)", 0.65),
]

SHORT_PHRASE_MAX = 80

# Domaines de rattachement
IA_KEYWORDS = [
    "ia", "intelligence artificielle", "machine learning", "deep learning",
    "apprentissage automatique", "apprentissage profond", "réseau de neurones",
    "réseaux de neurones", "réseau neuronal", "nlp", "traitement du langage",
    "computer vision", "vision par ordinateur", "reconnaissance d'image",
    "reconnaissance faciale", "rag", "génération augmentée",
    "générative", "generative", "llm", "grand modèle de langage",
    "transformeur", "transformers", "transformer", "embedding",
    "plongement", "vecteur", "similarité sémantique", "recherche sémantique",
    "prompt", "prompt engineering", "ia générative",
    "data science", "science des données", "data mining",
    "classification", "régression", "clustering", "réduction de dimension",
    "big data", "données massives", "data engineering",
]


def _clean_phrase(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip()
    phrase = re.sub(r"^[,\s;:]+", "", phrase)
    phrase = re.sub(r"[,\s;:]+$", "", phrase)
    return clean_text(phrase)


def _find_tool_only(text: str) -> list[ExtractedSkill]:
    """Extract standalone tool/technology mentions not caught by verb patterns."""
    results: list[ExtractedSkill] = []
    seen: set[str] = set()
    for pattern, confidence in TOOL_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            tool = m.group(1)
            if tool.lower() in seen:
                continue
            seen.add(tool.lower())

            # Check if the tool is already near an action verb (avoids duplicate)
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 10)
            before = text[start:m.start()].strip().lower()
            
            # Skip if the tool is clearly an object of a preceding verb
            is_action_context = any(
                re.search(rf"\b{v}\b", before)
                for v in ["utiliser", "utilisation", "avec", "via", "sur",
                          "développer", "développement", "programmer",
                          "programmation", "administrer", "configurer",
                          "installer", "installation", "déployer", "déploiement",
                          "implémenter", "implémentation", "manipuler",
                          "manipulation", "créer", "création", "gérer", "gestion"]
            )
            if is_action_context:
                confidence = 0.50  # Lower standalone tool confidence in action context

            results.append(ExtractedSkill(
                source_label=tool,
                normalized_label=tool,
                type="tool",
                source_text=tool,
                start=m.start(),
                end=m.end(),
                confidence=confidence,
                method="rule_tool_pattern",
            ))
    return results


def _extract_action_skills(text: str) -> list[ExtractedSkill]:
    """Extract skills through action verb patterns."""
    results: list[ExtractedSkill] = []
    seen_phrases: set[str] = set()

    for pattern, skill_type, confidence in VERB_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            phrase = _clean_phrase(m.group(1))
            if not phrase or len(phrase) > SHORT_PHRASE_MAX:
                continue

            key = phrase.lower()
            if key in seen_phrases:
                continue
            seen_phrases.add(key)

            if skill_type == "tool":
                actual_type = "tool"
            elif skill_type == "knowledge":
                actual_type = "knowledge"
            elif skill_type == "soft_skill":
                actual_type = "soft_skill"
            else:
                actual_type = "technical_skill"

            # Check if this contains a tool mention for better classification
            for tool_pattern, _ in TOOL_PATTERNS:
                if re.search(tool_pattern, phrase, re.IGNORECASE):
                    if actual_type == "technical_skill":
                        actual_type = "tool_with_context"
                    break

            normalized = phrase[0].upper() + phrase[1:] if phrase else phrase

            results.append(ExtractedSkill(
                source_label=phrase,
                normalized_label=normalized,
                type=actual_type,
                source_text=text[m.start():m.end()].strip(),
                start=m.start(),
                end=m.end(),
                confidence=confidence,
                method="rule_verb_pattern",
            ))

    return results


def _extract_knowledge(text: str) -> list[ExtractedSkill]:
    """Extract knowledge expressions."""
    results: list[ExtractedSkill] = []
    seen_phrases: set[str] = set()

    for pattern, confidence in KNOWLEDGE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            phrase = _clean_phrase(m.group(1))
            if not phrase or len(phrase) > SHORT_PHRASE_MAX:
                continue

            key = phrase.lower()
            if key in seen_phrases:
                continue
            seen_phrases.add(key)

            normalized = phrase[0].upper() + phrase[1:] if phrase else phrase

            results.append(ExtractedSkill(
                source_label=phrase,
                normalized_label=normalized,
                type="knowledge",
                source_text=text[m.start():m.end()].strip(),
                start=m.start(),
                end=m.end(),
                confidence=confidence,
                method="rule_knowledge_pattern",
            ))

    return results


def _deduplicate(skills: list[ExtractedSkill]) -> list[ExtractedSkill]:
    """Remove near-duplicate extractions, keeping the highest confidence."""
    from collections import OrderedDict

    # Group by normalized label (lowercase)
    groups: dict[str, list[ExtractedSkill]] = OrderedDict()
    for s in skills:
        key = s.normalized_label.lower()
        groups.setdefault(key, []).append(s)

    result: list[ExtractedSkill] = []
    for key, group in groups.items():
        # Keep the one with highest confidence
        best = max(group, key=lambda s: s.confidence)
        result.append(best)

    # Sort by start position then confidence
    result.sort(key=lambda s: (s.start, -s.confidence))
    return result


def _categorize_ia_skills(skills: list[ExtractedSkill]) -> list[ExtractedSkill]:
    """Tag skills with IA-related categorization where relevant."""
    for skill in skills:
        text_lower = (skill.source_label + " " + skill.normalized_label).lower()
        if any(kw in text_lower for kw in IA_KEYWORDS):
            skill.type = "technical_skill"
    return skills


def extract_skills(text: str) -> list[ExtractedSkill]:
    """Main extraction function. Combines all extraction methods."""
    text = clean_text(text)
    if not text:
        return []

    results: list[ExtractedSkill] = []

    # 1. Action verb patterns
    results.extend(_extract_action_skills(text))

    # 2. Knowledge patterns
    results.extend(_extract_knowledge(text))

    # 3. Tool mentions (standalone)
    results.extend(_find_tool_only(text))

    # 4. Categorize IA skills
    results = _categorize_ia_skills(results)

    # 5. Deduplicate
    results = _deduplicate(results)

    return results


def tag_with_ia_categories(
    skills: list[ExtractedSkill],
    ia_predictions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Tag extracted skills with IA family categories from the 18-label model.
    
    The 18-label model is used SECONDARILY:
    - It does NOT determine WHAT skills are present
    - It suggests which IA families each extracted skill might relate to
    """
    if not ia_predictions:
        return [{
            "source_label": s.source_label,
            "normalized_label": s.normalized_label,
            "type": s.type,
            "source_text": s.source_text,
            "start": s.start,
            "end": s.end,
            "confidence": s.confidence,
            "method": s.method,
            "referential_id": s.referential_id,
            "referential_source": s.referential_source,
            "ia_categories": [],
        } for s in skills]

    # Build a text from all extracted labels for IA classification
    ia_text = " ".join(
        s.normalized_label + " " + s.source_label
        for s in skills
    ).lower()

    # Map IA model predictions to this text
    ia_categories = [
        p.get("label", "") for p in ia_predictions
        if p.get("probability", 0) >= 0.35
    ] if ia_predictions else []

    result = []
    for s in skills:
        skill_ia_cats = []
        text_lower = (s.normalized_label + " " + s.source_label).lower()
        
        for cat in ia_categories:
            cat_lower = cat.lower()
            if cat_lower in text_lower:
                skill_ia_cats.append(cat)
            elif any(kw in text_lower for kw in IA_KEYWORDS):
                skill_ia_cats.append(cat)
                break

        result.append({
            "source_label": s.source_label,
            "normalized_label": s.normalized_label,
            "type": s.type,
            "source_text": s.source_text,
            "start": s.start,
            "end": s.end,
            "confidence": s.confidence,
            "method": s.method,
            "referential_id": s.referential_id,
            "referential_source": s.referential_source,
            "ia_categories": skill_ia_cats if skill_ia_cats else [],
        })

    return result
