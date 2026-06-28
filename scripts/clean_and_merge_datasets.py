from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

COMMON_COLUMNS = [
    "formation_id",
    "intitule",
    "description",
    "objectifs",
    "programme",
    "public_cible",
    "prerequis",
    "niveau",
    "modalite",
    "duree",
    "certification",
    "codes_rome",
    "organisme",
    "source_dataset",
    "texte_modele",
    "competences_ia",
    "competences_ia_suggerees",
    "est_lie_ia",
    "statut_annotation",
    "source_file",
    "source_row",
    "formation_group_id",
    "cleaning_flags",
]

COMPETENCE_REFERENTIAL = [
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "IA générative",
    "LLM",
    "RAG",
    "Computer Vision",
    "Data Science",
    "Data Engineering",
    "MLOps",
    "Python pour l’IA",
    "Traitement de données",
    "Modèles prédictifs",
    "Apprentissage supervisé",
    "Apprentissage non supervisé",
    "Reinforcement Learning",
    "Déploiement de modèles",
    "Éthique de l’IA",
    "Gouvernance de l’IA",
    "Gestion de projet IA",
    "Prompt Engineering",
    "No-code / Low-code",
    "Automatisation",
    "Big Data",
    "Visualisation",
    "Séries temporelles",
]

COMPETENCE_SYNONYMS = {
    "Machine Learning": [
        "machine learning",
        "ml",
        "apprentissage automatique",
        "apprentissage automatique supervisé",
    ],
    "Deep Learning": ["deep learning", "dl", "reseaux de neurones profonds"],
    "NLP": [
        "nlp",
        "traitement automatique du langage",
        "traitement du langage naturel",
        "tal",
        "language processing",
    ],
    "IA générative": [
        "ia generative",
        "intelligence artificielle generative",
        "generative ai",
        "ai generative",
        "ia generatif",
    ],
    "LLM": [
        "llm",
        "large language model",
        "large language models",
        "modele de langage",
        "modeles de langage",
    ],
    "RAG": [
        "rag",
        "retrieval augmented generation",
    ],
    "Computer Vision": [
        "computer vision",
        "vision par ordinateur",
    ],
    "Data Science": [
        "data science",
        "science des donnees",
    ],
    "Data Engineering": [
        "data engineering",
        "ingenierie des donnees",
    ],
    "MLOps": [
        "mlops",
        "machine learning operations",
    ],
    "Python pour l’IA": [
        "python pour l ia",
        "python pour l ia",
        "python applique a l ia",
        "python applique a ia",
        "python et ia",
    ],
    "Traitement de données": [
        "traitement de donnees",
        "preparation des donnees",
        "data preprocessing",
    ],
    "Modèles prédictifs": [
        "modeles predictifs",
        "modeles de prediction",
        "predictive modeling",
    ],
    "Apprentissage supervisé": ["apprentissage supervise", "supervised learning"],
    "Apprentissage non supervisé": [
        "apprentissage non supervise",
        "unsupervised learning",
    ],
    "Reinforcement Learning": [
        "reinforcement learning",
        "apprentissage par renforcement",
    ],
    "Déploiement de modèles": [
        "deploiement de modeles",
        "model deployment",
    ],
    "Éthique de l’IA": [
        "ethique de l ia",
        "ethique ia",
        "ethique de l intelligence artificielle",
        "ethique ia rgpd",
        "ethique ia et rgpd",
    ],
    "Gouvernance de l’IA": [
        "gouvernance de l ia",
        "gouvernance ia",
        "ia governance",
    ],
    "Gestion de projet IA": [
        "gestion de projet ia",
        "pilotage de projet ia",
    ],
    "Prompt Engineering": [
        "prompt engineering",
        "ingenierie de prompt",
    ],
    "No-code / Low-code": [
        "no code",
        "low code",
        "no code low code",
        "no code low code automation",
    ],
    "Automatisation": [
        "automatisation",
        "automation",
        "workflow automation",
    ],
    "Big Data": [
        "big data",
    ],
    "Visualisation": [
        "visualisation",
        "visualization",
        "dataviz",
    ],
    "Séries temporelles": [
        "series temporelles",
        "time series",
    ],
}

AI_SIGNAL_PATTERNS = [
    (r"\bintelligence artificielle\b", 3.0, "Intelligence artificielle"),
    (r"\bia generative\b", 3.0, "IA générative"),
    (r"\bmachine learning\b", 3.0, "Machine Learning"),
    (r"\bdeep learning\b", 3.0, "Deep Learning"),
    (r"\breseaux? de neurones\b", 3.0, "Réseaux de neurones"),
    (r"\bnlp\b", 3.0, "NLP"),
    (r"\btraitement automatique du langage\b", 3.0, "NLP"),
    (r"\bllm\b", 3.0, "LLM"),
    (r"\blarge language models?\b", 3.0, "LLM"),
    (r"\brag\b", 3.0, "RAG"),
    (r"\bretrieval augmented generation\b", 3.0, "RAG"),
    (r"\bcomputer vision\b", 3.0, "Computer Vision"),
    (r"\bvision par ordinateur\b", 3.0, "Computer Vision"),
    (r"\bmlops\b", 3.0, "MLOps"),
    (r"\breinforcement learning\b", 3.0, "Reinforcement Learning"),
    (r"\bapprentissage supervise\b", 3.0, "Apprentissage supervisé"),
    (r"\bapprentissage non supervise\b", 3.0, "Apprentissage non supervisé"),
    (r"\bpython\b.{0,40}\bia\b", 2.5, "Python pour l’IA"),
    (r"\bmodeles? predictifs?\b", 2.5, "Modèles prédictifs"),
    (r"\bprompt engineering\b", 2.0, "Prompt Engineering"),
    (r"\bdata science\b", 1.5, "Data Science"),
    (r"\bdata engineering\b", 1.5, "Data Engineering"),
    (r"\bchatgpt\b", 1.5, "IA générative"),
    (r"\bgpt\b", 1.5, "LLM"),
    (r"\btransformers?\b", 1.0, "LLM"),
    (r"\bembedding[s]?\b", 1.0, "LLM"),
    (r"\bvector database\b", 1.0, "RAG"),
    (r"\bfine[- ]?tuning\b", 1.0, "LLM"),
]

NON_AI_SIGNAL_PATTERNS = [
    r"\banglais\b",
    r"\bfrancais\b",
    r"\bespagnol\b",
    r"\ballemand\b",
    r"\bitralien\b",
    r"\bcommunication\b",
    r"\bcomptabilite\b",
    r"\bgestion\b",
    r"\bvente\b",
    r"\bcommerce\b",
    r"\bmarketing\b",
    r"\brestauration\b",
    r"\bcuisine\b",
    r"\bsante\b",
    r"\bcoiffure\b",
    r"\besthetique\b",
    r"\blogistique\b",
    r"\btransport\b",
    r"\bdroit\b",
    r"\bsecurite\b",
    r"\badministration\b",
    r"\bressources humaines\b",
    r"\brh\b",
    r"\bbureautique\b",
    r"\bimmobilier\b",
    r"\bformation linguistique\b",
]

TEXT_COLUMNS_ORDER = [
    ("intitule", "Intitulé"),
    ("description", "Description"),
    ("objectifs", "Objectifs"),
    ("programme", "Programme"),
    ("public_cible", "Public cible"),
    ("prerequis", "Prérequis"),
    ("niveau", "Niveau"),
    ("modalite", "Modalité"),
    ("duree", "Durée"),
    ("certification", "Certification"),
    ("codes_rome", "Codes ROME"),
    ("organisme", "Organisme"),
]

SOURCE_COLUMN_ALIASES = {
    "intitule de la formation": "intitule",
    "intitule": "intitule",
    "organisme de formation": "organisme",
    "organisme": "organisme",
    "type de certification": "certification_type",
    "code certification": "certification_code",
    "niveau": "niveau",
    "codes rome": "codes_rome",
    "competences ia extraites": "competences_ia",
    "competences majeures": "description",
    "modalite": "modalite",
    "duree": "duree",
    "public cible": "public_cible",
    "tags trendradar": "source_tags",
    "tags": "source_tags",
    "✅ relu / validé (oui/non)": "review_status",
    "🗒 corrections / remarques": "remarks",
    "#": "source_index",
}

NULL_LIKE = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "nr",
    "non renseigné",
    "non renseigne",
    "-",
    "--",
}


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = normalize_unicode(text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = text.replace("\n", " ")
    text = collapse_whitespace(text)
    if text.lower() in NULL_LIKE:
        return ""
    return text


def normalized_lookup(text: Any) -> str:
    text = clean_text(text)
    text = strip_accents(text).lower()
    text = re.sub(r"[_/|,;:()\\[\]{}<>«»“”\"'`´]", " ", text)
    text = re.sub(r"[^a-z0-9+&.-]+", " ", text)
    return collapse_whitespace(text)


def is_null_like(value: Any) -> bool:
    return clean_text(value) == ""


def stable_hash(*parts: str, length: int = 16) -> str:
    payload = "||".join(parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:length]


def parse_multi_values(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parts = re.split(r"\s*\|\s*|\s*;\s*|\s*\n+\s*", text)
    cleaned = []
    for part in parts:
        part = collapse_whitespace(part)
        if part and part.lower() not in NULL_LIKE:
            cleaned.append(part)
    return cleaned


def normalize_duration(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    low = normalized_lookup(text)
    low = low.replace("heures", "h").replace("heure", "h").replace("hrs", "h")
    low = low.replace("jours", "j").replace("jour", "j")
    low = low.replace("semaines", "sem").replace("semaine", "sem")
    low = low.replace("mois", "mois")
    low = low.replace("ans", "an").replace("annees", "an").replace("année", "an").replace("annee", "an")
    low = re.sub(r"\b(h|j|sem|mois|an)\b", r" \1", low)
    low = re.sub(r"\s+", " ", low).strip()
    match = re.match(r"^(\d+(?:[.,]\d+)?)\s*(h|j|sem|mois|an)$", low)
    if match:
        number = match.group(1).replace(",", ".")
        if number.endswith(".0"):
            number = number[:-2]
        return f"{number} {match.group(2)}"
    return clean_text(value)


def normalize_level(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    norm = normalized_lookup(text)
    level_map = {
        "debutant": "Débutant",
        "initiation": "Débutant",
        "intermediaire": "Intermédiaire",
        "avance": "Avancé",
        "expert": "Expert",
        "tous niveaux": "Tous niveaux",
        "tous niveaux confondus": "Tous niveaux",
        "bac2 deug bt dut niveau 5": "Bac+2 / Niveau 5",
        "bac3 licence niveau 6": "Bac+3 / Niveau 6",
        "bac4 master niveau 7": "Bac+4 / Niveau 7",
        "bac5 master niveau 7": "Bac+5 / Niveau 7",
    }
    for key, mapped in level_map.items():
        if key in norm:
            return mapped
    return collapse_whitespace(text)


def normalize_modalite(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    norm = normalized_lookup(text)
    if "distance" in norm or "en ligne" in norm or "remote" in norm:
        return "À distance"
    if "presentiel" in norm:
        return "Présentiel"
    if "hybride" in norm or "mixte" in norm:
        return "Hybride"
    return collapse_whitespace(text)


def normalize_codes_rome(value: Any) -> str:
    parts = parse_multi_values(value)
    if not parts:
        return ""
    codes = []
    for part in parts:
        norm = re.sub(r"\s+", "", part.upper())
        if norm and norm not in codes:
            codes.append(norm)
    return " | ".join(codes)


def normalize_certification(cert_type: Any, cert_code: Any) -> str:
    type_text = clean_text(cert_type)
    code_text = clean_text(cert_code)
    if type_text and code_text:
        return f"{type_text} {code_text}"
    return type_text or code_text


def canonicalize_competence(value: str) -> str | None:
    norm = normalized_lookup(value)
    if not norm:
        return None
    for canonical, synonyms in COMPETENCE_SYNONYMS.items():
        canonical_norm = normalized_lookup(canonical)
        if norm == canonical_norm:
            return canonical
        for synonym in synonyms:
            syn_norm = normalized_lookup(synonym)
            if norm == syn_norm or syn_norm in norm or norm in syn_norm:
                return canonical
    return None


def normalize_competence_field(value: Any) -> list[str]:
    items = parse_multi_values(value)
    normalized = []
    for item in items:
        canonical = canonicalize_competence(item)
        if canonical:
            normalized.append(canonical)
        else:
            fallback = collapse_whitespace(item)
            if fallback and fallback.lower() not in {"distance", "présentiel", "presentiel"}:
                normalized.append(fallback)
    deduped = []
    seen = set()
    for item in normalized:
        key = normalized_lookup(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def build_text_modele(row: pd.Series) -> str:
    parts = []
    for column, prefix in TEXT_COLUMNS_ORDER:
        value = clean_text(row.get(column, ""))
        if value:
            parts.append(f"{prefix} : {value}")
    return "\n".join(parts)


def build_group_key(intitule: str, organisme: str = "", certification: str = "", modalite: str = "", duree: str = "") -> str:
    fields = [normalized_lookup(intitule), normalized_lookup(organisme), normalized_lookup(certification), normalized_lookup(modalite), normalized_lookup(duree)]
    title = collapse_whitespace(fields[0])
    title = re.sub(r"\b(formation|cours|stage|atelier|module|initiation|perfectionnement|certification|programme)\b", " ", title)
    title = collapse_whitespace(title)
    return stable_hash(title, *fields[1:])


def build_formation_id(row: pd.Series) -> str:
    return stable_hash(
        normalized_lookup(row.get("source_file", "")),
        str(row.get("source_row", "")),
        normalized_lookup(row.get("intitule", "")),
        normalized_lookup(row.get("organisme", "")),
        normalized_lookup(row.get("certification", "")),
        normalized_lookup(row.get("codes_rome", "")),
        normalized_lookup(row.get("modalite", "")),
        normalized_lookup(row.get("duree", "")),
    )


def classify_ai_status(row: pd.Series) -> tuple[str, Any, list[str]]:
    competences = row.get("competences_ia", "")
    has_labels = bool(clean_text(competences))
    text_blocks = [
        clean_text(row.get("intitule", "")),
        clean_text(row.get("description", "")),
        clean_text(row.get("objectifs", "")),
        clean_text(row.get("programme", "")),
        clean_text(row.get("certification", "")),
        clean_text(row.get("codes_rome", "")),
    ]
    combined_text = " ".join(block for block in text_blocks if block)
    norm_text = normalized_lookup(combined_text)

    ai_matches: list[str] = []
    ai_score = 0.0
    for pattern, weight, label in AI_SIGNAL_PATTERNS:
        if re.search(pattern, norm_text):
            ai_score += weight
            ai_matches.append(label)

    non_ai_matches = [label for label in NON_AI_SIGNAL_PATTERNS if re.search(label, norm_text)]
    non_ai_score = len(non_ai_matches)

    if has_labels:
        return "ia_confirmee", 1, ai_matches

    if ai_score >= 2.0:
        return "a_verifier", pd.NA, ai_matches

    if ai_score > 0.0 and any(
        re.search(pattern, normalized_lookup(row.get("intitule", "")))
        for pattern, *_ in AI_SIGNAL_PATTERNS[:20]
    ):
        return "a_verifier", pd.NA, ai_matches

    if non_ai_score >= 1 and ai_score == 0.0:
        return "non_ia_confirmee", 0, []

    if ai_matches:
        return "a_verifier", pd.NA, ai_matches

    return "a_verifier", pd.NA, []


def suggest_competences(row: pd.Series) -> list[str]:
    text_blocks = [
        clean_text(row.get("intitule", "")),
        clean_text(row.get("description", "")),
        clean_text(row.get("objectifs", "")),
        clean_text(row.get("programme", "")),
        clean_text(row.get("certification", "")),
    ]
    combined_text = normalized_lookup(" ".join(text_blocks))
    suggestions = []
    for pattern, _, label in AI_SIGNAL_PATTERNS:
        if re.search(pattern, combined_text):
            canonical = label
            if canonical in COMPETENCE_REFERENTIAL and canonical not in suggestions:
                suggestions.append(canonical)
    return suggestions


def detect_fuzzy_column(source_columns: Iterable[str], target_norm: str) -> str | None:
    normalized_map = {normalized_lookup(col): col for col in source_columns}
    if target_norm in normalized_map:
        return normalized_map[target_norm]
    candidates = difflib.get_close_matches(target_norm, list(normalized_map.keys()), n=1, cutoff=0.72)
    if candidates:
        return normalized_map[candidates[0]]
    for norm_col, original in normalized_map.items():
        if target_norm in norm_col or norm_col in target_norm:
            return original
    return None


def map_source_columns(df: pd.DataFrame) -> tuple[dict[str, str], list[dict[str, str]]]:
    mapping: dict[str, str] = {}
    report: list[dict[str, str]] = []
    for source_col in df.columns:
        norm = normalized_lookup(source_col)
        target = SOURCE_COLUMN_ALIASES.get(norm)
        if target is None:
            target = None
            if "intitule" in norm and "formation" in norm:
                target = "intitule"
            elif "organisme" in norm:
                target = "organisme"
            elif "competence" in norm and "ia" in norm:
                target = "competences_ia"
            elif "rome" in norm:
                target = "codes_rome"
            elif "modalite" in norm:
                target = "modalite"
            elif "duree" in norm:
                target = "duree"
            elif "niveau" in norm:
                target = "niveau"
        if target:
            mapping[source_col] = target
            report.append(
                {
                    "source_column": source_col,
                    "target_column": target,
                    "action": "mapped",
                }
            )
        else:
            report.append(
                {
                    "source_column": source_col,
                    "target_column": "",
                    "action": "ignored",
                }
            )
    return mapping, report


def read_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=object, keep_default_na=True)


def column_letter(index: int) -> str:
    letters = ""
    while index >= 0:
        index, rem = divmod(index, 26)
        letters = chr(65 + rem) + letters
        index -= 1
    return letters


def column_index(ref: str) -> int:
    value = 0
    for ch in ref:
        value = value * 26 + (ord(ch) - 64)
    return value


def read_xlsx_file(path: Path) -> dict[str, pd.DataFrame]:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", ns)
        }
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", ns):
                texts = [t.text or "" for t in item.findall(".//main:t", ns)]
                shared_strings.append("".join(texts))

        def cell_value(cell: ET.Element) -> Any:
            cell_type = cell.attrib.get("t")
            value = cell.find("main:v", ns)
            if cell_type == "s" and value is not None:
                return shared_strings[int(value.text or 0)]
            if cell_type == "inlineStr":
                inline = cell.find("main:is", ns)
                if inline is not None:
                    texts = [t.text or "" for t in inline.findall(".//main:t", ns)]
                    return "".join(texts)
            if value is None:
                return ""
            return value.text or ""

        sheets: dict[str, pd.DataFrame] = {}
        for sheet in workbook.find("main:sheets", ns):
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{ns['rel']}}}id"]
            target = rel_map[rel_id]
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_root = ET.fromstring(zf.read(target))
            rows: list[dict[str, Any]] = []
            for row in sheet_root.findall(".//main:sheetData/main:row", ns):
                row_data: dict[str, Any] = {}
                for cell in row.findall("main:c", ns):
                    ref = cell.attrib.get("r", "")
                    col = re.match(r"[A-Z]+", ref)
                    if not col:
                        continue
                    row_data[col.group(0)] = cell_value(cell)
                rows.append(row_data)
            if not rows:
                sheets[sheet_name] = pd.DataFrame()
                continue
            max_col = max((max((column_index(c) for c in row.keys()), default=0) for row in rows), default=0)
            headers = [rows[0].get(column_letter(i - 1), "") for i in range(1, max_col + 1)]
            data_rows = []
            for row in rows[1:]:
                data_rows.append([row.get(column_letter(i - 1), "") for i in range(1, max_col + 1)])
            sheets[sheet_name] = pd.DataFrame(data_rows, columns=headers)
        return sheets


def inspect_dataframe(df: pd.DataFrame, name: str, sample_rows: int = 3) -> dict[str, Any]:
    return {
        "file": name,
        "shape": f"{df.shape[0]} x {df.shape[1]}",
        "columns": json.dumps(list(df.columns), ensure_ascii=False),
        "dtypes": json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()}, ensure_ascii=False),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": json.dumps(df.isna().sum().to_dict(), ensure_ascii=False),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": json.dumps(df.head(sample_rows).replace({pd.NA: None}).to_dict(orient="records"), ensure_ascii=False, default=str),
    }


def detect_input_files(root: Path) -> dict[str, list[Path]]:
    return {
        "csv": sorted(root.glob("*.csv")),
        "xlsx": sorted(root.glob("*.xlsx")),
        "ipynb": sorted(root.glob("*.ipynb")),
    }


def source_kind_from_columns(columns: Iterable[str]) -> str:
    norms = {normalized_lookup(col) for col in columns}
    if any("competences ia extraites" in col for col in norms):
        return "ia"
    if any("competences majeures" in col for col in norms):
        return "generalist"
    return "unknown"


def standardize_source_dataframe(
    df: pd.DataFrame,
    source_file: Path,
    source_dataset: str,
    source_kind: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    mapping, mapping_report = map_source_columns(df)
    standardized = pd.DataFrame(index=df.index)
    raw_source_columns = set(df.columns)

    for source_col, target_col in mapping.items():
        standardized[target_col] = df[source_col]

    for column in [
        "intitule",
        "description",
        "objectifs",
        "programme",
        "public_cible",
        "prerequis",
        "niveau",
        "modalite",
        "duree",
        "certification",
        "codes_rome",
        "organisme",
        "competences_ia",
    ]:
        if column not in standardized.columns:
            standardized[column] = ""

    cert_type_col = detect_fuzzy_column(df.columns, normalized_lookup("Type de certification"))
    cert_code_col = detect_fuzzy_column(df.columns, normalized_lookup("Code certification"))
    if cert_type_col or cert_code_col:
        cert_types = df[cert_type_col] if cert_type_col else pd.Series([""] * len(df), index=df.index)
        cert_codes = df[cert_code_col] if cert_code_col else pd.Series([""] * len(df), index=df.index)
        standardized["certification"] = [
            normalize_certification(
                cert_types.iloc[pos] if pos < len(cert_types) else "",
                cert_codes.iloc[pos] if pos < len(cert_codes) else "",
            )
            for pos in range(len(df))
        ]

    standardized["source_dataset"] = source_dataset
    standardized["source_file"] = source_file.name
    standardized["source_row"] = list(range(1, len(df) + 1))

    standardized["intitule"] = standardized["intitule"].apply(clean_text)
    standardized["description"] = standardized["description"].apply(clean_text)
    standardized["objectifs"] = standardized["objectifs"].apply(clean_text)
    standardized["programme"] = standardized["programme"].apply(clean_text)
    standardized["public_cible"] = standardized["public_cible"].apply(clean_text)
    standardized["prerequis"] = standardized["prerequis"].apply(clean_text)
    standardized["niveau"] = standardized["niveau"].apply(normalize_level)
    standardized["modalite"] = standardized["modalite"].apply(normalize_modalite)
    standardized["duree"] = standardized["duree"].apply(normalize_duration)
    standardized["certification"] = standardized["certification"].apply(clean_text)
    standardized["codes_rome"] = standardized["codes_rome"].apply(normalize_codes_rome)
    standardized["organisme"] = standardized["organisme"].apply(clean_text)

    if source_kind == "ia":
        comp_col = detect_fuzzy_column(df.columns, normalized_lookup("Compétences IA extraites"))
        if comp_col:
            standardized["competences_ia"] = df[comp_col].apply(lambda v: " | ".join(normalize_competence_field(v)))
    else:
        standardized["competences_ia"] = ""

    standardized["texte_modele"] = standardized.apply(build_text_modele, axis=1)
    standardized["formation_id"] = standardized.apply(build_formation_id, axis=1)
    standardized["formation_group_id"] = standardized.apply(
        lambda row: build_group_key(row.get("intitule", ""), row.get("organisme", ""), row.get("certification", ""), row.get("modalite", ""), row.get("duree", "")),
        axis=1,
    )

    cleaning_flags = []
    for idx, row in standardized.iterrows():
        flags = []
        if not row["intitule"]:
            flags.append("missing_title")
        if not row["texte_modele"]:
            flags.append("empty_text")
        cleaning_flags.append(" | ".join(flags))
    standardized["cleaning_flags"] = cleaning_flags

    standardized["competences_ia"] = standardized["competences_ia"].apply(lambda v: " | ".join(normalize_competence_field(v)) if clean_text(v) else "")

    statuses = standardized.apply(classify_ai_status, axis=1, result_type="expand")
    standardized["statut_annotation"] = statuses[0]
    standardized["est_lie_ia"] = statuses[1]
    standardized["competences_ia_suggerees"] = standardized.apply(
        lambda row: " | ".join(suggest_competences(row)) if row["statut_annotation"] == "a_verifier" else "",
        axis=1,
    )
    standardized.loc[
        (standardized["statut_annotation"] == "ia_confirmee") & standardized["competences_ia"].eq(""),
        "statut_annotation",
    ] = "a_verifier"
    standardized.loc[standardized["statut_annotation"] == "a_verifier", "est_lie_ia"] = pd.NA

    if source_kind == "generalist":
        source_tags_col = detect_fuzzy_column(df.columns, normalized_lookup("Tags"))
        if source_tags_col:
            standardized["source_tags"] = df[source_tags_col].apply(clean_text)
        else:
            standardized["source_tags"] = ""
    else:
        standardized["source_tags"] = ""

    report = mapping_report
    ignored = [item for item in report if item["action"] == "ignored"]
    if ignored:
        report.append(
            {
                "source_column": "(summary)",
                "target_column": "",
                "action": f"{len(ignored)} ignored columns",
            }
        )
    standardized.attrs["column_mapping_report"] = report
    standardized.attrs["source_raw_columns"] = sorted(raw_source_columns)
    return standardized, report


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    signature_cols = [
        "intitule",
        "description",
        "objectifs",
        "programme",
        "public_cible",
        "prerequis",
        "niveau",
        "modalite",
        "duree",
        "certification",
        "codes_rome",
        "organisme",
        "source_dataset",
        "competences_ia",
        "competences_ia_suggerees",
        "statut_annotation",
    ]
    before = len(df)
    normalized = df.copy()
    for col in signature_cols:
        normalized[col] = normalized[col].map(clean_text)
    deduped = df.loc[~normalized.duplicated(signature_cols, keep="first")].copy()
    removed = before - len(deduped)
    return deduped, removed


def detect_near_duplicates(df: pd.DataFrame, threshold: float = 0.88) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if df.empty:
        df = df.copy()
        df["near_duplicate_cluster_id"] = []
        return df, []

    working = df.copy().reset_index(drop=True)
    working["_title_norm"] = working["intitule"].map(normalized_lookup)
    working["_bucket"] = working["_title_norm"].str.slice(0, 18)
    cluster_ids = [""] * len(working)
    cluster_reports: list[dict[str, Any]] = []
    cluster_counter = 0

    for bucket, indices in working.groupby("_bucket").groups.items():
        bucket_indices = list(indices)
        reps: list[tuple[str, int]] = []
        for idx in bucket_indices:
            title = working.at[idx, "_title_norm"]
            assigned = None
            for rep_title, rep_cluster in reps:
                ratio = difflib.SequenceMatcher(None, title, rep_title).ratio()
                if ratio >= threshold:
                    assigned = rep_cluster
                    break
            if assigned is None:
                cluster_counter += 1
                assigned = f"dup_{cluster_counter:05d}"
                reps.append((title, assigned))
            cluster_ids[idx] = assigned

    working["near_duplicate_cluster_id"] = cluster_ids
    for cluster_id, group in working.groupby("near_duplicate_cluster_id"):
        if len(group) > 1:
            cluster_reports.append(
                {
                    "cluster_id": cluster_id,
                    "size": int(len(group)),
                    "example_titles": " | ".join(group["intitule"].head(3).tolist()),
                }
            )
    return working.drop(columns=["_title_norm", "_bucket"]), cluster_reports


def ensure_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in COMMON_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column not in {"est_lie_ia", "source_row"} else pd.NA
    ordered = [column for column in COMMON_COLUMNS if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]


def build_report_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(columns=["section", "metric", "value", "details"])
    df = pd.DataFrame(entries)
    for column in ["section", "metric", "value", "details"]:
        if column not in df.columns:
            df[column] = ""
    return df[["section", "metric", "value", "details"]]


def detect_competence_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for item in df.get("competences_ia", pd.Series(dtype=str)).fillna(""):
        for competence in parse_multi_values(item):
            counter[competence] += 1
    if not counter:
        return pd.DataFrame(columns=["competence", "occurrences"])
    return pd.DataFrame(sorted(counter.items(), key=lambda x: (-x[1], x[0])), columns=["competence", "occurrences"])


def detect_ai_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("statut_annotation", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def detect_class_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for status, group in df.groupby("statut_annotation", dropna=False):
        stats.append(
            {
                "section": "category_stats",
                "metric": str(status),
                "value": int(len(group)),
                "details": f"unique_groups={group['formation_group_id'].nunique()}",
            }
        )
    return build_report_frame(stats)


def write_simple_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    def esc(text: Any) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def cell_xml(ref: str, value: Any) -> str:
        if value is None or value == "":
            return ""
        if isinstance(value, bool):
            return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
        if isinstance(value, (int, float)) and not pd.isna(value):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return f'<c r="{ref}"><v>{value}</v></c>'
        if pd.isna(value):
            return ""
        text = esc(value)
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'

    sheet_names = list(sheets.keys())
    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    for idx, sheet_name in enumerate(sheet_names, start=1):
        workbook_xml.append(f'<sheet name="{esc(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    workbook_xml.extend(["</sheets>", "</workbook>"])

    rels_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx in range(1, len(sheet_names) + 1):
        rels_xml.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels_xml.append(
        '<Relationship Id="rId{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'.format(
            len(sheet_names) + 1
        )
    )
    rels_xml.append("</Relationships>")

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>
"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
""")
        zf.writestr("xl/workbook.xml", "\n".join(workbook_xml))
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(rels_xml))
        zf.writestr("xl/styles.xml", styles_xml)

        for idx, (sheet_name, frame) in enumerate(sheets.items(), start=1):
            frame = frame.copy().astype(object)
            frame = frame.where(pd.notna(frame), "")
            rows_xml = []
            header = list(frame.columns)
            rows_xml.append('<row r="1">')
            for col_idx, value in enumerate(header, start=1):
                ref = f"{column_letter(col_idx - 1)}1"
                rows_xml.append(cell_xml(ref, value))
            rows_xml.append("</row>")
            for row_idx, (_, row) in enumerate(frame.iterrows(), start=2):
                rows_xml.append(f'<row r="{row_idx}">')
                for col_idx, value in enumerate(row.tolist(), start=1):
                    ref = f"{column_letter(col_idx - 1)}{row_idx}"
                    rows_xml.append(cell_xml(ref, value))
                rows_xml.append("</row>")
            sheet_xml = [
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                "<sheetData>",
                *rows_xml,
                "</sheetData>",
                "</worksheet>",
            ]
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", "\n".join(sheet_xml))


def print_inspection_summary(inspections: list[dict[str, Any]]) -> None:
    print("\n=== INSPECTION DES FICHIERS DÉTECTÉS ===")
    for item in inspections:
        print(f"\nFichier : {item['file']}")
        print(f"Dernière forme connue : {item['shape']}")
        print("Colonnes :", item["columns"])
        print("Types :", item["dtypes"])
        print("Valeurs manquantes totales :", item["missing_values"])
        print("Doublons exacts :", item["duplicate_rows"])
        print("Exemples :", item["sample_rows"])


def build_processed_outputs(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detected = detect_input_files(root)
    print("\nFichiers détectés :")
    for kind, paths in detected.items():
        for path in paths:
            print(f"- [{kind}] {path.name}")

    inspections: list[dict[str, Any]] = []
    source_frames: list[tuple[Path, pd.DataFrame, str]] = []
    source_reports: list[dict[str, Any]] = []

    for csv_path in detected["csv"]:
        df = read_csv_file(csv_path)
        inspections.append(inspect_dataframe(df, csv_path.name))
        kind = source_kind_from_columns(df.columns)
        source_frames.append((csv_path, df, kind))

    for xlsx_path in detected["xlsx"]:
        workbook = read_xlsx_file(xlsx_path)
        for sheet_name, df in workbook.items():
            inspections.append(inspect_dataframe(df, f"{xlsx_path.name}::{sheet_name}"))
            kind = source_kind_from_columns(df.columns)
            source_frames.append((xlsx_path, df, kind))

    print_inspection_summary(inspections)

    cleaned_frames: list[pd.DataFrame] = []
    all_mapping_reports: list[dict[str, Any]] = []

    for source_file, frame, kind in source_frames:
        source_dataset = "dataset_ia" if kind == "ia" else "dataset_generaliste"
        if frame.empty:
            continue
        standardized, mapping_report = standardize_source_dataframe(frame, source_file, source_dataset, kind)
        cleaned_frames.append(standardized)
        for item in mapping_report:
            all_mapping_reports.append(
                {
                    "section": "column_mapping",
                    "metric": source_file.name,
                    "value": item["source_column"],
                    "details": f"{item['action']} -> {item['target_column']}",
                }
            )

    if not cleaned_frames:
        raise RuntimeError("Aucun fichier source compatible n'a été détecté.")

    merged = pd.concat(cleaned_frames, ignore_index=True, sort=False)
    before_dedup = len(merged)
    merged, removed_exact = drop_exact_duplicates(merged)
    merged, near_duplicate_clusters = detect_near_duplicates(merged)
    merged["texte_modele"] = merged["texte_modele"].map(clean_text)
    merged = ensure_common_columns(merged)

    report_rows: list[dict[str, Any]] = []
    report_rows.extend(all_mapping_reports)
    report_rows.append(
        {
            "section": "summary",
            "metric": "rows_before_dedup",
            "value": before_dedup,
            "details": "",
        }
    )
    report_rows.append(
        {
            "section": "summary",
            "metric": "rows_after_dedup",
            "value": len(merged),
            "details": "",
        }
    )
    report_rows.append(
        {
            "section": "summary",
            "metric": "exact_duplicates_removed",
            "value": removed_exact,
            "details": "",
        }
    )
    report_rows.append(
        {
            "section": "summary",
            "metric": "near_duplicate_groups",
            "value": len(near_duplicate_clusters),
            "details": json.dumps(near_duplicate_clusters[:10], ensure_ascii=False),
        }
    )

    category_stats = detect_ai_label_distribution(merged)
    for _, row in category_stats.iterrows():
        report_rows.append(
            {
                "section": "category",
                "metric": str(row["statut_annotation"]),
                "value": int(row["count"]),
                "details": "",
            }
        )

    comp_dist = detect_competence_distribution(merged)
    for _, row in comp_dist.head(50).iterrows():
        report_rows.append(
            {
                "section": "competence",
                "metric": str(row["competence"]),
                "value": int(row["occurrences"]),
                "details": "",
            }
        )

    training_df = merged[merged["statut_annotation"].isin(["ia_confirmee", "non_ia_confirmee"])].copy()
    verifier_df = merged[merged["statut_annotation"] == "a_verifier"].copy()
    ia_df = merged[merged["statut_annotation"] == "ia_confirmee"].copy()
    non_ia_df = merged[merged["statut_annotation"] == "non_ia_confirmee"].copy()

    train_path = output_dir / "dataset_entrainement.csv"
    verify_path = output_dir / "dataset_a_verifier.csv"
    all_path = output_dir / "dataset_formations_nettoye.csv"
    xlsx_path = output_dir / "dataset_formations_nettoye.xlsx"

    for df in [merged, training_df, verifier_df]:
        df["est_lie_ia"] = df["est_lie_ia"].astype("Int64")

    merged.to_csv(all_path, index=False, encoding="utf-8")
    training_df.to_csv(train_path, index=False, encoding="utf-8")
    verifier_df.to_csv(verify_path, index=False, encoding="utf-8")

    report_df = build_report_frame(report_rows)
    competence_ref_df = pd.DataFrame(
        {
            "competence": COMPETENCE_REFERENTIAL,
            "synonymes": [
                " | ".join(COMPETENCE_SYNONYMS.get(comp, []))
                for comp in COMPETENCE_REFERENTIAL
            ],
        }
    )
    write_simple_xlsx(
        xlsx_path,
        {
            "Toutes_les_formations": merged,
            "Entrainement": training_df,
            "IA_confirmee": ia_df,
            "Non_IA_confirmee": non_ia_df,
            "A_verifier": verifier_df,
            "Rapport_nettoyage": report_df,
            "Referentiel_competences": competence_ref_df,
        },
    )

    return {
        "merged": merged,
        "training": training_df,
        "verifier": verifier_df,
        "ia": ia_df,
        "non_ia": non_ia_df,
        "report": report_df,
        "competence_distribution": comp_dist,
        "near_duplicate_groups": near_duplicate_clusters,
        "removed_exact": removed_exact,
        "rows_before": before_dedup,
        "rows_after": len(merged),
        "files": {
            "all": all_path,
            "training": train_path,
            "verifier": verify_path,
            "xlsx": xlsx_path,
        },
    }


def run_quality_checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    merged = result["merged"]
    training_df = result["training"]
    checks = []

    checks.append(
        {
            "check": "no_duplicate_ids",
            "passed": not merged["formation_id"].duplicated().any(),
            "value": int(merged["formation_id"].duplicated().sum()),
        }
    )
    checks.append(
        {
            "check": "no_exact_duplicates_in_training",
            "passed": not training_df.duplicated().any(),
            "value": int(training_df.duplicated().sum()),
        }
    )
    checks.append(
        {
            "check": "no_empty_input_text",
            "passed": not training_df["texte_modele"].map(clean_text).eq("").any(),
            "value": int(training_df["texte_modele"].map(clean_text).eq("").sum()),
        }
    )
    checks.append(
        {
            "check": "ia_confirmed_have_competences",
            "passed": not merged.loc[merged["statut_annotation"] == "ia_confirmee", "competences_ia"].map(clean_text).eq("").any(),
            "value": int(
                merged.loc[merged["statut_annotation"] == "ia_confirmee", "competences_ia"].map(clean_text).eq("").sum()
            ),
        }
    )
    checks.append(
        {
            "check": "non_ia_no_competences",
            "passed": merged.loc[merged["statut_annotation"] == "non_ia_confirmee", "competences_ia"].map(clean_text).eq("").all(),
            "value": int(
                merged.loc[merged["statut_annotation"] == "non_ia_confirmee", "competences_ia"].map(clean_text).ne("").sum()
            ),
        }
    )
    checks.append(
        {
            "check": "no_leakage_token_names",
            "passed": not merged["texte_modele"].fillna("").str.contains("Compétences IA|competences_ia|Tags TrendRadar", case=False, regex=True).any(),
            "value": int(merged["texte_modele"].fillna("").str.contains("Compétences IA|competences_ia|Tags TrendRadar", case=False, regex=True).sum()),
        }
    )
    return checks


def print_summary(result: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    merged = result["merged"]
    training_df = result["training"]
    verifier_df = result["verifier"]
    ia_df = result["ia"]
    non_ia_df = result["non_ia"]
    comp_dist = result["competence_distribution"]

    print("\n=== RÉSUMÉ FINAL ===")
    print(f"Lignes avant nettoyage : {result['rows_before']}")
    print(f"Lignes après nettoyage : {result['rows_after']}")
    print(f"Doublons exacts supprimés : {result['removed_exact']}")
    print(f"Formations IA confirmées : {len(ia_df)}")
    print(f"Formations non-IA confirmées : {len(non_ia_df)}")
    print(f"Formations à vérifier : {len(verifier_df)}")
    print(f"Compétences distinctes : {len(comp_dist)}")
    print("\nRépartition catégories :")
    print(merged["statut_annotation"].value_counts(dropna=False).to_string())
    print("\nRépartition compétences :")
    if not comp_dist.empty:
        print(comp_dist.head(25).to_string(index=False))
    else:
        print("(aucune)")
    print("\nContrôles qualité :")
    for check in checks:
        state = "OK" if check["passed"] else "ECHEC"
        print(f"- {check['check']} : {state} (valeur={check['value']})")
    print("\nFichiers générés :")
    for label, path in result["files"].items():
        print(f"- {label} : {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoyer, normaliser et fusionner les datasets de formations.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Répertoire racine du projet")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="Répertoire de sortie")
    args = parser.parse_args()

    result = build_processed_outputs(args.root, args.output_dir)
    checks = run_quality_checks(result)
    print_summary(result, checks)

    failed = [check for check in checks if not check["passed"]]
    if failed:
        details = ", ".join(f"{check['check']}={check['value']}" for check in failed)
        raise RuntimeError(f"Contrôles qualité non satisfaits : {details}")


if __name__ == "__main__":
    main()
