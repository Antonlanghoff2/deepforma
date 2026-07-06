from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match


CATEGORY_ORDER = (
    "cadrage de projet IA",
    "analyse des besoins métiers",
    "gouvernance et éthique",
    "RGPD et sécurité",
    "data quality",
    "préparation des données",
    "statistiques",
    "Machine Learning",
    "Deep Learning",
    "NLP et LLM",
    "Computer Vision",
    "séries temporelles",
    "Data Engineering",
    "MLOps",
    "déploiement",
    "monitoring",
    "gestion de projet",
    "visualisation et communication",
)


CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cadrage de projet IA",
        (
            "cadrage",
            "besoin métier",
            "besoins métiers",
            "analyse des besoins",
            "pilotage",
            "stratégie",
            "atelier",
            "feuille de route",
        ),
    ),
    (
        "analyse des besoins métiers",
        (
            "besoin métier",
            "besoins métiers",
            "métier",
            "usage",
            "cas d'usage",
            "cas d’usage",
        ),
    ),
    (
        "gouvernance et éthique",
        (
            "gouvernance",
            "éthique",
            "ethique",
            "responsable",
            "conformité",
            "conformite",
            "compliance",
            "biais",
        ),
    ),
    (
        "RGPD et sécurité",
        (
            "rgpd",
            "gdpr",
            "donnée personnelle",
            "données personnelles",
            "sécurité",
            "securite",
            "privacy",
            "anonymisation",
            "chiffrement",
        ),
    ),
    (
        "data quality",
        (
            "data quality",
            "qualité des données",
            "qualite des donnees",
            "qualité data",
            "qualite data",
            "qualité",
            "qualite",
        ),
    ),
    (
        "préparation des données",
        (
            "préparation des données",
            "preparation des donnees",
            "nettoyage",
            "normalisation",
            "prétraitement",
            "pretraitement",
            "feature engineering",
        ),
    ),
    (
        "statistiques",
        (
            "statistique",
            "statistiques",
            "probabilité",
            "probabilite",
            "test d'hypothèse",
            "test d'hypothese",
            "intervalle de confiance",
        ),
    ),
    (
        "Machine Learning",
        (
            "machine learning",
            "apprentissage automatique",
            "apprentissage machine",
            "régression",
            "regression",
            "classification",
            "clustering",
            "svm",
            "random forest",
            "xgboost",
            "validation croisée",
            "validation croisee",
            "pca",
        ),
    ),
    (
        "Deep Learning",
        (
            "deep learning",
            "apprentissage profond",
            "réseaux de neurones",
            "reseaux de neurones",
            "cnn",
            "rnn",
            "lstm",
            "transformers",
            "transfer learning",
        ),
    ),
    (
        "NLP et LLM",
        (
            "nlp",
            "traitement du langage naturel",
            "traitement automatique du langage",
            "llm",
            "langue préentraîné",
            "langue preentraine",
            "word2vec",
            "bert",
            "gpt",
            "tokenisation",
            "tokenization",
            "embeddings",
            "rag",
            "langchain",
            "llamaindex",
        ),
    ),
    (
        "Computer Vision",
        (
            "computer vision",
            "vision par ordinateur",
            "vision",
            "image",
            "images",
        ),
    ),
    (
        "séries temporelles",
        (
            "séries temporelles",
            "series temporelles",
            "time series",
            "série chronologique",
            "serie chronologique",
        ),
    ),
    (
        "Data Engineering",
        (
            "data engineering",
            "etl",
            "elt",
            "pipeline",
            "pipelines",
            "sql",
            "spark",
            "data lake",
            "lakehouse",
            "airflow",
            "dbt",
        ),
    ),
    (
        "MLOps",
        (
            "mlops",
            "ml flow",
            "mlflow",
            "git",
            "docker",
            "kubernetes",
            "fastapi",
            "ci/cd",
            "continuous integration",
            "continuous delivery",
            "model registry",
        ),
    ),
    (
        "déploiement",
        (
            "mise en production",
            "mise en prod",
            "déploiement",
            "deploiement",
            "production",
        ),
    ),
    (
        "monitoring",
        (
            "monitoring",
            "surveillance",
            "drift",
            "observabilité",
            "observabilite",
        ),
    ),
    (
        "gestion de projet",
        (
            "gestion de projet",
            "pilotage",
            "coordination",
            "planning",
            "roadmap",
        ),
    ),
    (
        "visualisation et communication",
        (
            "visualisation",
            "dataviz",
            "data visualization",
            "communication",
            "reporting",
            "storytelling",
        ),
    ),
)


SUBCATEGORY_RULES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "Machine Learning": (
        ("Régression", ("régression", "regression")),
        ("Classification", ("classification", "svm", "random forest", "xgboost")),
        ("Clustering", ("clustering", "k-means", "kmeans", "dbscan")),
        ("Réduction de dimension", ("pca", "dimension")),
        ("Validation", ("validation croisée", "validation croisee", "cross validation")),
    ),
    "Deep Learning": (
        ("Réseaux de neurones", ("réseaux de neurones", "reseaux de neurones", "neural network")),
        ("CNN", ("cnn", "convolutional")),
        ("RNN", ("rnn", "recurrent")),
        ("LSTM", ("lstm", "long short-term memory")),
        ("Transformers", ("transformer", "transformers")),
        ("Transfer Learning", ("transfer learning", "apprentissage par transfert")),
    ),
    "NLP et LLM": (
        ("Tokenisation", ("tokenisation", "tokenization")),
        ("Embeddings", ("embeddings", "embedding")),
        ("Word2Vec", ("word2vec",)),
        ("BERT", ("bert",)),
        ("GPT", ("gpt",)),
        ("Classification de texte", ("classification de texte", "text classification")),
        ("Analyse de sentiment", ("analyse de sentiment", "sentiment analysis")),
        ("RAG / orchestration LLM", ("rag", "langchain", "llamaindex")),
    ),
    "MLOps": (
        ("CI/CD", ("ci/cd", "continuous integration", "continuous delivery")),
        ("Conteneurisation", ("docker", "kubernetes", "conteneur")),
        ("Suivi du cycle de vie ML", ("mlflow", "model registry")),
        ("Déploiement", ("déploiement", "mise en production", "production")),
        ("Monitoring", ("monitoring", "surveillance", "drift")),
    ),
    "Data Engineering": (
        ("ETL", ("etl", "elt")),
        ("SQL", ("sql",)),
        ("Orchestration", ("airflow", "pipeline", "pipelines", "dbt")),
        ("Stockage", ("data lake", "lakehouse")),
    ),
    "RGPD et sécurité": (
        ("RGPD", ("rgpd", "gdpr", "privacy")),
        ("Sécurité", ("sécurité", "securite", "chiffrement")),
        ("Anonymisation", ("anonymisation", "masquage")),
    ),
    "cadrage de projet IA": (
        ("Analyse des besoins métiers", ("besoin métier", "besoins métiers", "cas d'usage", "cas d’usage")),
        ("Gouvernance", ("gouvernance", "compliance")),
        ("Stratégie", ("stratégie", "strategie", "feuille de route")),
    ),
    "analyse des besoins métiers": (
        ("Recueil des besoins", ("recueillir", "analyse des besoins", "besoin métier")),
        ("Cas d'usage", ("cas d'usage", "cas d’usage")),
    ),
    "gouvernance et éthique": (
        ("Conformité", ("conformité", "conformite", "compliance")),
        ("Éthique", ("éthique", "ethique", "biais")),
    ),
    "préparation des données": (
        ("Nettoyage", ("nettoyage", "cleaning")),
        ("Normalisation", ("normalisation",)),
        ("Prétraitement", ("prétraitement", "pretraitement")),
    ),
    "statistiques": (
        ("Tests statistiques", ("test", "hypothèse", "hypothese")),
        ("Probabilités", ("probabilité", "probabilite")),
    ),
    "Computer Vision": (
        ("Vision par ordinateur", ("vision par ordinateur",)),
        ("Imagerie", ("image", "images")),
    ),
    "séries temporelles": (
        ("Prévision", ("prévision", "prevision", "forecast")),
    ),
    "déploiement": (
        ("Mise en production", ("mise en production", "mise en prod")),
    ),
    "monitoring": (
        ("Surveillance", ("surveillance", "drift", "observabilité", "observabilite")),
    ),
    "gestion de projet": (
        ("Pilotage", ("pilotage", "planning", "roadmap")),
    ),
    "visualisation et communication": (
        ("Reporting", ("reporting",)),
        ("Dataviz", ("visualisation", "dataviz")),
    ),
}


TECH_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python", "python 3"),
    "SQL": ("sql", "postgresql", "postgres", "mysql"),
    "TensorFlow": ("tensorflow", "tensor flow"),
    "PyTorch": ("pytorch", "py torch"),
    "Keras": ("keras",),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes",),
    "MLflow": ("mlflow", "ml flow"),
    "FastAPI": ("fastapi", "fast api"),
    "CI/CD": ("ci/cd", "cicd", "continuous integration", "continuous delivery"),
    "AWS": ("aws", "amazon web services", "sagemaker"),
    "Azure": ("azure",),
    "GCP": ("gcp", "google cloud"),
    "scikit-learn": ("scikit learn", "scikit-learn", "sklearn"),
    "Transformers": ("transformer", "transformers"),
    "LLM": ("llm", "large language model", "large language models"),
    "RAG": ("rag",),
    "LangChain": ("langchain",),
    "LlamaIndex": ("llamaindex",),
    "BERT": ("bert",),
    "GPT": ("gpt",),
    "Word2Vec": ("word2vec",),
    "CNN": ("cnn",),
    "RNN": ("rnn",),
    "LSTM": ("lstm",),
    "SVM": ("svm", "support vector machine"),
    "Random Forest": ("random forest", "randomforest"),
    "XGBoost": ("xgboost", "xg boost"),
    "K-Means": ("k-means", "k means", "kmeans"),
    "DBSCAN": ("dbscan",),
    "PCA": ("pca", "principal component analysis"),
    "Git": ("git",),
    "RGPD": ("rgpd", "gdpr"),
}


def _matched_rule(text: str, rules: Iterable[tuple[str, tuple[str, ...]]]) -> str | None:
    norm = normalize_for_match(text)
    if not norm:
        return None
    for label, hints in rules:
        for hint in hints:
            if normalize_for_match(hint) in norm:
                return label
    return None


def derive_category(label: str | None, description: str | None = None, *, block: str | None = None) -> str:
    text = " ".join(part for part in (clean_text(label), clean_text(description), clean_text(block)) if part)
    matched = _matched_rule(text, CATEGORY_RULES)
    return matched or "Other"


def derive_subcategory(label: str | None, description: str | None = None, *, category: str | None = None) -> str:
    current_category = clean_text(category) or derive_category(label, description)
    rules = SUBCATEGORY_RULES.get(current_category)
    if not rules:
        return clean_text(label)[:80] or "Général"
    text = " ".join(part for part in (clean_text(label), clean_text(description)) if part)
    matched = _matched_rule(text, rules)
    return matched or clean_text(label)[:80] or "Général"


def technical_keywords(*values: str | None) -> list[str]:
    text = " ".join(clean_text(value) for value in values if clean_text(value))
    normalized = normalize_for_match(text)
    keywords: list[str] = []
    seen: set[str] = set()
    for canonical, aliases in TECH_KEYWORD_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_for_match(alias)
            if alias_norm and alias_norm in normalized and canonical not in seen:
                seen.add(canonical)
                keywords.append(canonical)
                break
    return keywords


def normalize_market_alias(label: str | None) -> str:
    text = clean_text(label)
    if not text:
        return ""
    norm = normalize_for_match(text)
    if norm in {"ml", "machine learning", "apprentissage automatique", "apprentissage machine"}:
        return "Machine Learning"
    if norm in {"dl", "deep learning", "apprentissage profond"}:
        return "Deep Learning"
    if norm in {"nlp", "traitement du langage naturel", "traitement automatique du langage"}:
        return "NLP"
    if norm in {"llm", "large language model", "large language models"}:
        return "LLM"
    if norm in {"mise en production", "mise en prod"}:
        return "déploiement"
    if norm in {"workflows automatises", "workflows automatisés", "continuous integration", "continuous delivery"}:
        return "CI/CD"
    if norm in {"conteneurs", "conteneur", "conteneurisation"}:
        return "Docker"
    if norm in {"randomforest"}:
        return "Random Forest"
    if norm in {"tensor flow"}:
        return "TensorFlow"
    if norm in {"transformer"}:
        return "Transformers"
    if norm in {"reseaux neuronaux", "réseaux neuronaux"}:
        return "Réseaux de neurones"
    if norm in {"apprentissage profond"}:
        return "Deep Learning"
    if norm in {"apprentissage automatique"}:
        return "Machine Learning"
    if norm in {"python 3"}:
        return "Python"
    if norm in {"postgresql", "mysql"}:
        return "SQL"
    return text


@dataclass(frozen=True, slots=True)
class CertificationTaxonomy:
    category: str
    subcategory: str
    technical_keywords: list[str]
    normalized_label: str


def infer_skill_taxonomy(
    label: str | None,
    description: str | None = None,
    aliases: Iterable[str] | None = None,
    *,
    block: str | None = None,
    activity: str | None = None,
    origin_document: str | None = None,
) -> CertificationTaxonomy:
    alias_text = " ".join(clean_text(alias) for alias in aliases or [] if clean_text(alias))
    category = derive_category(label, description, block=block)
    subcategory = derive_subcategory(label, description, category=category)
    keywords = technical_keywords(label, description, alias_text, block, activity, origin_document)
    normalized_label = normalize_market_alias(label) or clean_text(label)
    return CertificationTaxonomy(
        category=category,
        subcategory=subcategory,
        technical_keywords=keywords,
        normalized_label=normalized_label,
    )

