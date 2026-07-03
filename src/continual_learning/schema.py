from __future__ import annotations

SCHEMA_VERSION = 1

OFFER_STATUSES = (
    "pending",
    "approved",
    "corrected",
    "rejected",
    "excluded",
    "used_for_training",
)

ANNOTATION_PROVENANCE = (
    "human_review",
    "france_travail_api",
    "exact_reference_match",
    "semantic_match",
    "model_prediction",
    "imported_gold_dataset",
)

VALIDATION_MODES = (
    "explicit",
    "implicit",
)

DETAILED_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS continual_learning_offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    description_original TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    location_label TEXT,
    territory TEXT,
    job_family TEXT,
    structured_skills_json TEXT NOT NULL,
    predicted_skills_json TEXT NOT NULL,
    detected_forms_json TEXT NOT NULL,
    offsets_json TEXT NOT NULL,
    confidence_json TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    model_version TEXT,
    validation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(validation_status IN ('pending', 'approved', 'corrected', 'rejected', 'excluded', 'used_for_training')),
    validation_actor TEXT,
    validation_date TEXT,
    validation_note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    raw_payload_json TEXT NOT NULL,
    UNIQUE(offer_id, content_version)
);

CREATE INDEX IF NOT EXISTS idx_continual_learning_offers_status
    ON continual_learning_offers(validation_status);
CREATE INDEX IF NOT EXISTS idx_continual_learning_offers_territory
    ON continual_learning_offers(territory);
CREATE INDEX IF NOT EXISTS idx_continual_learning_offers_job_family
    ON continual_learning_offers(job_family);
CREATE INDEX IF NOT EXISTS idx_continual_learning_offers_model_version
    ON continual_learning_offers(model_version);
CREATE INDEX IF NOT EXISTS idx_continual_learning_offers_collected_at
    ON continual_learning_offers(collected_at);

CREATE TABLE IF NOT EXISTS continual_learning_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_row_id INTEGER NOT NULL REFERENCES continual_learning_offers(id) ON DELETE CASCADE,
    offer_id TEXT NOT NULL,
    content_version TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    surface_form TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT 'SKILL',
    start INTEGER,
    end INTEGER,
    start_key INTEGER NOT NULL DEFAULT -1,
    end_key INTEGER NOT NULL DEFAULT -1,
    confidence REAL NOT NULL DEFAULT 0.0,
    source TEXT NOT NULL,
    provenance TEXT NOT NULL
        CHECK(provenance IN ('human_review', 'france_travail_api', 'exact_reference_match', 'semantic_match', 'model_prediction', 'imported_gold_dataset')),
    is_explicit INTEGER NOT NULL DEFAULT 1,
    text_sentence TEXT,
    referential_code TEXT,
    referential_label TEXT,
    validation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(validation_status IN ('pending', 'approved', 'corrected', 'rejected', 'excluded', 'used_for_training')),
    correction_json TEXT NOT NULL DEFAULT '{}',
    rejected_reason TEXT,
    validated_at TEXT,
    validated_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(offer_id, content_version, normalized_name, start_key, end_key, provenance, source)
);

CREATE INDEX IF NOT EXISTS idx_continual_learning_annotations_offer_row
    ON continual_learning_annotations(offer_row_id);
CREATE INDEX IF NOT EXISTS idx_continual_learning_annotations_status
    ON continual_learning_annotations(validation_status);
CREATE INDEX IF NOT EXISTS idx_continual_learning_annotations_provenance
    ON continual_learning_annotations(provenance);

CREATE TABLE IF NOT EXISTS continual_learning_validation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_row_id INTEGER NOT NULL REFERENCES continual_learning_offers(id) ON DELETE CASCADE,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS continual_learning_model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    date TEXT NOT NULL,
    git_hash TEXT NOT NULL,
    base_model TEXT,
    dataset_hashes_json TEXT NOT NULL,
    example_count INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    taxonomy_json TEXT NOT NULL DEFAULT '{}',
    referential_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL
        CHECK(state IN ('candidate', 'production', 'rejected', 'archived')),
    previous_version TEXT,
    model_dir TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_continual_learning_registry_state
    ON continual_learning_model_registry(state);
"""
