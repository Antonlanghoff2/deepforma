# Data Flow

This document describes the main business data flow in DeepForma.

## 1. Job Offers

1. Fetch offers by territory from France Travail.
2. Normalize the payload.
3. Extract explicit and implicit skills.
4. Deduplicate and normalize labels.
5. Compare the result with the market and the user profile.

## 2. France Compétences and ROME

1. Download the official RNCP / RS data.
2. Parse certifications, blocks, and skills.
3. Parse ROME jobs, skills, and links.
4. Build mappings between RNCP and ROME.
5. Build the canonical referential used by the extractor and recommender.

## 3. CPF Training Catalog

1. Load the CPF catalog.
2. Normalize the training text.
3. Extract training skills and map them to canonical labels.
4. Build candidate pairs and training datasets.
5. Train the CPF recommender.

## 4. Referential PDF Extraction

1. Read the PDF with the existing loader.
2. Split the document into blocks and sections.
3. Extract candidate skills and candidate multilabel families.
4. Export annotation files.
5. Validate humans before generating training data.

## 5. Web Analysis

1. Receive the request in Flask.
2. Build the market context through services.
3. Assemble the `AnalysisResult` object.
4. Render JSON or HTML.

## Output Discipline

- Keep source text and provenance when possible.
- Keep generated labels separate from validated labels.
- Never overwrite unrelated columns or fields during enrichment.
- Do not inject unverified skills as positives.
