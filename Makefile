PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
COLLECT_ARGS ?=
CPF_SOURCE_URL ?=
CPF_SOURCE_FILE ?= data/raw/Dataset_Generaliste_CPF_V3.xlsx
CPF_RAW_DIR ?= data/raw/cpf
CPF_RAW_FILE ?= $(CPF_RAW_DIR)/cpf_catalog.csv
CPF_PROCESSED_DIR ?= data/processed
CPF_FORMATIONS_NORMALIZED ?= data/processed/cpf/formations_normalized.parquet
CPF_FORMATIONS ?= data/processed/cpf/formations_with_skills.parquet
CPF_INSPECT_REPORT ?= data/processed/reports/cpf_v3_inspection.json
CPF_INDEX_METADATA ?= data/indexes/cpf/metadata.parquet
CPF_INDEX_FILE ?= data/indexes/cpf/faiss.index
CPF_INDEX_MANIFEST ?= data/indexes/cpf/index_manifest.json
CPF_OFFERS_DIR ?= data/france_travail/normalized
CPF_TRAIN_DIR ?= data/training
CPF_TRAIN ?= $(CPF_TRAIN_DIR)/cpf_train.jsonl
CPF_VALIDATION ?= $(CPF_TRAIN_DIR)/cpf_validation.jsonl
CPF_TEST ?= $(CPF_TRAIN_DIR)/cpf_test.jsonl
CPF_PAIRS ?= $(CPF_TRAIN_DIR)/cpf_pairs.jsonl
CPF_REVIEW ?= $(CPF_TRAIN_DIR)/cpf_pairs_review.csv
CPF_BASE_MODEL ?= sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CPF_MODEL_OUTPUT ?= models/cpf-recommender
DEPLOY_SUDO ?= sudo -E
DEPLOY_SERVICE ?= deepforma
DEPLOY_HEALTH_URL ?= http://127.0.0.1:8001/health
DEPLOY_ROOT ?= /opt/deepforma
CPF_EPOCHS ?= 3
CPF_BATCH_SIZE ?= 16
CPF_LEARNING_RATE ?= 2e-5
CPF_MAX_PAIRS_PER_FORMATION ?= 10
CPF_MAX_TRAIN_SAMPLES ?=
CPF_WARMUP_RATIO ?= 0.1
CPF_MAX_SEQ_LENGTH ?= 256
CPF_LOSS ?= MultipleNegativesRankingLoss
CPF_SEED ?= 42
CPF_MIN_SKILL_COVERAGE ?= 0.05
CPF_MIN_SEMANTIC_SIMILARITY ?= 0.2
CPF_MAX_QUERIES ?= 100
CPF_DEVICE ?=
CPF_GRADIENT_ACCUMULATION ?= 2
CPF_MIXED_PRECISION ?= true

# ----- France Compétences RNCP/RS -----
FRANCE_COMPETENCES_DATASET_SLUG ?= repertoire-national-des-certifications-professionnelles-et-repertoire-specifique
FRANCE_COMPETENCES_INCLUDE_RNCP ?= true
FRANCE_COMPETENCES_INCLUDE_RS ?= true
FRANCE_COMPETENCES_ACTIVE_ONLY ?= true
FRANCE_COMPETENCES_KEEP_EVALUATION ?= false
FRANCE_COMPETENCES_FORCE_DOWNLOAD ?= false
FRANCE_COMPETENCES_TIMEOUT ?= 60
FRANCE_COMPETENCES_RAW_DIR ?= data/raw/france_competences
FRANCE_COMPETENCES_INTERIM_DIR ?= data/interim/france_competences
FRANCE_COMPETENCES_PROCESSED_DIR ?= data/processed/france_competences
FRANCE_COMPETENCES_TRAINING_DIR ?= data/training/france_competences

# ----- IA Classifier variables -----
IA_DATASET ?= data/raw/Dataset_IA_V9_synth.xlsx
IA_TAXONOMY ?= config/ia_taxonomy_v2.json
IA_PROCESSED_DIR ?= data/processed
IA_BASE_MODEL ?= camembert-base
IA_MODEL_OUTPUT ?= models/ia-classifier-v2
IA_EPOCHS ?= 10
IA_BATCH_SIZE ?= 16
IA_LEARNING_RATE ?= 2e-5
IA_DEVICE ?=

# ----- CPF Generaliste variables -----
CPF_GENERAL_DATASET ?= data/raw/Dataset_Generaliste_CPF_V4.xlsx
CPF_GENERAL_PROCESSED_DIR ?= data/processed/cpf
CPF_GENERAL_PAIRS ?= data/processed/cpf/pairs_generalistes.jsonl
CPF_BASE_MODEL ?= sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CPF_MODEL_OUTPUT ?= models/cpf-recommender

.PHONY: install-dev collect-france-travail build-review-queue export-approved-training-data train-continual evaluate-candidate promote-candidate deploy-candidate rollback-model cpf-download cpf-source-check cpf-inspect cpf-prepare cpf-enrich-skills cpf-embed cpf-check-imports cpf-test cpf-build-pairs cpf-train-v3 cpf-train cpf-evaluate cpf-reindex cpf-v3-all cpf-all test ia-prepare ia-train ia-evaluate ia-all cpf-general-prepare cpf-pairs cpf-general-all deploy-check deploy-install deploy-update deploy-restart deploy-status deploy-logs deploy-apache-test deploy-nginx-test import-referential-preview validate-referential-import approve-referential-import audit-referential-pdfs build-referential-annotations export-referential-training-data train-referential-section-model train-referential-ner evaluate-referential-models test-referential-import deploy-referential-models

install-dev:
	$(PYTHON) -m pip install -e .

collect-france-travail:
	$(PYTHON) -m src.jobs.collect_france_travail_offers $(COLLECT_ARGS)

cpf-source-check:
	@$(PYTHON) -c "from data.cpf_loader import resolve_cpf_source; print(resolve_cpf_source('$(CPF_SOURCE_FILE)'))"

cpf-download:
	@if [ -z "$(strip $(CPF_SOURCE_URL))" ] && [ -z "$(strip $(CPF_SOURCE_FILE))" ]; then 		echo "Set CPF_SOURCE_URL or CPF_SOURCE_FILE before running cpf-download or cpf-all"; 		exit 1; 	fi
	$(PYTHON) scripts/download_cpf_catalog.py --output-dir "$(CPF_RAW_DIR)" $(if $(strip $(CPF_SOURCE_FILE)),--source-file "$(CPF_SOURCE_FILE)",) $(if $(strip $(CPF_SOURCE_URL)),--source-url "$(CPF_SOURCE_URL)",)

cpf-inspect: cpf-source-check
	$(PYTHON) scripts/inspect_cpf_dataset.py --input "$(CPF_SOURCE_FILE)" --output "$(CPF_INSPECT_REPORT)"

cpf-prepare: cpf-inspect
	$(PYTHON) scripts/prepare_cpf_dataset.py --input "$(CPF_SOURCE_FILE)" --output-dir "$(CPF_PROCESSED_DIR)"

cpf-enrich-skills: cpf-prepare
	$(PYTHON) scripts/extract_cpf_skills.py --input "$(CPF_FORMATIONS_NORMALIZED)" --output "$(CPF_FORMATIONS)"

cpf-embed: cpf-enrich-skills
	$(PYTHON) scripts/build_cpf_embeddings.py --input "$(CPF_FORMATIONS)" --metadata "$(CPF_INDEX_METADATA)" --index "$(CPF_INDEX_FILE)" --manifest "$(CPF_INDEX_MANIFEST)"

cpf-check-imports:
	$(PYTHON) -c "import deepforma; import deepforma.cpf"

cpf-test:
	$(PYTHON) -m pytest -q tests/test_cpf_pipeline.py

cpf-build-pairs: cpf-enrich-skills
	$(PYTHON) scripts/build_cpf_training_pairs.py --formations "$(CPF_FORMATIONS)" --offers-dir "$(CPF_OFFERS_DIR)" --output "$(CPF_PAIRS)" --review-output "$(CPF_REVIEW)" --train-output "$(CPF_TRAIN)" --validation-output "$(CPF_VALIDATION)" --test-output "$(CPF_TEST)" --min-skill-coverage $(CPF_MIN_SKILL_COVERAGE) --min-semantic-similarity $(CPF_MIN_SEMANTIC_SIMILARITY) --max-queries $(CPF_MAX_QUERIES)

cpf-train-v3: cpf-build-pairs
	$(PYTHON) scripts/train_cpf_recommender.py --train "$(CPF_TRAIN)" --validation "$(CPF_VALIDATION)" --base-model "$(CPF_BASE_MODEL)" --output-dir "$(CPF_MODEL_OUTPUT)" --epochs $(CPF_EPOCHS) --batch-size $(CPF_BATCH_SIZE) --learning-rate $(CPF_LEARNING_RATE) --warmup-ratio $(CPF_WARMUP_RATIO) --max-seq-length $(CPF_MAX_SEQ_LENGTH) --loss $(CPF_LOSS) --seed $(CPF_SEED) --gradient-accumulation $(CPF_GRADIENT_ACCUMULATION) $(if $(strip $(CPF_DEVICE)),--device "$(CPF_DEVICE)",) $(if $(filter true,$(CPF_MIXED_PRECISION)),--mixed-precision,--no-mixed-precision)

cpf-evaluate: cpf-train-v3
	$(PYTHON) scripts/evaluate_cpf_recommender.py --test "$(CPF_TEST)" --formations "$(CPF_FORMATIONS)" --base-model "$(CPF_BASE_MODEL)" --fine-tuned-model "$(CPF_MODEL_OUTPUT)/final"

cpf-reindex: cpf-evaluate
	$(PYTHON) scripts/build_cpf_embeddings.py --input "$(CPF_FORMATIONS)" --model "$(CPF_MODEL_OUTPUT)/final" --metadata "$(CPF_INDEX_METADATA)" --index "$(CPF_INDEX_FILE)" --manifest "$(CPF_INDEX_MANIFEST)"

cpf-v3-all: cpf-reindex

cpf-all: cpf-general-all

# ----- IA Classifier -----
ia-prepare:
	$(PYTHON) scripts/prepare_ia_training_dataset.py \
		--input "$(IA_DATASET)" \
		--output-dir "$(IA_PROCESSED_DIR)" \
		--taxonomy "$(IA_TAXONOMY)"

ia-train: ia-prepare
	$(PYTHON) scripts/train_ia_multilabel_classifier.py \
		--input-dir "$(IA_PROCESSED_DIR)" \
		--output-dir "$(IA_MODEL_OUTPUT)" \
		--base-model "$(IA_BASE_MODEL)" \
		--epochs $(IA_EPOCHS) --batch-size $(IA_BATCH_SIZE) --lr $(IA_LEARNING_RATE) \
		$(if $(strip $(IA_DEVICE)),--device "$(IA_DEVICE)",)

ia-evaluate:
	$(PYTHON) scripts/evaluate_ia_multilabel_classifier.py \
		--model-dir "$(IA_MODEL_OUTPUT)/final" \
		--test-file "$(IA_PROCESSED_DIR)/ia_multilabel_test.jsonl" \
		--output-dir reports \
		--taxonomy "$(IA_TAXONOMY)"

ia-all: ia-train ia-evaluate

# ----- CPF Generaliste -----
cpf-general-prepare:
	$(PYTHON) scripts/prepare_general_cpf_dataset.py \
		--input "$(CPF_GENERAL_DATASET)" \
		--output-dir "$(CPF_GENERAL_PROCESSED_DIR)"

cpf-pairs: cpf-general-prepare
	$(PYTHON) scripts/build_cpf_training_pairs.py \
		--input "$(CPF_GENERAL_PROCESSED_DIR)/formations_generalistes.jsonl" \
		--output-dir "$(CPF_GENERAL_PROCESSED_DIR)" \
		--output-pairs "pairs_generalistes.jsonl" \
		--max-pairs-per-formation $(CPF_MAX_PAIRS_PER_FORMATION)

cpf-train: cpf-pairs
	$(PYTHON) scripts/train_cpf_recommender.py \
		--input-pairs "$(CPF_GENERAL_PAIRS)" \
		--output-dir "$(CPF_MODEL_OUTPUT)" \
		--base-model "$(CPF_BASE_MODEL)" \
		--epochs $(CPF_EPOCHS) --batch-size $(CPF_BATCH_SIZE) \
		$(if $(strip $(CPF_MAX_TRAIN_SAMPLES)),--max-train-samples $(CPF_MAX_TRAIN_SAMPLES),)

cpf-general-all: cpf-train

# ----- Referential Import -----
REFERENTIAL_INPUT ?=
REFERENTIAL_REPORT ?= reports/referential_import.json
REFERENTIAL_OUTPUT ?= data/referentials/imported/
REFERENTIAL_DB ?= data/referentials/referential_imports.sqlite3
REFERENTIAL_PDF_DIR ?= data/raw/referentiel
REFERENTIAL_AUDIT_REPORT ?= reports/referential_pdf_audit.json
REFERENTIAL_CANDIDATES ?= data/annotation/referential_candidates.jsonl
REFERENTIAL_TRAIN_DIR ?= data/training
REFERENTIAL_SECTION_MODEL_OUTPUT ?= models/referential-section-classifier
REFERENTIAL_NER_MODEL_OUTPUT ?= models/referential-skill-ner
REFERENTIAL_SECTION_BASE_MODEL ?= camembert-base
REFERENTIAL_NER_BASE_MODEL ?= camembert-base
REFERENTIAL_SECTION_BATCH_SIZE ?= 8
REFERENTIAL_SECTION_EPOCHS ?= 5
REFERENTIAL_SECTION_LEARNING_RATE ?= 2e-5
REFERENTIAL_SECTION_DEVICE ?=
REFERENTIAL_SECTION_FP16 ?= true
REFERENTIAL_NER_BATCH_SIZE ?= 4
REFERENTIAL_NER_GRADIENT_ACCUMULATION ?= 4
REFERENTIAL_NER_EPOCHS ?= 5
REFERENTIAL_NER_LEARNING_RATE ?= 2e-5
REFERENTIAL_NER_DEVICE ?=
REFERENTIAL_NER_FP16 ?= true
REFERENTIAL_MULTILABEL_MODEL_OUTPUT ?= models/referential-multilabel
REFERENTIAL_MULTILABEL_BASE_MODEL ?= camembert-base
REFERENTIAL_MULTILABEL_BATCH_SIZE ?= 4
REFERENTIAL_MULTILABEL_EPOCHS ?= 5
REFERENTIAL_MULTILABEL_LEARNING_RATE ?= 2e-5
REFERENTIAL_MULTILABEL_DEVICE ?=
REFERENTIAL_MULTILABEL_FP16 ?= true

audit-referential-pdfs:
	$(PYTHON) scripts/audit_referential_pdfs.py --input-dir "$(REFERENTIAL_PDF_DIR)" --output-report "$(REFERENTIAL_AUDIT_REPORT)"

build-referential-ner-candidates: audit-referential-pdfs
	$(PYTHON) scripts/build_referential_ner_candidates.py --input-dir "$(REFERENTIAL_PDF_DIR)" --output "data/annotation/referential_ner_candidates.jsonl"

build-referential-multilabel-candidates: audit-referential-pdfs
	$(PYTHON) scripts/build_referential_multilabel_candidates.py --input-dir "$(REFERENTIAL_PDF_DIR)" --output "data/annotation/referential_multilabel_candidates.jsonl"

build-referential-annotations: build-referential-ner-candidates

export-referential-training-data:
	$(PYTHON) scripts/export_referential_training_data.py --ner-input "data/annotation/referential_ner_candidates.jsonl" --multilabel-input "data/annotation/referential_multilabel_candidates.jsonl" --output-dir "$(REFERENTIAL_TRAIN_DIR)"

train-referential-section-model: export-referential-training-data
	$(PYTHON) scripts/train_referential_section_classifier.py --train "$(REFERENTIAL_TRAIN_DIR)/referential_sections_train.jsonl" --validation "$(REFERENTIAL_TRAIN_DIR)/referential_sections_validation.jsonl" --test "$(REFERENTIAL_TRAIN_DIR)/referential_sections_test.jsonl" --base-model "$(REFERENTIAL_SECTION_BASE_MODEL)" --output-dir "$(REFERENTIAL_SECTION_MODEL_OUTPUT)" --batch-size $(REFERENTIAL_SECTION_BATCH_SIZE) --epochs $(REFERENTIAL_SECTION_EPOCHS) --learning-rate $(REFERENTIAL_SECTION_LEARNING_RATE) $(if $(strip $(REFERENTIAL_SECTION_DEVICE)),--device "$(REFERENTIAL_SECTION_DEVICE)",) $(if $(filter true,$(REFERENTIAL_SECTION_FP16)),--fp16,)

train-referential-ner: export-referential-training-data
	$(PYTHON) scripts/train_referential_skill_ner.py --train "$(REFERENTIAL_TRAIN_DIR)/referential_ner_train.jsonl" --validation "$(REFERENTIAL_TRAIN_DIR)/referential_ner_validation.jsonl" --test "$(REFERENTIAL_TRAIN_DIR)/referential_ner_test.jsonl" --base-model "$(REFERENTIAL_NER_BASE_MODEL)" --output-dir "$(REFERENTIAL_NER_MODEL_OUTPUT)" --batch-size $(REFERENTIAL_NER_BATCH_SIZE) --gradient-accumulation-steps $(REFERENTIAL_NER_GRADIENT_ACCUMULATION) --epochs $(REFERENTIAL_NER_EPOCHS) --learning-rate $(REFERENTIAL_NER_LEARNING_RATE) $(if $(strip $(REFERENTIAL_NER_DEVICE)),--device "$(REFERENTIAL_NER_DEVICE)",) $(if $(filter true,$(REFERENTIAL_NER_FP16)),--fp16,)

train-referential-multilabel: export-referential-training-data
	$(PYTHON) scripts/train_referential_multilabel.py --train "$(REFERENTIAL_TRAIN_DIR)/referential_multilabel_train.jsonl" --validation "$(REFERENTIAL_TRAIN_DIR)/referential_multilabel_validation.jsonl" --test "$(REFERENTIAL_TRAIN_DIR)/referential_multilabel_test.jsonl" --base-model "$(REFERENTIAL_MULTILABEL_BASE_MODEL)" --output-dir "$(REFERENTIAL_MULTILABEL_MODEL_OUTPUT)" --batch-size $(REFERENTIAL_MULTILABEL_BATCH_SIZE) --epochs $(REFERENTIAL_MULTILABEL_EPOCHS) --learning-rate $(REFERENTIAL_MULTILABEL_LEARNING_RATE) $(if $(strip $(REFERENTIAL_MULTILABEL_DEVICE)),--device "$(REFERENTIAL_MULTILABEL_DEVICE)",) $(if $(filter true,$(REFERENTIAL_MULTILABEL_FP16)),--fp16,)

evaluate-referential-models: export-referential-training-data
	$(PYTHON) scripts/evaluate_referential_models.py --ner-test "$(REFERENTIAL_TRAIN_DIR)/referential_ner_test.jsonl" --multilabel-test "$(REFERENTIAL_TRAIN_DIR)/referential_multilabel_test.jsonl" --ner-model "$(REFERENTIAL_NER_MODEL_OUTPUT)/final" --multilabel-model "$(REFERENTIAL_MULTILABEL_MODEL_OUTPUT)/final" --output-dir reports

test-referential-ml-dl: evaluate-referential-models
	$(PYTHON) -m pytest -q tests/test_referential_learning.py tests/test_taxonomy.py

import-referential-preview:
	@if [ -z "$(strip $(REFERENTIAL_INPUT))" ]; then \
		echo "Set REFERENTIAL_INPUT before running import-referential-preview"; \
		exit 1; \
	fi
	$(PYTHON) scripts/import_referential.py --input "$(REFERENTIAL_INPUT)" --dry-run --report "$(REFERENTIAL_REPORT)" --output "$(REFERENTIAL_OUTPUT)" --store-path "$(REFERENTIAL_DB)"

validate-referential-import: import-referential-preview

approve-referential-import:
	@if [ -z "$(strip $(REFERENTIAL_INPUT))" ]; then \
		echo "Set REFERENTIAL_INPUT before running approve-referential-import"; \
		exit 1; \
	fi
	$(PYTHON) scripts/import_referential.py --input "$(REFERENTIAL_INPUT)" --approve --report "$(REFERENTIAL_REPORT)" --output "$(REFERENTIAL_OUTPUT)" --store-path "$(REFERENTIAL_DB)"



test:
	$(PYTHON) -m pytest -q

test-referential-import:
	$(PYTHON) -m pytest -q tests/test_referential_learning.py tests/test_referential_import.py

deploy-referential-models:
	$(PYTHON) scripts/deploy_referential_models.py --source-root "$(REFERENTIAL_SECTION_MODEL_OUTPUT)/.." --target-root "$(DEPLOY_ROOT)/models"
DEPLOY_SCRIPTS := scripts/deploy_ubuntu.sh scripts/update_production.sh scripts/rollback_production.sh

deploy-check:
	bash -n $(DEPLOY_SCRIPTS)
	$(PYTHON) -m pytest -q tests/test_deployment.py

deploy-install:
	$(DEPLOY_SUDO) bash scripts/deploy_ubuntu.sh

deploy-update:
	$(DEPLOY_SUDO) bash scripts/update_production.sh

deploy-restart:
	$(DEPLOY_SUDO) systemctl restart $(DEPLOY_SERVICE)
	curl --fail --silent --show-error $(DEPLOY_HEALTH_URL)

deploy-status:
	$(DEPLOY_SUDO) systemctl status $(DEPLOY_SERVICE)

deploy-logs:
	$(DEPLOY_SUDO) journalctl -u $(DEPLOY_SERVICE) -f

deploy-apache-test:
	apache2ctl configtest

deploy-nginx-test:
	apache2ctl configtest


# ----- Continual Learning -----
CONTINUAL_DB ?= data/continual_learning/continual_learning.sqlite3
CONTINUAL_REVIEW_QUEUE ?= data/continual_learning/review_queue.jsonl
CONTINUAL_APPROVED_EXPORT ?= data/continual_learning/approved_training.jsonl
CONTINUAL_BASE_DATASET ?= $(CONTINUAL_APPROVED_EXPORT)
CONTINUAL_INCREMENTAL_DATASET ?= $(CONTINUAL_APPROVED_EXPORT)
CONTINUAL_VALIDATION_DATASET ?= data/continual_learning/validation.jsonl
CONTINUAL_TEST_DATASET ?= data/continual_learning/test_fixed.jsonl
CONTINUAL_OUTPUT_DIR ?= models/skill-extractor/candidates/latest
CONTINUAL_VERSION ?= latest
CONTINUAL_BASE_MODEL ?= camembert-base
CONTINUAL_RESUME_FROM_MODEL ?=
CONTINUAL_MAX_SAMPLES ?=
CONTINUAL_SEED ?= 42
CONTINUAL_DEVICE ?=
CONTINUAL_FP16 ?= true
CONTINUAL_BATCH_SIZE ?= 8
CONTINUAL_EPOCHS ?= 3
CONTINUAL_MIN_APPROVED_SAMPLES ?= 500
CONTINUAL_PRODUCTION_LINK ?= models/skill-extractor/production
CONTINUAL_SERVICE_NAME ?= deepforma
CONTINUAL_HEALTH_URL ?= http://127.0.0.1:8001/health

build-review-queue:
	$(PYTHON) scripts/build_review_queue.py --limit 200 --output "$(CONTINUAL_REVIEW_QUEUE)" --db-path "$(CONTINUAL_DB)"

export-approved-training-data:
	$(PYTHON) scripts/export_continual_training_dataset.py --output "$(CONTINUAL_APPROVED_EXPORT)" --db-path "$(CONTINUAL_DB)" --minimum-provenance semantic_match --include-human-reviewed --include-france-travail-api --exclude-model-only

train-continual: export-approved-training-data
	$(PYTHON) scripts/train_continual_skill_extractor.py --base-dataset "$(CONTINUAL_BASE_DATASET)" --incremental-dataset "$(CONTINUAL_INCREMENTAL_DATASET)" --validation-dataset "$(CONTINUAL_VALIDATION_DATASET)" --test-dataset "$(CONTINUAL_TEST_DATASET)" --base-model "$(CONTINUAL_BASE_MODEL)" --output-dir "$(CONTINUAL_OUTPUT_DIR)" $(if $(strip $(CONTINUAL_RESUME_FROM_MODEL)),--resume-from-model "$(CONTINUAL_RESUME_FROM_MODEL)",) $(if $(strip $(CONTINUAL_MAX_SAMPLES)),--max-samples $(CONTINUAL_MAX_SAMPLES),) $(if $(strip $(CONTINUAL_DEVICE)),--device "$(CONTINUAL_DEVICE)",) $(if $(filter true,$(CONTINUAL_FP16)),--fp16,) --batch-size $(CONTINUAL_BATCH_SIZE) --epochs $(CONTINUAL_EPOCHS) --seed $(CONTINUAL_SEED)

evaluate-candidate:
	$(PYTHON) scripts/compare_model_versions.py --candidate-model "$(CONTINUAL_OUTPUT_DIR)" --production-model "$(CONTINUAL_PRODUCTION_LINK)" --base-dataset "$(CONTINUAL_BASE_DATASET)" --test-dataset "$(CONTINUAL_TEST_DATASET)" --output-dir reports $(if $(strip $(CONTINUAL_DEVICE)),--device "$(CONTINUAL_DEVICE)",)

promote-candidate:
	$(PYTHON) scripts/promote_continual_model.py --model-dir "$(CONTINUAL_OUTPUT_DIR)" --version "$(CONTINUAL_VERSION)" --registry-path models/skill-extractor/registry.json --production-link "$(CONTINUAL_PRODUCTION_LINK)"

deploy-candidate:
	$(PYTHON) scripts/deploy_continual_model.py --model-dir "$(CONTINUAL_OUTPUT_DIR)" --production-link "$(CONTINUAL_PRODUCTION_LINK)" --service-name "$(CONTINUAL_SERVICE_NAME)" --health-url "$(CONTINUAL_HEALTH_URL)"

rollback-model:
	$(PYTHON) scripts/rollback_continual_model.py --registry-path models/skill-extractor/registry.json --production-link "$(CONTINUAL_PRODUCTION_LINK)" --service-name "$(CONTINUAL_SERVICE_NAME)" --health-url "$(CONTINUAL_HEALTH_URL)"


# ----- France Compétences RNCP/RS pipeline -----
france-competences-download:
	$(PYTHON) scripts/download_france_competences.py --output-dir "$(FRANCE_COMPETENCES_RAW_DIR)" --dataset-slug "$(FRANCE_COMPETENCES_DATASET_SLUG)" $(if $(filter true,$(FRANCE_COMPETENCES_INCLUDE_RNCP)),,--no-rncp) $(if $(filter true,$(FRANCE_COMPETENCES_INCLUDE_RS)),,--no-rs) $(if $(filter true,$(FRANCE_COMPETENCES_FORCE_DOWNLOAD)),--force,) --timeout $(FRANCE_COMPETENCES_TIMEOUT)

france-competences-inspect:
	$(PYTHON) scripts/inspect_france_competences_archive.py --input "$(FRANCE_COMPETENCES_RAW_DIR)" --output "$(FRANCE_COMPETENCES_INTERIM_DIR)/archive_report.json"

france-competences-normalize:
	$(PYTHON) scripts/normalize_france_competences.py --input "$(FRANCE_COMPETENCES_RAW_DIR)" --output-dir "$(FRANCE_COMPETENCES_PROCESSED_DIR)" $(if $(filter true,$(FRANCE_COMPETENCES_KEEP_EVALUATION)),--keep-evaluation,)

france-competences-build-training:
	$(PYTHON) scripts/build_france_competences_training_dataset.py --skills "$(FRANCE_COMPETENCES_PROCESSED_DIR)/skills.parquet" --output-dir "$(FRANCE_COMPETENCES_TRAINING_DIR)"

france-competences-all: france-competences-download france-competences-inspect france-competences-normalize france-competences-build-training

# ----- RNCP / ROME referentials -----
import-france-competences:
	$(PYTHON) scripts/import_france_competences.py --active-only --write

import-rome-referential:
	$(PYTHON) scripts/import_rome_referential.py --write

map-rncp-to-rome:
	$(PYTHON) scripts/map_rncp_to_rome.py --write

build-unified-skill-referential:
	$(PYTHON) scripts/build_unified_skill_referential.py --write

enrich-offers-with-rome-rncp:
	$(PYTHON) scripts/enrich_offers_with_rome_rncp.py --input data/france_travail/normalized --write

build-rome-rncp-training-dataset:
	$(PYTHON) scripts/build_rome_rncp_training_dataset.py

train-skill-extractor:
	$(PYTHON) scripts/train_skill_extractor.py --train data/training/skill_extraction/train.jsonl --validation data/training/skill_extraction/validation.jsonl --test data/training/skill_extraction/test.jsonl

