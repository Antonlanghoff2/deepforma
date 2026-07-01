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

.PHONY: install-dev collect-france-travail cpf-download cpf-source-check cpf-inspect cpf-prepare cpf-enrich-skills cpf-embed cpf-check-imports cpf-test cpf-build-pairs cpf-train-v3 cpf-train cpf-evaluate cpf-reindex cpf-v3-all cpf-all test ia-prepare ia-train ia-evaluate ia-all cpf-general-prepare cpf-pairs cpf-general-all deploy-check deploy-install deploy-update deploy-restart deploy-status deploy-logs deploy-nginx-test

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

test:
	$(PYTHON) -m pytest -q
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

deploy-nginx-test:
	$(DEPLOY_SUDO) nginx -t

