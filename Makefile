PYTHON ?= python
COLLECT_ARGS ?=
CPF_SOURCE_URL ?=
CPF_SOURCE_FILE ?=
CPF_RAW_DIR ?= data/raw/cpf
CPF_RAW_FILE ?= $(CPF_RAW_DIR)/cpf_catalog.csv
CPF_PREPARE_OUTPUT_DIR ?= data
CPF_PREPARED_PARQUET ?= $(CPF_PREPARE_OUTPUT_DIR)/processed/cpf/formations.parquet
CPF_INSPECT_REPORT ?= data/reports/cpf_schema_report.json
CPF_INDEX_METADATA ?= data/indexes/cpf/metadata.parquet
CPF_INDEX_FILE ?= data/indexes/cpf/faiss.index
CPF_INDEX_MANIFEST ?= data/indexes/cpf/index_manifest.json

.PHONY: install-dev collect-france-travail cpf-download cpf-inspect cpf-prepare cpf-embed cpf-check-imports cpf-test cpf-all test

install-dev:
	$(PYTHON) -m pip install -e .

collect-france-travail:
	$(PYTHON) -m src.jobs.collect_france_travail_offers $(COLLECT_ARGS)

cpf-download:
	@if [ -z "$(strip $(CPF_SOURCE_URL))" ] && [ -z "$(strip $(CPF_SOURCE_FILE))" ]; then 		echo "Set CPF_SOURCE_URL or CPF_SOURCE_FILE before running cpf-download or cpf-all"; 		exit 1; 	fi
	$(PYTHON) scripts/download_cpf_catalog.py --output-dir "$(CPF_RAW_DIR)" $(if $(strip $(CPF_SOURCE_FILE)),--source-file "$(CPF_SOURCE_FILE)",) $(if $(strip $(CPF_SOURCE_URL)),--source-url "$(CPF_SOURCE_URL)",)

cpf-inspect:
	$(PYTHON) scripts/inspect_cpf_catalog.py --input "$(CPF_RAW_FILE)" --output "$(CPF_INSPECT_REPORT)"

cpf-prepare:
	$(PYTHON) scripts/prepare_cpf_catalog.py --input "$(CPF_RAW_FILE)" --output-dir "$(CPF_PREPARE_OUTPUT_DIR)"

cpf-embed:
	$(PYTHON) scripts/build_cpf_embeddings.py --input "$(CPF_PREPARED_PARQUET)" --metadata "$(CPF_INDEX_METADATA)" --index "$(CPF_INDEX_FILE)" --manifest "$(CPF_INDEX_MANIFEST)"

cpf-check-imports:
	$(PYTHON) -c "import deepforma; import deepforma.cpf"

cpf-test:
	$(PYTHON) -m pytest -q tests/test_cpf_pipeline.py

cpf-all: cpf-download cpf-inspect cpf-prepare cpf-embed cpf-test

test:
	$(PYTHON) -m pytest -q
