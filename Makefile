.PHONY: collect-france-travail test

collect-france-travail:
	python -m src.jobs.collect_france_travail_offers $(ARGS)

test:
	python -m pytest -q
