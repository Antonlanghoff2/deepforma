# Overlay de déploiement: GNU Make charge GNUmakefile avant Makefile.
include Makefile

DEPLOY_SUDO ?= sudo -E
DEPLOY_SERVICE ?= deepforma.service

.PHONY: deploy deploy-install deploy-check deploy-update deploy-restart deploy-status deploy-logs deploy-backup deploy-rollback deploy-prepare-data

deploy deploy-install:
	$(DEPLOY_SUDO) bash deploy/install.sh

deploy-check:
	bash -n deploy/common.sh deploy/install.sh deploy/update.sh deploy/restart.sh deploy/check.sh deploy/backup.sh deploy/rollback.sh deploy/prepare-data.sh
	$(PYTHON) -m pytest -q tests/test_deployment.py tests/test_deploy_automation.py

deploy-update:
	$(DEPLOY_SUDO) bash deploy/update.sh

deploy-restart:
	$(DEPLOY_SUDO) bash deploy/restart.sh

deploy-status:
	$(DEPLOY_SUDO) systemctl status $(DEPLOY_SERVICE) --no-pager

deploy-logs:
	$(DEPLOY_SUDO) journalctl -u $(DEPLOY_SERVICE) -n 100 --no-pager

deploy-backup:
	$(DEPLOY_SUDO) bash deploy/backup.sh

deploy-rollback:
	$(DEPLOY_SUDO) bash deploy/rollback.sh

deploy-prepare-data:
	bash deploy/prepare-data.sh
