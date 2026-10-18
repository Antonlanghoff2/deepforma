#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
for arg in "$@"; do case "$arg" in --dry-run) DRY_RUN=true;; --help) echo "Usage: $0 [--dry-run]"; exit 0;; *) die "Argument inconnu: $arg";; esac; done
require_root; run systemctl restart "$SERVICE_NAME"
if ! is_true "$DRY_RUN"; then sleep 2; health_check; fi
log "Service $SERVICE_NAME redémarré."
