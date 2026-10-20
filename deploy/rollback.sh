#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"; TARGET=""
while [[ $# -gt 0 ]]; do case "$1" in --commit) shift; [[ $# -gt 0 ]] || die "--commit exige un SHA"; TARGET="$1";; --dry-run) DRY_RUN=true;; --help) echo "Usage: $0 [--commit SHA] [--dry-run]"; exit 0;; *) die "Argument inconnu: $1";; esac; shift; done
require_root; [[ -d "$APP_DIR/.git" ]] || die "Dépôt absent: $APP_DIR"; [[ -n "$TARGET" ]] || TARGET="$(cat "$DEPLOY_STATE_DIR/previous_commit" 2>/dev/null || true)"; [[ -n "$TARGET" ]] || die "Aucune version précédente enregistrée."
runuser -u "$APP_USER" -- git -C "$APP_DIR" cat-file -e "$TARGET^{commit}" || die "Commit de rollback invalide: $TARGET"; [[ -z "$(runuser -u "$APP_USER" -- git -C "$APP_DIR" status --porcelain)" ]] || die "Modifications locales détectées; rollback refusé."
run_as_app git -C "$APP_DIR" checkout --detach "$TARGET"; install_python_dependencies; script_args=(); is_true "$DRY_RUN" && script_args+=(--dry-run); "$SCRIPT_DIR/restart.sh" "${script_args[@]}"; if ! is_true "$DRY_RUN"; then health_check; fi; record_commit
log "Rollback terminé vers $TARGET; les données et modèles n’ont pas été supprimés."
