#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
for arg in "$@"; do case "$arg" in --dry-run) DRY_RUN=true;; --help) echo "Usage: $0 [--dry-run]"; exit 0;; *) die "Argument inconnu: $arg";; esac; done
require_root; [[ -d "$APP_DIR/.git" ]] || die "Dépôt absent: $APP_DIR"; [[ -z "$(runuser -u "$APP_USER" -- git -C "$APP_DIR" status --porcelain)" ]] || die "Modifications locales détectées; mise à jour refusée."
old_commit="$(runuser -u "$APP_USER" -- git -C "$APP_DIR" rev-parse HEAD)"; run mkdir -p "$DEPLOY_STATE_DIR"; run bash -c "printf '%s\\n' '$old_commit' > '$DEPLOY_STATE_DIR/previous_commit'"
script_args=(); is_true "$DRY_RUN" && script_args+=(--dry-run); "$SCRIPT_DIR/backup.sh" "${script_args[@]}"
run_as_app git -C "$APP_DIR" fetch --prune origin "$REPO_BRANCH"; run_as_app git -C "$APP_DIR" merge --ff-only "origin/$REPO_BRANCH"; install_python_dependencies
if ! "$SCRIPT_DIR/check.sh"; then warn "Contrôles échoués; restauration de $old_commit"; run_as_app git -C "$APP_DIR" checkout --detach "$old_commit"; install_python_dependencies; "$SCRIPT_DIR/restart.sh" "${script_args[@]}"; die "Mise à jour annulée."; fi
"$SCRIPT_DIR/restart.sh" "${script_args[@]}"; record_commit; log "Mise à jour terminée."
