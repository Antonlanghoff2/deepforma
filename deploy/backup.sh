#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
for arg in "$@"; do case "$arg" in --dry-run) DRY_RUN=true;; --help) echo "Usage: $0 [--dry-run]"; exit 0;; *) die "Argument inconnu: $arg";; esac; done
require_root; timestamp="$(date -u +%Y%m%dT%H%M%SZ)"; archive="$BACKUP_DIR/deepforma-$timestamp.tar.gz"
run mkdir -p "$BACKUP_DIR"; run chmod 0700 "$BACKUP_DIR"
run tar -czf "$archive" --exclude='models/*/*.safetensors' --exclude='models/*/pytorch_model.bin' -C "$(dirname "$APP_DIR")" "$(basename "$APP_DIR")/.env" "$(basename "$APP_DIR")/.deploy" "$(basename "$APP_DIR")/config" "$(basename "$APP_DIR")/data" "$(basename "$APP_DIR")/models" "/etc/systemd/system/$SERVICE_NAME" "/etc/apache2/sites-available/$APP_NAME.conf"
if ! is_true "$DRY_RUN"; then find "$BACKUP_DIR" -maxdepth 1 -type f -name 'deepforma-*.tar.gz' -printf '%T@ %p\n' | sort -nr | awk "NR > $RETENTION_COUNT {print \$2}" | xargs -r rm -f; fi
log "Sauvegarde créée: $archive"
