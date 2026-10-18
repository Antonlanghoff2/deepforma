#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"; ENABLE_CERTBOT="${ENABLE_CERTBOT:-false}"; SSL_EMAIL="${SSL_EMAIL:-}"
for arg in "$@"; do case "$arg" in --dry-run) DRY_RUN=true;; --help) echo "Usage: $0 [--dry-run]"; exit 0;; *) die "Argument inconnu: $arg";; esac; done
require_root; [[ -f /etc/os-release ]] || die "/etc/os-release absent"; . /etc/os-release; [[ "${ID:-}" == ubuntu ]] || die "Distribution non supportée: ${ID:-inconnue}"
if [[ ! -d "$APP_DIR/.git" ]]; then run mkdir -p "$(dirname "$APP_DIR")"; run git clone --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$APP_DIR"; fi
if ! is_true "$DRY_RUN"; then export DEBIAN_FRONTEND=noninteractive; apt-get update; apt-get install -y ca-certificates curl git openssh-client build-essential apache2 python3.12 python3.12-venv python3.12-dev certbot python3-certbot-apache; curl --fail --silent --show-error --max-time 10 https://github.com/ >/dev/null; fi
ensure_group_user; ensure_directories; if [[ ! -f "$ENV_FILE" ]]; then run install -m 0600 -o "$APP_USER" -g "$APP_GROUP" "$SCRIPT_DIR/deepforma.env.example" "$ENV_FILE"; else log "$ENV_FILE préservé"; fi
if command -v a2enmod >/dev/null 2>&1; then run a2enmod proxy proxy_http headers ssl rewrite; fi
install_python_dependencies
if is_true "$DRY_RUN"; then log "Dry-run: contrôles de données et modèles différés"; else check_data_paths; check_models || die "Modèles requis absents; transférez-les puis relancez."; fi
write_systemd_unit; write_apache_http_config; validate_apache
if is_true "$ENABLE_CERTBOT"; then [[ -n "$SSL_EMAIL" ]] || die "SSL_EMAIL est requis avec ENABLE_CERTBOT=true"; dns_ip="$(getent ahostsv4 "$DOMAIN" | awk 'NR==1 {print $1}')"; current_ip="$(curl -4 --fail --silent https://api.ipify.org)"; [[ -n "$dns_ip" && "$dns_ip" == "$current_ip" ]] || die "DNS de $DOMAIN ($dns_ip) ne pointe pas vers ce VPS ($current_ip)."; run systemctl reload apache2; run certbot --apache --non-interactive --agree-tos --email "$SSL_EMAIL" --domain "$DOMAIN" --redirect --keep-until-expiring; fi
validate_apache; run systemctl enable --now "$SERVICE_NAME"; if ! is_true "$DRY_RUN"; then sleep 2; health_check; fi; record_commit; log "Installation Deepforma terminée."
