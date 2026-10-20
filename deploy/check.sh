#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
ERRORS=0; WARNINGS=0
ok(){ printf '[OK] %s\n' "$*"; }; warning(){ WARNINGS=$((WARNINGS+1)); printf '[WARNING] %s\n' "$*"; }; error(){ ERRORS=$((ERRORS+1)); printf '[ERROR] %s\n' "$*"; }
for arg in "$@"; do case "$arg" in --dry-run) DRY_RUN=true;; --help) echo "Usage: $0 [--dry-run]"; exit 0;; *) die "Argument inconnu: $arg";; esac; done
echo '=== Deepforma deployment check ==='
if is_true "$DRY_RUN"; then
    log "Dry-run: vérifications système uniquement simulées; aucune modification effectuée."
    printf 'Bilan: 0 erreur(s), 0 avertissement(s)\n'
    exit 0
fi
if [[ -f /etc/os-release ]]; then . /etc/os-release; ok "OS: ${PRETTY_NAME:-unknown}"; else error "/etc/os-release absent"; fi
command -v python3 >/dev/null 2>&1 && ok "Python système: $(python3 --version 2>&1)" || error "Python système absent"
[[ -x "$VENV_DIR/bin/python" ]] && ok "Python applicatif: $(run_as_app "$VENV_DIR/bin/python" --version 2>&1)" || error "Venv absent: $VENV_DIR"
if [[ -x "$VENV_DIR/bin/python" ]]; then
    run_as_app "$VENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)' && ok "Python applicatif 3.12" || error "Le venv n'utilise pas Python 3.12"
    run_as_app env PYTHONPATH="$APP_DIR/src" "$VENV_DIR/bin/python" -c 'import wsgi; assert callable(wsgi.application)' && ok "Point d'entrée WSGI wsgi:application" || error "Point d'entrée WSGI invalide"
fi
[[ -x "$VENV_DIR/bin/gunicorn" ]] && ok "Gunicorn présent" || error "Gunicorn absent"
if command -v systemctl >/dev/null 2>&1; then systemctl is-active --quiet "$SERVICE_NAME" && ok "Service actif" || error "Service inactif"; else warning "systemctl indisponible"; fi
if command -v apache2ctl >/dev/null 2>&1; then apache2ctl configtest >/dev/null 2>&1 && ok "Apache configtest" || error "Apache configuration invalide"; else warning "Apache absent"; fi
if command -v ss >/dev/null 2>&1; then ss -ltn | awk '$4 ~ /127\.0\.0\.1:8001$/ {found=1} END {exit !found}' && ok "Gunicorn écoute sur 127.0.0.1:8001" || error "Port 127.0.0.1:8001 non ouvert"; fi
for path in "$APP_DIR/data/raw" "$APP_DIR/data/processed" "$APP_DIR/data/processed/cpf"; do [[ -d "$path" && -w "$path" ]] && ok "Données accessibles: $path" || error "Données non accessibles: $path"; done
check_models && ok "Modèles principaux valides" || error "Modèles principaux incomplets"
[[ -d "$APP_DIR/data/processed/cpf" ]] && ok "CPF: répertoire présent" || error "CPF: répertoire absent"
df -h "$APP_DIR" 2>/dev/null | tail -1 | awk '{print "[OK] Espace disque: "$4" libres ("$5" utilisés)"}'
free -h 2>/dev/null | awk '/Mem:/ {print "[OK] RAM: "$7" disponible"}'
dns_ip="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk 'NR==1 {print $1}')"; [[ -n "$dns_ip" ]] && ok "DNS $DOMAIN: $dns_ip" || warning "DNS non résolu: $DOMAIN"
[[ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]] && ok "Certificat présent" || warning "Certificat Let's Encrypt absent"
command -v journalctl >/dev/null 2>&1 && journalctl -u "$SERVICE_NAME" -p err -n 5 --no-pager 2>/dev/null || true
printf 'Bilan: %d erreur(s), %d avertissement(s)\n' "$ERRORS" "$WARNINGS"; (( ERRORS == 0 ))
