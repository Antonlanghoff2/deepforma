#!/usr/bin/env bash
set -Eeuo pipefail
APP_NAME="${APP_NAME:-deepforma}"; APP_USER="${APP_USER:-deepforma}"; APP_GROUP="${APP_GROUP:-deepforma}"; APP_DIR="${APP_DIR:-/opt/deepforma/app}"; REPO_URL="${REPO_URL:-git@github.com:Antonlanghoff2/deepforma.git}"; REPO_BRANCH="${REPO_BRANCH:-main}"; DOMAIN="${DOMAIN:-deepforma.hephaestos.eu}"; APP_HOST="${APP_HOST:-127.0.0.1}"; APP_PORT="${APP_PORT:-8001}"; SERVICE_NAME="${SERVICE_NAME:-deepforma.service}"; DEPLOY_STATE_DIR="${DEPLOY_STATE_DIR:-$APP_DIR/.deploy}"; BACKUP_DIR="${BACKUP_DIR:-/var/backups/deepforma}"; HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${APP_PORT}/health}"; DRY_RUN="${DRY_RUN:-false}"; RETENTION_COUNT="${RETENTION_COUNT:-7}"; GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-300}"; UV_BIN="${UV_BIN:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; VENV_DIR="$APP_DIR/.venv"; ENV_FILE="$APP_DIR/.env"; SYSTEMD_UNIT="/etc/systemd/system/$SERVICE_NAME"; APACHE_AVAILABLE="/etc/apache2/sites-available/$APP_NAME.conf"
log(){ printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }; warn(){ printf '[WARNING] %s\n' "$*" >&2; }; die(){ printf '[ERROR] %s\n' "$*" >&2; exit 1; }; is_true(){ case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }; run(){ if is_true "$DRY_RUN"; then printf '+ '; printf '%q ' "$@"; printf '\n'; else "$@"; fi; }; require_root(){ [[ "${EUID:-$(id -u)}" -eq 0 || "${DRY_RUN,,}" == true ]] || die "Droits root requis; utilisez sudo."; }; require_file(){ [[ -f "$1" ]] || die "Fichier requis absent: $1"; }; on_error(){ local code=$?; warn "Échec ligne $1 (code $code)"; exit "$code"; }; trap 'on_error "$LINENO"' ERR
ensure_group_user(){ if ! getent group "$APP_GROUP" >/dev/null 2>&1; then run groupadd --system "$APP_GROUP"; fi; if ! id "$APP_USER" >/dev/null 2>&1; then run useradd --system --create-home --home-dir "/home/$APP_USER" --shell /bin/bash --gid "$APP_GROUP" "$APP_USER"; fi; if id ubuntu >/dev/null 2>&1; then run usermod -a -G "$APP_GROUP" ubuntu; fi; }
ensure_directories(){ local path; for path in "$APP_DIR" "$APP_DIR/data/raw" "$APP_DIR/data/processed/cpf" "$APP_DIR/models" "$APP_DIR/logs" "$APP_DIR/.cache/huggingface" "$DEPLOY_STATE_DIR"; do run mkdir -p "$path"; run chown "$APP_USER:$APP_GROUP" "$path"; run chmod 2775 "$path"; done; if [[ -f "$ENV_FILE" ]]; then run chown "$APP_USER:$APP_GROUP" "$ENV_FILE"; run chmod 0600 "$ENV_FILE"; fi; }
check_models(){ local root="${DEEPFORMA_MODELS_DIR:-$APP_DIR/models}" missing=0 path required; for path in "$root/binary_ia_v2/final" "$root/multilabel_competences_v2/final"; do if [[ ! -d "$path" ]]; then warn "Modèle absent: $path"; missing=1; continue; fi; for required in config.json model.safetensors tokenizer_config.json; do [[ -f "$path/$required" ]] || { warn "Artefact modèle absent: $path/$required"; missing=1; }; done; done; if (( missing )); then warn "Transférez les modèles historiques vers $root; aucun téléchargement automatique."; return 1; fi; }
check_data_paths(){ local path; for path in "$APP_DIR/data/raw" "$APP_DIR/data/processed" "$APP_DIR/data/processed/cpf"; do runuser -u "$APP_USER" -- test -d "$path" && runuser -u "$APP_USER" -- test -w "$path" || { warn "Répertoire CPF non accessible en écriture par $APP_USER: $path"; return 1; }; done; }
run_as_app(){ if is_true "$DRY_RUN"; then printf '+ runuser -u %q -- env HOME=%q PATH=%q ' "$APP_USER" "/home/$APP_USER" "/home/$APP_USER/.local/bin:/usr/local/bin:/usr/bin:/bin"; printf '%q ' "$@"; printf '\n'; return; fi; runuser -u "$APP_USER" -- env HOME="/home/$APP_USER" PATH="/home/$APP_USER/.local/bin:/usr/local/bin:/usr/bin:/bin" "$@"; }
python_version(){ run_as_app "$VENV_DIR/bin/python" --version 2>&1; }
validate_wsgi(){ log "Validation du point d'entrée WSGI wsgi:application"; run_as_app env PYTHONPATH="$APP_DIR/src" "$VENV_DIR/bin/python" -c 'import wsgi; assert callable(wsgi.application)'; }
install_python_dependencies(){
    if is_true "$DRY_RUN"; then
        log "Dry-run: .venv et dépendances seraient installés sous $APP_USER"
        return 0
    fi
    require_file "$APP_DIR/requirements.txt"
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        run_as_app python3 -m venv "$VENV_DIR"
    fi
    local version major minor
    version="$(python_version)"
    read -r _ major minor _ < <(printf '%s\n' "$version" | tr '.' ' ')
    (( major > 3 || (major == 3 && minor >= 11) )) || die "Python >= 3.11 requis par pyproject.toml; trouvé $version. Utilisez Ubuntu 24.04+ ou installez Python 3.11+."
    run_as_app "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    run_as_app "$VENV_DIR/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
    run_as_app "$VENV_DIR/bin/python" -m pip install --no-deps -e "$APP_DIR"
    validate_wsgi
}
write_systemd_unit(){
    run install -m 0644 "$SCRIPT_DIR/systemd/deepforma.service" "$SYSTEMD_UNIT"
    run systemctl daemon-reload
}
write_apache_http_config(){
    if is_true "$DRY_RUN"; then
        log "Dry-run: configuration Apache générée pour $DOMAIN et $APP_DIR"
    else
        local temp
        temp="$(mktemp)"
        sed -e "s|deepforma.hephaestos.eu|$DOMAIN|g" \
            -e "s|/opt/deepforma/app|$APP_DIR|g" \
            "$SCRIPT_DIR/apache/deepforma.conf" > "$temp"
        install -m 0644 "$temp" "$APACHE_AVAILABLE"
        rm -f "$temp"
    fi
    run a2ensite "$APP_NAME.conf"
}
validate_apache(){ run apache2ctl configtest; }
health_check(){ curl --fail --silent --show-error --max-time 15 "$HEALTH_URL" >/dev/null; }
record_commit(){ if is_true "$DRY_RUN"; then log "Dry-run: le commit déployé serait enregistré"; return 0; fi; mkdir -p "$DEPLOY_STATE_DIR"; git -C "$APP_DIR" rev-parse HEAD > "$DEPLOY_STATE_DIR/current_commit"; }
