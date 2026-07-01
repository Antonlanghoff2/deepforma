#!/usr/bin/env bash
set -Eeuo pipefail

APP_NAME="${APP_NAME:-deepforma}"
APP_USER="${APP_USER:-deepforma}"
APP_GROUP="${APP_GROUP:-deepforma}"
APP_DIR="${APP_DIR:-/opt/deepforma}"
REPO_URL="${REPO_URL:-git@github.com:Antonlanghoff2/deepforma.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
DOMAIN="${DOMAIN:-deepforma.hephaestos.eu}"
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8001}"
GUNICORN_APP="${GUNICORN_APP:-src.web_app:create_app()}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-300}"
ENABLE_SSL="${ENABLE_SSL:-false}"
SSL_EMAIL="${SSL_EMAIL:-}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-true}"
RUN_TESTS="${RUN_TESTS:-true}"
RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-false}"
DRY_RUN="${DRY_RUN:-false}"

SYSTEMD_UNIT_PATH="/etc/systemd/system/${APP_NAME}.service"
NGINX_AVAILABLE_PATH="/etc/nginx/sites-available/${APP_NAME}.conf"
NGINX_ENABLED_PATH="/etc/nginx/sites-enabled/${APP_NAME}.conf"
DEPLOY_STATE_DIR="${APP_DIR}/.deploy"
LAST_COMMIT_FILE="${DEPLOY_STATE_DIR}/last_deployed_commit"
PREVIOUS_COMMIT_FILE="${DEPLOY_STATE_DIR}/previous_deployed_commit"
APP_ENV_FILE="${APP_DIR}/.env"
VENV_DIR="${APP_DIR}/.venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

on_error() {
  local line="$1"
  local code="$?"
  warn "Échec à la ligne $line (code $code)"
  show_failure_logs
  exit "$code"
}

trap 'on_error ${LINENO}' ERR

run() {
  if is_true "$DRY_RUN"; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_sh() {
  if is_true "$DRY_RUN"; then
    printf '+ %s\n' "$*"
  else
    bash -lc "$*"
  fi
}

require_root() {
  [[ "${EUID:-$(id -u)}" -eq 0 ]] || die "Ce script doit être exécuté avec des droits root."
}

ensure_group() {
  if ! getent group "$APP_GROUP" >/dev/null 2>&1; then
    log "Création du groupe système $APP_GROUP"
    run groupadd --system "$APP_GROUP"
  fi
}

ensure_user() {
  if ! id "$APP_USER" >/dev/null 2>&1; then
    log "Création de l'utilisateur système $APP_USER"
    run useradd --system --create-home --home-dir "/home/$APP_USER" --shell /bin/bash --gid "$APP_GROUP" "$APP_USER"
  else
    run usermod --gid "$APP_GROUP" --home "/home/$APP_USER" --shell /bin/bash "$APP_USER"
  fi
}

ensure_dirs() {
  local dirs=(
    "$APP_DIR"
    "$APP_DIR/data"
    "$APP_DIR/models"
    "$APP_DIR/logs"
    "$APP_DIR/uploads"
    "$APP_DIR/.cache"
    "$APP_DIR/.cache/huggingface"
    "$APP_DIR/.cache/huggingface/transformers"
    "$DEPLOY_STATE_DIR"
  )
  local dir
  for dir in "${dirs[@]}"; do
    run mkdir -p "$dir"
  done
  run chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
}

ensure_system_packages() {
  if ! is_true "$INSTALL_SYSTEM_PACKAGES"; then
    log "INSTALL_SYSTEM_PACKAGES=false, installation système ignorée."
    return
  fi
  log "Installation des paquets système nécessaires"
  run apt-get update
  run env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git nginx openssh-client python3 python3-dev python3-pip python3-venv build-essential
  if is_true "$ENABLE_SSL"; then
    run env DEBIAN_FRONTEND=noninteractive apt-get install -y certbot python3-certbot-nginx
  fi
}

ensure_git_access() {
  case "$REPO_URL" in
    git@*)
      local home="/home/$APP_USER"
      if ! sudo -u "$APP_USER" bash -lc 'test -f "$HOME/.ssh/id_ed25519" || test -f "$HOME/.ssh/id_rsa"'; then
        cat <<EOF
Aucune clé SSH n'a été trouvée pour l'utilisateur $APP_USER.
Créez une clé puis ajoutez la clé publique comme Deploy Key sur GitHub :

sudo -u $APP_USER mkdir -p $home/.ssh
sudo -u $APP_USER ssh-keygen -t ed25519 -f $home/.ssh/id_ed25519 -N ""
sudo cat $home/.ssh/id_ed25519.pub
EOF
        die "Accès Git SSH non vérifié."
      fi
      if ! sudo -u "$APP_USER" env GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' git ls-remote "$REPO_URL" "$REPO_BRANCH" >/dev/null 2>&1; then
        die "Impossible d'accéder au dépôt Git via SSH avec l'utilisateur $APP_USER."
      fi
      ;;
    https://*|http://*)
      if ! git ls-remote "$REPO_URL" "$REPO_BRANCH" >/dev/null 2>&1; then
        die "Impossible d'accéder au dépôt Git via HTTPS."
      fi
      ;;
    *)
      warn "Schéma Git inhabituel: $REPO_URL"
      ;;
  esac
}

ensure_repo() {
  if [[ ! -d "$APP_DIR/.git" ]]; then
    log "Clonage du dépôt dans $APP_DIR"
    run git clone --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$APP_DIR"
    return
  fi

  local status
  status="$(git -C "$APP_DIR" status --porcelain)"
  if [[ -n "$status" ]]; then
    printf '%s\n' "$status" >&2
    die "Le dépôt contient des modifications locales. Refus de continuer."
  fi

  run git -C "$APP_DIR" fetch --prune origin
  if git -C "$APP_DIR" show-ref --verify --quiet "refs/heads/$REPO_BRANCH"; then
    run git -C "$APP_DIR" checkout "$REPO_BRANCH"
  else
    run git -C "$APP_DIR" checkout -B "$REPO_BRANCH" "origin/$REPO_BRANCH"
  fi
  run git -C "$APP_DIR" pull --ff-only origin "$REPO_BRANCH"
}

ensure_env_file() {
  if [[ ! -f "$APP_ENV_FILE" ]]; then
    local example="$PROJECT_ROOT/deploy/deepforma.env.example"
    [[ -f "$example" ]] || die "Fichier modèle absent: $example"
    log "Création de $APP_ENV_FILE"
    if is_true "$DRY_RUN"; then
      printf '+ cp %q %q\n' "$example" "$APP_ENV_FILE"
    else
      install -m 0600 -o "$APP_USER" -g "$APP_GROUP" "$example" "$APP_ENV_FILE"
    fi
  else
    run chown "$APP_USER:$APP_GROUP" "$APP_ENV_FILE"
    run chmod 600 "$APP_ENV_FILE"
  fi
}

ensure_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Création de l'environnement virtuel"
    run python3 -m venv "$VENV_DIR"
  fi
  run "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  if [[ -f "$APP_DIR/requirements.txt" ]]; then
    run "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
  fi
  run "$VENV_DIR/bin/pip" install -e "$APP_DIR"
  run "$VENV_DIR/bin/pip" install gunicorn
}

maybe_download_models() {
  if ! is_true "$DOWNLOAD_MODELS"; then
    log "DOWNLOAD_MODELS=false, téléchargement des modèles ignoré."
    return
  fi
  if [[ -x "$APP_DIR/scripts/download_models.py" ]]; then
    run "$VENV_DIR/bin/python" "$APP_DIR/scripts/download_models.py"
  else
    warn "Aucun script officiel de téléchargement des modèles n'a été trouvé."
  fi
}

run_tests_if_requested() {
  if ! is_true "$RUN_TESTS"; then
    log "RUN_TESTS=false, tests ignorés."
    return
  fi
  log "Exécution de la suite de tests Python"
  run_sh "cd $(printf '%q' "$APP_DIR") && $(printf '%q' "$VENV_DIR/bin/python") -m pytest -q"
}

run_migrations_if_requested() {
  if ! is_true "$RUN_MIGRATIONS"; then
    log "RUN_MIGRATIONS=false, migrations ignorées."
    return
  fi
  if [[ -f "$APP_DIR/alembic.ini" && -x "$VENV_DIR/bin/alembic" ]]; then
    run_sh "cd $(printf '%q' "$APP_DIR") && $(printf '%q' "$VENV_DIR/bin/alembic") upgrade head"
  elif [[ -x "$APP_DIR/scripts/migrate.py" ]]; then
    run "$VENV_DIR/bin/python" "$APP_DIR/scripts/migrate.py"
  else
    log "Aucune migration détectée, étape ignorée."
  fi
}

write_systemd_unit() {
  log "Mise à jour du service systemd $APP_NAME"
  if is_true "$DRY_RUN"; then
    printf '+ write %s\n' "$SYSTEMD_UNIT_PATH"
    return
  fi
  cat > "$SYSTEMD_UNIT_PATH" <<EOF
[Unit]
Description=Deepforma application
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_ENV_FILE
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=$APP_DIR/.cache/huggingface
Environment=TRANSFORMERS_CACHE=$APP_DIR/.cache/huggingface/transformers
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=$VENV_DIR/bin/gunicorn --bind $APP_HOST:$APP_PORT --workers $GUNICORN_WORKERS --threads $GUNICORN_THREADS --timeout $GUNICORN_TIMEOUT $GUNICORN_APP
Restart=always
RestartSec=5
TimeoutStartSec=300
TimeoutStopSec=30
KillSignal=SIGTERM
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=false
ReadWritePaths=$APP_DIR/data $APP_DIR/models $APP_DIR/logs $APP_DIR/.cache/huggingface
UMask=0027

[Install]
WantedBy=multi-user.target
EOF
  chmod 0644 "$SYSTEMD_UNIT_PATH"
}

write_nginx_config() {
  log "Mise à jour de la configuration Nginx"
  if is_true "$DRY_RUN"; then
    printf '+ write %s\n' "$NGINX_AVAILABLE_PATH"
    return
  fi
  local static_block=""
  if [[ -d "$APP_DIR/static" ]]; then
    static_block=$(cat <<'EOF_STATIC'
    location /static/ {
        alias /opt/deepforma/static/;
        access_log off;
        expires 30d;
        add_header Cache-Control "public, max-age=2592000";
    }

EOF_STATIC
)
  fi
  cat > "$NGINX_AVAILABLE_PATH" <<EOF
server {
    listen 80;
    server_name ${DOMAIN:-_};

    client_max_body_size 50M;

${static_block}    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
        proxy_buffering off;
    }
}
EOF
  chmod 0644 "$NGINX_AVAILABLE_PATH"
  ln -sfn "$NGINX_AVAILABLE_PATH" "$NGINX_ENABLED_PATH"
}

reload_system_services() {
  run systemctl daemon-reload
  run systemctl enable "$APP_NAME"
  run systemctl restart "$APP_NAME"
  run nginx -t
  run systemctl restart nginx
}

check_health() {
  log "Vérification de la route santé"
  curl --fail --silent --show-error "http://$APP_HOST:$APP_PORT/health" >/dev/null
}

show_failure_logs() {
  warn "Affichage des journaux de diagnostic"
  systemctl status "$APP_NAME" --no-pager || true
  journalctl -u "$APP_NAME" -n 100 --no-pager || true
  if [[ -f /var/log/nginx/error.log ]]; then
    tail -n 100 /var/log/nginx/error.log || true
  fi
}

enable_https_if_requested() {
  if ! is_true "$ENABLE_SSL"; then
    return
  fi
  [[ -n "${DOMAIN:-}" && "$DOMAIN" != "_" ]] || die "ENABLE_SSL=true exige un DOMAIN valide."
  [[ -n "$SSL_EMAIL" ]] || die "ENABLE_SSL=true exige SSL_EMAIL."

  local resolved public_ip
  resolved="$(getent ahostsv4 "$DOMAIN" | awk 'NR==1 {print $1}' | head -n1 || true)"
  public_ip="$(curl -fsS https://api.ipify.org || true)"
  log "DNS $DOMAIN -> ${resolved:-inconnu}"
  if [[ -n "$resolved" && -n "$public_ip" && "$resolved" != "$public_ip" ]]; then
    warn "Le domaine ne pointe pas encore clairement vers l'IP publique du serveur ($public_ip)."
    warn "Certbot est ignoré pour le moment; Nginx HTTP reste actif."
    return
  fi

  if command -v certbot >/dev/null 2>&1; then
    if ! certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email "$SSL_EMAIL" --redirect; then
      warn "Certbot a échoué; le service HTTP reste disponible."
    fi
  else
    warn "Certbot n'est pas disponible."
  fi
}

record_commit_state() {
  local commit
  commit="$(git -C "$APP_DIR" rev-parse HEAD)"
  if is_true "$DRY_RUN"; then
    printf '+ echo %s > %s\n' "$commit" "$LAST_COMMIT_FILE"
    return
  fi
  printf '%s\n' "$commit" > "$LAST_COMMIT_FILE"
  chown "$APP_USER:$APP_GROUP" "$LAST_COMMIT_FILE"
  chmod 0644 "$LAST_COMMIT_FILE"
}

rollback_to_previous_commit() {
  local commit="$1"
  [[ -n "$commit" ]] || return 0
  log "Restauration du commit précédent $commit"
  run systemctl stop "$APP_NAME"
  run git -C "$APP_DIR" checkout --force "$commit"
  ensure_venv
  run systemctl restart "$APP_NAME"
  check_health
}

main_deploy() {
  require_root
  ensure_group
  ensure_user
  ensure_dirs
  ensure_system_packages
  ensure_git_access
  ensure_repo
  ensure_env_file
  ensure_venv
  maybe_download_models
  run_tests_if_requested
  run_migrations_if_requested
  write_systemd_unit
  write_nginx_config
  reload_system_services
  check_health
  enable_https_if_requested
  record_commit_state
  log "Déploiement terminé avec succès."
}

main_update() {
  require_root
  ensure_group
  ensure_user
  ensure_dirs
  ensure_git_access
  ensure_repo
  local previous_commit current_commit
  previous_commit="$(git -C "$APP_DIR" rev-parse HEAD)"
  if is_true "$DRY_RUN"; then
    printf '+ echo %s > %s\n' "$previous_commit" "$PREVIOUS_COMMIT_FILE"
  else
    printf '%s\n' "$previous_commit" > "$PREVIOUS_COMMIT_FILE"
    chown "$APP_USER:$APP_GROUP" "$PREVIOUS_COMMIT_FILE"
  fi
  log "Commits à appliquer:"
  git -C "$APP_DIR" log --oneline "HEAD..origin/$REPO_BRANCH" || true
  run git -C "$APP_DIR" pull --ff-only origin "$REPO_BRANCH"
  current_commit="$(git -C "$APP_DIR" rev-parse HEAD)"
  ensure_env_file
  ensure_venv
  maybe_download_models
  run_tests_if_requested
  run_migrations_if_requested
  write_systemd_unit
  write_nginx_config
  reload_system_services
  if ! check_health; then
    show_failure_logs
    rollback_to_previous_commit "$previous_commit"
    die "La mise à jour a échoué et le rollback a été tenté."
  fi
  if is_true "$DRY_RUN"; then
    printf '+ echo %s > %s\n' "$current_commit" "$LAST_COMMIT_FILE"
  else
    printf '%s\n' "$current_commit" > "$LAST_COMMIT_FILE"
    chown "$APP_USER:$APP_GROUP" "$LAST_COMMIT_FILE"
    chmod 0644 "$LAST_COMMIT_FILE"
  fi
  log "Mise à jour terminée avec succès."
}

main_rollback() {
  require_root
  ensure_group
  ensure_user
  ensure_dirs
  ensure_repo
  local commit="${1:-}"
  if [[ -z "$commit" ]]; then
    if [[ -f "$LAST_COMMIT_FILE" ]]; then
      commit="$(tr -d '[:space:]' < "$LAST_COMMIT_FILE")"
    else
      die "Aucun commit de rollback trouvé. Utilisez --commit SHA."
    fi
  fi
  [[ -n "$commit" ]] || die "Commit de rollback vide."
  run systemctl stop "$APP_NAME"
  run git -C "$APP_DIR" checkout --force "$commit"
  ensure_env_file
  ensure_venv
  run_tests_if_requested
  run_migrations_if_requested
  write_systemd_unit
  write_nginx_config
  reload_system_services
  if ! check_health; then
    show_failure_logs
    die "Le rollback n'a pas permis de restaurer un état sain."
  fi
  log "Rollback terminé avec succès vers $commit."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main_deploy "$@"
fi
