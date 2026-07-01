#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy_ubuntu.sh"

ROLLBACK_COMMIT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      shift
      [[ $# -gt 0 ]] || die "--commit exige un SHA."
      ROLLBACK_COMMIT="$1"
      ;;
    --help)
      cat <<'EOF'
Usage: bash scripts/rollback_production.sh [--commit SHA]
EOF
      exit 0
      ;;
    *)
      die "Argument inconnu: $1"
      ;;
  esac
  shift
done

main_rollback "$ROLLBACK_COMMIT"
