#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
INPUT="${CPF_GENERAL_INPUT:-$APP_DIR/data/raw/Dataset_Generaliste_CPF_V4.xlsx}"
[[ -f "$INPUT" ]] || die "Entrée CPF absente: $INPUT"
check_data_paths || die "Les répertoires CPF ne sont pas accessibles en écriture."
[[ -x "$VENV_DIR/bin/python" ]] || die "Venv absent: $VENV_DIR"
exec make -C "$APP_DIR" cpf-general-prepare CPF_GENERAL_INPUT="$INPUT"
