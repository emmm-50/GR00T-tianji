#!/usr/bin/env bash
# Source from any directory to activate the project-local Orin environment.
_GR00T_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_GR00T_DIR}/.venv/bin/activate" || return $?
export PYTHONPATH="${_GR00T_DIR}/upstream${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_CACHE="${_GR00T_DIR}/.cache/huggingface/hub"
source "${_GR00T_DIR}/upstream/scripts/activate_orin.sh" || return $?
unset _GR00T_DIR
