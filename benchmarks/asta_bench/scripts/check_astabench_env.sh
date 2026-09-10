#!/usr/bin/env bash
set -euo pipefail

missing=0
for key in HF_TOKEN ASTA_TOOL_KEY; do
  value="${!key:-}"
  if [[ -z "${value}" || "${value}" == "<"*">" ]]; then
    echo "${key}=missing"
    missing=1
  else
    echo "${key}=set"
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi
