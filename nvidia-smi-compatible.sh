#!/usr/bin/env bash
set -euo pipefail

# Temporary monitoring helper for the current boot.  The running kernel module
# is 595.71.05 while Ubuntu's userspace libraries have already advanced to
# 595.84.  A reboot makes this wrapper unnecessary.
readonly NVML_COMPAT_DIR="/home/mjp/.local/lib/nvidia-595.71.05"
readonly NVML_COMPAT_LIB="${NVML_COMPAT_DIR}/libnvidia-ml.so.595.71.05"

if [[ ! -f "${NVML_COMPAT_LIB}" ]]; then
    echo "Compatible NVML library not found: ${NVML_COMPAT_LIB}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${NVML_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec /usr/bin/nvidia-smi "$@"
