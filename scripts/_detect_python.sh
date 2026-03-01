# Detect the right Python for this environment.
# Source this file; it sets PYTHON to the best available interpreter.
#
# Priority:
#   1. MIMIC_PYTHON env var (explicit override, e.g. in .env)
#   2. $HOME/.conda/envs/mimic_cdm (micromamba / conda default)
#   3. /opt/anaconda3/envs/mimic_cdm (macOS Anaconda)
#   4. Dynamic conda detection (any conda install)
#   5. bare "python" on PATH (works inside Apptainer, conda activate, etc.)

if [ -n "${MIMIC_PYTHON:-}" ]; then
    PYTHON="$MIMIC_PYTHON"
elif [ -x "$HOME/.conda/envs/mimic_cdm/bin/python" ]; then
    PYTHON="$HOME/.conda/envs/mimic_cdm/bin/python"
elif [ -x "/opt/anaconda3/envs/mimic_cdm/bin/python" ]; then
    PYTHON="/opt/anaconda3/envs/mimic_cdm/bin/python"
elif command -v conda &>/dev/null; then
    _CONDA_PREFIX="$(conda info --envs 2>/dev/null | awk '/mimic_cdm/{print $NF}')"
    if [ -n "$_CONDA_PREFIX" ] && [ -x "$_CONDA_PREFIX/bin/python" ]; then
        PYTHON="$_CONDA_PREFIX/bin/python"
    else
        PYTHON="python"
    fi
else
    PYTHON="python"
fi
