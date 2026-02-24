# Detect the right Python for this environment.
# Source this file; it sets PYTHON to the best available interpreter.
#
# Priority:
#   1. MIMIC_PYTHON env var (explicit override, e.g. in .env)
#   2. micromamba mimic_cdm env (if it exists on this machine)
#   3. bare "python" on PATH (works inside Apptainer, conda activate, etc.)

if [ -n "${MIMIC_PYTHON:-}" ]; then
    PYTHON="$MIMIC_PYTHON"
elif [ -x "$HOME/.conda/envs/mimic_cdm/bin/python" ]; then
    PYTHON="$HOME/.conda/envs/mimic_cdm/bin/python"
else
    PYTHON="python"
fi
