# Source from any working directory. Paths are relative to this script's location.
ZJ_WAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STABLEWM_HOME="$ZJ_WAM_ROOT/data"
export HF_HOME="$ZJ_WAM_ROOT/data/hf_cache"
export TORCH_HOME="$ZJ_WAM_ROOT/data/torch_cache"
export MPLCONFIGDIR="$ZJ_WAM_ROOT/artifacts/setup/matplotlib"
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$ZJ_WAM_ROOT/external/stable-worldmodel:$ZJ_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
