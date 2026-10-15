"""Resolve one known vendored DINO module without replacing HuggingFace datasets."""
import importlib.util
import sys
from pathlib import Path
from scripts.retrained_dinowm_core import SOURCE,verify_assets,FrozenRetrainedDinoWM


def load_dino(env):
    # Verify the frozen third-party source before executing its one data-interface
    # module. LeWM/transformers may already own the top-level `datasets` package.
    verify_assets(env)
    name='datasets.traj_dset';path=SOURCE/'third_party/dino_wm/datasets/traj_dset.py'
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve()==path.resolve()
    else:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    return FrozenRetrainedDinoWM(env)
