"""Invoke untouched official evaluation with bounded GPU allocation."""
import os
import runpy
import sys
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.08,0)
mode=sys.argv.pop(1)
script = (root/'external/le-wm/eval.py' if mode=='lewm' else
          root/'external/stable-worldmodel/scripts/plan/eval_wm.py')
sys.argv[0]=str(script)
runpy.run_path(str(script),run_name='__main__')
