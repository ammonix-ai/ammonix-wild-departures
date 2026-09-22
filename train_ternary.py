"""Fit and calibrate the decision layer on the ternary states.

Runs the released train.py unchanged, pointed at ternary/features.npz, writing the classifier,
prototypes and fit report into ternary/. Falls back to a CPU fit if this xgboost build has no
CUDA support. The released classifier and features stay untouched.
"""
import runtime
from runtime import ROOT
import runpy, sys

runtime.FEATURES = ROOT / 'ternary' / 'features.npz'
args = sys.argv[1:]
if not any(a.startswith('--output') for a in args):
    args += ['--output', str(ROOT / 'ternary')]
sys.argv = ['train.py'] + args
try:
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
except Exception as e:                       # e.g. XGBoostError: CUDA not available in this build
    if '--device' in args and 'cpu' in args:
        raise
    print(f'GPU fit failed ({type(e).__name__}: {str(e)[:160]}); refitting on the CPU', flush=True)
    args = [a for a in args if a not in ('--device', 'cuda')] + ['--device', 'cpu']
    sys.argv = ['train.py'] + args
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
