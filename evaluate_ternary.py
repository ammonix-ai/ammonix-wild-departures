"""Score the held-out photographs from the ternary states with the ternary decision layer.

Runs the released evaluate_features.py unchanged, pointed at ternary/features.npz and the
classifier folder ternary/. No GPU, no photographs, no language model.
"""
import runtime
from runtime import ROOT
import runpy, sys

runtime.FEATURES = ROOT / 'ternary' / 'features.npz'
sys.argv = ['evaluate_features.py', '--classifier', str(ROOT / 'ternary')] + sys.argv[1:]
runpy.run_path(str(ROOT / 'evaluate_features.py'), run_name='__main__')
