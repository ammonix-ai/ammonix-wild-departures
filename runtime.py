"""Paths and fixed definitions of the Wild Departures demonstration.

Nothing here points at a particular workstation. The browser replay and the
retraining of the decision layer need only this folder. Fresh inference also
needs the model checkpoint: name its folder in settings.json (see
settings.example.json) or in the environment variable AMMONIX_MODEL_DIR.
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / 'features' / 'features.npz'
# Scratch output of image collection and visual audits; never part of a release.
WORK = ROOT / 'work'

_settings = json.loads((ROOT / 'settings.json').read_text('utf-8')) if (ROOT / 'settings.json').exists() else {}
for _dependency in _settings.get('dependency_paths', []):
    if _dependency not in sys.path:
        sys.path.append(_dependency)

CHECKPOINT = 'unsloth/Qwen3.8-27B-unsloth-bnb-4bit'
CHECKPOINT_REVISION = '8aa5f05d26b7205477066e1449e0af13f762a299'
MODEL = Path(os.environ.get('AMMONIX_MODEL_DIR') or _settings.get('model') or ROOT / 'models' / 'Qwen3.8-27B-bnb4')

GROUPS = {
    'MAMMALS': 'dog cat horse cow sheep goat pig elephant lion tiger bear rabbit giraffe zebra monkey'.split(),
    'BIRDS & REPTILES': 'eagle owl chicken duck parrot penguin flamingo ostrich crocodile alligator snake turtle lizard'.split(),
    'ARTHROPODS': 'bee butterfly ant beetle grasshopper dragonfly ladybug'.split(),
}
CLASSES = sum(GROUPS.values(), [])
GATES = {s: g for g, values in GROUPS.items() for s in values}
PROMPT = 'Identify the animal species. Return only the species name.'
# Keep PROMPT unchanged: the trained classifier consumes features from that prompt.
TRADITIONAL_PROMPT = 'What animal species is shown, and is it a mammal, bird, reptile, or arthropod? Answer in one sentence.'
TRADITIONAL_MAX_NEW_TOKENS = 128


def configure_transformers():
    # This image-only pipeline uses the supported PIL backend, not torchvision.
    import transformers.utils as u
    import transformers.utils.import_utils as iu
    for name in ('is_torchvision_available', 'is_torchvision_v2_available', 'is_librosa_available', 'is_torchaudio_available', 'is_sklearn_available'):
        setattr(iu, name, lambda: False)
        if hasattr(u, name): setattr(u, name, lambda: False)


_packed = None


def load_features(rows):
    """The frozen 5,120-value states of these manifest rows, from the one packed file."""
    global _packed
    import numpy as np
    if _packed is None:
        with np.load(FEATURES, allow_pickle=False) as arrays:
            _packed = ({identity: index for index, identity in enumerate(arrays['ids'].tolist())}, arrays['features'].copy())
    index, matrix = _packed
    return np.stack([matrix[index[row['id']]] for row in rows])
