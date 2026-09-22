"""Compare the ternary states with the released FP16 states of the same photographs.

For every photograph that has both states: the cosine similarity between the two 5,120-value
vectors, and the released FP16 classifier applied unchanged to the ternary state, which shows
whether the decision transfers across precisions without refitting. Needs ternary/features.npz.
"""
import runtime
from runtime import ROOT, CLASSES, load_features
import json
import numpy as np
import xgboost as xgb
from train import probs

rows = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
with np.load(ROOT / 'ternary' / 'features.npz', allow_pickle=False) as z:
    ternary = dict(zip(z['ids'].tolist(), z['features']))
both = [r for r in rows if r['id'] in ternary]
if not both:
    raise SystemExit('No ternary states yet; run extract_ternary.py first.')
F = load_features(both)
T = np.stack([ternary[r['id']] for r in both])
cos = (F * T).sum(1) / np.linalg.norm(F, axis=1) / np.linalg.norm(T, axis=1)
print(f'{len(both)} photographs with both states')
print(f'cosine similarity FP16 vs ternary state: median {np.median(cos):.4f}, min {cos.min():.4f}, max {cos.max():.4f}')
print(f'vector norms: FP16 median {np.median(np.linalg.norm(F, axis=1)):.1f}, ternary median {np.median(np.linalg.norm(T, axis=1)):.1f}')
worst = np.argsort(cos)[:5]
print('lowest cosine:', [(both[i]['id'], both[i]['species'], round(float(cos[i]), 3)) for i in worst])

model = xgb.Booster()
model.load_model(str(ROOT / 'ammonix-xgboost.ubj'))
model.set_param({'nthread': 1, 'device': 'cpu'})
config = json.loads((ROOT / 'training.json').read_text())
known = [i for i, r in enumerate(both) if r['species'] in CLASSES]
for name, X in [('FP16 states', F), ('ternary states', T)]:
    P = probs(model.inplace_predict(X, predict_type='margin'), config['temperature'])
    pred = P.argmax(1)
    for split in ['test', 'validation', 'train']:
        idx = [i for i in known if both[i]['split'] == split]
        if idx:
            y = np.array([CLASSES.index(both[i]['species']) for i in idx])
            hits = int(np.sum(pred[idx] == y))
            print(f'  released FP16 classifier on {name:14s} {split:10s}: {hits}/{len(idx)} ({hits / len(idx):.1%})')
