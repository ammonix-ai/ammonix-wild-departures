"""Extract the frozen 5,120-value state of every manifest photo into features/features.npz.

Needs the model checkpoint and all photographs under data/ (only the 71 replay
photos are shipped; collect.py and collect_inat.py fetch the others from their
sources). Resumable: states already in the packed file are kept.
"""
import runtime
from runtime import ROOT, FEATURES
from engine import Engine
import json, numpy as np

rows=json.loads((ROOT/'manifest.json').read_text('utf-8'))
done={}
if FEATURES.exists():
    with np.load(FEATURES,allow_pickle=False) as packed:
        done=dict(zip(packed['ids'].tolist(),packed['features']))
missing=[r for r in rows if r['id'] not in done]
absent=[r['image'] for r in missing if not (ROOT/r['image']).exists()]
if absent:
    raise SystemExit(f'{len(absent)} photographs are not under data/ yet, for example {absent[0]}. Collect them first.')
engine=Engine()
# Separate warmup; never included in reported latency.
from PIL import Image
for _ in range(3): engine.feature(Image.new('RGB',(384,384),(40,100,60)))
for count,row in enumerate(missing,1):
    done[row['id']],timing=engine.feature(ROOT/row['image'])
    if count%25==0 or count==len(missing):
        FEATURES.parent.mkdir(exist_ok=True)
        ids=[r['id'] for r in rows if r['id'] in done]
        np.savez_compressed(FEATURES,ids=np.array(ids),features=np.stack([done[i] for i in ids]).astype(np.float32))
        print('features',len(ids),timing,flush=True)
print('EXTRACTION_DONE',flush=True)
