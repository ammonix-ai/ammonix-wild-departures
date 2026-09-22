"""Extract the ternary 5,120-value state of every manifest photograph into ternary/features.npz.

The loop of extract.py against engine_ternary.Engine. Photographs that are not under data/ yet
are skipped (run recollect.py first), so the packed file can be grown over several runs.
Per-photograph timings go to ternary/extract-timings.json. The released features are untouched.
"""
import runtime
from runtime import ROOT
from engine_ternary import Engine
import json, time
import numpy as np

OUT = ROOT / 'ternary'
FEATURES = OUT / 'features.npz'
TIMINGS = OUT / 'extract-timings.json'

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    rows = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
    done = {}
    if FEATURES.exists():
        with np.load(FEATURES, allow_pickle=False) as packed:
            done = dict(zip(packed['ids'].tolist(), packed['features']))
    timings = json.loads(TIMINGS.read_text('utf-8')) if TIMINGS.exists() else {}
    missing = [r for r in rows if r['id'] not in done]
    available = [r for r in missing if (ROOT / r['image']).exists()]
    print(f'{len(done)} states present, {len(missing)} to compute, {len(available)} of them have their photograph', flush=True)
    if not available:
        raise SystemExit('Nothing to do.')

    def save():
        ids = [r['id'] for r in rows if r['id'] in done]
        tmp = FEATURES.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, ids=np.array(ids), features=np.stack([done[i] for i in ids]).astype(np.float32))
        tmp.replace(FEATURES)
        TIMINGS.write_text(json.dumps(timings, indent=1))

    engine = Engine(mode='feature')
    try:
        start = time.perf_counter()
        for count, row in enumerate(available, 1):
            done[row['id']], t = engine.feature(ROOT / row['image'])
            timings[row['id']] = t
            if count % 25 == 0 or count == len(available):
                save()
                print('features', len(done), '/', len(rows), {k: round(v) for k, v in t.items()}, f'{(time.perf_counter() - start) / count:.2f}s per photograph', flush=True)
    finally:
        save()
        engine.close()
    print('EXTRACTION_DONE', len(done), 'of', len(rows), flush=True)
