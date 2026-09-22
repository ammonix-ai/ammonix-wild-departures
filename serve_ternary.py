"""Browser replay of the ternary run, with optional live measurement on the ternary model.

Serves the released page from ternary/replay with the results of benchmark_ternary.py, at
http://127.0.0.1:8767 by default (the released replay of the FP16 run stays at serve.py).

With --live, the page's "Measure this image again" button re-measures the chosen photograph on
this machine: the state and decision come from llama-server in embeddings mode, the generated
answer from the same weights served in generation mode. One server process cannot do both and
two copies of the model do not fit into 8 GB, so the server is restarted between the two steps
(about 15 to 20 s each way). The restart is never part of a reported timing.
"""
import runtime
from runtime import ROOT
import argparse, json, shutil, threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from benchmark import fallback
from benchmark_ternary import Classifier, traditional

REPLAY = ROOT / 'ternary' / 'replay'
REPLAY_ROWS = ('test', 'ood_test')


def refresh_replay_folder():
    """The released page and photographs next to the ternary results (or the partial progress file)."""
    REPLAY.mkdir(parents=True, exist_ok=True)
    (REPLAY / 'brand').mkdir(exist_ok=True)
    (REPLAY / 'data').mkdir(exist_ok=True)
    for name in ('index.html', 'scene.js', 'credits.html'):
        shutil.copy2(ROOT / name, REPLAY / name)
    for p in (ROOT / 'brand').iterdir():
        shutil.copy2(p, REPLAY / 'brand' / p.name)
    results = ROOT / 'ternary' / 'results.json'
    progress = ROOT / 'ternary' / 'benchmark-progress.json'
    if results.exists():
        shutil.copy2(results, REPLAY / 'results.json')
        (REPLAY / 'benchmark-progress.json').unlink(missing_ok=True)
    elif progress.exists():
        shutil.copy2(progress, REPLAY / 'benchmark-progress.json')
        (REPLAY / 'results.json').unlink(missing_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
    for r in manifest:
        if r['split'] in REPLAY_ROWS and not (REPLAY / r['image']).exists():
            shutil.copy2(ROOT / r['image'], REPLAY / r['image'])
    return manifest


class Live:
    """Both measurement paths on the ternary weights, switching the server mode as needed."""

    def __init__(self):
        self.engine = None
        self.classifier = Classifier()

    def mode(self, wanted):
        if self.engine is not None and self.engine.mode == wanted:
            return self.engine
        if self.engine is not None:
            self.engine.close()
        from engine_ternary import Engine
        self.engine = Engine(mode=wanted, warmup=1)
        return self.engine

    def classify(self, row):
        path = ROOT / row['image']
        engine = self.mode('feature')
        engine.erase_cache()
        x, t = engine.feature(path)
        rec, _ = self.classifier.predict(x)
        rec.update(t)
        rec['ammonix_ms'] = t['feature_ms'] + rec['classifier_ms']
        engine = self.mode('generate')
        engine.erase_cache()
        answer, gms, completion = traditional(engine, path)
        rec.update(traditional_ms=gms, traditional_answer=answer, traditional_completion=completion, qwen_ms=gms, qwen_answer=answer,
                   final_label=rec['prediction'], final_gate=rec['gate'], fallback_ms=0)
        if rec['escalated']:
            engine.erase_cache()
            rec.update(fallback(engine, path))
        rec.update(id=row['id'], species=row['species'], image=row['image'], ood=row['split'] == 'ood_test', correct=rec['prediction'] == row['species'])
        return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='re-measure clicked photographs on the ternary model')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    manifest = refresh_replay_folder()
    lock = threading.Lock()
    live = Live() if args.live else None
    if live:
        live.mode('feature')          # load the model before the first click

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(REPLAY), **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/api/status':
                self._json({'live': live is not None})
                return
            super().do_GET()

        def do_POST(self):
            if self.path != '/api/classify' or live is None:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length > 1024:
                    raise ValueError('Request too large')
                requested = json.loads(self.rfile.read(length))['id']
                row = next(r for r in manifest if r['id'] == requested and r['split'] in REPLAY_ROWS)
                with lock:
                    rec = live.classify(row)
                self._json(rec)
            except Exception as e:
                self.send_error(400, str(e))

    print(f'Wild Departures, ternary run: http://127.0.0.1:{args.port}  live={live is not None}', flush=True)
    try:
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    finally:
        if live and live.engine:
            live.engine.close()
