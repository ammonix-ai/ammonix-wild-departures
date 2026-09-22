"""Wild Departures on the ternary model: the animal airport, live in the browser.

    python serve_ternary.py --live   the 21 curated passengers of the film, and any photograph you
                                     drop in, are decided and answered live by the 1.75-bit
                                     Bonsai 2 27B on this machine's GPU
    python serve_ternary.py          replay of the measured laptop run in ternary/results.json;
                                     no GPU, no model

Opens http://127.0.0.1:8767. In live mode one llama-server process of the PrismML fork holds the
weights once and serves both paths: the pre-answer state for the Ammonix decision through its
embeddings endpoint and the generated answers through its completion endpoint. The cache is
erased before every measured request. Every time shown in the page is the wall-clock time of that
request on this machine, measured once, not a mean of trials.

With the environment variable OPENAI_API_KEY set, the page can compare against GPT-6 at maximum
reasoning instead of the local generated answer (one paid request per passenger, never made
without the choice in the page). The key never enters any result.
"""
import runtime
from runtime import ROOT, CLASSES, PROMPT, TRADITIONAL_PROMPT, TRADITIONAL_MAX_NEW_TOKENS
import argparse, base64, http.client, json, os, re, threading, time, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlsplit
from benchmark import fallback, normalize
from benchmark_ternary import Classifier, traditional, gpu_name

PORT = 8767
PASSENGERS = json.loads((ROOT / 'passengers.json').read_text('utf-8'))
RESULTS = ROOT / 'ternary' / 'results.json'
MODEL_LABEL = 'Ternary Bonsai 2 27B · PTQ1_0 · 1.75 bits per weight'
GPT6_MODEL = 'gpt-6-astra'
# Files the page may fetch. Everything else in the folder (settings, models, logs) stays private.
PUBLIC = ('/airport.html', '/airport.js', '/credits.html', '/passengers.json', '/brand/', '/data/', '/ternary/results.json')
MAX_UPLOAD = 12 << 20


def decode_image(field):
    """The photograph of a request: a passenger id, or an uploaded JPEG/PNG/WebP as base64."""
    if 'id' in field:
        row = next((r for r in PASSENGERS['rows'] if r['id'] == field['id']), None)
        if row is None:
            raise ValueError('Unknown passenger id')
        return (ROOT / row['image']).read_bytes(), row
    data = str(field.get('image', ''))
    data = data.split(',', 1)[1] if data.startswith('data:') else data
    raw = base64.b64decode(data, validate=True)
    if not (raw[:3] == b'\xff\xd8\xff' or raw[:8] == b'\x89PNG\r\n\x1a\n' or raw[:4] == b'RIFF'):
        raise ValueError('The upload is not a JPEG, PNG or WebP photograph')
    return raw, {'id': 'upload-' + str(int(time.time() * 1000)), 'species': None, 'gate': None, 'ood': False, 'upload': True}


def gpt6_answer(raw):
    """One request to GPT-6 at maximum reasoning, as in benchmark_gpt6.py; returns (answer, ms, complete)."""
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    payload = {'model': GPT6_MODEL, 'reasoning': {'effort': 'max'}, 'store': False, 'max_output_tokens': 8192,
               'input': [{'role': 'user', 'content': [
                   {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'), 'detail': 'high'},
                   {'type': 'input_text', 'text': TRADITIONAL_PROMPT}]}]}
    started = time.perf_counter()
    connection = http.client.HTTPSConnection('api.openai.com', timeout=300)
    try:
        connection.request('POST', '/v1/responses', body=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = json.loads(response.read())
    finally:
        connection.close()
    elapsed = (time.perf_counter() - started) * 1000
    answer = ''.join(c.get('text', '') for o in data.get('output', []) if o.get('type') == 'message' for c in o.get('content', []) if c.get('type') == 'output_text')
    complete = response.status == 200 and data.get('status') == 'completed' and bool(answer.strip())
    if not complete:
        error = json.dumps(data.get('error') or data.get('incomplete_details') or {'http_status': response.status})
        answer = 'GPT-6 request failed: ' + re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', error)[:300]
    return answer.strip(), elapsed, complete


class Live:
    """Both measurement paths on the ternary weights: one server process, one request at a time."""

    def __init__(self):
        from engine_ternary import Engine
        self.classifier = Classifier()
        self.engine = Engine(mode='both', warmup=2)
        self.lock = threading.Lock()
        self.props = self.engine.props()
        self.gpu = gpu_name()

    def record(self, kind, rec):
        """Every live measurement is appended to logs/live-<day>.jsonl, so a demonstration leaves evidence."""
        try:
            (ROOT / 'logs').mkdir(exist_ok=True)
            entry = {k: v for k, v in rec.items() if k != 'probabilities'}
            entry.update(kind=kind, at=time.strftime('%Y-%m-%dT%H:%M:%S'), gpu=self.gpu)
            with open(ROOT / 'logs' / time.strftime('live-%Y-%m-%d.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except OSError:
            pass

    def decide(self, raw):
        """System One: state, classifier, unknown-animal check; System Two only when escalated."""
        with self.lock:
            self.engine.erase_cache()
            x, t = self.engine.feature(raw)
            rec, p = self.classifier.predict(x)
            rec.update(t)
            rec['ammonix_ms'] = t['feature_ms'] + rec['classifier_ms']
            rec['probabilities'] = [float(v) for v in p]
            rec.update(final_label=rec['prediction'], final_gate=rec['gate'], fallback_ms=0)
            if rec['escalated']:
                self.engine.erase_cache()
                rec.update(fallback(self.engine, raw))
        return rec

    def answer(self, raw, rival):
        """The full generated answer of the same model, or of GPT-6 when chosen and configured."""
        if rival == 'gpt6':
            if not os.environ.get('OPENAI_API_KEY', '').strip():
                raise ValueError('OPENAI_API_KEY is not set in this terminal')
            text, ms, complete = gpt6_answer(raw)
            return dict(rival='gpt6', traditional_answer=text, traditional_ms=ms, traditional_completion={'complete': complete, 'model': GPT6_MODEL, 'reasoning_effort': 'max'})
        with self.lock:
            self.engine.erase_cache()
            text, ms, completion = traditional(self.engine, raw)
        return dict(rival='local', traditional_answer=text, traditional_ms=ms, traditional_completion=completion)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--live', action='store_true', help='decide and answer live on the ternary model')
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    live = Live() if args.live else None
    gpu = gpu_name()
    status = {
        'live': live is not None, 'model': MODEL_LABEL, 'gpu': gpu,
        'server': (live.props.get('build_info') if live else None),
        'gpt6': bool(os.environ.get('OPENAI_API_KEY', '').strip()) and live is not None,
        'replay': RESULTS.exists(), 'passengers': len(PASSENGERS['rows']), 'classes': CLASSES,
        'feature_prompt': PROMPT, 'traditional_prompt': TRADITIONAL_PROMPT, 'traditional_max_new_tokens': TRADITIONAL_MAX_NEW_TOKENS,
    }

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                self.send_response(302); self.send_header('Location', '/airport.html'); self.end_headers(); return
            if path == '/api/status':
                return self._json(status)
            if path == '/api/passengers':
                return self._json(PASSENGERS)
            if path == '/api/replay':
                if not RESULTS.exists():
                    return self._json({'error': 'no measured run in ternary/results.json'}, 404)
                measured = json.loads(RESULTS.read_text('utf-8'))
                return self._json({'gpu': measured.get('gpu'), 'model': measured.get('model'), 'report': measured.get('report'),
                                   'rows': {r['id']: r for r in measured['rows']}})
            if any(path == p or (p.endswith('/') and path.startswith(p)) for p in PUBLIC):
                return super().do_GET()
            self.send_error(404)

        def do_POST(self):
            path = urlsplit(self.path).path
            if live is None or path not in ('/api/decide', '/api/answer'):
                return self._json({'error': 'not available' if live is None else 'unknown endpoint'}, 404)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length > MAX_UPLOAD:
                    raise ValueError('Request too large')
                field = json.loads(self.rfile.read(length))
                raw, row = decode_image(field)
                if path == '/api/decide':
                    rec = live.decide(raw)
                    rec.update(id=row['id'], species=row.get('species'), gold_gate=row.get('gate'), ood=bool(row.get('ood')), upload=bool(row.get('upload')))
                    rec['correct'] = (rec['prediction'] == row['species']) if row.get('species') and not row.get('ood') else None
                    live.record('decide', rec)
                else:
                    rec = live.answer(raw, field.get('rival', 'local'))
                    rec['id'] = row['id']
                    # Whether the free answer names the reference species (released normalizer; informative only).
                    rec['answer_names_species'] = (normalize(rec['traditional_answer']) == row['species']) if row.get('species') else None
                    live.record('answer', rec)
                self._json(rec)
            except Exception as e:
                self._json({'error': f'{type(e).__name__}: {e}'}, 400)

    url = f'http://127.0.0.1:{args.port}/'
    print(f'Wild Departures, ternary model: {url}  live={live is not None}  gpu={gpu}', flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            live.engine.close()


if __name__ == '__main__':
    main()
