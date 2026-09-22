"""Frozen ternary Qwen3.8-27B (Bonsai 2 27B, PTQ1_0), served locally by the PrismML llama.cpp fork.

Counterpart of engine.Engine for the 1.75-bit GGUF weights with the Q8_0 vision projector. The
state is the same quantity as in engine.py: the normalized final-layer hidden state at the last
prompt position, conditioned on the photograph and the fixed prompt. llama-server started with
`--embeddings --pooling last` returns exactly that vector (the output of the final norm, which
Hugging Face calls last_hidden_state), un-normalized when the request asks for embd_normalize -1.

An Engine runs in one of three modes. 'both' (the default, used by the live airport) starts one
llama-server with `--embeddings --pooling last` and uses it for the state and for generation: the
server switches its context between the two per request, so one copy of the weights serves both
paths. 'feature' and 'generate' are the single-purpose modes of the laptop benchmark, one server
process each. Paths: `python setup.py` puts llama-server under llama/ and the two GGUF files under
models/ inside this folder; ../runtime next to the repository (bin/llama-server.exe, models/*.gguf)
is the older layout and still found; the environment variables AMMONIX_TERNARY_RUNTIME,
AMMONIX_TERNARY_MODEL and AMMONIX_TERNARY_MMPROJ override everything.
AMMONIX_TERNARY_MMPROJ_CPU=1 keeps the vision projector in system RAM (about 600 MB of GPU memory
freed, but about 4 s of CPU image encoding per photograph on this laptop); by default it is on the GPU.
"""
import runtime
from runtime import ROOT, PROMPT
import atexit, base64, io, json, os, subprocess, time, urllib.error, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

RUNTIME = Path(os.environ.get('AMMONIX_TERNARY_RUNTIME') or ROOT.parent / 'runtime')
MODEL_FILE = 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
MMPROJ_FILE = 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'


def _first_existing(*candidates):
    for path in candidates:
        if path is not None and Path(path).exists():
            return Path(path)
    return Path(candidates[0])


def _server_binary():
    name = 'llama-server.exe' if os.name == 'nt' else 'llama-server'
    local = ROOT / 'llama'
    found = next((p for p in local.rglob(name) if p.is_file()), None) if local.exists() else None
    return _first_existing(os.environ.get('AMMONIX_TERNARY_RUNTIME') and RUNTIME / 'bin' / name, found, RUNTIME / 'bin' / name)


SERVER_EXE = _server_binary()
MODEL = _first_existing(os.environ.get('AMMONIX_TERNARY_MODEL'), ROOT / 'models' / MODEL_FILE, RUNTIME / 'models' / MODEL_FILE)
MMPROJ = _first_existing(os.environ.get('AMMONIX_TERNARY_MMPROJ'), ROOT / 'models' / MMPROJ_FILE, RUNTIME / 'models' / MMPROJ_FILE)
PORT = int(os.environ.get('AMMONIX_TERNARY_PORT', '8090'))
OUT = ROOT / 'ternary'
LOGS = OUT / 'logs'
# The server's media marker stands for the photograph in the prompt string; mtmd expands it to
# <|vision_start|> ... <|vision_end|> for Qwen VL models. This build randomizes the marker per
# process unless LLAMA_MEDIA_MARKER pins it, so the Engine pins it and re-reads it from /props.
MARKER = '<__media__>'
FEATURE_DIM = 5120
# Pixel budget of the released run: 65,536 to 147,456 pixels. One vision token covers 32x32 pixels
# (patch 16, spatial merge 2), so this is 64 to 144 vision tokens.
IMAGE_MIN_TOKENS, IMAGE_MAX_TOKENS = 64, 144


def render(prompt, marker=MARKER):
    """The released chat template with add_generation_prompt=True and enable_thinking=False."""
    return f'<|im_start|>user\n{marker}{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class Engine:
    def __init__(self, mode='both', warmup=3):
        if mode not in ('both', 'feature', 'generate'):
            raise ValueError(mode)
        self.mode = mode
        self.base = f'http://127.0.0.1:{PORT}'
        self.proc = None
        self.last_generation = None
        for path, what in ((SERVER_EXE, 'llama-server of the PrismML fork'), (MODEL, 'ternary model'), (MMPROJ, 'vision projector')):
            if not path.exists():
                raise FileNotFoundError(f'The {what} is missing: {path}. Run `python setup.py` first.')
        if self._alive():
            raise RuntimeError(f'Something already answers on {self.base}; stop it before starting an Engine.')
        args = [str(SERVER_EXE), '-m', str(MODEL), '--mmproj', str(MMPROJ),
                '--host', '127.0.0.1', '--port', str(PORT),
                '-ngl', '99', '-fa', 'on', '-c', '2048', '--parallel', '1',
                '--image-min-tokens', str(IMAGE_MIN_TOKENS), '--image-max-tokens', str(IMAGE_MAX_TOKENS),
                '--cache-ram', '0', '--slots', '--slot-save-path', str(LOGS / 'slots'), '--jinja', '--no-webui']
        # Default: the vision projector is offloaded to the GPU like the language model (the whole
        # server then occupies about 7.2 GiB of the 8 GiB card). AMMONIX_TERNARY_MMPROJ_CPU=1 keeps it
        # in system RAM instead: about 600 MB of GPU memory freed, but image encoding on this
        # laptop's CPU costs about 4 s per photograph (measured) instead of well under a second.
        self.mmproj_on_cpu = os.environ.get('AMMONIX_TERNARY_MMPROJ_CPU', '0') in ('1', 'true', 'yes')
        if self.mmproj_on_cpu:
            args.append('--no-mmproj-offload')
        if mode in ('feature', 'both'):
            args += ['--embeddings', '--pooling', 'last']
        self.args = args
        (LOGS / 'slots').mkdir(parents=True, exist_ok=True)   # --slot-save-path enables the slot erase action
        self.log = LOGS / f'server-{mode}.log'
        self._logf = open(self.log, 'ab')
        self._logf.write(('\n=== ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + ' '.join(args) + '\n').encode())
        self._logf.flush()
        env = dict(os.environ, LLAMA_MEDIA_MARKER=MARKER)
        self.proc = subprocess.Popen(args, stdout=self._logf, stderr=subprocess.STDOUT, cwd=str(SERVER_EXE.parent), env=env)
        atexit.register(self.close)
        self._wait_ready()
        self.marker = self.props().get('media_marker') or MARKER
        # Separate warmup; never included in reported latency.
        for _ in range(warmup):
            image = Image.new('RGB', (384, 384), (40, 100, 60))
            if mode in ('feature', 'both'):
                self.feature(image)
            if mode in ('generate', 'both'):
                self.generate(image)

    # ----- server lifecycle -----
    def _alive(self):
        try:
            with urllib.request.urlopen(self.base + '/health', timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_ready(self, timeout=900):
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f'llama-server exited with code {self.proc.returncode}; see {self.log}')
            if self._alive():
                return
            time.sleep(1)
        raise TimeoutError(f'llama-server not ready after {timeout}s; see {self.log}')

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        try:
            self._logf.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- HTTP helpers -----
    def _post(self, path, body, timeout=900):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{path} -> HTTP {e.code}: {e.read()[:600]!r}')

    def props(self):
        with urllib.request.urlopen(self.base + '/props', timeout=30) as r:
            return json.loads(r.read())

    def erase_cache(self):
        """Drop the slot's KV cache so a repeated photograph is processed from scratch."""
        return self._post('/slots/0?action=erase', {})

    def template_check(self, prompt=PROMPT):
        """Render the fixed prompt with the server's own copy of the chat template and compare with render()."""
        res = self._post('/apply-template', {'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': False}})
        server = res['prompt']
        ours = render(prompt, self.marker).replace(self.marker, '')
        return server == ours, server, ours

    def _payload(self, image):
        """A photograph as base64: a path, the raw JPEG/PNG bytes, or a PIL image."""
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.convert('RGB').save(buf, 'JPEG', quality=95)
            raw = buf.getvalue()
        elif isinstance(image, (bytes, bytearray)):
            raw = bytes(image)
        else:
            raw = Path(image).read_bytes()
        return base64.b64encode(raw).decode('ascii')

    # ----- the two paths -----
    def feature(self, image, prompt=PROMPT):
        if self.mode == 'generate':
            raise RuntimeError('This Engine was started in generate mode; use Engine(mode="feature") or the default mode "both".')
        start = time.perf_counter()
        b64 = self._payload(image)
        processed = time.perf_counter()
        res = self._post('/embeddings', {'content': {'prompt_string': render(prompt, self.marker), 'multimodal_data': [b64]}, 'embd_normalize': -1})
        done = time.perf_counter()
        emb = np.asarray(res[0]['embedding'], dtype=np.float32)
        if emb.ndim == 2:
            assert emb.shape[0] == 1, emb.shape   # one pooled vector per prompt
            emb = emb[0]
        assert emb.shape == (FEATURE_DIM,) and np.isfinite(emb).all(), emb.shape
        # preprocess_ms: reading and encoding the file on the client. The server decodes and resizes
        # the photograph inside the request, so that work is part of qwen_ms here.
        return emb, {'preprocess_ms': (processed - start) * 1000, 'qwen_ms': (done - processed) * 1000, 'feature_ms': (done - start) * 1000}

    def generate(self, image, prompt=PROMPT, max_new_tokens=24):
        if self.mode == 'feature':
            raise RuntimeError('This Engine was started in feature mode; use Engine(mode="generate") or the default mode "both".')
        start = time.perf_counter()
        b64 = self._payload(image)
        res = self._post('/completion', {'prompt': {'prompt_string': render(prompt, self.marker), 'multimodal_data': [b64]},
                                         'n_predict': max_new_tokens, 'temperature': 0, 'top_k': 1,
                                         'cache_prompt': False, 'n_probs': 0})
        elapsed = (time.perf_counter() - start) * 1000
        answer = str(res.get('content', '')).strip()
        ended = res.get('stop_type') == 'eos'
        timings = res.get('timings') or {}
        self.last_generation = {'generated_tokens': int(res.get('tokens_predicted', 0)), 'finish_reason': 'eos' if ended else 'length', 'complete': ended,
                                'prompt_tokens': int(res.get('tokens_evaluated', 0)),
                                'server_prompt_ms': timings.get('prompt_ms'), 'server_predicted_ms': timings.get('predicted_ms')}
        return answer, elapsed


def log_highlights(path, keys=('image_min_pixels', 'image_max_pixels', 'model buffer size', 'compute buffer size', 'KV self size', 'load_tensors', 'pooling', 'n_ctx ', 'warmup', 'error', 'failed')):
    lines = path.read_text('utf-8', errors='replace').splitlines()
    return [l for l in lines[-400:] if any(k in l for k in keys)]


if __name__ == '__main__':
    from runtime import TRADITIONAL_PROMPT, load_features
    rows = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
    sample = next(r for r in rows if r['id'] == '75068269')      # data/dog-75068269.jpg, a shipped test photograph
    print('runtime:', RUNTIME, '| model:', MODEL.name, '| mmproj:', MMPROJ.name, flush=True)
    e = Engine(mode='feature')
    try:
        same, server, ours = e.template_check()
        print('chat template identical to render():', same)
        if not same:
            print('server:', repr(server)); print('ours:  ', repr(ours))
        for color in [(20, 130, 80), (130, 20, 80)]:
            x, t = e.feature(Image.new('RGB', (384, 384), color)); print('synthetic', x.shape, round(float(np.linalg.norm(x)), 1), {k: round(v) for k, v in t.items()}, flush=True)
        x, t = e.feature(ROOT / sample['image'])
        fp16 = load_features([sample])[0]
        cos = float(x @ fp16 / np.linalg.norm(x) / np.linalg.norm(fp16))
        print(f"{sample['image']}: ternary norm {np.linalg.norm(x):.1f}, FP16 norm {np.linalg.norm(fp16):.1f}, cosine {cos:.4f}, timing {({k: round(v) for k, v in t.items()})}", flush=True)
        for l in log_highlights(e.log): print('  log |', l)
    finally:
        e.close()
    e = Engine(mode='generate')
    try:
        answer, ms = e.generate(ROOT / sample['image'], TRADITIONAL_PROMPT, max_new_tokens=128)
        print(f'traditional answer ({ms:.0f} ms): {answer!r}', e.last_generation, flush=True)
        answer, ms = e.generate(ROOT / sample['image'], PROMPT, max_new_tokens=24)
        print(f'species answer ({ms:.0f} ms): {answer!r}', e.last_generation, flush=True)
    finally:
        e.close()
