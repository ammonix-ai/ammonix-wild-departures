"""Re-collect the photographs that are not shipped, from the sources recorded in manifest.json.

Only the 71 replay photographs are in the repository. This script downloads every other
manifest row from its recorded `original` URL and reproduces the processing of collect.py
(Wikimedia Commons originals: EXIF transpose, RGB, thumbnail 960x960, JPEG quality 91) and
collect_inat.py (iNaturalist `medium` files: EXIF transpose, RGB, JPEG quality 93). The
SHA-256 of each downloaded source file is compared with the manifest and recorded.

Resumable: photographs already under data/ are skipped. Progress and outcomes are written to
ternary/recollect-log.json. Nothing in the released files is modified.
"""
import runtime
from runtime import ROOT
import argparse, hashlib, io, json, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image, ImageOps

# Some Commons originals exceed PIL's default 89-megapixel guard; the files come from the sources
# recorded in the manifest, so the guard is lifted (collect.py processed the same originals).
Image.MAX_IMAGE_PIXELS = None

OUT = ROOT / 'ternary'
LOG = OUT / 'recollect-log.json'
UA = {'User-Agent': 'AmmonixAnimalDemo/1.0 (https://github.com/ammonix-ai/ammonix-wild-departures; research re-collection of the released manifest)'}
COMMONS = 'https://commons.wikimedia.org'

lock = threading.Lock()
log = {}
counts = {'ok': 0, 'sha_mismatch': 0, 'failed': 0, 'present': 0}


class Throttle:
    """One request at a time per source, at most one per `interval` seconds. A 429 or 503 pauses
    the whole source for the server's Retry-After (Wikimedia answers 429 with Retry-After 600)."""

    def __init__(self, interval, initial_pause=0.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_allowed = time.monotonic() + initial_pause

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_allowed:
                time.sleep(self.next_allowed - now)
            self.next_allowed = time.monotonic() + self.interval

    def pause(self, seconds):
        with self.lock:
            self.next_allowed = max(self.next_allowed, time.monotonic() + seconds)


def save_log():
    OUT.mkdir(exist_ok=True)
    tmp = LOG.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(counts=counts, rows=log), indent=1), encoding='utf-8')
    tmp.replace(LOG)


def fetch(session, url, throttle):
    last = None
    limited = 0
    for attempt in range(8):
        throttle.wait()
        try:
            r = session.get(url, headers=UA, timeout=120)
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            try:
                wait = min(900.0, float(r.headers.get('Retry-After', 60) or 60))
            except ValueError:
                wait = 60.0
            print(f'{r.status_code} from {url.split("/")[2]}: pausing that source for {wait:.0f}s', flush=True)
            throttle.pause(wait)
            last = r.status_code
            limited += 1
            if limited >= 3:
                break
            continue
        if r.status_code == 404:
            return None, 404
        if r.status_code >= 400:
            last = r.status_code
            time.sleep(3 * (attempt + 1))
            continue
        return r.content, r.status_code
    return None, last


def process(row, raw):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    if row['source'].startswith(COMMONS):
        im.thumbnail((960, 960))
        quality = 91
    else:
        quality = 93
    target = ROOT / row['image']
    tmp = target.with_suffix('.part.jpg')
    im.save(tmp, 'JPEG', quality=quality)
    tmp.replace(target)


def work(row, session, throttle):
    if (ROOT / row['image']).exists():
        with lock:
            counts['present'] += 1
        return
    url = row['original']
    raw, status = fetch(session, url, throttle)
    if raw is None and status == 404 and 'inaturalist' in url:
        # iNaturalist occasionally re-encoded a photo under the other extension.
        alt = url.replace('medium.jpeg', 'medium.jpg') if url.endswith('.jpeg') else url.replace('medium.jpg', 'medium.jpeg')
        if alt != url:
            raw, status = fetch(session, alt, throttle)
    if raw is None:
        with lock:
            counts['failed'] += 1
            log[row['id']] = dict(status='failed', http=status, species=row['species'], split=row['split'], url=url)
        return
    sha_ok = hashlib.sha256(raw).hexdigest() == row['sha256']
    try:
        process(row, raw)
    except Exception as e:
        with lock:
            counts['failed'] += 1
            log[row['id']] = dict(status='failed', error=type(e).__name__, species=row['species'], split=row['split'], url=url)
        return
    with lock:
        counts['ok' if sha_ok else 'sha_mismatch'] += 1
        log[row['id']] = dict(status='ok', sha256_match=sha_ok, source_bytes=len(raw), species=row['species'], split=row['split'])


def run(rows, workers, throttle, label):
    session = requests.Session()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(lambda r: work(r, session, throttle), rows):
            done += 1
            if done % 25 == 0 or done == len(rows):
                with lock:
                    save_log()
                    print(label, done, '/', len(rows), dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commons-workers', type=int, default=1, help='Wikimedia asks for gentle, serialized access; keep at 1')
    parser.add_argument('--commons-interval', type=float, default=1.0, help='seconds between Commons requests')
    parser.add_argument('--commons-pause', type=float, default=0.0, help='seconds to wait before the first Commons request (after an earlier 429)')
    parser.add_argument('--inat-workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if LOG.exists():
        log = json.loads(LOG.read_text('utf-8')).get('rows', {})
        log = {k: v for k, v in log.items() if v.get('status') != 'failed'}   # retry earlier failures
    rows = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
    missing = [r for r in rows if not (ROOT / r['image']).exists()]
    commons = [r for r in missing if r['source'].startswith(COMMONS)]
    inat = [r for r in missing if not r['source'].startswith(COMMONS)]
    print(f'{len(rows)} manifest rows, {len(rows) - len(missing)} present, {len(commons)} Commons + {len(inat)} iNaturalist to fetch', flush=True)
    start = time.perf_counter()
    threads = [threading.Thread(target=run, args=(commons, args.commons_workers, Throttle(args.commons_interval, args.commons_pause), 'commons')),
               threading.Thread(target=run, args=(inat, args.inat_workers, Throttle(0.05), 'inat'))]
    for t in threads: t.start()
    for t in threads: t.join()
    save_log()
    failed = [k for k, v in log.items() if v.get('status') == 'failed']
    present = sum((ROOT / r['image']).exists() for r in rows)
    print(json.dumps(dict(seconds=round(time.perf_counter() - start), counts=counts, photographs_present=present, of=len(rows), failed_ids=failed), indent=1), flush=True)
    print('RECOLLECT_DONE', flush=True)
