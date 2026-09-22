"""Fetch the pinned model files and llama.cpp binaries into this folder. Standard library only.

    python setup.py              model + vision projector + llama.cpp for this platform
    python setup.py --model      only the two GGUF files (about 6.6 GB)
    python setup.py --llama      only the llama.cpp release archives
    python setup.py --cuda 12.4  build for an older NVIDIA driver (default 13.3; Linux also 12.8)
    python setup.py --cpu        CPU-only build (Linux / macOS; slow, for checking only)

Every file is pinned by size and SHA-256; a partial download resumes where it stopped.
The model is Ternary Bonsai 2 27B by PrismML (Qwen3.8-27B with ternary weights, Apache 2.0),
which needs PrismML's llama.cpp fork: stock llama.cpp refuses its PTQ1_0 packing.
"""
from pathlib import Path
import argparse
import hashlib
import platform
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent

MODEL_REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
MODEL_REVISION = '6ed5e12bf84b7a63069882c91dd9e9218647d17b'
HF = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/'
MODEL_FILES = {
    'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf': (HF + 'Ternary-Bonsai-2-27B-PTQ1_0.gguf', 5946648928,
        '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'),
    'models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf': (HF + 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf', 629246976,
        '6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903'),
}

LLAMA_RELEASE = 'prism-b10709-9a9394a'
GH = f'https://github.com/PrismML-Eng/llama.cpp/releases/download/{LLAMA_RELEASE}/'
CUDART = {
    '13.3': ('cudart-llama-bin-win-cuda-13.3-x64.zip', 390970417, '1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e'),
    '12.4': ('cudart-llama-bin-win-cuda-12.4-x64.zip', 391443627, '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'),
}
# (name, size, sha256) as published by the GitHub release API; the first archive holds llama-server.
LLAMA_ARCHIVES = {
    ('Windows', 'x64', '13.3'): [(f'llama-{LLAMA_RELEASE}-bin-win-cuda-13.3-x64.zip', 145031500, 'd656f217172c489706df40951e46bef647eb1a81eb8eeb1398c8f38bd7fa9725'), CUDART['13.3']],
    ('Windows', 'x64', '12.4'): [(f'llama-{LLAMA_RELEASE}-bin-win-cuda-12.4-x64.zip', 253442371, 'f565c8428c1f108311f65ed97f02425188b3aa3c745c2bc597521bbd24bcbbc9'), CUDART['12.4']],
    ('Linux', 'x64', '13.3'): [(f'llama-{LLAMA_RELEASE}-bin-linux-cuda-13.3-x64.tar.gz', 146381070, '7e01a434e513b373026c347cd008502ab04f6307d1cab71fcd4cea212b4fdbb0')],
    ('Linux', 'x64', '12.8'): [(f'llama-{LLAMA_RELEASE}-bin-linux-cuda-12.8-x64.tar.gz', 167241119, '8aec67eb023b251712c7e6490f367b5671bf587eced1436a9b85f4a90c3b7d3d')],
    ('Linux', 'x64', '12.4'): [(f'llama-{LLAMA_RELEASE}-bin-linux-cuda-12.4-x64.tar.gz', 260869644, 'f542fdcc818562359e947db65e0b11c4658dd5ca3bd240490448252e817d8e7a')],
    ('Linux', 'x64', 'cpu'): [(f'llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz', 17108139, '48b487f00fd2b27bc3ef77c701b43c1c23a4af484d2a203ae87d0efc41506728')],
    ('Darwin', 'arm64', 'cpu'): [(f'llama-{LLAMA_RELEASE}-bin-macos-arm64.tar.gz', 11500187, 'f9cdf245fb7b832f1996dd776b321d4ae1f23b6d88c380100f636742c3a980ff')],
}


def download(url, target, size, sha256):
    target = Path(target)
    if target.exists() and target.stat().st_size == size:
        print(f'{target.name}: already present ({size:,} bytes)', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    digest = hashlib.sha256()
    done = 0
    if part.exists():
        with part.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                digest.update(chunk)
                done += len(chunk)
        if done >= size:
            part.unlink(); done = 0; digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={'User-Agent': 'ammonix-wild-departures-setup'})
    if done:
        request.add_header('Range', f'bytes={done}-')
    print(f'{target.name}: downloading {size:,} bytes' + (f' (resuming at {done:,})' if done else ''), flush=True)
    started = time.time(); last = 0
    with urllib.request.urlopen(request, timeout=120) as response, part.open('ab' if done else 'wb') as f:
        if done and response.status != 206:
            raise SystemExit(f'{url}: the server ignored the resume request; delete {part} and retry')
        for chunk in iter(lambda: response.read(1 << 20), b''):
            f.write(chunk); digest.update(chunk); done += len(chunk)
            if time.time() - last > 5:
                rate = done / max(1e-6, time.time() - started) / 1e6
                print(f'  {done / size:6.1%}  {done / 1e9:6.2f} / {size / 1e9:.2f} GB  {rate:5.1f} MB/s', flush=True); last = time.time()
    if done != size:
        raise SystemExit(f'{target.name}: received {done:,} bytes, expected {size:,}; run setup again to resume')
    if digest.hexdigest() != sha256:
        part.unlink()
        raise SystemExit(f'{target.name}: SHA-256 mismatch, the download was discarded')
    part.replace(target)
    print(f'{target.name}: verified', flush=True)


def fetch_llama(cuda):
    machine = 'x64' if platform.machine().lower() in ('amd64', 'x86_64') else 'arm64'
    key = (platform.system(), machine, cuda)
    if key not in LLAMA_ARCHIVES:
        raise SystemExit(f'No pinned llama.cpp archive for {key}. Build the PrismML fork at release {LLAMA_RELEASE} '
                         '(https://github.com/PrismML-Eng/llama.cpp) and put llama-server with its libraries in llama/.')
    folder = ROOT / 'llama'; folder.mkdir(exist_ok=True)
    destination = folder
    for name, size, sha256 in LLAMA_ARCHIVES[key]:
        archive = folder / name
        download(GH + name, archive, size, sha256)
        print(f'{archive.name}: extracting', flush=True)
        if archive.suffix == '.zip':
            with zipfile.ZipFile(archive) as z:
                z.extractall(destination)
        else:
            with tarfile.open(archive) as t:
                t.extractall(destination)
        server = next((p for p in folder.rglob('llama-server*') if p.is_file() and p.suffix in ('', '.exe')), None)
        if server is None:
            raise SystemExit('llama-server was not found after extraction')
        destination = server.parent   # runtime libraries of later archives go next to the server
    print(f'llama.cpp {LLAMA_RELEASE} ready: {server}', flush=True)


def already_available():
    """Files that engine_ternary.py will find anyway: the older ../runtime layout next to the
    repository, or paths given in the AMMONIX_TERNARY_* environment variables."""
    import os
    runtime = Path(os.environ.get('AMMONIX_TERNARY_RUNTIME') or ROOT.parent / 'runtime')
    models = [Path(os.environ.get('AMMONIX_TERNARY_MODEL') or runtime / 'models' / 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'),
              Path(os.environ.get('AMMONIX_TERNARY_MMPROJ') or runtime / 'models' / 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf')]
    server = runtime / 'bin' / ('llama-server.exe' if platform.system() == 'Windows' else 'llama-server')
    return all(p.exists() for p in models), server.exists(), models, server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='store_true', help='only the GGUF files')
    parser.add_argument('--llama', action='store_true', help='only the llama.cpp binaries')
    parser.add_argument('--cuda', default='13.3', help='CUDA runtime of the build: 13.3 (default), 12.4, or 12.8 on Linux')
    parser.add_argument('--cpu', action='store_true', help='CPU-only build (Linux, macOS)')
    parser.add_argument('--force', action='store_true', help='download even when the files exist in ../runtime or the AMMONIX_TERNARY_* paths')
    args = parser.parse_args()
    everything = not (args.model or args.llama)
    have_models, have_server, models, server = already_available()
    if args.model or everything:
        if have_models and not args.force:
            print('model files already available: ' + ', '.join(str(p) for p in models), flush=True)
        else:
            for relative, (url, size, sha256) in MODEL_FILES.items():
                download(url, ROOT / relative, size, sha256)
    if args.llama or everything:
        if have_server and not args.force:
            print(f'llama-server already available: {server}', flush=True)
        else:
            fetch_llama('cpu' if args.cpu or platform.system() == 'Darwin' else args.cuda)
    print('SETUP_DONE', flush=True)
