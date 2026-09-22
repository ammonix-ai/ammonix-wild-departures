"""Measure the ternary decision layer against the ternary traditional answer on the replay photographs.

The protocol of benchmark.py: the 70 held-out photographs in the same shuffled order plus the
unknown animal, three trials each, medians reported, everything sequential at batch size 1 with
no reused cache. Two deviations are forced by the server: the state and the generated answer
come from two separate llama-server sessions of the same weights (the embeddings mode cannot
generate), so all Ammonix trials run first and all traditional trials second instead of
alternating per photograph; and both paths receive the frozen JPEG bytes and let the server
resize them, so image preprocessing is measured inside the request. Before every measured
request the slot's cache is erased.

Writes ternary/results.json, ternary/raw-trials.json and ternary/benchmark-progress.json. The
released results are untouched. Needs the refit classifier in ternary/ (train_ternary.py).
"""
import runtime
from runtime import ROOT, CLASSES, GATES, PROMPT, TRADITIONAL_PROMPT, TRADITIONAL_MAX_NEW_TOKENS
import hashlib, json, random, subprocess, threading, time
import numpy as np
import xgboost as xgb
from train import probs, metrics
from benchmark import Classifier as ReleasedClassifier, normalize, fallback
from engine_ternary import Engine, MODEL, MMPROJ, IMAGE_MIN_TOKENS, IMAGE_MAX_TOKENS

OUT = ROOT / 'ternary'


class Telemetry(threading.Thread):
    """Samples GPU temperature, clocks, power and memory with nvidia-smi while the benchmark runs.

    A laptop GPU throttles under sustained load (this one drops from 1,635 to about 510 MHz at
    81 C), so the thermal condition is recorded next to every timing it influenced."""
    FIELDS = ['timestamp', 'temperature.gpu', 'clocks.sm', 'clocks.mem', 'power.draw', 'memory.used', 'utilization.gpu',
              'clocks_throttle_reasons.sw_thermal_slowdown', 'clocks_throttle_reasons.hw_thermal_slowdown', 'clocks_throttle_reasons.sw_power_cap']

    def __init__(self, path, interval=2.0):
        super().__init__(daemon=True)
        self.path, self.interval, self.rows = path, interval, []
        self.stop_event = threading.Event()

    def run(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(','.join(self.FIELDS) + '\n')
            while not self.stop_event.is_set():
                try:
                    out = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(self.FIELDS), '--format=csv,noheader,nounits'],
                                         capture_output=True, text=True, timeout=15).stdout.strip()
                    if out:
                        f.write(out + '\n')
                        f.flush()
                        self.rows.append([c.strip() for c in out.split(',')])
                except Exception:
                    pass
                self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=20)

    def summary(self):
        def column(i):
            values = []
            for r in self.rows:
                try:
                    values.append(float(r[i]))
                except (ValueError, IndexError):
                    pass
            return values
        temperature, sm, power, memory = column(1), column(2), column(4), column(5)
        thermal = [r[7] for r in self.rows if len(r) > 7]
        if not sm:
            return None
        return dict(samples=len(self.rows), interval_s=self.interval,
                    temperature_c=dict(median=float(np.median(temperature)), max=max(temperature)),
                    sm_clock_mhz=dict(median=float(np.median(sm)), min=min(sm), max=max(sm)),
                    power_w_median=float(np.median(power)) if power else None,
                    memory_used_mib_max=max(memory) if memory else None,
                    share_of_samples_thermally_throttled=sum(1 for x in thermal if x == 'Active') / len(thermal) if thermal else None)


class Classifier(ReleasedClassifier):
    """The refit decision layer in ternary/, scored exactly like the released one."""
    def __init__(self, folder=OUT):
        self.model = xgb.Booster()
        self.model.load_model(str(folder / 'ammonix-xgboost.ubj'))
        self.model.set_param({'nthread': 1, 'device': 'cpu'})
        self.config = json.loads((folder / 'training.json').read_text())
        self.novel = dict(np.load(folder / 'novelty.npz'))


def traditional(engine, path):
    """Like benchmark.traditional, but an answer that hits the token limit is recorded, not fatal."""
    answer, latency = engine.generate(path, TRADITIONAL_PROMPT, max_new_tokens=TRADITIONAL_MAX_NEW_TOKENS)
    return answer, latency, dict(engine.last_generation)


def gpu_name():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return 'unknown'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def run():
    rows = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
    test = [r for r in rows if r['split'] == 'test']
    random.Random(70135).shuffle(test)
    assert len(test) == 70 and all(sum(r['species'] == c for r in test) == 2 for c in CLASSES)
    test += [r for r in rows if r['split'] == 'ood_test']
    classifier = Classifier()
    trials_by_id = {}
    probabilities_by_id = {}
    telemetry = Telemetry(OUT / 'gpu-telemetry.csv')
    telemetry.start()

    # Phase 1: the Ammonix path (state + classifier), three trials per photograph.
    engine = Engine(mode='feature')
    try:
        props = engine.props()
        projector = 'in system RAM (CPU)' if engine.mmproj_on_cpu else 'on the GPU'
        same_template, _, _ = engine.template_check()
        for _ in range(3):
            x, _ = engine.feature(ROOT / test[0]['image'])
            classifier.predict(x)
        for i, row in enumerate(test):
            path = ROOT / row['image']
            trials, ps = [], []
            for repeat in range(3):
                engine.erase_cache()
                x, t = engine.feature(path)
                result, p = classifier.predict(x)
                result.update(t)
                result['ammonix_ms'] = t['feature_ms'] + result['classifier_ms']
                trials.append(result)
                ps.append(p)
            trials_by_id[row['id']] = trials
            probabilities_by_id[row['id']] = ps
            print('ammonix', i + 1, row['species'], '->', trials[0]['prediction'], [round(t['ammonix_ms']) for t in trials], 'escalated', trials[0]['escalated'], flush=True)
    finally:
        engine.close()

    # Phase 2: the traditional path (generated one-sentence answer), three trials per photograph,
    # then the System Two fallback for escalated photographs.
    records, probabilities, raw_trials = [], [], []
    engine = Engine(mode='generate')
    try:
        for _ in range(3):
            traditional(engine, ROOT / test[0]['image'])
        for i, row in enumerate(test):
            path = ROOT / row['image']
            trials = trials_by_id[row['id']]
            for repeat in range(3):
                engine.erase_cache()
                answer, gms, completion = traditional(engine, path)
                trials[repeat].update(traditional_ms=gms, traditional_answer=answer, traditional_completion=completion, qwen_ms=gms, qwen_answer=answer)
            print('traditional', i + 1, row['species'], [round(t['traditional_ms']) for t in trials], repr(trials[0]['traditional_answer'][:70]), flush=True)
        for i, row in enumerate(test):
            trials = trials_by_id[row['id']]
            rec = dict(trials[0])
            for key in ['preprocess_ms', 'traditional_ms', 'qwen_ms', 'feature_ms', 'classifier_ms', 'ammonix_ms']:
                rec[key] = float(np.median([t[key] for t in trials]))
            rec.update(id=row['id'], species=row['species'], image=row['image'], source=row['source'], ood=row['split'] == 'ood_test', correct=rec['prediction'] == row['species'])
            rec['final_label'] = rec['prediction']
            rec['final_gate'] = rec['gate']
            rec['fallback_ms'] = 0
            if rec['escalated']:
                engine.erase_cache()
                rec.update(fallback(engine, ROOT / row['image']))
            if not rec['ood']:
                rec['final_correct'] = normalize(rec['final_label']) == row['species']
            records.append(rec)
            probabilities.append(probabilities_by_id[row['id']][0])
            raw_trials.append(dict(id=row['id'], trials=trials))
            (OUT / 'benchmark-progress.json').write_text(json.dumps(records, indent=2))
    finally:
        engine.close()
        telemetry.stop()

    known = records[:70]
    report = metrics(np.array([CLASSES.index(r['species']) for r in test[:70]]), np.stack(probabilities[:70]))
    report.update(median_ammonix_ms=float(np.median([r['ammonix_ms'] for r in known])),
                  median_traditional_ms=float(np.median([r['traditional_ms'] for r in known])),
                  median_qwen_ms=float(np.median([r['traditional_ms'] for r in known])),
                  median_classifier_ms=float(np.median([r['classifier_ms'] for r in known])),
                  median_feature_ms=float(np.median([r['feature_ms'] for r in known])),
                  p95_ammonix_ms=float(np.percentile([r['ammonix_ms'] for r in known], 95)),
                  known_escalations=int(sum(r['escalated'] for r in known)),
                  traditional_incomplete=int(sum(not r['traditional_completion']['complete'] for r in known)),
                  traditional_generated_tokens_median=float(np.median([r['traditional_completion']['generated_tokens'] for r in known])),
                  kangaroo_rejected=bool(records[-1]['escalated']), kangaroo_answer=records[-1]['final_label'], test_images=70)
    result = dict(
        measured=True,
        model='Qwen3.8-27B in ternary weights (Bonsai 2 27B, PTQ1_0, 1.75 bits per weight)',
        checkpoint='prism-ml/Ternary-Bonsai-2-27B-gguf',
        checkpoint_files={MODEL.name: sha256(MODEL), MMPROJ.name: sha256(MMPROJ)},
        execution=f'PrismML llama.cpp fork (CUDA), ternary g128 language weights on the GPU, Q8_0 vision projector {projector}; llama-server, batch 1; image budget {IMAGE_MIN_TOKENS}-{IMAGE_MAX_TOKENS} vision tokens (65,536-147,456 pixels); no thinking; greedy decoding',
        gpu=gpu_name(),
        feature='final layer normalized hidden state at last prompt position; predicts first answer token; 5120 values; llama-server --embeddings --pooling last, un-normalized',
        feature_prompt=PROMPT, traditional_prompt=TRADITIONAL_PROMPT, traditional_max_new_tokens=TRADITIONAL_MAX_NEW_TOKENS,
        traditional_answer_processing='Full decoded answer; no normalization or parsing.',
        repetitions=3,
        deviations=['State and generated answer come from two llama-server sessions of the same weights: all Ammonix trials ran first, all traditional trials second, instead of alternating per photograph.',
                    'Both paths receive the frozen JPEG bytes; the server decodes and resizes the photograph, so preprocessing is inside the measured request.',
                    'The slot cache was erased before every measured request; the host prompt cache was disabled.',
                    'An answer that reached the token limit is kept and counted in traditional_incomplete instead of aborting the run.'],
        server={k: props.get(k) for k in ('build_info', 'model_path', 'modalities', 'default_generation_settings') if k in props},
        gpu_telemetry=telemetry.summary(),
        chat_template_matches_engine_render=bool(same_template),
        manifest_sha256=hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest(),
        classifier_sha256=sha256(OUT / 'ammonix-xgboost.ubj'), novelty_sha256=sha256(OUT / 'novelty.npz'),
        report=report, rows=records)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    (OUT / 'raw-trials.json').write_text(json.dumps(raw_trials, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
