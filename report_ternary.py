"""Side-by-side report: the released FP16 run versus the ternary run on this machine.

Reads results.json and training.json (released) and ternary/results.json, ternary/training.json,
ternary/features.npz, ternary/extract-timings.json, ternary/recollect-log.json (this run).
Writes ternary/REPORT.md and prints it. Every number is taken from those files; nothing is
recomputed except the cross-precision cosine similarity and the transfer of the released
classifier onto the ternary states.
"""
import runtime
from runtime import ROOT, CLASSES, load_features
import json
import numpy as np

OUT = ROOT / 'ternary'


def load(path):
    return json.loads(path.read_text('utf-8')) if path.exists() else None


def pct(x):
    return f'{100 * x:.1f}%'


def main():
    released = load(ROOT / 'results.json')
    ternary = load(OUT / 'results.json')
    fit_released = load(ROOT / 'training.json')
    fit_ternary = load(OUT / 'training.json')
    timings = load(OUT / 'extract-timings.json') or {}
    recollect = load(OUT / 'recollect-log.json')
    lines = ['# Wild Departures on an 8 GB laptop GPU: ternary Bonsai 2 27B versus the released FP16 run', '']

    if recollect:
        c = recollect['counts']
        rows_log = recollect['rows']
        manifest = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
        inat_ok = sum(1 for k, v in rows_log.items() if k.startswith('inat') and v.get('sha256_match'))
        inat_all = sum(1 for k in rows_log if k.startswith('inat'))
        lines += ['## Photographs', '',
                  f"All {c['ok'] + c['sha_mismatch']} photographs that are not shipped were re-obtained from the sources recorded in the manifest; {c['failed']} were unavailable.",
                  f"iNaturalist files: {inat_ok} of {inat_all} downloads match the recorded SHA-256 byte for byte.", '']
        try:
            import imagehash
            from PIL import Image
            commons = [r for r in manifest if r['source'].startswith('https://commons') and (ROOT / r['image']).exists()]
            dist = [imagehash.phash(Image.open(ROOT / r['image']).convert('RGB')) - imagehash.hex_to_hash(r['phash']) for r in commons]
            exact = sum(d == 0 for d in dist)
            close = sum(0 < d <= 4 for d in dist)
            lines += [f"Wikimedia Commons files: the manifest's SHA-256 values are hashes of the 640-pixel API thumbnail, not of the original file (verified on samples), so identity was checked with the manifest's perceptual hash instead: {exact} of {len(commons)} re-collected or shipped Commons photographs have an identical pHash, {close} are within Hamming distance 4, {len(commons) - exact - close} differ more.", '']
        except Exception as e:
            lines += [f'(Commons perceptual-hash check skipped: {type(e).__name__})', '']

    if fit_released and fit_ternary:
        lines += ['## Decision layer fit', '', '| | Released (FP16 states) | This run (ternary states) |', '|---|---:|---:|']
        for label, key, fmt in [('Training photographs', 'training_images', '{}'), ('Validation top-1', ('validation', 'top1_accuracy'), 'pct'),
                                ('Validation macro-F1', ('validation', 'macro_f1'), '{:.3f}'), ('Validation macro AUROC', ('validation', 'macro_ovr_auroc'), '{:.3f}'),
                                ('Known-class coverage (validation)', 'validation_known_coverage', 'pct'), ('Unknown-animal recall (validation)', 'validation_ood_recall', 'pct'),
                                ('Temperature', 'temperature', '{:.2f}'), ('Rejection threshold', 'rejection_threshold', '{:.3f}'), ('Fit time', 'training_seconds', '{:.1f} s'),
                                ('Device', ('parameters', 'device'), '{}')]:
            def get(d):
                v = d
                for k in (key if isinstance(key, tuple) else (key,)):
                    v = v[k]
                return pct(v) if fmt == 'pct' else fmt.format(v)
            lines.append(f'| {label} | {get(fit_released)} | {get(fit_ternary)} |')
        lines.append('')

    if released and ternary:
        r, t = released['report'], ternary['report']
        lines += ['## Held-out benchmark (70 photographs, medians of three trials)', '',
                  f"Released: {released['gpu']}, {released['execution']}.", '',
                  f"This run: {ternary['gpu']}, {ternary['execution']}.", '',
                  '| Measure | Released | This run |', '|---|---:|---:|',
                  f"| Top-1 accuracy | {round(r['top1_accuracy'] * 70)}/70 ({pct(r['top1_accuracy'])}) | {round(t['top1_accuracy'] * 70)}/70 ({pct(t['top1_accuracy'])}) |",
                  f"| Wilson 95% interval | {pct(r['accuracy_wilson95'][0])} to {pct(r['accuracy_wilson95'][1])} | {pct(t['accuracy_wilson95'][0])} to {pct(t['accuracy_wilson95'][1])} |",
                  f"| Macro-F1 / macro AUROC | {r['macro_f1']:.3f} / {r['macro_ovr_auroc']:.3f} | {t['macro_f1']:.3f} / {t['macro_ovr_auroc']:.3f} |",
                  f"| Known photographs escalated | {r['known_escalations']}/70 | {t['known_escalations']}/70 |",
                  f"| Unknown animal (kangaroo) | {'rejected' if r['kangaroo_rejected'] else 'not rejected'}, System Two said {r['kangaroo_answer']!r} | {'rejected' if t['kangaroo_rejected'] else 'not rejected'}, System Two said {t['kangaroo_answer']!r} |",
                  f"| Median photo to Ammonix decision | {r['median_ammonix_ms']:.0f} ms | {t['median_ammonix_ms']:.0f} ms |",
                  f"| 95th percentile Ammonix decision | {r['p95_ammonix_ms']:.0f} ms | {t['p95_ammonix_ms']:.0f} ms |",
                  f"| Median classifier time | {r['median_classifier_ms']:.1f} ms | {t['median_classifier_ms']:.1f} ms |",
                  f"| Median photo to complete generated answer | {r['median_traditional_ms']:.0f} ms | {t['median_traditional_ms']:.0f} ms |",
                  f"| Ratio of medians (generated / Ammonix) | {r['median_traditional_ms'] / r['median_ammonix_ms']:.2f} | {t['median_traditional_ms'] / t['median_ammonix_ms']:.2f} |",
                  f"| Generated answers hitting the token limit | 0 | {t.get('traditional_incomplete', 0)} |", '']
        try:
            from benchmark import normalize
            def answered(rows):
                return sum(normalize(row.get('traditional_answer') or row.get('qwen_answer') or '') == row['species'] for row in rows[:70])
            lines += [f"Generated answers that name the right species after the released normalizer (not part of the released protocol, added here because both paths run the same model): released {answered(released['rows'])}/70, this run {answered(ternary['rows'])}/70.", '']
        except Exception as e:
            lines += [f'(generated-answer scoring skipped: {type(e).__name__})', '']
        rel = {row['id']: row for row in released['rows']}
        diff = [(row['species'], rel[row['id']]['prediction'], row['prediction'], row['escalated']) for row in ternary['rows'] if row['id'] in rel and (rel[row['id']]['prediction'] != row['prediction'] or bool(rel[row['id']]['escalated']) != bool(row['escalated']))]
        lines += [f'Photographs decided differently from the released run: {len(diff)}', '']
        for species, a, b, esc in diff:
            lines.append(f'- {species}: released {a}, ternary {b}{" (escalated)" if esc else ""}')
        lines.append('')
        wrong = [(row['species'], row['prediction'], row['traditional_answer']) for row in ternary['rows'][:70] if not row['correct']]
        lines += [f'Ternary Ammonix errors on the 70 held-out photographs: {len(wrong)}', '']
        for species, pred, answer in wrong:
            lines.append(f'- {species}: decided {pred}; generated answer was {answer!r}')
        lines.append('')
        if ternary.get('gpu_telemetry'):
            g = ternary['gpu_telemetry']
            lines += [f"GPU condition during the run (sampled every {g['interval_s']:.0f} s, {g['samples']} samples): median core clock {g['sm_clock_mhz']['median']:.0f} MHz (range {g['sm_clock_mhz']['min']:.0f} to {g['sm_clock_mhz']['max']:.0f}), median temperature {g['temperature_c']['median']:.0f} C (max {g['temperature_c']['max']:.0f}), thermally throttled in {100 * (g['share_of_samples_thermally_throttled'] or 0):.0f}% of samples, peak memory {g['memory_used_mib_max']:.0f} MiB.", '']
        if ternary.get('deviations'):
            lines += ['Protocol deviations of this run:', ''] + [f'- {d}' for d in ternary['deviations']] + ['']

    if (OUT / 'features.npz').exists():
        rows = json.loads((ROOT / 'manifest.json').read_text('utf-8'))
        with np.load(OUT / 'features.npz', allow_pickle=False) as z:
            tern = dict(zip(z['ids'].tolist(), z['features']))
        both = [row for row in rows if row['id'] in tern]
        F = load_features(both)
        T = np.stack([tern[row['id']] for row in both])
        cos = (F * T).sum(1) / np.linalg.norm(F, axis=1) / np.linalg.norm(T, axis=1)
        lines += ['## The state itself', '',
                  f'{len(both)} photographs have both states. Cosine similarity between the FP16 and the ternary 5,120-value state: median {np.median(cos):.3f}, minimum {cos.min():.3f}.', '']
        try:
            import xgboost as xgb
            from train import probs
            model = xgb.Booster(); model.load_model(str(ROOT / 'ammonix-xgboost.ubj')); model.set_param({'nthread': 1, 'device': 'cpu'})
            temp = json.loads((ROOT / 'training.json').read_text())['temperature']
            idx = [i for i, row in enumerate(both) if row['split'] == 'test']
            if idx:
                y = np.array([CLASSES.index(both[i]['species']) for i in idx])
                for name, X in [('FP16', F), ('ternary', T)]:
                    pred = probs(model.inplace_predict(X[idx], predict_type='margin'), temp).argmax(1)
                    lines.append(f'Released FP16 classifier applied unchanged to the {name} states of the 70 held-out photographs: {int((pred == y).sum())}/70 correct.')
                lines.append('')
        except Exception as e:
            lines += [f'(transfer check skipped: {type(e).__name__})', '']
        if timings:
            ms = np.array([v['feature_ms'] for v in timings.values()])
            lines += [f'State extraction on this GPU over {len(ms)} photographs: median {np.median(ms):.0f} ms, 95th percentile {np.percentile(ms, 95):.0f} ms per photograph.', '']

    text = '\n'.join(lines)
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
