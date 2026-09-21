"""One sequential cloud API pass; existing local benchmark remains frozen.

Run in a shell with OPENAI_API_KEY set. Credentials never enter result files.
"""
from pathlib import Path
import argparse
import base64
import collections
import datetime
import hashlib
import http.client
import json
import os
import re
import statistics
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'gpt6-benchmark'
MODEL = 'gpt-6-astra'
PROMPT = 'What animal species is shown, and is it a mammal, bird, reptile, or arthropod? Answer in one sentence.'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def summary(rows):
    known = [r for r in rows if not r['ood']]
    complete = [r for r in known if r.get('complete')]
    values = [r['latency_ms'] for r in complete]
    usage = collections.Counter()
    for row in rows:
        u = row.get('usage') or {}
        usage['input_tokens'] += u.get('input_tokens', 0)
        usage['cached_input_tokens'] += (u.get('input_tokens_details') or {}).get('cached_tokens', 0)
        usage['cache_write_tokens'] += (u.get('input_tokens_details') or {}).get('cache_write_tokens', 0)
        usage['output_tokens'] += u.get('output_tokens', 0)
        usage['reasoning_tokens'] += (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    # Token-based estimate at published standard prices; provider billing is authoritative.
    uncached = max(0, usage['input_tokens'] - usage['cached_input_tokens'] - usage['cache_write_tokens'])
    cost = (uncached * 10 + usage['cached_input_tokens'] + usage['cache_write_tokens'] * 12.5 + usage['output_tokens'] * 50) / 1e6
    return {
        'known_attempted': len(known), 'known_completed': len(complete),
        'known_failed_or_incomplete': len(known) - len(complete),
        'ood_attempted': sum(r['ood'] for r in rows),
        'median_complete_answer_ms': statistics.median(values) if values else None,
        'p95_complete_answer_ms': percentile(values, .95),
        'min_complete_answer_ms': min(values) if values else None,
        'max_complete_answer_ms': max(values) if values else None,
        'mean_complete_answer_ms': statistics.mean(values) if values else None,
        'usage_all_requests': dict(usage),
        'estimated_standard_token_cost_usd': cost,
        'accuracy': None,
        'accuracy_note': 'Full answers preserved; scoring is a separate post-hoc review.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    local_path = ROOT / 'results.json'
    local = json.loads(local_path.read_text('utf-8'))
    manifest_path = ROOT / 'manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == local['manifest_sha256']
    assert local['traditional_prompt'] == PROMPT
    records = local['rows']
    known = [r for r in records if not r['ood']]
    assert len(known) == 70
    counts = collections.Counter(r['species'] for r in known)
    assert len(counts) == 35 and set(counts.values()) == {2}
    images = []
    for row in records:
        path = ROOT / row['image']
        data = path.read_bytes()
        assert data[:2] == b'\xff\xd8'
        images.append({k: row[k] for k in ('id', 'species', 'image', 'ood')} | {
            'jpeg_sha256': hashlib.sha256(data).hexdigest(), 'jpeg_bytes': len(data)
        })
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'model': MODEL, 'reasoning_effort': 'max',
            'held_out_images': len(known), 'ood_images': len(records)-len(known),
            'image_bytes': sum(r['jpeg_bytes'] for r in images), 'prompt': PROMPT}))
        return
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        raise SystemExit('OPENAI_API_KEY is missing in this terminal. No API requests made.')
    DEST.mkdir(exist_ok=True)
    output_path = DEST / 'results.json'
    if output_path.exists():
        raise SystemExit('A benchmark result already exists; refusing duplicate paid requests.')
    report = {
        'schema_version': 1, 'state': 'running', 'started_at': utc(),
        'model_requested': MODEL, 'reasoning_effort': 'max', 'service_tier_requested': 'default',
        'endpoint': 'https://api.openai.com/v1/responses', 'store': False,
        'prompt': PROMPT, 'max_output_tokens': 8192, 'repetitions': 1,
        'concurrency': 1, 'automatic_retries': 0, 'timeout_seconds': 300,
        'input_image': 'Exact frozen local JPEG bytes, base64 data URL, detail=high; no filename or label sent.',
        'timing': 'Client wall clock from reading the image through receiving and decoding the complete HTTP response; includes encoding, network, provider queuing, reasoning, and answer generation. Persistent HTTPS connection, first connection setup included. No artificial delays.',
        'completion_rule': 'HTTP 200, response status completed, nonempty full output_text, no refusal.',
        'manifest_sha256': local['manifest_sha256'],
        'local_results_sha256': hashlib.sha256(local_path.read_bytes()).hexdigest(),
        'local_reference': {'model': local['model'], 'repetitions': local['repetitions'], 'report': local['report']},
        'comparison_limits': [
            'Cloud GPT-6 at maximum reasoning versus local model with thinking disabled: not the same model, hardware, or deployment.',
            'API receives frozen source JPEGs at high detail; local model resizes to at most 147456 pixels.',
            'One observation per image here versus medians of three local observations; descriptive comparison, not a controlled architectural speedup claim.',
            'Unknown provider pretraining overlap; images are held out from Ammonix training.',
            'Failed and incomplete responses are retained and excluded from complete-answer latency summaries, with counts reported.',
        ], 'images': images, 'rows': [],
    }
    write_json(output_path, report)
    connection = http.client.HTTPSConnection('api.openai.com', timeout=300)
    try:
        for index, row in enumerate(images):
            if (ROOT / 'stop-gpt6-benchmark').exists():
                report['state'] = 'stopped_by_request'
                break
            record = dict(row, index=index+1, started_at=utc())
            started = time.perf_counter()
            try:
                jpeg = (ROOT / row['image']).read_bytes()
                assert hashlib.sha256(jpeg).hexdigest() == row['jpeg_sha256']
                payload = {
                    'model': MODEL, 'reasoning': {'effort': 'max'},
                    'service_tier': 'default', 'store': False, 'max_output_tokens': 8192,
                    'input': [{'role': 'user', 'content': [
                        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii'), 'detail': 'high'},
                        {'type': 'input_text', 'text': PROMPT},
                    ]}],
                }
                body = json.dumps(payload).encode('utf-8')
                connection.request('POST', '/v1/responses', body=body, headers={
                    'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                })
                response = connection.getresponse()
                raw = response.read()
                data = json.loads(raw)
                answer = ''.join(c.get('text', '') for o in data.get('output', [])
                    if o.get('type') == 'message' for c in o.get('content', [])
                    if c.get('type') == 'output_text')
                refusals = [c.get('refusal', '') for o in data.get('output', [])
                    if o.get('type') == 'message' for c in o.get('content', [])
                    if c.get('type') == 'refusal']
                latency = (time.perf_counter() - started) * 1000
                record.update(http_status=response.status, status=data.get('status'),
                    latency_ms=latency, answer=answer, refusals=refusals,
                    complete=response.status == 200 and data.get('status') == 'completed' and bool(answer.strip()) and not refusals,
                    model_returned=data.get('model'), reasoning_returned=data.get('reasoning'),
                    service_tier_returned=data.get('service_tier'), usage=data.get('usage'),
                    response_id=data.get('id'), request_id=response.getheader('x-request-id'),
                    incomplete_details=data.get('incomplete_details'))
                if data.get('error'):
                    # Provider errors can echo credentials; redact before writing anything.
                    err = json.dumps(data['error']).replace(key, '[REDACTED]')
                    record['error'] = json.loads(re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', err))
            except Exception as exc:
                # Exception strings may include request details, so persist only their type.
                record.update(complete=False, status='client_error', latency_ms=(time.perf_counter()-started)*1000,
                    error={'type': type(exc).__name__}, answer='')
            record['finished_at'] = utc()
            report['rows'].append(record)
            report['summary'] = summary(report['rows'])
            report['updated_at'] = utc()
            if record.get('http_status') != 200 or record.get('status') == 'client_error':
                report['state'] = 'blocked_by_api_error'
            write_json(output_path, report)
            print(json.dumps({'image': index+1, 'total': len(images), 'complete': record['complete'],
                'seconds': round(record['latency_ms']/1000, 3), 'http_status': record.get('http_status'),
                'state': report['state']}, ensure_ascii=True), flush=True)
            if report['state'] != 'running':
                break
        else:
            report['state'] = 'finished'
    finally:
        connection.close()
        report['finished_at'] = utc()
        report['summary'] = summary(report['rows'])
        write_json(output_path, report)
    print(json.dumps({'state': report['state'], 'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
