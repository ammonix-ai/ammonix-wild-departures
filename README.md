# Wild Departures: the Ammonix decision layer on photographs

Seventy animal photographs arrive at an airport and must be sent to one of three gates. Each
photograph is handled twice by the **same frozen vision-language model**:

- **Ammonix** reads the model's internal state at the position just before it would write its
  first answer token, and a small classifier on those 5,120 values decides. Nothing is generated.
- **Traditional LLM** lets the model write a one-sentence answer.

| Measured on 70 held-out photographs | Result |
|---|---:|
| Top-1 accuracy of the Ammonix decision | 67/70 (95.7%; Wilson 95% interval 88.1% to 98.5%) |
| Macro-F1 / one-vs-rest macro AUROC | 0.954 / 1.000 |
| Median time, photograph to Ammonix decision | 372 ms |
| Median time, photograph to complete generated answer | 1,509 ms (ratio of medians 4.06) |
| Known photographs escalated to System Two | 1/70 |
| One photograph of an unknown animal (kangaroo) | rejected, then identified by System Two |
| Training the decision layer (1,009 photographs, 35 classes) | 12.1 s on the GPU |

Both paths ran sequentially on one NVIDIA RTX PRO 6000 Blackwell (96 GB), three trials per
photograph, medians reported. The decision time includes loading and preprocessing the
photograph, the forward pass, the classifier and the unknown-animal check.

A comparison of the same task against GPT-6 is shown in the video on the Ammonix website.
Its recorded results are not part of this repository; `benchmark_gpt6.py` lets you measure
it yourself with your own API key.

## Three ways to use this repository

### 1. Watch the measured replay: no GPU, no model, no packages

```text
python -m http.server 8766
```

Open http://127.0.0.1:8766. The scene replays the recorded decisions and timings of all 71
photographs from `results.json`.

### 2. Check and refit the decision layer: no GPU, no photographs

`features/features.npz` holds the frozen 5,120-value state of all 1,282 photographs
(train, validation, test and the unknown-animal sets). Needs `numpy` and `xgboost`.

```text
python evaluate_features.py                       re-derives 67/70 and agrees with results.json on 71/71
python train.py --device cpu                      refits the classifier into retrained/ (about 2.5 minutes on a CPU)
python evaluate_features.py --classifier retrained
```

`train.py --device cuda` is the setting the released classifier was fitted with. A CPU fit is
not bit-identical; when tried it gave the same 67/70 on the held-out photographs.

### 3. Fresh inference: needs the checkpoint and a large GPU

Download `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` at revision
`8aa5f05d26b7205477066e1449e0af13f762a299` into a folder and name it in `settings.json` (copy
`settings.example.json`) or in `AMMONIX_MODEL_DIR`. The loader decodes the 4-bit checkpoint to
FP16, which occupies about 55 GB of GPU memory, so a GPU with roughly 60 GB free is needed.
Python packages: `torch`, `transformers`, `safetensors`, `accelerate`, `pillow`, `numpy`, `xgboost`.

```text
python serve.py --live        replay plus live classification of the 71 shipped photographs
python benchmark.py           re-measures all 71 photographs, three trials each
```

`extract.py` recomputes the states for every photograph. Only the 71 replay photographs are
shipped; `manifest.json` lists source, author, licence and SHA-256 of all 1,282, and
`collect.py` / `collect_inat.py` fetched them from Wikimedia Commons and iNaturalist.
`curate.py` froze the splits from the candidate lists under `data/`.

## What is in here

| File | Content |
|---|---|
| `index.html`, `scene.js`, `results.json`, `data/*.jpg` | The replay and its 71 photographs |
| `features/features.npz` | 1,282 x 5,120 frozen states, float32 |
| `ammonix-xgboost.ubj`, `novelty.npz`, `training.json` | Released decision layer, unknown-animal prototypes, fit report |
| `raw-trials.json`, `verification.json`, `split-counts.json` | All timing trials, dataset checks, per-class split sizes |
| `manifest.json`, `all-image-credits.json`, `credits.html` | Provenance and credits of every photograph |
| `engine.py`, `runtime.py`, `train.py`, `benchmark.py`, `serve.py`, `extract.py`, `evaluate_features.py` | Model loading and state extraction, settings, fitting, measurement, serving |
| `collect*.py`, `curate.py`, `audit*.py` | How the dataset was gathered, split and visually audited |
| `benchmark_gpt6.py` | Optional comparison against a cloud model with your own key |

## How the state is defined

The feature is `model.model(...).last_hidden_state[0, -1]`: the normalized final-layer hidden
state at the last prompt position, conditioned on the photograph and the fixed prompt
"Identify the animal species. Return only the species name." It is not a token embedding, not
a pooled vision feature, and not a state obtained after generating anything. Changing the
prompt, the checkpoint or the image preprocessing (PIL backend, patch size 16, merge size 2,
pixel budget 65,536 to 147,456) changes the state, and the classifier must then be refitted.

An unknown animal is rejected when the largest temperature-scaled class probability times the
largest cosine similarity to the training-class prototypes falls below a threshold chosen on
validation data. A rejected photograph goes to a short generated answer.

## Limits

- 70 held-out photographs, two per class: a demonstration, not a generalization claim. The one
  kangaroo illustrates the rejection path; it is not an out-of-distribution benchmark.
- "Held out" refers to this classifier's training and calibration, with photographers kept in
  one split per class and perceptual-hash de-duplication. Whether the language model saw these
  public photographs during its own pretraining is unknown.
- An AUROC of 1.000 is a ranking measure, not 100% accuracy. The class score is not a
  calibrated probability of being correct.
- The generated answers were not scored for accuracy; only their completion time is reported.
- Labels and image suitability had one visual audit, not an independent expert annotation.
- The model runs in a reconstructed FP16 form, not an optimized low-latency deployment.

## Licences

Research use of the code and data files is free under the **Ammonix Research License** (see
[LICENSE](LICENSE)): research, education, and evaluation are permitted; **any commercial use
requires a separate commercial license** — contact licensing@ammonix.ai. This software is
deliberately not distributed under an OSI-approved open-source license. **The photographs are
not covered by it.** Each photograph under `data/` remains under its own Creative Commons
or public-domain terms; see [PHOTO-LICENSES.md](PHOTO-LICENSES.md).
