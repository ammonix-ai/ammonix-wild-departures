# Wild Departures on an 8 GB laptop GPU: ternary Bonsai 2 27B versus the released FP16 run

## Photographs

All 364 photographs that are not shipped were re-obtained from the sources recorded in the manifest; 0 were unavailable.
iNaturalist files: 837 of 837 downloads match the recorded SHA-256 byte for byte.

Wikimedia Commons files: the manifest's SHA-256 values are hashes of the 640-pixel API thumbnail, not of the original file (verified on samples), so identity was checked with the manifest's perceptual hash instead: 356 of 396 re-collected or shipped Commons photographs have an identical pHash, 40 are within Hamming distance 4, 0 differ more.

## Decision layer fit

| | Released (FP16 states) | This run (ternary states) |
|---|---:|---:|
| Training photographs | 1009 | 1009 |
| Validation top-1 | 95.4% | 97.1% |
| Validation macro-F1 | 0.953 | 0.972 |
| Validation macro AUROC | 0.995 | 0.999 |
| Known-class coverage (validation) | 98.9% | 98.3% |
| Unknown-animal recall (validation) | 100.0% | 100.0% |
| Temperature | 0.75 | 0.75 |
| Rejection threshold | 0.243 | 0.159 |
| Fit time | 12.1 s | 85.2 s |
| Device | cuda | cuda |

## Held-out benchmark (70 photographs, medians of three trials)

Released: NVIDIA RTX PRO 6000 Blackwell 96 GB, NF4 checkpoint reconstructed to FP16; torch SDPA; eager linear attention; batch 1; max image pixels 147456; no thinking.

This run: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB, PrismML llama.cpp fork (CUDA), ternary g128 language weights on the GPU, Q8_0 vision projector on the GPU; llama-server, batch 1; image budget 64-144 vision tokens (65,536-147,456 pixels); no thinking; greedy decoding.

| Measure | Released | This run |
|---|---:|---:|
| Top-1 accuracy | 67/70 (95.7%) | 68/70 (97.1%) |
| Wilson 95% interval | 88.1% to 98.5% | 90.2% to 99.2% |
| Macro-F1 / macro AUROC | 0.954 / 1.000 | 0.970 / 1.000 |
| Known photographs escalated | 1/70 | 0/70 |
| Unknown animal (kangaroo) | rejected, System Two said 'kangaroo' | rejected, System Two said 'kangaroo' |
| Median photo to Ammonix decision | 372 ms | 1290 ms |
| 95th percentile Ammonix decision | 426 ms | 2029 ms |
| Median classifier time | 4.1 ms | 3.3 ms |
| Median photo to complete generated answer | 1509 ms | 2838 ms |
| Ratio of medians (generated / Ammonix) | 4.06 | 2.20 |
| Generated answers hitting the token limit | 0 | 0 |

Generated answers that name the right species after the released normalizer (not part of the released protocol, added here because both paths run the same model): released 64/70, this run 64/70.

Photographs decided differently from the released run: 4

- dragonfly: released butterfly, ternary dragonfly
- dog: released dog, ternary bear
- penguin: released duck, ternary penguin
- kangaroo: released pig, ternary zebra (escalated)

Ternary Ammonix errors on the 70 held-out photographs: 2

- dog: decided bear; generated answer was 'The animal shown is a husky, which is a mammal.'
- beetle: decided grasshopper; generated answer was 'The animal shown is a stick insect (Phasmatodea), which is an arthropod.'

GPU condition during the run (sampled every 2 s, 467 samples): median core clock 600 MHz (range 210 to 1635), median temperature 79 C (max 81), thermally throttled in 97% of samples, peak memory 7925 MiB.

Protocol deviations of this run:

- State and generated answer come from two llama-server sessions of the same weights: all Ammonix trials ran first, all traditional trials second, instead of alternating per photograph.
- Both paths receive the frozen JPEG bytes; the server decodes and resizes the photograph, so preprocessing is inside the measured request.
- The slot cache was erased before every measured request; the host prompt cache was disabled.
- An answer that reached the token limit is kept and counted in traditional_incomplete instead of aborting the run.

## The state itself

1282 photographs have both states. Cosine similarity between the FP16 and the ternary 5,120-value state: median 0.651, minimum 0.344.

Released FP16 classifier applied unchanged to the FP16 states of the 70 held-out photographs: 67/70 correct.
Released FP16 classifier applied unchanged to the ternary states of the 70 held-out photographs: 41/70 correct.

State extraction on this GPU over 1282 photographs: median 1642 ms, 95th percentile 2042 ms per photograph.
