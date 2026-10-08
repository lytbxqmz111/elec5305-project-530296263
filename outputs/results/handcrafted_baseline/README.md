# Handcrafted Baseline: Ordinary Result

## Protocol

- Representation: 76 aggregated MFCC/delta, spectral, temporal, and attack features.
- Training data: all official-train notes from the six selected families.
- StandardScaler fit: train only.
- LinearSVC fit: train only.
- C selection: validation macro-F1 over 0.01, 0.1, and 1.0.
- Ordinary test condition: MIDI 43-69, without pitch balancing.
- Test evaluation: one run after C was selected.

## Results

| Split | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 10,421 | 0.5702 | 0.6288 |
| Ordinary test | 1,244 | 0.6359 | 0.6816 |

Selected C: **0.1**

## Ordinary-test recall by family

| Family | Recall |
|---|---:|
| bass | 0.4355 |
| brass | 0.7955 |
| guitar | 0.5359 |
| keyboard | 0.6743 |
| organ | 0.6734 |
| string | 0.9699 |

## Reproduce

```powershell
python scripts\extract_handcrafted_features.py --config configs\handcrafted_baseline.yaml --cache-tag ordinary
python scripts\train_handcrafted_baseline.py --config configs\handcrafted_baseline.yaml --cache-tag ordinary
```

Feature parts are cached under `data/cache/handcrafted/ordinary/` and are ignored by Git. Re-running extraction resumes completed parts.
