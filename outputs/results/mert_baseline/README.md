# Frozen MERT + LinearSVC: Ordinary Result

## Protocol

- MERT representation: final hidden layer, time-mean pooled, 768 dimensions.
- Training data: all official-train notes from the six selected families.
- StandardScaler and LinearSVC fit: train only.
- C selection: validation macro-F1 over 0.01, 0.1, and 1.0.
- Ordinary test: MIDI 43-69, without pitch balancing.
- Test evaluated once after validation selection.

## Results

| Condition | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 10,421 | 0.6318 | 0.6612 |
| Ordinary test | 1,244 | 0.6712 | 0.6982 |

Selected C: **1**

## Ordinary-test recall by family

| Family | Recall |
|---|---:|
| bass | 0.5774 |
| brass | 0.7727 |
| guitar | 0.5598 |
| keyboard | 0.6743 |
| organ | 0.7387 |
| string | 0.8571 |

## Reproduce

```powershell
python scripts\train_mert_baseline.py --config configs\mert_baseline.yaml
```
