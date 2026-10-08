# Coarse Pitch-Probe Results

## Protocol

- Target: lower, middle, or upper register within the frozen MIDI 43-69 interval.
- Representation: the same cached handcrafted features or frozen MERT embeddings used for family recognition.
- StandardScaler and LinearSVC fit: official train split only.
- C selection: official validation macro-F1 only.
- Evaluation: ordinary test and the same 594-note pitch-balanced test used for family robustness.

## Results beside family robustness

| Representation | Balanced family macro-F1 | Ordinary pitch macro-F1 | Balanced pitch macro-F1 |
|---|---:|---:|---:|
| handcrafted | 0.6707 | 0.7845 | 0.7650 |
| mert | 0.7055 | 0.9856 | 0.9783 |

High pitch predictability is not inherently undesirable: a useful music representation may encode both pitch and timbre. The central question is whether family recognition remains robust while pitch is still decodable.

Per-bin recall, prediction tables, confusion matrices, and per-family pitch-probe results are saved beside this report.

## Reproduce

```powershell
python scripts\train_pitch_probe.py --config configs\pitch_probe.yaml
```
