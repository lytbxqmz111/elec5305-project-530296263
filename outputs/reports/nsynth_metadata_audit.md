# NSynth Metadata Audit and Controlled-Subset Decision

## Predeclared selection rule

- Require at least 4 distinct source instruments in both validation and test.
- Consider 5-8 eligible families.
- Search contiguous intervals of 24-60 semitones, in steps of 3, divisible into three equal bins.
- Require every split x family x register cell to contain at least 20 notes, 2 instruments, and 65% of pitches in the bin.
- Require at least 4 shared pitches per split and register; a supported pitch has at least 5 notes from 2 instruments in every selected family.
- Rank feasible results by: more families, wider interval, stronger shared-pitch support, larger weakest cell, greater coverage, then total usable notes.

## Selected design

- Families: bass, brass, guitar, keyboard, organ, string
- Common MIDI interval: 43-69 (27 semitones)
- Lower register: 43-51
- Middle register: 52-60
- Upper register: 61-69
- Shared supported pitches by split/register: {'train': {'lower': [43, 44, 45, 46, 47, 48, 49, 50, 51], 'middle': [52, 53, 54, 55, 56, 57, 58, 59, 60], 'upper': [61, 62, 63, 64, 65, 66, 67, 68, 69]}, 'valid': {'lower': [43, 44, 45, 46, 47, 48, 49, 50, 51], 'middle': [52, 53, 54, 55, 56, 57, 58, 59, 60], 'upper': [61, 62, 63, 64, 65, 66, 67, 68, 69]}, 'test': {'lower': [43, 45, 49, 51], 'middle': [55, 58, 59, 60], 'upper': [61, 62, 64, 65]}}
- Ineligible families: flute, mallet, synth_lead, vocal
- Eligible but not selected by the joint-overlap rule: reed

## Primary register-balanced manifest

- Rows: 6,078
- Rows by split: {'train': 3600, 'valid': 1884, 'test': 594}
- Register-bin and velocity counts are identical across selected families within each split.
- Maximum source-instrument share in a register x velocity cell: 50.0%
- Random seed: 5305
- Official train/validation/test labels are preserved, and train instruments remain disjoint from evaluation instruments.

## Strict exact-pitch sensitivity manifest

- Rows: 8,364
- Rows by split: {'train': 6480, 'valid': 1764, 'test': 120}
- Uses only pitches jointly supported by every selected family.
- Exact pitch x velocity counts are identical across families, with the instrument cap checked per exact pitch.
- The strict test subset is too small for the primary per-register analysis and is retained only as a sensitivity check.

## Core comparison protocol

- Use the same six-family class universe in every condition.
- Train one classifier per representation on official-train data and tune it with validation data only.
- Compare the same fitted classifier on the unbalanced and primary balanced test conditions over the same MIDI interval.
- Treat the full-range test and strict exact-pitch manifest as secondary analyses.

## Interpretation boundary

The selection uses metadata only. Test audio, acoustic features, embeddings, predictions, and test scores were not inspected. The saved item identifiers freeze the evaluation subset before classifier development.
