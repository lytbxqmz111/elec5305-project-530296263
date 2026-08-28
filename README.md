# Investigating the Effect of Acoustic Feature Representation on Musical Instrument Recognition

## Project Proposal

[View the Project Proposal (PDF)](./Project_Proposal.pdf)

This project investigates how different acoustic feature representations affect the performance of automatic musical instrument recognition.

Musical instruments produce distinctive sounds through differences in their spectral, temporal, and harmonic characteristics. However, different instruments can share similar acoustic properties, making instrument recognition a challenging music information retrieval task. This project focuses on understanding which acoustic features are most informative for distinguishing musical instruments and whether combining different feature types improves classification performance.

## Research Question

**Which acoustic features are most informative for distinguishing different musical instruments, and how does the choice of feature representation affect classification performance across different instrument groups?**

The project will investigate the following questions:

1. Which feature group provides the strongest classification performance when used independently?
2. Do combinations of different feature groups provide complementary information and improve performance?
3. Does the relative importance of acoustic features vary across different instrument families?

## Datasets

The primary dataset will be **NSynth**, which contains 305,979 four-second monophonic musical notes sampled at 16 kHz from 1,006 instruments. Each note includes annotations such as instrument family.

Additional datasets may be used for external validation:

- **IRMAS** – a dataset for predominant instrument recognition in musical audio.
- **Medley-solos-DB** – a cross-collection dataset containing solo recordings from multiple instrument categories.

These datasets will be evaluated separately rather than directly combined because they have different recording conditions and label taxonomies.

## Acoustic Features

Three main feature groups will be investigated.

### 1. MFCC-based Features

- MFCCs
- Delta MFCCs
- Delta-Delta MFCCs

MFCCs provide a compact representation of the spectral envelope and are widely used in audio classification.

### 2. Spectral Features

- Spectral centroid
- Spectral bandwidth
- Spectral roll-off
- Spectral flatness
- Spectral flux

These features describe the distribution and variation of spectral energy and are related to the timbral characteristics of instruments.

### 3. Temporal Features

- RMS energy
- Zero-crossing rate
- Temporal envelope statistics
- Attack characteristics

These features capture temporal properties such as onset and decay behaviour.

## Methodology

The project will use a controlled machine-learning framework to compare different acoustic representations.

Audio recordings will first be preprocessed and converted into fixed-length segments. Feature representations will then be extracted using short-time Fourier transform (STFT)-based and time-domain analysis.

A Support Vector Machine (SVM) will be used as the main classifier. The same classifier configuration, data split, and evaluation procedure will be maintained across experiments so that differences in performance can primarily be attributed to the feature representation.

## Experiments

### Experiment 1 – Baseline

An MFCC-based SVM classifier will establish the baseline performance.

### Experiment 2 – Feature Comparison

MFCC, spectral, and temporal representations will be evaluated separately using the same experimental setup.

### Experiment 3 – Feature Combination

The following combinations will be compared:

- MFCC + Spectral
- MFCC + Temporal
- MFCC + Spectral + Temporal

This experiment will investigate whether combining complementary acoustic information improves classification performance.

### Experiment 4 – Instrument-Family Analysis

The performance of different feature representations will be compared across instrument families to investigate whether different groups depend more strongly on spectral or temporal characteristics.

### Experiment 5 – Acoustic Error Analysis

Confusion matrices will be examined to identify commonly confused instruments. Representative examples will then be analysed using spectra, spectrograms, harmonic structure, and temporal envelopes to investigate possible acoustic explanations for classification errors.

## Evaluation

The main evaluation metrics will be:

- **Accuracy** – overall classification performance.
- **Macro F1-score** – balanced evaluation across instrument classes.
- **Confusion matrix** – analysis of class-specific errors.

The results will be used to determine which acoustic feature representations are most informative and whether feature combinations provide measurable improvements.

## Tools

The project will be implemented primarily in Python using:

- [Librosa](https://librosa.org/) – audio processing and feature extraction
- [NumPy](https://numpy.org/) – numerical computation
- [SciPy](https://scipy.org/) – signal processing
- [Scikit-learn](https://scikit-learn.org/) – SVM classification and evaluation
- [Matplotlib](https://matplotlib.org/) – visualisation

## References
For the complete reference list, see the project proposal.
