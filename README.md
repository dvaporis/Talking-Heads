# GRID Visual Phoneme Prediction

[![CI](https://github.com/dvaporis/Talking-Heads/actions/workflows/ci.yml/badge.svg)](https://github.com/dvaporis/Talking-Heads/actions/workflows/ci.yml)

This repository develops a **GRID-based visual phoneme prediction pipeline**: lip-crop video is converted into frame-aligned phoneme predictions, then decoded into words with GRID grammar. An audio-visual extension fuses the lip video with the audio track, is tested under controlled audio and video corruption, and is compared against fine-tuned state-of-the-art audio-only and video-only recognisers.

Older RAVDESS, reconstruction, contrastive-learning, alignment, and diffusion experiments are retained for provenance, but are not required for the active GRID pipeline.

## Demo

![Subtitle-generation proof of concept](Report/figures/swwp2n_lipcrop_pred_subtitled.gif)

The model predicts **set white with p 2 now** and renders the words at frame-derived timings. [Open the original MP4](Report/figures/swwp2n_lipcrop_pred_subtitled.mp4) · [Predicted subtitles](Report/figures/swwp2n_lipcrop_pred.srt) · [Ground truth](Report/figures/swwp2n_lipcrop_gt.srt)

This is a qualitative single-clip proof of concept, not an additional aggregate metric.

## Experiments

The thesis evaluation ([`Report/Chapters/Evaluation.tex`](Report/Chapters/Evaluation.tex)) has three experiments, all on GRID speaker `s1`:

| # | Experiment | What it covers | Entry points |
| --- | --- | --- | --- |
| 1 | Visual phoneme prediction from silent video | (1) Six-variant ablation study. (2) The selected model in depth, with GRID word decoding. | `ablation_experiment.py`, `train_lip_viseme.py` |
| 2 | Audio-visual phoneme prediction under corruption | (1) Audio SNR sweep, audio-visual vs audio-only. (2) Video obstruction sweep, audio-visual vs video-only. | `run_av_fusion_snr_experiment.py`, `run_av_fusion_video_experiment.py` |
| 3 | Comparison with state-of-the-art unimodal models | The audio-visual model against two audio-only and two video-only pretrained models, all fine-tuned on the same corrupted data. | `train_av_fusion.py`, `run_baseline_comparison.py` |

Commands for each experiment are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Results

All results use GRID's fixed vocabulary and sentence grammar, so they are not unrestricted-vocabulary lip reading. The full figure collection is in [`Report/figures`](Report/figures).

### Experiment 1: visual phoneme prediction

| Metric | Result | Scope |
| --- | ---: | --- |
| Word-correct prediction rate | **87.1%** | Main constrained test evaluation. |
| Exact sentence match | **38.0%** | Main constrained test evaluation. |
| Main-run phoneme TER | **12.41%** | Separate constrained visual evaluation run. |
| Best ablation frame accuracy | **84.74%** | BiLSTM full-loss baseline, before collapsing repeated frame labels into a sequence. |

![Test phoneme confusion matrix](Report/figures/constrained_confusion_matrix.png)

#### Ablation summary

| Variant | Test TER | Test frame accuracy | Validation-accuracy AUC* |
| --- | ---: | ---: | ---: |
| BiLSTM + full loss | 0.1219 | 0.8474 | 0.7840 |
| Transformer + full loss | 0.1566 | 0.8394 | 0.7176 |
| No sequence encoder | 0.2464 | 0.8122 | 0.6006 |
| BiLSTM + CTC only | 0.1869 | 0.4743 | 0.6005 |
| BiLSTM + frame-CE only | 0.1419 | 0.8401 | 0.7402 |
| BiLSTM + hard targets | 0.1289 | 0.8456 | 0.7696 |

\* AUC is the trapezoidal area under the per-epoch validation-accuracy curve, normalized by the number of recorded epochs.

The 12.41% main-run TER and 0.1219 ablation-baseline TER are **different runs**, not contradictory values. The ablation baseline was retrained as V1 with its own output directory and conditions. Ablation results are single runs, so the 0.007-point difference between soft full loss and hard targets is not evidence of a statistically reliable advantage. The clearer result is the sequence-encoder ablation: BiLSTM TER 0.1219 versus projection-only TER 0.2464. CTC-only also illustrates an objective trade-off: sequence TER is 0.1869 while frame accuracy is 0.4743 because CTC optimizes collapsed sequence structure rather than framewise labels.

### Experiment 2: audio-visual fusion under corruption

Each sweep trains the audio-visual model and a single-modality version of the same model on corrupted audio and video. The SNR sweep sets the audio to −5, 0, 5, 10 and 15 dB SNR, with about 30 of each clip's 75 video frames obstructed. The obstruction sweep obstructs 10, 15, 20, 25 or 30 frames, with the audio fixed at 15 dB SNR. In both sweeps the audio-visual model reaches a lower best validation TER than the single-modality model at every setting. Its audio gate rises as the audio gets cleaner or more video frames are obstructed.

![Test TER versus audio SNR](Report/figures/test_ter_vs_snr_db.png)

![Test TER versus video obstruction](Report/figures/test_ter_vs_video_obstruct_frames.png)

### Experiment 3: comparison with state-of-the-art unimodal models

Four publicly available pretrained models were fine-tuned on the same corrupted data, split, 32-phoneme vocabulary, loss and TER metric as the proposed audio-visual model. Every clip has additive white or babble noise at 0 dB SNR. 31.1% of clips also have a 0.6 s (15-frame) audio dropout. Every clip has one contiguous 25-frame video span that is zeroed, frozen or blurred.

| Model | Input | Parameters (trainable) | Epochs run (best) | Val TER | Test TER | ΔTER vs AV [95% CI] | AV better / worse / tie (clips) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Proposed audio-visual** | audio + video | **3.6M** | 92 (80) | **0.144** | **0.171** | — | — |
| [Allosaurus](https://github.com/xinjli/allosaurus) (eng2102) | audio | 10.8M | 63 (51) | 0.184 | 0.186 | +0.014 [−0.013, +0.043] | 58 / 61 / 31 |
| [icefall](https://github.com/k2-fsa/icefall) TDNN-LSTM-CTC (TIMIT) | audio | 10.9M | 78 (66) | 0.235 | 0.251 | +0.079 [+0.054, +0.105] | 96 / 32 / 22 |
| [LipNet](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch) | video | 6.6M | 83 (71) | 0.298 | 0.335 | +0.164 [+0.135, +0.194] | 115 / 16 / 19 |
| [Auto-AVSR](https://github.com/mpc001/auto_avsr) (VSR, CTC path) | video | 182.1M (28.4M) | 58 (46) | 0.311 | 0.363 | +0.192 [+0.165, +0.220] | 126 / 13 / 11 |

ΔTER is each model's test TER minus the audio-visual model's, with a 95% paired-bootstrap confidence interval (10,000 resamples of the 150 test clips). The last column counts the test clips on which the audio-visual model made fewer, more or equal phoneme errors.

The audio-visual model has the best validation and test TER while being the smallest model. Its lead over Allosaurus is not significant: the interval includes zero, and the per-clip record is nearly even. It is clearly better than icefall, LipNet and Auto-AVSR. The audio-only models beat both video-only models, consistent with visual speech recognition being the harder task. When reading the table, keep in mind:

- The audio-visual model receives the per-frame audio and video reliability signals as input; the baselines do not.
- Auto-AVSR is the only baseline whose fine-tuning setup was tuned. Six configurations were compared on validation TER; the reported one trains the top 2 Conformer blocks with Auto-AVSR's native augmentation. The other baselines were fine-tuned end to end.
- LipNet was pretrained on other GRID speakers (not `s1`); Auto-AVSR was pretrained on LRS2, LRS3, VoxCeleb2 and AVSpeech with a different crop style.
- This is a single speaker and a single corruption level, evaluated at the phoneme level only (no WER).

## Evaluation scope

The archived results use 1,000 paired clips from GRID speaker `s1`. The visual and audio-visual trainers both use a seeded random shuffle with `seed=42`, then a 70/15/15 within-speaker split. Experiment 3 reuses the audio-visual trainer's split for every model:

| Split | Clips |
| --- | ---: |
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| Total | 1,000 |

This split is not speaker-independent and can contain repeated GRID sentence structures across splits. GRID has 34 speakers; no held-out-speaker result is claimed yet. The package includes `grid_phoneme.split_by_speaker()` to support a future leakage-resistant experiment. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full scope and limitations.

Hardware and wall-clock training time were not recorded in a machine-readable manifest for the archived runs, so no hardware or timing claim is made here.

## How it works

```text
GRID videos in s1/
    -> crop_lips.py
GRID lip crops in s1_lip_crops/
    -> extract_phonemes.py
phonemes_s1_aligned/phoneme_predictions.csv
    -> train_lip_viseme.py or train_av_fusion.py
checkpoints, metrics, and plots
```

`extract_phonemes.py` decodes six-character GRID stems, creates canonical eSpeak phonemes, runs wav2vec2 forced alignment, and converts timings to 25 FPS frame labels. The CSV is the shared contract for visual training and audio-visual fusion.

The active entry points are `crop_lips.py`, `save_all_lip_videos.py`, `extract_phonemes.py`, `train_lip_viseme.py`, `train_av_fusion.py`, `add_av_noise.py`, `precompute_mel_spectograms.py`, `ablation_experiment.py`, `evaluate_lip_lstm.py`, the two AV sweep runners, `run_baseline_comparison.py`, and the plotting/augmentation helpers.

`run_baseline_comparison.py` imports the sample pairing, split, vocabulary, loss and TER decoding from `train_av_fusion.py`, so the Experiment 3 baselines share the audio-visual model's data, targets and metric.

## Installation and usage

The active pipeline was tested on Python 3.14.0; the package metadata supports Python 3.11 through 3.14. Python 3.11 is shown below for broadest wheel availability. Install pinned active dependencies with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
```

The active pipeline also needs FFmpeg, eSpeak-NG, a compatible PyTorch build, and GRID videos under `s1/`. Experiment 3 additionally needs the `allosaurus` package and third-party code and weights under the git-ignored `external/` folder; these keep their own licences and are not part of the core requirements. For the complete visual and AV command sequence, see [`docs/QUICKSTART.md`](docs/QUICKSTART.md). For the per-experiment commands, baseline setup, and split details, see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). For debugging, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

The dependency files are intentionally split:

- [`requirements-core.txt`](requirements-core.txt): pinned active GRID dependencies.
- [`requirements-dev.txt`](requirements-dev.txt): core dependencies plus pytest.
- [`requirements-legacy.txt`](requirements-legacy.txt): unpinned historical package names only; it is not a lockfile. Install it in a separate environment and never combine it with core.

The active wav2vec2 extractor was checked under Transformers 5.2.0 and Hugging Face Hub 1.5.0: the three APIs it uses import successfully and `extract_phonemes.py --help` runs. A full extraction still requires model download, eSpeak-NG, and usable media.

## Repository hygiene

- [`grid_phoneme/`](grid_phoneme) is an importable package for shared GRID paths, stem normalization, and speaker grouping.
- [`tests/`](tests) contains focused domain tests for GRID decoding, viseme soft targets, Levenshtein/TER, and shared contracts.
- [`pyproject.toml`](pyproject.toml) defines package metadata and the supported Python range.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) installs the active requirements, compiles the repository, and runs tests on Python 3.11 and 3.14.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) documents local validation.
- [`LICENSE`](LICENSE) contains the MIT license.

The root scripts remain at the repository root for compatibility with legacy imports and subprocess entry points. Shared active utilities have been moved into `grid_phoneme`; future refactoring can migrate individual scripts behind compatibility wrappers.

## Legacy and retained work

These scripts are retained for completeness but are not part of the final GRID visual phoneme workflow:

- RAVDESS acquisition and preprocessing: `download_kaggle_dataset.py`, `download_only_videos.py`, `download_unzip_videos.py`, `extract_ravdess_phonemes.py`, `preprocess_ravdess_audio.py`, `preprocess_mels_full.py`.
- Audio representation experiments: `train_ravdess_mel_cpc.py`, `train_mfcc_contrastive.py`, `train_time_domain_contrastive.py`, `test_mel_reconstruction.py`, `test_mfcc_reconstruction.py`.
- Alignment and diffusion: `train_audio_video_alignment.py`, `evaluate_alignment.py`, `train_diffusion_mel.py`, `train_video_conditioned_mel_diffusion.py`, `evaluate_video_conditioned_mel_diffusion.py`.
- Earlier visual variants and decoders: `train_lip_lstm.py`, `train_lip_lstm_ctc.py`, `train_lip_transformer.py`, `extract_durations.py`, `wav_eval.py`, `wav_eval2.py`, `train_video_frame_order.py`, `train_video_contrastive.py`, and `visualize_video_augmentations.py`.

The old output folders remain as provenance. Their dependency list is deliberately separate and explicitly not historically reproducible because the original legacy environment was not preserved.
