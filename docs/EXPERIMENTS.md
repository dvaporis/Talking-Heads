# Experiments

The thesis evaluation has three experiments, all on GRID speaker `s1`:

1. **Visual phoneme prediction from silent video**: an ablation study, then the selected model in depth with GRID word decoding.
2. **Audio-visual phoneme prediction under corruption**: an audio SNR sweep and a video obstruction sweep, each comparing the fused model with a single-modality version of it.
3. **Comparison with state-of-the-art unimodal models**: the audio-visual model against four pretrained audio-only and video-only models, all fine-tuned on the same corrupted data.

The results are summarised in the root [README](../README.md#results).

## Reproducibility scope

The archived headline results use 1,000 paired clips from GRID speaker `s1`. The visual and audio-visual trainers both perform a seeded random shuffle with `seed=42`, then take 70% for training, 15% for validation, and the remainder for testing. Experiment 3 reuses the audio-visual trainer's split for every model:

| Split | Clips |
| --- | ---: |
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| Total | 1,000 |

This is a within-speaker split. It is not speaker-independent, and repeated GRID sentence structures can occur across the splits. The repository includes `grid_phoneme.split_by_speaker()` for future speaker-held-out experiments, but no held-out-speaker result is claimed here.

The archived runs do not record hardware or wall-clock training time in a machine-readable experiment manifest. Those details should be recorded before treating a new run as a full reproduction.

## Experiment 1: visual phoneme prediction

### Ablation

Run the six visual variants:

```bash
python ablation_experiment.py \
    --video-dir s1_lip_crops \
    --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
    --output-dir ablation_results
```

The tracked presentation figure is [`Report/figures/ablation_val_ter.png`](../Report/figures/ablation_val_ter.png). The numerical table in the README is sourced from the tracked [`Report/ablation_summary.json`](../Report/ablation_summary.json), copied from the historical ablation output. The generated `ablation_results/` directory remains ignored.

The ablation uses single runs, so small differences should not be treated as statistically established. The BiLSTM full-loss baseline has test TER 0.1219 versus 0.1289 for hard targets and 0.1419 for frame-CE only. Those margins are modest and may be affected by seed variance. The clearer result is the sequence-encoder ablation: the BiLSTM baseline at 0.1219 is substantially better than the projection-only variant at 0.2464.

The CTC-only variant is informative: it has relatively poor frame accuracy (0.4743) but a better sequence TER (0.1869), because CTC optimizes collapsed sequence structure rather than framewise label accuracy. This is an objective difference, not evidence that the two metrics measure the same behavior.

### Selected model

The selected BiLSTM full-loss model is trained with `train_lip_viseme.py` and evaluated with `evaluate_lip_lstm.py`; see [QUICKSTART.md](QUICKSTART.md#train-and-evaluate) for the commands.

## Experiment 2: audio-visual fusion under corruption

### Audio SNR sweep

```bash
python run_av_fusion_snr_experiment.py
```

Default SNR values are `-5, 0, 5, 10, 15` dB. Plot the summary:

```bash
python plot_av_fusion_sweep.py \
    --summary-json snr_sweep_results/snr_sweep_summary.json \
    --output-dir snr_sweep_plots
```

### Video obstruction sweep

```bash
python run_av_fusion_video_experiment.py
```

Default obstruction lengths are 10, 15, 20, 25, and 30 frames. Use `--phase prepare` or `--phase train` to resume specific stages and `--skip-existing` to avoid repeating completed runs.

## Experiment 3: comparison with state-of-the-art unimodal models

`run_baseline_comparison.py` fine-tunes external audio-only and video-only phoneme recognisers on the same corrupted GRID data as the audio-visual model. It imports the sample pairing, split, vocabulary, loss and TER decoding from `train_av_fusion.py`.

| Baseline | `--models` name | Input | Pretrained on |
| --- | --- | --- | --- |
| Allosaurus, English model `eng2102` | `allosaurus` | audio | 14 datasets in 11 languages, including Switchboard and LibriSpeech |
| icefall TDNN-LSTM-CTC | `icefall` | audio | TIMIT |
| LipNet (VIPL PyTorch re-implementation, unseen-speaker weights) | `lipnet` | video | GRID speakers other than `s1` |
| Auto-AVSR visual model (CTC path only) | `autoavsr` | video | LRS2, LRS3, VoxCeleb2, AVSpeech (3,291 h) |

### Third-party setup

The baseline code and weights are not vendored. They live in the git-ignored `external/` folder (except Allosaurus, which is a pip package) and keep their own licences:

| Baseline | Source | Expected location |
| --- | --- | --- |
| Allosaurus (GPL-3.0) | `pip install allosaurus`, then `python -m allosaurus.bin.download_model -m eng2102` | `site-packages/allosaurus/pretrained/eng2102` |
| icefall (Apache-2.0) | [luomingshuang/icefall_asr_timit_tdnn_lstm_ctc](https://huggingface.co/luomingshuang/icefall_asr_timit_tdnn_lstm_ctc) | `external/icefall_timit/exp/pretrained_average_16_25.pt`, `external/icefall_timit/data/lang_phone/tokens.txt` |
| LipNet | `git clone https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch external/LipNet-PyTorch` | unseen-speaker checkpoint in `external/LipNet-PyTorch/pretrain/` |
| Auto-AVSR (Apache-2.0) | `git clone https://github.com/mpc001/auto_avsr external/auto_avsr`, plus the `vsr_trlrs2lrs3vox2avsp_base.pth` checkpoint from its model zoo | `external/auto_avsr/weights/vsr_trlrs2lrs3vox2avsp_base.pth` |

Check the setup before training. This loads every selected checkpoint, decodes one clean clip zero-shot, and runs one forward pass on a corrupted clip:

```bash
python run_baseline_comparison.py --check-setup --models icefall allosaurus lipnet autoavsr
```

### Run

1. Build the corrupted set: 0 dB SNR on every clip and exactly 25 obstructed video frames per clip. The audio dropout (probability 0.3, 15 frames) and the noise types and video corruption modes are the `add_av_noise.py` defaults.

   ```bash
   python add_av_noise.py \
       --output-dir baseline_noise_s1 \
       --audio-snr-db-range 0 0 \
       --video-obstruct-frames 25 \
       --video-obstruct-jitter 0

   python precompute_mel_spectograms.py \
       --noise-manifest baseline_noise_s1/noise_manifest.csv
   ```

2. Train the proposed audio-visual model on that set:

   ```bash
   python train_av_fusion.py \
       --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
       --noise-manifest baseline_noise_s1/noise_manifest_with_mel.csv \
       --output-dir baseline_results/av_fusion
   ```

3. Fine-tune the baselines. `run_baseline_comparison.py` reads `baseline_noise_s1/noise_manifest_with_mel.csv` and writes to `baseline_results/` by default. The reported Auto-AVSR row uses 2 trainable Conformer blocks and Auto-AVSR's native augmentation:

   ```bash
   python run_baseline_comparison.py --models icefall allosaurus lipnet
   python run_baseline_comparison.py --models autoavsr --autoavsr-train-blocks 2 --autoavsr-augment
   ```

   The script's default `--models` list also includes `vallr`, which is not part of the reported comparison, so pass `--models` explicitly.

Each model writes `best_model.pt`, `history.json`, `test_metrics.json`, `test_predictions.csv` and a confusion matrix to `baseline_results/<model>/`. A combined `baseline_results/summary.csv` is merged across runs, so models can be trained separately.

### Fine-tuning protocol

- Each baseline keeps its pretrained body and gets a new output layer over the 32 GRID phonemes plus blank. Rows for phonemes that exist in the model's original inventory (TIMIT or IPA) are initialised from the old weights. LipNet (characters) and Auto-AVSR (subword units) reuse only the blank row.
- The loss is frame-level cross-entropy with the viseme-aware soft targets, the insertion penalty, and 0.3 × CTC, without the audio-visual model's auxiliary CTC heads. The 25 fps frame targets are resampled onto each model's output rate.
- AdamW with learning rate 1e-4 for pretrained layers and 1e-3 for the new head, batch size 4. The schedule matches the audio-visual model: 5-epoch linear warm-up, halving after 6 epochs without improvement, early stopping after 12, at most 100 epochs. The best-validation-TER checkpoint is evaluated on the test set.
- Audio baselines get time and frequency masking; video baselines get random horizontal flips.
- icefall's BatchNorm statistics are re-estimated once on the GRID training clips and then kept frozen during fine-tuning (`--icefall-bn recalibrate`, the default).
- Auto-AVSR is the only baseline whose setup was tuned. Six configurations (0, 1 or 2 trainable Conformer blocks, with or without native augmentation) were compared on validation TER only. The other baselines are fine-tuned end to end.

### Statistical comparison

The ΔTER confidence intervals in the README come from a paired bootstrap over the 150 test clips (10,000 resamples). The better/worse/tie counts compare per-clip phoneme edit distances. Both use the per-clip `test_predictions.csv` files. The repository does not include a script for this step. The baseline files carry a `stem` column; `train_av_fusion.py`'s file does not, but it lists the test clips in the same order.

### Limitations

- The audio-visual model receives the per-frame audio and video reliability signals as input; the baselines do not.
- One speaker, one corruption level, and phoneme-level evaluation only, so no WER comparison with the baselines' published numbers is possible.
- Both video baselines receive this repository's lip crops, resized to each model's input size, rather than their native crop style. This particularly affects Auto-AVSR.

## Speaker-independent extension

A proper generalization experiment should select complete speakers for train, validation, and test without crossing speaker boundaries. For example, reserve one GRID speaker entirely for testing, train on several other speakers, and report the same word, sentence, phoneme TER, and frame metrics. Do not compare that result directly with the current s1 within-speaker result without labeling the split change.
