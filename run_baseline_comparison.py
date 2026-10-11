#!/usr/bin/env python3
"""Fine-tune external audio-only / video-only phoneme recognisers on the same
corrupted GRID data as train_av_fusion.py, for a like-for-like comparison
against the audio-visual fusion model.

Baselines
---------
    icefall     audio  TDNN-LSTM-CTC trained on TIMIT (k2-fsa/icefall, Apache-2.0)
    allosaurus  audio  BiLSTM phone recogniser, English model eng2102 (xinjli/allosaurus, GPL-3.0)
    lipnet      video  LipNet, VIPL PyTorch re-implementation, unseen-speaker weights
    vallr       video  VALLR V1 video-to-phoneme stage (MarshallT-99/VALLR, CC BY-NC 4.0)
    autoavsr    video  Auto-AVSR VSR, ResNet-18 + Conformer, CTC path (mpc001/auto_avsr, Apache-2.0)

Every baseline keeps its pretrained body and gets a fresh output layer over
the same 32-phoneme vocabulary (+ blank) that train_av_fusion.py builds. Rows
for phonemes that have a counterpart in the model's original inventory
(TIMIT / IPA / ARPAbet) are initialised from the old weights; LipNet
(characters) and Auto-AVSR (subword units) reuse only the blank row.

Third-party code and weights live in external/ (git-ignored, see README of
each project for its licence):

    external/icefall_timit/     exp/pretrained_average_16_25.pt, data/lang_phone/tokens.txt
    external/LipNet-PyTorch/    git clone, weights in pretrain/
    external/VALLR/             git clone, weights in weights/vallr.pth
    external/auto_avsr/         git clone, weights in weights/vsr_trlrs2lrs3vox2avsp_base.pth
    allosaurus                 pip package, eng2102 model in site-packages/allosaurus/pretrained/

Fine-tuning uses train_av_fusion.py's split, vocabulary, loss and metric:
frame-CE with viseme soft targets + 0.3 x CTC (no auxiliary heads), with
the 25 fps frame targets resampled onto each model's output grid; VALLR's
outputs are not time-aligned, so it is trained with CTC only. Results go to
<output-dir>/<model>/ (same files as train_av_fusion.py) and a combined
<output-dir>/summary.csv.

Usage
-----
    python run_baseline_comparison.py --check-setup
    python run_baseline_comparison.py --dry-run
    python run_baseline_comparison.py                      # all baselines
    python run_baseline_comparison.py --models icefall lipnet
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_av_fusion import (
    AVFusionLoss,
    AVLipReadingModel,
    BLANK_TOKEN,
    apply_specaugment,
    apply_vocab_remap,
    build_samples,
    build_vocab,
    greedy_decode,
    levenshtein,
    phoneme_report,
    split_samples,
)

logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("external")
ICEFALL_DIR = EXTERNAL_DIR / "icefall_timit"
LIPNET_DIR = EXTERNAL_DIR / "LipNet-PyTorch"
VALLR_DIR = EXTERNAL_DIR / "VALLR"

ICEFALL_CKPT = ICEFALL_DIR / "exp" / "pretrained_average_16_25.pt"
ICEFALL_TOKENS = ICEFALL_DIR / "data" / "lang_phone" / "tokens.txt"
LIPNET_CKPT = LIPNET_DIR / "pretrain" / (
    "LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt"
)
VALLR_CKPT = VALLR_DIR / "weights" / "vallr.pth"
ALLOSAURUS_MODEL = "eng2102"
AUTOAVSR_DIR = EXTERNAL_DIR / "auto_avsr"
AUTOAVSR_CKPT = AUTOAVSR_DIR / "weights" / "vsr_trlrs2lrs3vox2avsp_base.pth"
AUTOAVSR_UNITS = AUTOAVSR_DIR / "spm" / "unigram" / "unigram5000_units.txt"

BASELINE_NAMES = ("icefall", "allosaurus", "lipnet", "vallr", "autoavsr")

# ---------------------------------------------------------------------------
# Phoneme maps: our (eSpeak IPA) vocabulary -> each model's own inventory.
# Used only to initialise the new output layer; unmapped rows start random.
# Diphthongs are split into single symbols by extract_phonemes.py, so "a"
# and "e" are mostly the onsets of /aɪ aʊ/ and /eɪ/.
# ---------------------------------------------------------------------------

IPA_TO_TIMIT: Dict[str, str] = {   # TIMIT 39-phone set (ao->aa, ax->ah, zh->sh folded)
    "a": "ay", "b": "b", "d": "d", "e": "ey", "f": "f", "i": "iy", "j": "y",
    "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "s": "s", "t": "t",
    "u": "uw", "v": "v", "w": "w", "z": "z", "ð": "dh", "ɐ": "ah", "ɑ": "aa",
    "ɒ": "aa", "ɔ": "aa", "ə": "ah", "ɛ": "eh", "g": "g", "ɡ": "g", "ɪ": "ih",
    "ɹ": "r", "ʃ": "sh", "ʊ": "uh", "ʒ": "sh", "θ": "th",
}

IPA_TO_ALLOSAURUS: Dict[str, str] = {   # eng2102 is IPA already; only ɐ and ɒ are missing
    **{p: p for p in ["a", "b", "d", "e", "f", "i", "j", "k", "l", "m", "n", "p", "s", "t",
                      "u", "v", "w", "z", "ð", "ɑ", "ɔ", "ə", "ɛ", "ɪ", "ɹ", "ʃ", "ʊ", "ʒ", "θ"]},
    "ɐ": "ʌ", "ɒ": "ɑ", "g": "ɡ", "ɡ": "ɡ",
}

IPA_TO_ARPABET: Dict[str, str] = {
    "a": "AY", "b": "B", "d": "D", "e": "EY", "f": "F", "i": "IY", "j": "Y",
    "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "s": "S", "t": "T",
    "u": "UW", "v": "V", "w": "W", "z": "Z", "ð": "DH", "ɐ": "AH", "ɑ": "AA",
    "ɒ": "AA", "ɔ": "AO", "ə": "AH", "ɛ": "EH", "g": "G", "ɡ": "G", "ɪ": "IH",
    "ɹ": "R", "ʃ": "SH", "ʊ": "UH", "ʒ": "ZH", "θ": "TH",
}

# VALLR config.get_vocab(), index order (config.py needs configargparse, so copied here)
VALLR_VOCAB: List[str] = [
    "<pad>", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R",
    "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]

# LipNet dataset.py letters; CTC targets start at 1, so index 0 is the blank
LIPNET_LETTERS: List[str] = [" "] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _load_module(name: str, path: Path):
    """Import a third-party file under a unique module name (LipNet's is just
    `model.py`, which would clash with anything else called `model`)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pad_batch(items: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a list of (T, ...) tensors along time -> (B, T_max, ...), lengths."""
    lengths = torch.tensor([x.shape[0] for x in items], dtype=torch.long)
    out = items[0].new_zeros((len(items), int(lengths.max()), *items[0].shape[1:]))
    for i, x in enumerate(items):
        out[i, :x.shape[0]] = x
    return out, lengths


def _read_frames(path: Path, rgb: bool) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if rgb else bgr)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames


# ---------------------------------------------------------------------------
# Baseline interface
# ---------------------------------------------------------------------------

class Baseline(nn.Module):
    """Common wrapper. Subclasses set the class attributes, build `self.net`
    and implement load_input() (CPU, called from the Dataset) and forward().

    forward(x, lengths) -> (logits (B, T_out, V), out_lengths (B,))
    """

    name: str = ""
    modality: str = ""                 # "audio" | "video"
    step_sec: Optional[float] = None   # seconds per output step; None = not time-aligned
    head_attr: str = ""                # attribute path of the final nn.Linear
    source_blank: int = 0

    def __init__(self):
        super().__init__()
        self.source_symbols: List[str] = []   # index -> symbol of the *original* head
        self.source_map: Dict[str, str] = {}  # our phoneme -> source symbol
        self.output_symbols: List[str] = []   # index -> symbol of the *current* head
        self.frozen: List[nn.Module] = []     # kept in eval mode with requires_grad=False

    def freeze_modules(self, modules: Sequence[nn.Module]) -> None:
        """Freeze a *prefix* of the network. Because nothing before it needs
        gradients, autograd stores no activations for it, so frozen layers
        cost no training memory. They are also kept in eval mode (no dropout,
        fixed BatchNorm statistics)."""
        self.frozen = list(modules)
        for module in self.frozen:
            module.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True) -> "Baseline":
        super().train(mode)
        for module in getattr(self, "frozen", []):
            module.eval()
        return self

    def format_source(self, ids: Sequence[int]) -> str:
        """Render a decode of the *original* head (zero-shot check)."""
        return " ".join(self.source_symbols[i] for i in ids)

    def load_input(self, path: Path) -> torch.Tensor:
        raise NotImplementedError

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time augmentation of one load_input() result."""
        if self.modality == "audio":
            # Same masks as the AV model's mel SpecAugment, scaled to these
            # 10 ms feature frames (8 x 40 ms -> 32 frames) and feature sizes
            return apply_specaugment(x, time_mask_param=32, freq_mask_param=max(1, int(12 * x.shape[1] / 80)))
        if random.random() < 0.5:
            return x.flip(-1)   # horizontal flip (LipNet's own training augmentation)
        return x

    def input_path(self, sample) -> Path:
        return sample.audio_path if self.modality == "audio" else sample.video_path

    def _get_head(self) -> nn.Linear:
        obj = self
        for part in self.head_attr.split("."):
            obj = getattr(obj, part)
        return obj

    def _set_head(self, head: nn.Linear) -> None:
        *parents, last = self.head_attr.split(".")
        obj = self
        for part in parents:
            obj = getattr(obj, part)
        setattr(obj, last, head)

    def replace_head(self, vocab: Sequence[str]) -> List[str]:
        """Swap the output layer for blank + `vocab`, copying rows whose
        phoneme maps onto the source inventory. Returns unmapped phonemes."""
        old = self._get_head()
        new = nn.Linear(old.in_features, len(vocab) + 1).to(old.weight.device)
        src_index = {sym: i for i, sym in enumerate(self.source_symbols)}

        rows: List[Optional[int]] = [self.source_blank]
        rows += [src_index.get(self.source_map.get(tok, "")) for tok in vocab]
        unmapped = [tok for tok, r in zip(vocab, rows[1:]) if r is None]

        with torch.no_grad():
            for new_i, old_i in enumerate(rows):
                if old_i is not None:
                    new.weight[new_i] = old.weight[old_i]
                    new.bias[new_i] = old.bias[old_i]

        self._set_head(new)
        self.output_symbols = [BLANK_TOKEN] + list(vocab)
        return unmapped


# ---------------------------------------------------------------------------
# icefall TIMIT TDNN-LSTM-CTC (audio)
# ---------------------------------------------------------------------------

class TdnnLstm(nn.Module):
    """Copied from k2-fsa/icefall egs/timit/ASR/tdnn_lstm_ctc/model.py
    (Copyright 2021 Xiaomi Corp., Apache License 2.0) so that k2/lhotse are
    not needed. Changes: forward() returns logits instead of log_softmax,
    so the head can be trained with the shared loss, and takes optional
    `lengths` to zero padded frames before every Conv1d, so a clip's
    output does not depend on how much padding its batch has.

    Input (N, num_features, T) fbank -> (N, ceil(T / 3), num_classes)."""

    def __init__(self, num_features: int, num_classes: int, subsampling_factor: int = 3) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(num_features, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(512, 512, kernel_size=3, stride=self.subsampling_factor, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
        )
        self.lstms = nn.ModuleList([nn.LSTM(input_size=512, hidden_size=512, num_layers=1) for _ in range(4)])
        self.lstm_bnorms = nn.ModuleList([nn.BatchNorm1d(num_features=512, affine=False) for _ in range(5)])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is None:
            x = self.tdnn(x)
        else:
            for layer in self.tdnn:
                if isinstance(layer, nn.Conv1d):
                    # padded frames would otherwise be BN(ReLU(bias)) != 0 and leak
                    # into the last valid frame through the kernel
                    x = x * (torch.arange(x.shape[-1], device=x.device) < lengths[:, None]).unsqueeze(1)
                x = layer(x)
                if isinstance(layer, nn.Conv1d) and layer.stride[0] > 1:
                    lengths = (lengths - 1) // layer.stride[0] + 1
        x = x.permute(2, 0, 1)  # (T, N, C)
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(2, 0, 1)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(1, 0)  # (N, T, C)
        return self.linear(x)


class IcefallBaseline(Baseline):
    name = "icefall"
    modality = "audio"
    step_sec = 0.03   # 10 ms fbank hop x subsampling 3
    head_attr = "net.linear"
    sample_rate = 16000

    def __init__(self):
        super().__init__()
        tokens = [line.split() for line in ICEFALL_TOKENS.read_text(encoding="utf-8").splitlines() if line.strip()]
        id2sym = {int(i): sym for sym, i in tokens}
        ckpt = torch.load(ICEFALL_CKPT, map_location="cpu", weights_only=False)["model"]
        num_classes = ckpt["linear.weight"].shape[0]   # 41: <eps>=blank ... dx (#0 is not an output)
        self.net = TdnnLstm(num_features=80, num_classes=num_classes)
        self.net.load_state_dict(ckpt, strict=True)
        self.source_symbols = [id2sym[i] for i in range(num_classes)]
        self.source_map = IPA_TO_TIMIT

    def load_input(self, path: Path) -> torch.Tensor:
        # lhotse Fbank defaults used by the icefall recipe: 80 bins, 25/10 ms,
        # povey window, snip_edges=False, waveform in [-1, 1], no CMVN.
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        feats = kaldi.fbank(
            torch.from_numpy(wav).unsqueeze(0), num_mel_bins=80, sample_frequency=self.sample_rate,
            frame_length=25.0, frame_shift=10.0, dither=0.0, energy_floor=1e-10,
            low_freq=20.0, high_freq=-400.0, window_type="povey", snip_edges=False,
        )
        return feats   # (T, 80)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x.transpose(1, 2), lengths)
        out_lengths = (lengths - 1) // self.net.subsampling_factor + 1
        return logits, out_lengths.clamp(max=logits.shape[1])

    # The BatchNorm layers are affine=False: their running statistics are the
    # model's only normalisation. They stay frozen during fine-tuning, so
    # zero-padded frames in a batch never enter them; recalibrate_batchnorm()
    # optionally re-estimates them on GRID first (AdaBN).

    def train(self, mode: bool = True) -> "IcefallBaseline":
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        return self

    @torch.no_grad()
    def recalibrate_batchnorm(self, inputs: Sequence[torch.Tensor], device: torch.device) -> None:
        """Replace the TIMIT running mean/var with a cumulative average over
        `inputs` (unpadded (T, 80) fbank clips from the *training* split),
        fed one clip at a time so padding cannot bias the statistics."""
        bns = [m for m in self.modules() if isinstance(m, nn.BatchNorm1d)]
        was_training = self.training
        self.eval()
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None   # cumulative moving average
            bn.train()
        for x in inputs:
            self(x.unsqueeze(0).to(device), torch.tensor([x.shape[0]], device=device))
        for bn in bns:
            bn.momentum = 0.1
        self.train(was_training)


# ---------------------------------------------------------------------------
# Allosaurus eng2102 (audio)
# ---------------------------------------------------------------------------

class AllosaurusBaseline(Baseline):
    name = "allosaurus"
    modality = "audio"
    step_sec = 0.01   # eng2102: 10 ms MFCC hop, feature_window=1 (no stacking)
    head_attr = "net.phone_layer"

    def __init__(self):
        super().__init__()
        from allosaurus.am.factory import read_am
        from allosaurus.model import get_model_path
        from allosaurus.pm.factory import read_pm

        model_path = get_model_path(ALLOSAURUS_MODEL)
        self.net = read_am(model_path, Namespace(device_id=-1))
        self.pm = read_pm(model_path, Namespace())
        phones = [line.split()[0] for line in (model_path / "phone.txt").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        self.source_symbols = ["<blk>"] + phones   # phone_layer index 0 is the CTC blank
        self.source_map = IPA_TO_ALLOSAURUS

    def load_input(self, path: Path) -> torch.Tensor:
        from allosaurus.audio import read_audio
        # 8 kHz, 40-dim hires MFCC, per-utterance ("speaker") CMVN, as in recognize()
        return torch.from_numpy(self.pm.compute(read_audio(str(path))))   # (T, 40)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Same as AllosaurusTorchModel.forward, but with enforce_sorted=False so
        # batches do not have to be sorted by length.
        packed = nn.utils.rnn.pack_padded_sequence(x.transpose(0, 1), lengths.cpu(), enforce_sorted=False)
        self.net.blstm_layer.flatten_parameters()
        hidden, _ = self.net.blstm_layer(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, total_length=x.shape[1])
        logits = self.net.phone_layer(hidden).transpose(0, 1)
        return logits, lengths


# ---------------------------------------------------------------------------
# LipNet, VIPL PyTorch version (video)
# ---------------------------------------------------------------------------

class LipNetBaseline(Baseline):
    name = "lipnet"
    modality = "video"
    step_sec = 0.04   # one output per 25 fps frame
    head_attr = "net.FC"

    def __init__(self):
        super().__init__()
        lipnet_module = _load_module("lipnet_model", LIPNET_DIR / "model.py")
        self.net = lipnet_module.LipNet()
        self.net.load_state_dict(torch.load(LIPNET_CKPT, map_location="cpu", weights_only=False), strict=True)
        self.source_symbols = ["<blk>"] + LIPNET_LETTERS
        self.source_map = {}   # characters, not phonemes: only the blank row is reused

    def load_input(self, path: Path) -> torch.Tensor:
        # dataset.py: BGR, cv2.resize to 128x64 (W x H) with LANCZOS4, / 255
        frames = [cv2.resize(f, (128, 64), interpolation=cv2.INTER_LANCZOS4) for f in _read_frames(path, rgb=False)]
        video = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        return video.permute(0, 3, 1, 2)   # (T, 3, 64, 128)

    def format_source(self, ids: Sequence[int]) -> str:
        return "".join(self.source_symbols[i] for i in ids)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net(x.permute(0, 2, 1, 3, 4)), lengths   # LipNet takes (B, C, T, H, W)


# ---------------------------------------------------------------------------
# VALLR V1 video-to-phoneme stage (video)
# ---------------------------------------------------------------------------

class VALLRBaseline(Baseline):
    """Rebuilds only the parts of VALLR V1 (Models/VALLR.py) that run in its
    forward(): VideoMAE-Base -> 4x Conv1d downsampling over the 1568 tokens
    -> adapter -> CTC head. The checkpoint also stores a full Wav2Vec2ForCTC
    body (~94M params) that forward() never uses; it is not instantiated.

    The released downsampling ends in AvgPool1d(5, stride=8), which leaves
    only 8 output steps per clip -- fewer than a GRID sentence has phonemes,
    so CTC cannot align. `keep_avgpool=False` (default) drops that
    parameter-free layer, giving 66 steps; the checkpoint loads unchanged.

    Output steps index the flattened (time x space) token sequence, so they
    are not aligned to video frames: CTC only, no frame-level targets."""

    name = "vallr"
    modality = "video"
    step_sec = None
    head_attr = "head"
    num_frames = 16   # VideoMAE positional table is fixed to 16 frames x 224 x 224

    def __init__(self, keep_avgpool: bool = False):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        adapter_dim, hidden = 256, 768
        self.videomae = VideoMAEModel(VideoMAEConfig())
        layers: List[nn.Module] = [
            nn.Conv1d(hidden, adapter_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(adapter_dim), nn.ReLU(),
            nn.Conv1d(adapter_dim, adapter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(adapter_dim), nn.ReLU(),
            nn.Conv1d(adapter_dim, adapter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(adapter_dim), nn.ReLU(),
            nn.Conv1d(adapter_dim, adapter_dim, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(adapter_dim), nn.ReLU(),
        ]
        if keep_avgpool:
            layers.append(nn.AvgPool1d(kernel_size=5, stride=8))
        self.downsampling = nn.Sequential(*layers)
        self.adapter = nn.Sequential(nn.Linear(adapter_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, len(VALLR_VOCAB))

        ckpt = torch.load(VALLR_CKPT, map_location="cpu", weights_only=False)
        self.checkpoint_params = sum(v.numel() for v in ckpt.values() if v.dtype.is_floating_point)
        if "videomae.embeddings.patch_embeddings.projection.weight" not in ckpt:
            raise RuntimeError(f"{VALLR_CKPT} is not a VALLR V1 (VideoMAE) checkpoint")

        def _sub(prefix: str) -> Dict[str, torch.Tensor]:
            return {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

        self.videomae.load_state_dict(_sub("videomae."), strict=True)
        self.downsampling.load_state_dict(_sub("downsampling."), strict=True)
        self.adapter.load_state_dict(_sub("adapter."), strict=True)
        self.head.load_state_dict(_sub("ctc_head."), strict=True)

        used = ("videomae.", "downsampling.", "adapter.", "ctc_head.", "wav2vec2.")
        leftover = [k for k in ckpt if not k.startswith(used)]
        if leftover:
            raise RuntimeError(f"Unexpected VALLR checkpoint keys: {leftover[:5]}")
        # Models/VALLR.py sets self.ctc_head = self.wav2vec2.lm_head, so both are stored
        self.head_matches_lm_head = torch.equal(ckpt["ctc_head.weight"], ckpt["wav2vec2.lm_head.weight"])

        self.source_symbols = VALLR_VOCAB
        self.source_map = IPA_TO_ARPABET

    def freeze_backbone(self, train_top_blocks: int) -> None:
        """Freeze the VideoMAE patch embedding and all but the top
        `train_top_blocks` of its 12 transformer blocks."""
        blocks = self.videomae.encoder.layer
        if not 0 <= train_top_blocks <= len(blocks):
            raise ValueError(f"train_top_blocks must be in [0, {len(blocks)}]")
        self.freeze_modules([self.videomae.embeddings, *blocks[:len(blocks) - train_top_blocks]])

    def load_input(self, path: Path) -> torch.Tensor:
        # main.load_videos(): 16 evenly spaced frames, RGB, 224x224, raw 0-255 floats
        frames = _read_frames(path, rgb=True)
        idx = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
        picked = [frames[i] if frames[i].shape[:2] == (224, 224) else cv2.resize(frames[i], (224, 224))
                  for i in idx]
        return torch.from_numpy(np.stack(picked).astype(np.float32)).permute(0, 3, 1, 2)   # (16, 3, 224, 224)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.videomae(pixel_values=x).last_hidden_state   # (B, 1568, 768)
        feats = self.downsampling(tokens.transpose(1, 2)).transpose(1, 2)
        logits = self.head(self.adapter(feats))
        out_lengths = torch.full_like(lengths, logits.shape[1])
        return logits, out_lengths


# ---------------------------------------------------------------------------
# Auto-AVSR visual speech recognition model (video)
# ---------------------------------------------------------------------------

class AutoAVSRBaseline(Baseline):
    """VSR model of Auto-AVSR (Ma et al., ICASSP 2023; mpc001/auto_avsr,
    Apache-2.0 code), checkpoint vsr_trlrs2lrs3vox2avsp_base: trained on
    LRS2 + LRS3 + VoxCeleb2 + AVSpeech (3291 h), 20.3% WER on LRS3.

    Only the CTC path is built: 3-D conv + ResNet-18 front-end ->
    Linear(512, 768) -> 12-block Conformer -> CTC layer, using the repo's
    own module code. The 6-block attention decoder (64.5M params) is not
    needed for CTC output and is not loaded. Temporal stride is 1
    throughout, so there is one output per video frame (25 fps)."""

    name = "autoavsr"
    modality = "video"
    step_sec = 0.04
    head_attr = "head"
    roi_size, crop_size = 96, 88     # mouth ROI and the 88x88 crop the network sees
    mean, std = 0.421, 0.165
    native_augment = False           # + RandomCrop(88) and AdaptiveTimeMask from the repo's training

    def __init__(self):
        super().__init__()
        if str(AUTOAVSR_DIR) not in sys.path:
            sys.path.insert(0, str(AUTOAVSR_DIR))   # its bundled `espnet` package
        from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
        from espnet.nets.pytorch_backend.frontend.resnet import video_resnet

        # architecture as in espnet/nets/pytorch_backend/e2e_asr_conformer.py (E2E, modality="video")
        self.frontend = video_resnet()
        self.proj_encoder = nn.Linear(512, 768)
        self.encoder = ConformerEncoder(attention_dim=768, attention_heads=12, linear_units=3072,
                                        num_blocks=12, cnn_module_kernel=31)

        ckpt = torch.load(AUTOAVSR_CKPT, map_location="cpu", weights_only=False)
        self.checkpoint_params = sum(v.numel() for v in ckpt.values() if v.dtype.is_floating_point)
        self.head = nn.Linear(768, ckpt["ctc.ctc_lo.weight"].shape[0])

        def _sub(prefix: str) -> Dict[str, torch.Tensor]:
            return {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

        self.frontend.load_state_dict(_sub("frontend."), strict=True)
        self.proj_encoder.load_state_dict(_sub("proj_encoder."), strict=True)
        self.encoder.load_state_dict(_sub("encoder."), strict=True)
        self.head.load_state_dict(_sub("ctc.ctc_lo."), strict=True)
        leftover = [k for k in ckpt if not k.startswith(("frontend.", "proj_encoder.", "encoder.", "ctc.", "decoder."))]
        if leftover:
            raise RuntimeError(f"Unexpected Auto-AVSR checkpoint keys: {leftover[:5]}")

        # index 0 = CTC blank, 1..5047 = SentencePiece units, last = <sos/eos>
        symbols = ["<blank>"] + [""] * (self.head.out_features - 1)
        for line in AUTOAVSR_UNITS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                piece, idx = line.rsplit(" ", 1)
                symbols[int(idx)] = piece
        symbols[-1] = "<sos/eos>"
        self.source_symbols = symbols
        self.source_map = {}   # subword units, not phonemes: only the blank row is reused

    def freeze_backbone(self, train_top_blocks: int) -> None:
        """Freeze the front-end, the projection and all but the top
        `train_top_blocks` of the 12 Conformer blocks."""
        blocks = list(self.encoder.encoders)
        if not 0 <= train_top_blocks <= len(blocks):
            raise ValueError(f"train_top_blocks must be in [0, {len(blocks)}]")
        self.freeze_modules([self.frontend, self.proj_encoder, self.encoder.embed,
                             *blocks[:len(blocks) - train_top_blocks]])

    def format_source(self, ids: Sequence[int]) -> str:
        return "".join(self.source_symbols[i] for i in ids).replace("▁", " ").strip()

    def load_input(self, path: Path) -> torch.Tensor:
        # Repo transforms on a 96x96 mouth ROI: /255, Grayscale, Normalize(0.421, 0.165),
        # then an 88x88 crop -- random in augment() (training, --autoavsr-augment),
        # central in forward() otherwise. Our lip crops are resized to the 96x96 ROI.
        frames = [cv2.cvtColor(cv2.resize(f, (self.roi_size, self.roi_size), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2GRAY) for f in _read_frames(path, rgb=False)]
        video = (np.stack(frames).astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(video).unsqueeze(1)   # (T, 1, 96, 96)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        x = super().augment(x)   # horizontal flip, as for the other video baselines
        if not self.native_augment:
            return x
        # VideoTransform("train") from datamodule/transforms.py: RandomCrop(88) ...
        top, left = (random.randint(0, self.roi_size - self.crop_size) for _ in range(2))
        x = x[..., top:top + self.crop_size, left:left + self.crop_size]
        # ... and AdaptiveTimeMask(10, 25), copied (Apache-2.0); it zeroes pixels
        # before Normalize, i.e. sets them to -mean/std here
        x = x.clone()
        length = x.shape[0]
        for t, t_len in torch.randint(0, 10, size=(int((length + 25 - 0.1) // 25), 2)).tolist():
            if length - t <= 0 or t == 0:
                continue
            t_start = random.randrange(0, length - t)
            x[t_start:t_start + t_len] = -self.mean / self.std
        return x

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.crop_size:   # un-augmented 96x96 ROI: CenterCrop(88) as in the test transform
            o = (self.roi_size - self.crop_size) // 2
            x = x[..., o:o + self.crop_size, o:o + self.crop_size]
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(1)
        feats = self.proj_encoder(self.frontend(x))   # frontend takes (B, T, 1, H, W) -> (B, T, 512)
        feats, _ = self.encoder(feats, mask)
        return self.head(feats), lengths


BASELINES = {
    "icefall": IcefallBaseline,
    "allosaurus": AllosaurusBaseline,
    "lipnet": LipNetBaseline,
    "vallr": VALLRBaseline,
    "autoavsr": AutoAVSRBaseline,
}


def build_baseline(name: str, args: argparse.Namespace) -> Baseline:
    """Construct a baseline with its fine-tuning configuration applied.
    (icefall's BatchNorm recalibration needs the training clips, so it is
    run by the training loop when --icefall-bn=recalibrate.)"""
    model = BASELINES[name]()
    if isinstance(model, VALLRBaseline):
        model.freeze_backbone(args.vallr_train_blocks)
    elif isinstance(model, AutoAVSRBaseline):
        model.freeze_backbone(args.autoavsr_train_blocks)
        model.native_augment = args.autoavsr_augment
    return model


# ---------------------------------------------------------------------------
# Setup check
# ---------------------------------------------------------------------------

def ctc_min_steps(tokens: Sequence[str]) -> int:
    """Fewest output steps CTC needs: one per token plus a blank between repeats."""
    return len(tokens) + sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def extract_clean_audio(stem: str, audio_source_dir: Path, out_dir: Path) -> Path:
    src = next((p for p in audio_source_dir.glob(f"{stem}.*") if p.suffix.lower() in {".mpg", ".mp4", ".wav"}), None)
    if src is None:
        raise FileNotFoundError(f"No clean source for {stem} in {audio_source_dir}")
    out = out_dir / f"{stem}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)],
                   check=True)
    return out


def prepare_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Split + vocabulary exactly as train_av_fusion.main() builds them."""
    all_samples = build_samples(Path(args.phoneme_csv), Path(args.noise_manifest), fps=args.fps)
    train_s, val_s, test_s = split_samples(all_samples, train_ratio=0.7, val_ratio=0.15, seed=args.seed)
    vocab, remap = build_vocab(train_s, min_count=args.min_phoneme_count)
    logger.info("Split -- train:%d val:%d test:%d | vocab (%d): %s",
                len(train_s), len(val_s), len(test_s), len(vocab), " ".join(vocab))
    return {
        "all": apply_vocab_remap(all_samples, remap),
        "train": apply_vocab_remap(train_s, remap),
        "val": apply_vocab_remap(val_s, remap),
        "test": apply_vocab_remap(test_s, remap),
        "vocab": vocab,
        "remap": remap,
    }


@torch.no_grad()
def check_setup(args: argparse.Namespace, device: torch.device) -> None:
    data = prepare_data(args)
    vocab, test_s, all_remapped = data["vocab"], data["test"], data["all"]

    need = [ctc_min_steps(s.canonical_tokens) for s in all_remapped]
    frames = [len(s.frame_tokens) for s in all_remapped]
    logger.info("CTC min steps over %d clips: max %d, mean %.1f | clip length max %d frames",
                len(need), max(need), float(np.mean(need)), max(frames))

    ours = AVLipReadingModel(vocab_size=len(vocab))
    rows = [("ours (AV fusion)", "audio+video", count_params(ours), count_params(ours, True), "0.04", "-")]

    clean_stem = args.clean_stem
    clean_video = Path(args.clean_video_dir) / f"{clean_stem}_lipcrop.mp4"
    ref_sample = next((s for s in all_remapped if s.stem == clean_stem), None)
    if ref_sample is not None:
        logger.info("Clean check clip %s: \"%s\" -> %s", clean_stem, ref_sample.sentence,
                    " ".join(ref_sample.canonical_tokens))
    noisy = test_s[0]

    with tempfile.TemporaryDirectory() as tmp:
        clean_audio = extract_clean_audio(clean_stem, Path(args.audio_source_dir), Path(tmp))

        for name in args.models:
            logger.info("=" * 70)
            model = build_baseline(name, args).to(device).eval()
            n_params = count_params(model)
            logger.info("[%s] %s, %s parameters", name, model.modality, f"{n_params:,}")
            if name == "vallr":
                logger.info("[vallr] checkpoint holds %s params (unused Wav2Vec2 body not loaded); "
                            "ctc_head == wav2vec2.lm_head: %s", f"{model.checkpoint_params:,}",
                            model.head_matches_lm_head)

            # 1) zero-shot with the original head on a clean clip: sanity check of
            #    weights + preprocessing before any fine-tuning
            x = model.load_input(clean_audio if model.modality == "audio" else clean_video)
            xb, lens = pad_batch([x])
            logits, out_lens = model(xb.to(device), lens.to(device))
            ids = greedy_decode(logits[0].cpu(), int(out_lens[0]), blank_id=model.source_blank)
            decoded = model.format_source(ids)
            logger.info("[%s] input %s -> output %s", name, tuple(x.shape), tuple(logits.shape))
            logger.info("[%s] zero-shot (clean %s): %s", name, clean_stem, decoded)

            # 2) new 32+1 head, initialised from the mapped source rows
            unmapped = model.replace_head(vocab)
            mapped = len(vocab) - len(unmapped)
            logger.info("[%s] new head %d outputs; %d/%d phonemes initialised from source%s", name,
                        len(vocab) + 1, mapped, len(vocab),
                        f" (random: {' '.join(unmapped)})" if unmapped else "")

            # 3) forward on a corrupted test clip with the new head
            x = model.load_input(model.input_path(noisy))
            xb, lens = pad_batch([x])
            logits, out_lens = model(xb.to(device), lens.to(device))
            steps = int(out_lens[0])
            logger.info("[%s] noisy %s: %d output steps, CTC needs %d", name, noisy.stem, steps,
                        ctc_min_steps(noisy.canonical_tokens))
            if name == "vallr":
                logger.info("[vallr] with the released AvgPool1d(5, 8) this would be %d steps "
                            "(dataset max need: %d)", (steps - 5) // 8 + 1, max(need))
                steps_all = steps
            else:
                steps_all = min(int(round(f * 0.04 / model.step_sec)) for f in frames)
            ok = sum(1 for n in need if n <= steps_all)
            rows.append((name, model.modality, count_params(model), count_params(model, True),
                         f"{model.step_sec}" if model.step_sec else "not aligned",
                         f"{ok}/{len(need)}"))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info("%-18s %-12s %12s %12s %12s %19s", "model", "modality", "params", "trainable", "sec/step",
                "CTC-feasible clips")
    for r in rows:
        logger.info("%-18s %-12s %12s %12s %12s %19s", r[0], r[1], f"{r[2]:,}", f"{r[3]:,}", r[4], r[5])


# ---------------------------------------------------------------------------
# Fine-tuning data
# ---------------------------------------------------------------------------

class BaselineDataset(torch.utils.data.Dataset):
    """Model input (from model.load_input) + the same targets as
    AVPhonemeDataset: 25 fps frame ids, alignment weights and CTC ids. Frame
    targets are resampled onto the model's output grid in run_batch(), once
    the output lengths are known.

    load_input is a bound method of the model, so this runs in the main
    process (num_workers=0). Audio features are small and deterministic and
    are cached; augmentation is applied on top of the cached copy."""

    def __init__(self, samples, vocab: Sequence[str], model: Baseline, augment: bool):
        self.samples = list(samples)
        self.model = model
        self.augment = augment
        self.tok2id = {tok: idx + 1 for idx, tok in enumerate(vocab)}
        self.cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def input(self, idx: int) -> torch.Tensor:
        if idx in self.cache:
            return self.cache[idx]
        x = self.model.load_input(self.model.input_path(self.samples[idx]))
        if self.model.modality == "audio":
            self.cache[idx] = x
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        x = self.input(idx)

        if self.augment:
            x = self.model.augment(x)

        frame_ids = [self.tok2id.get(t, 0) if t != "" else 0 for t in sample.frame_tokens]
        ctc_ids = [self.tok2id.get(t, 0) for t in sample.canonical_tokens if t in self.tok2id] or [0]
        return {
            "x": x,
            "frame_targets": torch.tensor(frame_ids, dtype=torch.long),
            "alignment_weights": torch.tensor(sample.alignment_weights[:len(frame_ids)], dtype=torch.float32),
            "ctc_targets": torch.tensor(ctc_ids, dtype=torch.long),
            "stem": sample.stem,
        }


def collate_baseline(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x, lengths = pad_batch([b["x"] for b in batch])
    frame_targets, frame_lengths = pad_batch([b["frame_targets"] for b in batch])   # pad = 0 = blank
    alignment_weights, _ = pad_batch([b["alignment_weights"] for b in batch])
    return {
        "x": x, "lengths": lengths,
        "frame_targets": frame_targets, "frame_lengths": frame_lengths,
        "alignment_weights": alignment_weights,
        "ctc_targets": torch.cat([b["ctc_targets"] for b in batch]),
        "ctc_lengths": torch.tensor([len(b["ctc_targets"]) for b in batch], dtype=torch.long),
        "stems": [b["stem"] for b in batch],
    }


def to_output_grid(values: torch.Tensor, frame_lengths: torch.Tensor, n_steps: int,
                   step_sec: float, fps: float) -> torch.Tensor:
    """(B, T_frames) per-video-frame values -> (B, n_steps): each output step
    takes the value of the video frame containing its centre time."""
    centres = (torch.arange(n_steps, device=values.device) + 0.5) * step_sec * fps
    src = torch.minimum(centres.long().unsqueeze(0), (frame_lengths - 1).clamp(min=0).unsqueeze(1))
    return values.gather(1, src)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_batch(model: Baseline, batch: Dict[str, Any], criterion: AVFusionLoss, device: torch.device,
              fps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    logits, out_lens = model(batch["x"].to(device), batch["lengths"].to(device))
    log_probs = F.log_softmax(logits, dim=-1)
    ctc_targets = batch["ctc_targets"].to(device)
    ctc_lengths = batch["ctc_lengths"].to(device)

    if model.step_sec is None:
        # not time-aligned (VALLR): CTC only
        loss = F.ctc_loss(log_probs.transpose(0, 1), ctc_targets, out_lens, ctc_lengths,
                          blank=0, zero_infinity=True)
        bd = {"frame_loss": 0.0, "ctc_loss": float(loss.detach()), "total_loss": float(loss.detach())}
        return logits, out_lens, loss, bd

    frame_lengths = batch["frame_lengths"].to(device)
    frame_targets = to_output_grid(batch["frame_targets"].to(device), frame_lengths, logits.shape[1],
                                   model.step_sec, fps)
    weights = to_output_grid(batch["alignment_weights"].to(device), frame_lengths, logits.shape[1],
                             model.step_sec, fps)
    loss, bd = criterion(logits, log_probs.transpose(0, 1), frame_targets, weights, out_lens,
                         ctc_targets, ctc_lengths)
    return logits, out_lens, loss, bd


def train_epoch(model, loader, criterion, optimizer, device, fps, grad_clip=1.0,
                max_batches: int = 0) -> Dict[str, float]:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    totals: Dict[str, float] = {}
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        _, _, loss, bd = run_batch(model, batch, criterion, device, fps)
        if not loss.isfinite():
            logger.warning("Non-finite loss -- skipping batch")
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        for k, v in bd.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        if max_batches and n >= max_batches:
            break
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def evaluate_baseline(model, loader, criterion, device, id2tok, fps,
                      max_batches: int = 0) -> Tuple[Dict[str, float], List[str], List[str], List[str]]:
    """Same decoding and TER as train_av_fusion.evaluate(): greedy CTC decode
    of the output, reference = 25 fps frame labels without blanks, repeats
    collapsed."""
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    all_preds: List[str] = []
    all_refs: List[str] = []
    all_stems: List[str] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        logits, out_lens, _, bd = run_batch(model, batch, criterion, device, fps)
        for k, v in bd.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

        for i in range(logits.shape[0]):
            pred = greedy_decode(logits[i].cpu(), int(out_lens[i]))
            L = int(batch["frame_lengths"][i])
            ref = [int(t) for t in batch["frame_targets"][i, :L].tolist() if t != 0]
            ref_collapsed = [t for j, t in enumerate(ref) if j == 0 or t != ref[j - 1]]
            all_preds.append(" ".join(id2tok.get(t, "?") for t in pred))
            all_refs.append(" ".join(id2tok.get(t, "?") for t in ref_collapsed))
            all_stems.append(batch["stems"][i])
        if max_batches and n >= max_batches:
            break

    avg = {k: v / max(1, n) for k, v in totals.items()}
    errs = sum(levenshtein(p.split(), r.split()) for p, r in zip(all_preds, all_refs))
    total = sum(max(1, len(r.split())) for r in all_refs)
    avg["ter"] = errs / total
    return avg, all_preds, all_refs, all_stems


def fine_tune(name: str, args: argparse.Namespace, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # A dry run must not touch <output-dir>: a real run may be writing there
    output_dir = Path(args.output_dir) / name
    log_handler: Optional[logging.Handler] = None
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logging.getLogger().addHandler(log_handler)
    t_start = time.time()

    try:
        vocab = data["vocab"]
        id2tok = {0: BLANK_TOKEN, **{i + 1: tok for i, tok in enumerate(vocab)}}
        if not args.dry_run:
            with (output_dir / "vocab.json").open("w", encoding="utf-8") as fh:
                json.dump({"blank_id": 0, "tokens": vocab, "remap": data["remap"]}, fh, indent=2,
                          ensure_ascii=False)

        model = build_baseline(name, args)
        unmapped = model.replace_head(vocab)
        model.to(device)
        n_params, n_trainable = count_params(model), count_params(model, trainable_only=True)
        logger.info("[%s] %s params, %s trainable | head rows from source: %d/%d", name, f"{n_params:,}",
                    f"{n_trainable:,}", len(vocab) - len(unmapped), len(vocab))

        train_ds = BaselineDataset(data["train"], vocab, model, augment=True)
        val_ds = BaselineDataset(data["val"], vocab, model, augment=False)
        test_ds = BaselineDataset(data["test"], vocab, model, augment=False)
        loader_kw = dict(batch_size=args.batch_size, num_workers=0, collate_fn=collate_baseline)
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

        if isinstance(model, IcefallBaseline) and args.icefall_bn == "recalibrate":
            n_cal = len(train_ds) if not args.dry_run else min(len(train_ds), 2 * args.batch_size)
            model.recalibrate_batchnorm([train_ds.input(i) for i in range(n_cal)], device)
            logger.info("[%s] BatchNorm statistics re-estimated on %d training clips", name, n_cal)

        criterion = AVFusionLoss(vocab=vocab, blank_id=0, ctc_weight=args.ctc_weight,
                                 insertion_penalty=args.insertion_penalty, aux_ctc_weight=0.0).to(device)

        head_ids = {id(p) for p in model._get_head().parameters()}
        body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        head = [p for p in model.parameters() if id(p) in head_ids]
        optimizer = torch.optim.AdamW([
            {"params": body, "lr": args.lr},
            {"params": head, "lr": args.lr * args.head_lr_mult},
        ], weight_decay=args.weight_decay)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
        )

        if args.dry_run:
            t0 = time.time()
            train_m = train_epoch(model, train_loader, criterion, optimizer, device, args.fps, args.grad_clip,
                                  max_batches=2)
            t_train = (time.time() - t0) / 2
            t0 = time.time()
            val_m, preds, refs, _ = evaluate_baseline(model, val_loader, criterion, device, id2tok, args.fps,
                                                       max_batches=2)
            t_eval = (time.time() - t0) / 2
            logger.info("[%s] dry run | train %s | val TER %.3f | ~%.2fs/train batch, ~%.2fs/eval batch "
                        "-> ~%.1f min/epoch | peak GPU %.2f GB", name,
                        {k: round(v, 3) for k, v in train_m.items()}, val_m["ter"], t_train, t_eval,
                        (len(train_loader) * t_train + len(val_loader) * t_eval) / 60,
                        torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0)
            logger.info("[%s] dry run | ref : %s", name, refs[0])
            logger.info("[%s] dry run | pred: %s", name, preds[0])
            return {}

        best_ter, best_epoch, no_improve = float("inf"), 0, 0
        history: Dict[str, List] = {"train_total": [], "train_frame": [], "train_ctc": [],
                                    "val_total": [], "val_ter": [], "lr": [], "epoch_sec": []}

        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            if epoch <= args.warmup_epochs:
                for g, base in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = base * epoch / args.warmup_epochs

            train_m = train_epoch(model, train_loader, criterion, optimizer, device, args.fps, args.grad_clip)
            val_m, _, _, _ = evaluate_baseline(model, val_loader, criterion, device, id2tok, args.fps)
            if epoch > args.warmup_epochs:
                scheduler.step(val_m["ter"])

            cur_lr = optimizer.param_groups[0]["lr"]
            history["train_total"].append(train_m.get("total_loss", 0.0))
            history["train_frame"].append(train_m.get("frame_loss", 0.0))
            history["train_ctc"].append(train_m.get("ctc_loss", 0.0))
            history["val_total"].append(val_m.get("total_loss", 0.0))
            history["val_ter"].append(val_m["ter"])
            history["lr"].append(cur_lr)
            history["epoch_sec"].append(time.time() - t_epoch)

            logger.info("[%s] Epoch %03d/%03d | train loss=%.4f (frame=%.4f ctc=%.4f) | val loss=%.4f TER=%.4f "
                        "| lr=%.2e | %.0fs", name, epoch, args.epochs, train_m.get("total_loss", 0),
                        train_m.get("frame_loss", 0), train_m.get("ctc_loss", 0), val_m.get("total_loss", 0),
                        val_m["ter"], cur_lr, history["epoch_sec"][-1])

            if val_m["ter"] < best_ter - 1e-4:
                best_ter, best_epoch, no_improve = val_m["ter"], epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_ter": best_ter,
                            "vocab": vocab, "config": vars(args)}, output_dir / "best_model.pt")
                logger.info("[%s]   saved best model (val TER %.4f)", name, best_ter)
            else:
                no_improve += 1
                if no_improve >= args.early_stop:
                    logger.info("[%s] Early stopping at epoch %d", name, epoch)
                    break

        with (output_dir / "history.json").open("w") as fh:
            json.dump(history, fh, indent=2)

        ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m, test_preds, test_refs, test_stems = evaluate_baseline(model, test_loader, criterion, device,
                                                                      id2tok, args.fps)
        logger.info("[%s] Test -- loss=%.4f TER=%.4f", name, test_m.get("total_loss", 0.0), test_m["ter"])

        phoneme_report(test_preds, test_refs, vocab, output_dir, title=f"Phoneme confusion matrix ({name})")
        with (output_dir / "test_metrics.json").open("w") as fh:
            json.dump(test_m, fh, indent=2)
        with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["stem", "reference", "prediction"])
            w.writeheader()
            w.writerows({"stem": s, "reference": r, "prediction": p}
                        for s, r, p in zip(test_stems, test_refs, test_preds))

        summary = {
            "model": name, "modality": model.modality, "params": n_params, "trainable_params": n_trainable,
            "best_epoch": best_epoch, "epochs_run": len(history["val_ter"]),
            "best_val_ter": round(best_ter, 4), "test_ter": round(test_m["ter"], 4),
            "test_loss": round(test_m.get("total_loss", 0.0), 4),
            "train_minutes": round((time.time() - t_start) / 60, 1),
        }
        logger.info("[%s] Done: %s", name, summary)
        return summary
    finally:
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    """Merge into an existing summary.csv so models can be run separately."""
    merged: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fh:
            merged = {r["model"]: r for r in csv.DictReader(fh)}
    merged.update({r["model"]: r for r in rows})
    order = [n for n in BASELINE_NAMES if n in merged] + [n for n in merged if n not in BASELINE_NAMES]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(merged[n] for n in order)


# ---------------------------------------------------------------------------
# Argument parser / main
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune external audio/video phoneme recognisers as baselines for the AV fusion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phoneme-csv", default="phonemes_s1_aligned/phoneme_predictions.csv")
    p.add_argument("--noise-manifest", default="baseline_noise_s1/noise_manifest_with_mel.csv",
                   help="Same corrupted set the AV model is trained on.")
    p.add_argument("--output-dir", default="baseline_results")
    p.add_argument("--models", nargs="+", choices=BASELINE_NAMES, default=list(BASELINE_NAMES))
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-phoneme-count", type=int, default=1,
                   help="Must match train_av_fusion.py so the vocabulary is identical.")

    # Optimisation: schedule/early stopping/loss weights as train_av_fusion.py;
    # lower LR because the bodies are pretrained, new head at head-lr-mult x LR
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--insertion-penalty", type=float, default=0.5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr-patience", type=int, default=6)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stop", type=int, default=12)
    p.add_argument("--dry-run", action="store_true",
                   help="Two train + two eval batches per model, print timing/memory, save nothing.")

    p.add_argument("--vallr-train-blocks", type=int, default=2,
                   help="VideoMAE blocks (of 12, counted from the top) left trainable; the patch embedding "
                        "and all lower blocks are frozen.")
    p.add_argument("--autoavsr-train-blocks", type=int, default=2,
                   help="Conformer blocks (of 12, counted from the top) left trainable; the ResNet front-end, "
                        "projection and all lower blocks are frozen.")
    p.add_argument("--autoavsr-augment", action="store_true",
                   help="Add Auto-AVSR's own training augmentations (random 88x88 crop of the 96x96 ROI, "
                        "AdaptiveTimeMask(10, 25)) on top of the horizontal flip.")
    p.add_argument("--icefall-bn", choices=["recalibrate", "timit"], default="recalibrate",
                   help="icefall BatchNorm stats (always frozen while fine-tuning): re-estimated once on the "
                        "GRID training clips, or kept from TIMIT.")

    p.add_argument("--check-setup", action="store_true",
                   help="Load every checkpoint, run a zero-shot clean decode and a noisy forward, then exit.")
    p.add_argument("--clean-stem", default="bbaf2n", help="Clip used for the zero-shot sanity check.")
    p.add_argument("--audio-source-dir", default="s1", help="Clean GRID .mpg files (audio for the clean check).")
    p.add_argument("--clean-video-dir", default="s1_lip_crops")
    return p


def main() -> None:
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.check_setup:
        check_setup(args, device)
        return

    data = prepare_data(args)
    summaries = []
    for name in args.models:
        logger.info("=" * 70)
        summary = fine_tune(name, args, data, device)
        if summary:
            summaries.append(summary)
            write_summary(summaries, Path(args.output_dir) / "summary.csv")

    if summaries:
        logger.info("=" * 70)
        for s in summaries:
            logger.info("%-11s %-6s params=%11s trainable=%11s best_epoch=%3d val_TER=%.4f test_TER=%.4f (%.0f min)",
                        s["model"], s["modality"], f"{s['params']:,}", f"{s['trainable_params']:,}",
                        s["best_epoch"], s["best_val_ter"], s["test_ter"], s["train_minutes"])


if __name__ == "__main__":
    main()
