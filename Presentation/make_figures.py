"""Generate the presentation-only figures in Presentation/figures/.

Uses clip bbaf2n of speaker s1 ("bin blue at f two now"):
  - lip-crop frames (input of the visual stream)
  - a full GRID frame (problem-definition slide)
  - the log Mel-Spectrogram of the audio (input of the audio stream),
    with the settings of the Methodology chapter
    (16 kHz, n_fft=1024, hop=round(16000/25)=640, 80 mel bands).

Usage (from the repository root):
    python Presentation/make_figures.py
"""

import subprocess
import tempfile
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Presentation" / "figures"
CLIP = "bbaf2n"
LIP_VIDEO = ROOT / "s1_lip_crops" / f"{CLIP}_lipcrop.mp4"
RAW_VIDEO = ROOT / "s1" / f"{CLIP}.mpg"
# 24-31 cover "bin bl(ue)": per-frame labels b ɪ ɪ - n - b l (phonemes_s1_aligned)
LIP_FRAMES = [0, *range(24, 32), 49]


def extract_frame(video: Path, index: int, dest: Path, size: int | None = None) -> None:
    vf = f"select=eq(n\\,{index})"
    if size:
        vf += f",scale={size}:{size}:flags=area"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", vf, "-vframes", "1", str(dest)],
        check=True,
    )


def mel_spectrogram(video: Path, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
        y, sr = librosa.load(wav, sr=16000)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=round(16000 / 25), n_mels=80
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(6, 2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
    ax.set_axis_off()
    fig.savefig(dest)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for idx in LIP_FRAMES:
        extract_frame(LIP_VIDEO, idx, OUT / f"lips_{CLIP}_f{idx:02d}.png", size=96)
    extract_frame(RAW_VIDEO, 49, OUT / f"grid_{CLIP}_f49.png")
    mel_spectrogram(RAW_VIDEO, OUT / f"mel_{CLIP}.png")
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
