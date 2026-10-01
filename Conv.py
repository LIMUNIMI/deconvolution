"""
Conv.py
=======
Genera il dataset di validazione: convolve ogni segnale anecoico (dry)
con ogni RIR e salva il segnale riverberato (wet).

Scelte, tutte allineate con main.py:
- frequenza di campionamento unica a 48 kHz per tutti i file;
- RIR ridotta a mono con la stessa regola di main.py (opzione --rir_mono,
  che deve avere lo stesso valore nei due script);
- dry multicanale: canale 0, come in evaluate.py e validated_metrics.py,
  cosi' il riferimento usato per le metriche e' esattamente il segnale
  che e' stato convoluto;
- RIR normalizzata in energia e tagliata al picco del suono diretto;
- wet normalizzato al picco (0.9) e salvato in PCM a 16 bit.
"""

import argparse
import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

TARGET_SR = 48000
BASE_DIR = Path(__file__).resolve().parent.parent      # ...\ground_truth


def rir_to_mono(rir: np.ndarray, mode: str = "mean") -> np.ndarray:
    
    rir = np.asarray(rir, dtype=np.float64)
    if rir.ndim == 1 or rir.shape[1] == 1:
        return rir.reshape(-1)
    if mode == "mean":
        return np.mean(rir, axis=1)
    if mode == "w":
        return rir[:, 0]
    raise ValueError(f"rir_mono non valido: {mode!r} (usa 'mean' o 'w')")


def resample_audio(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g)


def rir_subfolder(rir_path: Path) -> str:
    m = re.match(r"(\d+)", rir_path.stem)
    return f"rir{m.group(1)}" if m else rir_path.stem


def main():
    p = argparse.ArgumentParser(description="Genera i segnali wet del dataset di validazione")
    p.add_argument("--dry_dir", default=str(BASE_DIR / "dry"))
    p.add_argument("--rir_dir", default=str(BASE_DIR / "rir"))
    p.add_argument("--out_dir", default=str(BASE_DIR / "reverberated_v2"),
                   help="Cartella di output (nuova, per non sovrascrivere i wet esistenti)")
    p.add_argument("--rir_mono", choices=["mean", "w"], default="mean",
                   help="Deve coincidere con --rir_mono di main.py")
    args = p.parse_args()

    dry_files = sorted(Path(args.dry_dir).glob("*.wav"))
    rir_files = sorted(Path(args.rir_dir).glob("*.wav"))
    print(f"[INFO] dry: {args.dry_dir} ({len(dry_files)} file)")
    print(f"[INFO] RIR: {args.rir_dir} ({len(rir_files)} file), riduzione a mono: {args.rir_mono}")
    print(f"[INFO] output: {args.out_dir}\n")
    if not dry_files or not rir_files:
        raise RuntimeError("Nessun file .wav trovato: controlla --dry_dir e --rir_dir")

    for rir_path in rir_files:
        h, sr_h = sf.read(rir_path, always_2d=True)
        h = rir_to_mono(h, args.rir_mono)
        h = resample_audio(h, sr_h, TARGET_SR)
        h = h / np.sqrt(np.sum(h ** 2))          # normalizzazione in energia
        h = h[np.argmax(np.abs(h)):]              # taglio al suono diretto

        out_dir = Path(args.out_dir) / rir_subfolder(rir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        for dry_path in dry_files:
            x, sr_x = sf.read(dry_path, always_2d=True)
            x = resample_audio(x[:, 0], sr_x, TARGET_SR)

            y = fftconvolve(x, h, mode="full")
            y = y / np.max(np.abs(y)) * 0.9

            out_path = out_dir / f"{dry_path.stem}_{rir_path.stem}.wav"
            sf.write(out_path, y, TARGET_SR, subtype="PCM_16")
            print(f"  {out_dir.name}\\{out_path.name}")

    print(f"\nFine. Elabora i wet con main.py usando --no_sync --rir_mono {args.rir_mono}")


if __name__ == "__main__":
    main()
