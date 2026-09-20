from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


# =========================
# FUNZIONI DI SUPPORTO I/O
# =========================

def clean_path(s: str) -> str:
    """
    Pulisce un percorso inserito dall'utente o ottenuto via drag&drop su Windows:
    - rimuove virgolette singole/doppie
    - rimuove spazi iniziali/finali
    """
    if s is None:
        return ""
    return s.strip().strip('"').strip("'")


def to_2d(x: np.ndarray) -> np.ndarray:
    """
    Forza un array audio in forma [N, C]:
    - mono: [N] -> [N, 1]
    - stereo o più canali: già [N, C]
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Carica un file audio generico con soundfile.
    Ritorna:
      x: np.ndarray [N, C] in float64
      sr: sample rate
    """
    path = Path(path)
    x, sr = sf.read(path, always_2d=True)
    return x.astype(np.float64), int(sr)


def resample_multichannel(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    Resampling per canale, usando librosa.resample.
    Serve perché audio e RIR devono essere allo stesso sample rate.
    """
    if sr_in == sr_out:
        return to_2d(x)

    x = to_2d(x).astype(np.float64)
    ys = []
    for c in range(x.shape[1]):
        ys.append(librosa.resample(x[:, c], orig_sr=sr_in, target_sr=sr_out))
    return np.stack(ys, axis=1)


def peak_normalize(x: np.ndarray, target: float = 0.99) -> np.ndarray:
    """
    Normalizzazione peak per evitare clipping.
    target <= 0: disattiva
    """
    x = np.asarray(x, dtype=np.float64)
    if target <= 0:
        return x
    m = float(np.max(np.abs(x)) + 1e-12)
    return (target / m) * x


def save_audio(path: str | Path, x: np.ndarray, sr: int, float_out: bool = False) -> None:
    """
    Salva audio in:
    - PCM_16 (compatibile ovunque) se float_out=False
    - FLOAT (senza quantizzazione) se float_out=True
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = to_2d(np.asarray(x, dtype=np.float64))
    x = np.clip(x, -1.0, 1.0)

    if float_out:
        sf.write(path, x.astype(np.float32), sr, subtype="FLOAT")
    else:
        sf.write(path, x.astype(np.float32), sr, subtype="PCM_16")