from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# ============================
# UTILS
# ============================

def _mono(x: np.ndarray) -> np.ndarray:
    """Ritorna un segnale mono (canale 0 se multicanale)."""
    x = np.asarray(x, dtype=np.float64)
    return x[:, 0] if x.ndim == 2 else x


def compute_simple_snr(clean, processed):
    """SNR proxy senza reference perfetto."""
    noise = clean - processed
    return 10 * np.log10(
        np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-12)
    )


# ============================
# ONSET METRICS
# ============================

def compute_onset_late_metrics(
    wet: np.ndarray,
    dry: np.ndarray,
    fs: int,
    early_ms: float = 80.0,
    late_start_ms: float = 80.0,
    late_end_ms: float = 350.0,
    max_onsets: int = 12,
    min_separation_ms: float = 250.0,
):
    y_w = _mono(wet)
    y_d = _mono(dry)

    oenv = librosa.onset.onset_strength(y=y_w, sr=fs)
    on_frames = librosa.onset.onset_detect(
        onset_envelope=oenv,
        sr=fs,
        backtrack=True
    )
    on_samples = librosa.frames_to_samples(on_frames)

    if len(on_samples) == 0:
        return {"n_onsets": 0, "rows": []}

    min_sep = int((min_separation_ms / 1000.0) * fs)

    picked = []
    last = -10**18
    for s in on_samples:
        if s - last >= min_sep:
            picked.append(int(s))
            last = int(s)
        if len(picked) >= max_onsets:
            break

    eN = max(1, int((early_ms / 1000.0) * fs))
    l0 = int((late_start_ms / 1000.0) * fs)
    l1 = max(l0 + 1, int((late_end_ms / 1000.0) * fs))

    rows = []

    for s0 in picked:
        if s0 + l1 >= len(y_w) or s0 + l1 >= len(y_d):
            continue

        Ew_e = np.sum(y_w[s0:s0 + eN] ** 2)
        Ew_l = np.sum(y_w[s0 + l0:s0 + l1] ** 2) + 1e-12
        Ed_e = np.sum(y_d[s0:s0 + eN] ** 2)
        Ed_l = np.sum(y_d[s0 + l0:s0 + l1] ** 2) + 1e-12

        ELR_w = 10 * np.log10(Ew_e / Ew_l)
        ELR_d = 10 * np.log10(Ed_e / Ed_l)

        rows.append({
            "ELR_wet": ELR_w,
            "ELR_dry": ELR_d,
            "ELR_gain": ELR_d - ELR_w,
        })

    if len(rows) == 0:
        return {"n_onsets": 0, "rows": []}

    return {
        "n_onsets": len(rows),
        "ELR_wet_med": float(np.median([r["ELR_wet"] for r in rows])),
        "ELR_dry_med": float(np.median([r["ELR_dry"] for r in rows])),
        "ELR_gain_med": float(np.median([r["ELR_gain"] for r in rows])),
        "rows": rows,
    }


# ============================
# MAIN PLOT FUNCTION
# ============================

def _auto_hop(n_samples: int, n_fft: int, max_frames: int = 1500) -> int:
    """
    Calcola un hop_length per lo spettrogramma tale da non superare
    max_frames frame temporali, indipendentemente dalla durata del file.

    Necessario perché con hop fisso (es. 256) i file più lunghi generano
    spettrogrammi enormi che possono esaurire la memoria in fase di
    salvataggio (fig.savefig ad alto dpi su migliaia di frame).
    """
    hop = max(1, n_samples // max_frames)
    return max(hop, n_fft // 8)  # non scendere sotto una risoluzione minima


def save_analysis_plots(
    out_png: str | Path,
    wet: np.ndarray,
    dry: np.ndarray,
    fs: int,
    dbg: dict,
    zoom_ms: float = 300.0,
    n_fft_stft: int = 2048,
    hop: int | None = 256,
    dpi: int = 150,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    wet = np.asarray(wet, dtype=np.float64)
    dry = np.asarray(dry, dtype=np.float64)

    wet0 = wet[:, 0] if wet.ndim == 2 else wet
    dry0 = dry[:, 0] if dry.ndim == 2 else dry

    # --------------------
    # SPETTROGRAMMA: hop adattivo per evitare immagini enormi su file lunghi
    # --------------------
    hop_requested = hop if hop is not None else 256
    hop_spec = max(hop_requested, _auto_hop(len(wet0), n_fft_stft))

    # --------------------
    # METRICS
    # --------------------
    snr_proxy = compute_simple_snr(wet0, dry0)
    metrics = compute_onset_late_metrics(wet0, dry0, fs)

    # --------------------
    # FILTER DATA
    # --------------------
    H = dbg["H"]
    G = dbg["G"]
    freqs = dbg["freqs"]
    params = dbg.get("params", {})

    # --------------------
    # TIME AXIS
    # --------------------
    t = np.arange(len(wet0)) / fs
    Z = min(len(wet0), int((zoom_ms / 1000.0) * fs))

    # --------------------
    # FIGURE
    # --------------------
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

    # Waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, wet0, "b", alpha=0.6, label="Wet")
    ax1.plot(t, dry0, "r", alpha=0.6, label="Dry")
    ax1.set_title("Waveform completa")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Zoom
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[:Z], wet0[:Z], "b")
    ax2.plot(t[:Z], dry0[:Z], "r")
    ax2.set_title("Zoom attacco")
    ax2.grid(True, alpha=0.3)

    # Spectrogram Wet
    ax3 = fig.add_subplot(gs[1, 1])
    Sw = librosa.amplitude_to_db(
        np.abs(librosa.stft(wet0, n_fft=n_fft_stft, hop_length=hop_spec)),
        ref=np.max
    )
    librosa.display.specshow(Sw, sr=fs, hop_length=hop_spec, ax=ax3)
    ax3.set_title("Wet spectrogram")

    # Spectrogram Dry
    ax4 = fig.add_subplot(gs[1, 2])
    Sd = librosa.amplitude_to_db(
        np.abs(librosa.stft(dry0, n_fft=n_fft_stft, hop_length=hop_spec)),
        ref=np.max
    )
    librosa.display.specshow(Sd, sr=fs, hop_length=hop_spec, ax=ax4)
    ax4.set_title("Dry spectrogram")

    # Frequency response
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.plot(freqs, 20*np.log10(np.abs(H)+1e-12), "k", label="H")
    ax5.plot(freqs, 20*np.log10(np.abs(G)+1e-12), "m", label="G")
    ax5.set_xlim([0, fs/2])
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title("Frequency response")

    # TEXT PANEL
    axT = fig.add_subplot(gs[2, 2])
    axT.axis("off")

    txt = f"PARAMS:\n{params}\n\n"
    txt += f"SNR proxy:\n{snr_proxy:+.2f} dB\n\n"

    if metrics["n_onsets"] > 0:
        txt += (
            f"Onsets: {metrics['n_onsets']}\n"
            f"ELR wet: {metrics['ELR_wet_med']:+.2f}\n"
            f"ELR dry: {metrics['ELR_dry_med']:+.2f}\n"
            f"Gain:    {metrics['ELR_gain_med']:+.2f}\n"
        )
    else:
        txt += "No onsets detected"

    axT.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)