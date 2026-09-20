"""
apply_noise_gate.py
=====================
Applica il noise gate/espansore ai file gia' dereverberati (batch),
per ridurre il rumore di fondo percepito nei tratti silenziosi, senza
dover rilanciare tutta la pipeline di dereverberazione.

Uso:
    py apply_noise_gate.py --in_dir dereverb_out_test2/wiener --out_dir dereverb_out_gated/wiener
    py apply_noise_gate.py --in_dir dereverb_out_test2/wiener --out_dir dereverb_out_gated/wiener \
        --threshold_db -40 --range_db 18 --release_ms 300

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from noise_gate import apply_noise_gate


def save_gate_plot(out_png: Path, before: np.ndarray, after: np.ndarray,
                    gain_db: np.ndarray, fs: int, n_fft: int = 2048, dpi: int = 150) -> None:
    """
    Genera un grafico diagnostico del gate: waveform prima/dopo,
    spettrogrammi prima/dopo, e la curva di guadagno applicata nel
    tempo (mostra visivamente quando/quanto il gate interviene).
    """
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    before0 = before[:, 0] if before.ndim == 2 else before
    after0  = after[:, 0]  if after.ndim  == 2 else after

    # Hop adattivo per non generare spettrogrammi enormi su file lunghi
    max_frames = 1500
    hop = max(n_fft // 8, len(before0) // max_frames)

    t = np.arange(len(before0)) / fs

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, before0, "b", alpha=0.5, label="Prima del gate")
    ax1.plot(t, after0, "r", alpha=0.7, label="Dopo il gate")
    ax1.set_title("Waveform: prima vs dopo il gate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    S_before = librosa.amplitude_to_db(np.abs(librosa.stft(before0, n_fft=n_fft, hop_length=hop)), ref=np.max)
    librosa.display.specshow(S_before, sr=fs, hop_length=hop, ax=ax2)
    ax2.set_title("Spettrogramma PRIMA")

    ax3 = fig.add_subplot(gs[1, 1])
    S_after = librosa.amplitude_to_db(np.abs(librosa.stft(after0, n_fft=n_fft, hop_length=hop)), ref=np.max)
    librosa.display.specshow(S_after, sr=fs, hop_length=hop, ax=ax3)
    ax3.set_title("Spettrogramma DOPO")

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(t, gain_db, "purple")
    ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_title("Guadagno applicato dal gate nel tempo (0dB = nessuna attenuazione)")
    ax4.set_ylabel("dB")
    ax4.set_xlabel("Tempo (s)")
    ax4.grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_file(in_path: Path, out_path: Path, args, plot_dir: Path | None = None) -> None:
    x, fs = sf.read(in_path, always_2d=True)
    x = x.astype(np.float64)

    result = apply_noise_gate(
        x, fs,
        threshold_db = args.threshold_db,
        range_db     = args.range_db,
        knee_db      = args.knee_db,
        attack_ms    = args.attack_ms,
        release_ms   = args.release_ms,
        env_win_ms   = args.env_win_ms,
        return_gain  = plot_dir is not None,
    )
    if plot_dir is not None:
        out, gain_db = result
    else:
        out = result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, out.astype(np.float32), fs)
    print(f"  {in_path.name} -> {out_path.name}")

    if plot_dir is not None:
        png_path = plot_dir / f"Gate_{in_path.stem.replace('-', '_')}.png"
        save_gate_plot(png_path, x, out, gain_db, fs)
        print(f"    [PLOT] {png_path.name}")


def main():
    p = argparse.ArgumentParser(description="Applica un noise gate ai file dereverberati")
    p.add_argument("--in_file", default=None, help="Singolo file di input")
    p.add_argument("--out_file", default=None, help="Singolo file di output")
    p.add_argument("--in_dir", default=None, help="Cartella di file .wav da processare in batch")
    p.add_argument("--out_dir", default=None, help="Cartella di output per il batch")

    p.add_argument("--threshold_db", type=float, default=-45.0,
                    help="Soglia (relativa al picco) sotto cui il gate si chiude (dB)")
    p.add_argument("--range_db", type=float, default=18.0,
                    help="Attenuazione massima applicata (dB). Non azzera mai il segnale.")
    p.add_argument("--knee_db", type=float, default=6.0,
                    help="Larghezza della transizione morbida attorno alla soglia (dB)")
    p.add_argument("--attack_ms", type=float, default=3.0,
                    help="Velocita' di apertura del gate (ms)")
    p.add_argument("--release_ms", type=float, default=300.0,
                    help="Velocita' di chiusura del gate (ms). Piu' alto = decadimento "
                         "naturale delle note preservato meglio, ma il rumore impiega "
                         "piu' tempo a essere attenuato dopo che il segnale finisce.")
    p.add_argument("--env_win_ms", type=float, default=10.0,
                    help="Finestra per la stima dell'inviluppo RMS (ms)")

    p.add_argument("--plots", action="store_true",
                    help="Genera grafici diagnostici (waveform, spettrogrammi, "
                         "curva del guadagno nel tempo) per ogni file processato")
    p.add_argument("--plot_dir", default="gate_plots",
                    help="Cartella di output per i grafici (solo se --plots)")

    args = p.parse_args()

    plot_dir = Path(args.plot_dir) if args.plots else None

    if args.in_file:
        out_path = Path(args.out_file) if args.out_file else Path(args.in_file).with_stem(
            Path(args.in_file).stem + "_gated")
        process_file(Path(args.in_file), out_path, args, plot_dir=plot_dir)
    elif args.in_dir:
        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir) if args.out_dir else in_dir.parent / (in_dir.name + "_gated")
        files = sorted(in_dir.glob("*.wav"))
        print(f"[INFO] {len(files)} file da processare")
        print(f"[INFO] Parametri: threshold={args.threshold_db}dB  range={args.range_db}dB  "
              f"attack={args.attack_ms}ms  release={args.release_ms}ms")
        if plot_dir is not None:
            print(f"[INFO] Grafici in: {plot_dir}")
        print()
        for f in files:
            process_file(f, out_dir / f.name, args, plot_dir=plot_dir)
        print(f"\n[OK] Output in: {out_dir}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
