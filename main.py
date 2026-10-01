"""
main.py
=======
Entry point del sistema di de-riverberazione.

All'avvio chiede interattivamente due cose: il percorso del file o
della cartella audio da elaborare, e il percorso della RIR
dell'ambiente. Nessun'altra domanda, nessuna assunzione sul naming
dei file, nessuna logica di convoluzione qui dentro.

Uso base (interattivo):
    py main.py

Uso avanzato (per chi vuole controllare i parametri sperimentali,
es. per riprodurre il dataset di validazione della tesi):
    py main.py --no_auto --beta_rel 0.1 --plots --plot_dir out_plots
    py main.py --max_anticausal_ms 150     (limita il pre-eco con RIR misurate)
"""

from __future__ import annotations

import numpy as np
import argparse
from pathlib import Path

from io_utils import clean_path, load_audio, resample_multichannel, save_audio
from rir_utils import prepare_rir, drr_from_rir_db, auto_wiener_params, find_direct_path_offset, rir_to_mono
from wiener_dereverb import dereverb_wiener, apply_low_shelf_boost
from wiener_stft import dereverb_wiener_stft
from plot_utils import save_analysis_plots
from noise_gate import apply_noise_gate


# ============================
# MENU INTERATTIVO
# ============================

def ask_path(prompt: str) -> Path:
    while True:
        raw = input(prompt).strip().strip('"')
        p = Path(clean_path(raw))
        if p.exists():
            return p
        print(f"  -> Percorso non trovato: {p}. Riprova.\n")


def run_interactive_menu() -> tuple[Path, Path]:
    print("=" * 70)
    print("  DE-RIVERBERAZIONE — configurazione")
    print("=" * 70)
    input_path = ask_path(
        "\nPercorso del file audio o della cartella da elaborare: "
    )
    rir_path = ask_path(
        "\nPercorso del file della RIR dell'ambiente: "
    )
    return input_path, rir_path


# ============================
# CLI (parametri avanzati, opzionali, con default sensati)
# ============================

def build_parser():
    p = argparse.ArgumentParser(
        description="DeRiverberazione: Wiener stazionario + Wiener STFT in parallelo."
    )

    # percorsi: opzionali da CLI, se assenti vengono chiesti a menu
    p.add_argument("-i", "--input", default=None,
                   help="File o cartella audio da elaborare (se omesso, viene chiesto a menu)")
    p.add_argument("-r", "--rir", default=None,
                   help="File RIR (se omesso, viene chiesto a menu)")
    p.add_argument("--out_dir", default="dereverb_out",
                   help="Cartella radice di output")

    p.add_argument("--sync", action="store_true", default=False,
                   help="Attiva il taglio iniziale del segnale stimato per "
                        "cross-correlazione con la testa della RIR. Disattivato di "
                        "default: il filtro e' tempo-invariante e non richiede "
                        "allineamento, e su registrazioni musicali il taglio rimuove "
                        "parte del segnale (vedi Cap. 4).")
    p.add_argument("--no_sync", action="store_true", default=False,
                   help="Mantenuto per compatibilita' con i comandi precedenti: "
                        "la sincronizzazione e' gia' disattivata di default.")
    p.add_argument("--no_auto", action="store_true", default=False,
                   help="Disabilita la stima automatica dei parametri dalla RIR "
                        "(T60/DRR -> trunc_s/beta_rel/post_ms). Default: attiva.")

    p.add_argument("--float_out", action="store_true",
                   help="Salva in float32 invece di PCM_16")
    p.add_argument("--plots",    action="store_true",
                   help="Genera plot di analisi per entrambi i metodi")
    p.add_argument("--plot_dir", default="dereverb_plots",
                   help="Cartella radice plot (usata solo se --plots e' attivo)")
    p.add_argument("--sr", type=int, default=0,
                   help="Sample rate target (0 = usa SR del file input)")
    p.add_argument("--adaptive_beta", action="store_true",
                   help="Attiva beta adattiva per banda nello STFT")

    # Parametri RIR
    p.add_argument("--rir_trunc_s",  type=float, default=0.8)
    p.add_argument("--tukey_alpha",  type=float, default=0.25)
    p.add_argument("--no_causal",    action="store_true")
    p.add_argument("--rir_mono", choices=["mean", "w"], default="mean",
                   help="Riduzione a mono della RIR multicanale: 'mean' = media dei "
                        "canali, 'w' = canale 0 (W, omnidirezionale in AmbiX). "
                        "Deve coincidere con quella usata in Conv.py.")

    # Parametri filtro di Wiener
    p.add_argument("--beta_rel",  type=float, default=0.15)
    p.add_argument("--beta_scale", type=float, default=1.0,
                   help="Moltiplicatore applicato a beta_rel (auto o manuale). "
                        "Valori <1.0 rendono il filtro piu' aggressivo "
                        "(meno regolarizzazione, rischio di artefatti se troppo "
                        "basso); valori >1.0 lo rendono piu' prudente. "
                        "Es. 0.85 = 15%% piu' aggressivo del valore stimato.")
    p.add_argument("--hf_tilt",   type=float, default=0.0)
    p.add_argument("--hf_fc",     type=float, default=6000.0)
    p.add_argument("--beta_max",  type=float, default=10.0)
    p.add_argument("--delay_ms",  type=float, default=18.0)
    p.add_argument("--f_lo",      type=float, default=60.0)
    p.add_argument("--f_hi",      type=float, default=20000.0)
    p.add_argument("--gmax_db",   type=float, default=8.0)
    p.add_argument("--pre_ms",    type=float, default=15.0)
    p.add_argument("--post_ms",   type=float, default=600.0)
    p.add_argument("--post_window_cap_ms", type=float, default=2000.0)
    p.add_argument("--max_anticausal_ms", type=float, default=0.0,
                   help="Limita la parte anticausale del filtro inverso a questo numero "
                        "di millisecondi prima dell'istante corrente (0 = nessun limite). "
                        "Riduce il pre-eco quando la RIR misurata non coincide esattamente "
                        "con quella della registrazione, al prezzo di una de-riverberazione "
                        "meno efficace (vedi Cap. 5). Esempio: 150.")
    p.add_argument("--time_window", action="store_true", default=False,
                   help="Applica una finestra temporale alla risposta del filtro "
                        "(disattivata di default: tronca il filtro inverso e ne "
                        "annulla l'effetto, vedi Cap. 5).")
    p.add_argument("--shelf_db", type=float, default=10.0,
                   help="Guadagno dell'EQ low-shelf sotto i 350 Hz in post-processing "
                        "(0 = disattivato).")

    # Parametri STFT
    p.add_argument("--stft_n_fft",   type=int,   default=2048)
    p.add_argument("--stft_overlap", type=float, default=0.75)

    # Noise gate (opzionale, in aggiunta all'output senza gate, non al posto)
    p.add_argument("--gate", action="store_true", default=False,
                   help="Applica anche il noise gate all'output, salvandolo in "
                        "sottocartelle separate (wiener_gated/, wiener_stft_gated/) "
                        "senza sovrascrivere l'output senza gate.")
    p.add_argument("--gate_threshold_db", type=float, default=-45.0)
    p.add_argument("--gate_range_db",     type=float, default=18.0)
    p.add_argument("--gate_knee_db",      type=float, default=6.0)
    p.add_argument("--gate_attack_ms",    type=float, default=3.0)
    p.add_argument("--gate_release_ms",   type=float, default=300.0)
    p.add_argument("--gate_env_win_ms",   type=float, default=10.0)

    return p


# ============================
# POST-PROCESSING COMUNE
# ============================

def postprocess(dry: np.ndarray, x_rs: np.ndarray, fs: int, shelf_db: float = 10.0) -> np.ndarray:
    """Normalizzazione RMS + EQ compensativo low-shelf (Cap.4 Sez.4.3.6)."""
    rms_in  = np.sqrt(np.mean(x_rs[:, 0] ** 2))
    rms_out = np.sqrt(np.mean(dry[:, 0] ** 2)) + 1e-12
    dry    *= rms_in / rms_out
    if shelf_db != 0.0:
        dry = apply_low_shelf_boost(dry, fs, f0=350.0, gain_db=shelf_db)
    return dry


# ============================
# ELABORAZIONE DI UN SINGOLO FILE
# ============================

def apply_gate_and_save(dry: np.ndarray, fs: int, in_path: Path,
                         out_gated: Path, args, suffix: str) -> None:
    """Applica il noise gate a un output gia' de-riverberato e lo salva
    in una cartella separata, senza toccare l'output senza gate."""
    gated = apply_noise_gate(
        dry, fs,
        threshold_db = args.gate_threshold_db,
        range_db     = args.gate_range_db,
        knee_db      = args.gate_knee_db,
        attack_ms    = args.gate_attack_ms,
        release_ms   = args.gate_release_ms,
        env_win_ms   = args.gate_env_win_ms,
    )
    path_gated = out_gated / f"{in_path.stem}_{suffix}_gated.wav"
    save_audio(path_gated, gated, fs, float_out=args.float_out)
    print(f"  [gate]        → {path_gated.name}")


def process_one(in_path: Path, rir_mono: np.ndarray, sr_rir: int,
                 out_stat: Path, out_stft: Path, args,
                 plot_stat: Path | None, plot_stft: Path | None,
                 out_stat_gated: Path | None = None,
                 out_stft_gated: Path | None = None):

    print("=" * 70)
    print(f"Input: {in_path.name}")

    x, sr_x = load_audio(in_path)
    fs      = sr_x if args.sr == 0 else int(args.sr)

    x_rs   = resample_multichannel(x,        sr_x,   fs)
    rir_rs = resample_multichannel(rir_mono, sr_rir, fs).squeeze()

    beta_rel, trunc_s, post_ms = args.beta_rel, args.rir_trunc_s, args.post_ms
    if not args.no_auto:
        auto     = auto_wiener_params(rir_rs, fs)
        beta_rel = auto["beta_rel"]
        trunc_s  = auto["trunc_s"]
        post_ms  = auto["post_ms"]
        print(f"  [AUTO] trunc_s={trunc_s:.2f}s  beta_rel={beta_rel:.3f}  post_ms={post_ms:.0f}ms")

    if args.beta_scale != 1.0:
        beta_rel_prima = beta_rel
        beta_rel *= args.beta_scale
        print(f"  [SCALE] beta_rel {beta_rel_prima:.3f} -> {beta_rel:.3f} (x{args.beta_scale})")

    # --- sincronizzazione: disattivata di default (opzione --sync). Il filtro
    #     e' lineare e tempo-invariante: non serve allineare il segnale alla
    #     RIR, e il taglio iniziale rimuoverebbe parte della registrazione ---
    if args.sync and not args.no_sync:
        sync_offset = find_direct_path_offset(x_rs[:, 0], rir_rs, fs)
        if sync_offset > 0:
            print(f"  [SYNC] offset diretto: {sync_offset/fs*1000:.1f}ms — wet allineato alla RIR")
            x_rs = x_rs[sync_offset:, :]

    rir_proc, peak_idx = prepare_rir(
        rir         = rir_rs,
        fs          = fs,
        trunc_s     = trunc_s,
        tukey_alpha = args.tukey_alpha,
        make_causal = (not args.no_causal),
    )

    drr = drr_from_rir_db(rir_proc, fs)
    print(f"  RIR: peak={peak_idx} | DRR={drr:+.1f} dB")

    wiener_kwargs = dict(
        beta_rel=beta_rel, hf_tilt=args.hf_tilt, hf_fc=args.hf_fc,
        beta_max=args.beta_max, delay_ms=args.delay_ms, f_lo=args.f_lo,
        f_hi=args.f_hi, gmax_db=args.gmax_db, pre_ms=args.pre_ms,
        post_ms=post_ms, post_window_cap_ms=args.post_window_cap_ms,
        time_window=args.time_window,
    )
    if args.max_anticausal_ms > 0:
        # finestra solo sul lato anticausale: il lato causale resta illimitato
        wiener_kwargs.update(time_window=True, pre_ms=args.max_anticausal_ms,
                             post_ms=1e5, post_window_cap_ms=1e5)
    hop_stft = max(1, int(args.stft_n_fft * (1.0 - args.stft_overlap)))
    safe_stem = in_path.stem.replace("-", "_")

    dry_stat, dbg_stat = dereverb_wiener(audio=x_rs, rir_proc=rir_proc, fs=fs, **wiener_kwargs)
    dry_stat = postprocess(dry_stat, x_rs, fs, args.shelf_db)
    path_stat = out_stat / f"{in_path.stem}_wiener_dereverb.wav"
    save_audio(path_stat, dry_stat, fs, float_out=args.float_out)
    print(f"  [wiener]      → {path_stat.name}")
    if args.plots:
        save_analysis_plots(plot_stat / f"Analisi_{safe_stem}.png", x_rs, dry_stat, fs, dbg_stat)
    if args.gate:
        apply_gate_and_save(dry_stat, fs, in_path, out_stat_gated, args, "wiener")

    dry_stft, dbg_stft = dereverb_wiener_stft(audio=x_rs, rir_proc=rir_proc, fs=fs,
                                               n_fft_stft=args.stft_n_fft, hop=hop_stft,
                                               **wiener_kwargs)
    dry_stft = postprocess(dry_stft, x_rs, fs, args.shelf_db)
    path_stft = out_stft / f"{in_path.stem}_wiener_stft_dereverb.wav"
    save_audio(path_stft, dry_stft, fs, float_out=args.float_out)
    print(f"  [wiener_stft] → {path_stft.name}")
    if args.plots:
        save_analysis_plots(plot_stft / f"Analisi_{safe_stem}.png", x_rs, dry_stft, fs, dbg_stft)
    if args.gate:
        apply_gate_and_save(dry_stft, fs, in_path, out_stft_gated, args, "wiener_stft")


# ============================
# MAIN
# ============================

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # --- le uniche due domande: se non passate da CLI, si chiedono a menu ---
    if args.input and args.rir:
        input_path = Path(clean_path(args.input))
        rir_path   = Path(clean_path(args.rir))
        if not input_path.exists():
            raise RuntimeError(f"Percorso di input non trovato: {input_path}")
        if not rir_path.exists():
            raise RuntimeError(f"RIR non trovata: {rir_path}")
    else:
        input_path, rir_path = run_interactive_menu()

    in_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.wav"))
    if not in_files:
        raise RuntimeError(f"Nessun file .wav trovato in: {input_path}")

    print(f"\n[INFO] RIR:              {rir_path}")
    print(f"[INFO] File trovati:     {len(in_files)}")
    print(f"[INFO] Sincronizzazione: {'attiva (--sync)' if (args.sync and not args.no_sync) else 'disattiva'}\n")

    rir_x, sr_rir = load_audio(rir_path)
    # Stessa funzione usata da Conv.py: la RIR contenuta nel wet e quella
    # usata dal filtro devono essere ridotte a mono nello stesso modo.
    if rir_x.ndim > 1 and rir_x.shape[1] > 1:
        print(f"[INFO] RIR multicanale ({rir_x.shape[1]} ch), riduzione a mono: {args.rir_mono}")
    rir_mono = rir_to_mono(rir_x, args.rir_mono)

    out_stat = Path(args.out_dir) / "wiener"
    out_stft = Path(args.out_dir) / "wiener_stft"
    out_stat.mkdir(parents=True, exist_ok=True)
    out_stft.mkdir(parents=True, exist_ok=True)

    plot_stat = plot_stft = None
    if args.plots:
        plot_stat = Path(args.plot_dir) / "wiener"
        plot_stft = Path(args.plot_dir) / "wiener_stft"
        plot_stat.mkdir(parents=True, exist_ok=True)
        plot_stft.mkdir(parents=True, exist_ok=True)

    out_stat_gated = out_stft_gated = None
    if args.gate:
        out_stat_gated = Path(args.out_dir) / "wiener_gated"
        out_stft_gated = Path(args.out_dir) / "wiener_stft_gated"
        out_stat_gated.mkdir(parents=True, exist_ok=True)
        out_stft_gated.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Gate attivo: output aggiuntivo in {out_stat_gated} e {out_stft_gated}\n")

    for in_path in in_files:
        process_one(in_path, rir_mono, sr_rir, out_stat, out_stft, args, plot_stat, plot_stft,
                    out_stat_gated, out_stft_gated)

    print("\nFine.")


if __name__ == "__main__":
    main()
