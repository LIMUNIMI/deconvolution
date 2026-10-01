from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey


# ============================
# RIDUZIONE A MONO DELLA RIR (convenzione condivisa)
# ============================

def rir_to_mono(rir: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Riduce una RIR multicanale a un solo canale.

    Questa funzione e' usata SIA da Conv.py (generazione del wet) SIA da
    main.py (costruzione del filtro): in questo modo la RIR contenuta nel
    segnale e quella usata dal filtro sono sempre ridotte nello stesso modo.

    mode = "mean": media aritmetica dei canali (convenzione usata finora).
    mode = "w"   : solo il canale 0. Nel formato AmbiX (ordinamento ACN) il
                   canale 0 e' W, la componente omnidirezionale.
    """
    rir = np.asarray(rir, dtype=np.float64)
    if rir.ndim == 1 or rir.shape[1] == 1:
        return rir.reshape(-1)
    if mode == "mean":
        return np.mean(rir, axis=1)
    if mode == "w":
        return rir[:, 0]
    raise ValueError(f"rir_mono non valido: {mode!r} (usa 'mean' o 'w')")


# ============================
# PRE-PROCESSING DELLA RIR
# ============================

def prepare_rir(
    rir: np.ndarray,
    fs: int,
    trunc_s: float = 0.8,
    tukey_alpha: float = 0.25,
    make_causal: bool = True,
    pre_peak_ms: float = 4.0
) -> tuple[np.ndarray, int]:
    """

    Prepara una RIR reale per l'inversione:

    1) DC removal + normalizzazione
    2) Allineamento al direct-path (picco massimo)
    3) (Opzionale) causalizzazione: taglia la parte prima del direct-path (riduce pre-echo)
    4) Troncamento della coda (stabilità numerica e percettiva)
    5) Fade-out (meta' di finestra di Tukey) SOLO sulla coda, per evitare
       il taglio netto senza attenuare il suono diretto
    """
    rir = np.asarray(rir, dtype=np.float64).copy()
    rir = rir - np.mean(rir)
    rir = rir / (np.max(np.abs(rir)) + 1e-12)

    peak_idx = int(np.argmax(np.abs(rir)))  # fallback

    if make_causal:
        #   conserva pre_peak_ms ms prima del picco per includere l'onset
        #   del suono diretto e avere una transizione graduale
        pre_peak_samp = int(pre_peak_ms / 1000.0 * fs)
        start = max(0, peak_idx - pre_peak_samp)
        rir = rir[start:]
        # aggiorna peak_idx in base al nuovo inizio
        peak_idx = peak_idx - start

    L = int(max(1, trunc_s * fs))
    if len(rir) < L:
        rir = np.pad(rir, (0, L - len(rir)))
    else:
        rir = rir[:L]

    w = np.ones(len(rir))
    n_fade = int(len(rir) * tukey_alpha / 2)
    if n_fade > 1:
        w[-n_fade:] = tukey(2 * n_fade, alpha=1.0)[n_fade:]
    rir_proc = rir * w

    return rir_proc, peak_idx


def drr_from_rir_db(rir_proc: np.ndarray, fs: int, early_ms: float = 100.0) -> float:
    """
    DRR calcolata sulla RIR processata: rapporto energia early/late (in dB).
    """
    rir_proc = np.asarray(rir_proc, dtype=np.float64)
    early = int((early_ms / 1000.0) * fs)
    early = max(1, min(early, len(rir_proc) - 1))

    e_early = float(np.sum(rir_proc[:early] ** 2))
    e_late  = float(np.sum(rir_proc[early:] ** 2) + 1e-12)
    return 10.0 * np.log10(e_early / e_late)

def estimate_t60(rir: np.ndarray, fs: int) -> float:
    """
    Stima T60 via integrale di Schroeder (metodo T20 estrapolato).
    """
    h2 = rir ** 2
    peak_idx = int(np.argmax(np.abs(rir)))
    edc = np.cumsum(h2[peak_idx:][::-1])[::-1]  # energia dalla coda
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)

    # Usa il range -5dB / -25dB (T20) per robustezza, poi estrapola a T60
    fs_arr = np.arange(len(edc_db)) / fs
    mask = (edc_db < -5) & (edc_db > -25)
    if mask.sum() < 10:
        # fallback: trova dove scende sotto -20dB e moltiplica
        below = np.where(edc_db < -20)[0]
        return float(below[0] / fs * 3) if len(below) else 1.0

    coeffs = np.polyfit(fs_arr[mask], edc_db[mask], 1)  # fit lineare
    # slope in dB/s → T60 = -60 / slope
    slope = coeffs[0]
    t60 = -60.0 / slope if slope < -0.1 else 2.0
    return float(np.clip(t60, 0.05, 10.0))


def estimate_drr(rir: np.ndarray, fs: int,
                 direct_ms: float = 5.0, reverb_onset_ms: float = 50.0) -> float:
    """
    Stima DRR: energia della finestra diretta vs energia della coda.
    """
    peak_idx = int(np.argmax(np.abs(rir)))
    direct_end = peak_idx + int(direct_ms / 1000.0 * fs)
    reverb_start = peak_idx + int(reverb_onset_ms / 1000.0 * fs)

    e_direct = np.sum(rir[peak_idx:direct_end] ** 2) + 1e-30
    e_reverb = np.sum(rir[reverb_start:] ** 2) + 1e-30
    return float(10.0 * np.log10(e_direct / e_reverb))


def find_direct_path_offset(wet: np.ndarray, rir: np.ndarray, fs: int,
                             search_ms: float = 100.0) -> int:
    """
    Trova l'offset del suono diretto tramite cross-correlazione tra
    il segnale wet e la RIR.
    
    Cerca il picco solo nei primi search_ms millisecondi per evitare
    di trovare riflessioni tardive.
    """
    wet = np.asarray(wet, dtype=np.float64)
    rir = np.asarray(rir, dtype=np.float64)
    
    # Usa solo i primi search_ms ms della RIR come template
    search_samp = int(search_ms / 1000.0 * fs)
    rir_head    = rir[:search_samp]
    
    # Cross-correlazione tra wet e testa della RIR
    cc  = np.correlate(wet[:min(len(wet), search_samp * 4)], rir_head, mode='full')
    lag = np.argmax(np.abs(cc)) - (len(rir_head) - 1)
    
    return max(0, int(lag))


def auto_wiener_params(rir_raw: np.ndarray, fs: int) -> dict:
    t60 = estimate_t60(rir_raw, fs)
    drr = estimate_drr(rir_raw, fs)

    print(f"[AUTO] T60 stimato: {t60*1000:.0f} ms | DRR stimato: {drr:+.1f} dB")

   
    trunc_s = float(np.clip(t60 / 3.0, 0.2, 2.0))

    
    drr_factor = np.clip(1.0 + (-drr) / 10.0, 1.0, 4.0)   # >1 quando DRR < 0
    t60_factor  = np.clip(t60 / 0.5, 1.0, 3.0)              # >1 quando T60 > 0.5s
    beta_rel = float(np.clip(0.05 * drr_factor * t60_factor, 0.03, 0.5))

    
    post_ms = float(np.clip(trunc_s * 3000, 200, 1200))

    return {
        "trunc_s":  trunc_s,
        "beta_rel": beta_rel,
        "post_ms":  post_ms,
        "t60_s":    t60,
        "drr_db":   drr,
    }
