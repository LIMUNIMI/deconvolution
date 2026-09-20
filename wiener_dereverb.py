from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def apply_low_shelf_boost(audio: np.ndarray, fs: int,
                          f0: float = 350.0, gain_db: float = 10.0) -> np.ndarray:
    """Low-shelf EQ corretto: boost sotto f0, neutro sopra."""
    from scipy.signal import sosfilt
    
    A  = 10 ** (gain_db / 40.0)          # ampiezza (non potenza)
    w0 = 2 * np.pi * f0 / fs
    S  = 1.0                              # slope = 1 (standard)
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)

    b0 =  A * ((A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b1 =  2*A * ((A-1) - (A+1)*np.cos(w0))
    b2 =  A * ((A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a1 = -2 * ((A-1) + (A+1)*np.cos(w0))
    a2 =       (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha

    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    out = sosfilt(sos, audio, axis=0)

    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def build_wiener_filter(
    rir_proc: np.ndarray,
    fs: int,
    n_fft: int,
    beta_rel: float = 0.1,
    hf_tilt: float = 6.0,
    hf_fc: float = 7000.0,
    beta_max: float = 1e2,
    delay_ms: float = 8.0,
    f_lo: float = 80.0,
    f_hi: float = 15000.0,
    gmax_db: float = 15.0,
    pre_ms: float = 10.0,
    post_ms: float = 400,
    adaptive_beta: bool = False,
    post_window_cap_ms: float = 2000.0,
):
    
    rir_proc = np.asarray(rir_proc, dtype=np.float64)
    H = np.fft.rfft(rir_proc, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0/fs)
    H2 = np.abs(H)**2

    base = float(np.mean(H2) + 1e-12)
    beta0 = beta_rel * base

    if adaptive_beta:
        # Beta adattiva per banda: dove H² è alto → β più alto → G più conservativo
        # Riduce il guadagno anomalo nelle bande ben rappresentate dalla RIR (es. mid)
        # senza penalizzare le bande con H² basso (basse freq) dove β era già alto
        H2_norm = H2 / (np.mean(H2) + 1e-30)
        alpha   = 2.0
        beta_f  = beta0 * (1.0 + alpha * H2_norm)
        beta_f  = np.clip(beta_f, beta0 * 0.1, beta_max)
    else:
        # Beta flat (default) — comportamento originale
        fc     = max(float(hf_fc), 1.0)
        beta_f = beta0 * (1.0 + hf_tilt * (freqs / fc) ** 2)
        beta_f = np.minimum(beta_f, beta_max)

    G = np.conj(H) / (H2 + beta_f)

    transition = 80.0
    low  = 1.0 / (1.0 + np.exp(-(freqs - f_lo) / 80.0))    # stretta in basso
    high = 1.0 / (1.0 + np.exp( (freqs - f_hi) / 1500.0))  # morbida in alto
    G *= low * high

    gmax = 10 ** (gmax_db / 20.0)
    mag  = np.abs(G)
    G *= np.minimum(1.0, gmax / np.maximum(mag, 1e-12))

    delay_samp = int((delay_ms / 1000.0) * fs)
    G *= np.exp(-1j * 2.0 * np.pi * freqs * (delay_samp / fs))

    g = np.fft.irfft(G, n=n_fft)
    pk   = int(np.argmax(np.abs(g)))
    pre  = int(pre_ms * fs / 1000.0)
    post = int(min(post_ms, post_window_cap_ms) * fs / 1000.0)
    a, b = max(0, pk - pre), min(len(g), pk + post)
    win  = np.zeros_like(g)
    n_win = b - a
    if n_win > 4:
        t      = np.arange(n_win) - (pk - a)
        tau    = max(post / 6.0, 1.0)
        attack = np.where(t < 0,  np.exp( t / max(pre / 3.0, 1.0)), 1.0)
        decay  = np.where(t >= 0, np.exp(-t / tau), 1.0)
        win[a:b] = attack * decay
    G = np.fft.rfft(g * win, n=n_fft)

    return {
        "H": H,
        "G": G,
        "freqs": freqs,
        "beta0": beta0,
        "beta_f": beta_f,
        "delay_samp": delay_samp,
        "n_fft": n_fft,
        "params": {
            "beta_rel": beta_rel,
            "trunc_s": len(rir_proc) / fs,
            "post_ms": post_ms,
            "gmax_db": gmax_db,
        }
    }


def dereverb_wiener(audio: np.ndarray, rir_proc: np.ndarray, fs: int, **wiener_kwargs):
    x = to_2d(np.asarray(audio, dtype=np.float64))
    N, C = x.shape

    n_lin = N + len(rir_proc) - 1
    n_fft = next_pow_two(n_lin)

    dbg        = build_wiener_filter(rir_proc=rir_proc, fs=fs, n_fft=n_fft, **wiener_kwargs)
    G          = dbg["G"]
    delay_samp = dbg["delay_samp"]

    y = np.zeros_like(x)
    for c in range(C):
        Y      = np.fft.rfft(x[:, c], n=n_fft)
        x_full = np.fft.irfft(Y * G, n=n_fft)
        start  = delay_samp
        end    = start + N
        if end > len(x_full):
            x_full = np.pad(x_full, (0, end - len(x_full)))
        y[:, c] = x_full[start:end]

   
    return y, dbg


def next_pow_two(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x
