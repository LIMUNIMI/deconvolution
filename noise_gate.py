"""
noise_gate.py
==============
Gate/espansore verso il basso ("downward expander") per ridurre il
rumore di fondo percepito nei tratti silenziosi/di decadimento
dell'output de-riverberato — senza toccare i passaggi dove c'e'
segnale reale.


Questo modulo non "toglie" rumore in senso stretto — attenua
automaticamente il segnale quando il suo livello scende sotto una
soglia (quindi nei tratti silenziosi/decadimento), lasciando intatti i
passaggi con segnale reale. E' una tecnica standard in produzione audio
(gate/espansore), qui applicata come post-processing.
Parametri chiave
-----------------
threshold_db : sotto questo livello (relativo al picco del file) il
    gate inizia a chiudersi.
range_db     : attenuazione MASSIMA applicata quando il gate e'
    completamente chiuso (non azzera mai il segnale — un'attenuazione
    totale suonerebbe innaturale/a scatti).
knee_db      : larghezza della transizione morbida attorno alla soglia,
    per evitare un gate "a scatto" udibile.
attack_ms    : quanto velocemente il gate si apre quando il segnale
    supera la soglia (deve essere rapido, per non "mangiare" gli attacchi).
release_ms   : quanto velocemente il gate si richiude quando il segnale
    scende sotto la soglia (deve essere abbastanza lento da non troncare
    in modo innaturale il decadimento naturale delle note).
"""

from __future__ import annotations

import numpy as np


def _envelope_rms(x: np.ndarray, fs: int, win_ms: float = 10.0) -> np.ndarray:
    """Inviluppo RMS mobile (finestra rettangolare) via convoluzione."""
    win = max(1, int(win_ms / 1000.0 * fs))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    env = np.sqrt(np.convolve(x2, kernel, mode="same") + 1e-20)
    return env


def _smooth_gain_attack_release(gain_db: np.ndarray, fs: int,
                                 attack_ms: float, release_ms: float) -> np.ndarray:
    """
    Smoothing temporale asimmetrico (attack/release) del guadagno in dB.
    """
    N = len(gain_db)
    coeff_attack = np.exp(-1.0 / (max(attack_ms, 0.01) / 1000.0 * fs))
    coeff_release = np.exp(-1.0 / (max(release_ms, 0.01) / 1000.0 * fs))

    smoothed = np.empty(N)
    g_prev = gain_db[0]
    for i in range(N):
        target = gain_db[i]
        if target > g_prev:
            g_prev = coeff_attack * g_prev + (1 - coeff_attack) * target
        else:
            g_prev = coeff_release * g_prev + (1 - coeff_release) * target
        smoothed[i] = g_prev
    return smoothed


def _gate_curve(env_db: np.ndarray, threshold_db: float,
                 range_db: float, knee_db: float) -> np.ndarray:
    """Curva statica del gate (prima dello smoothing temporale), soft-knee."""
    lo = threshold_db - knee_db / 2.0
    hi = threshold_db + knee_db / 2.0

    gain_db = np.zeros_like(env_db)
    below = env_db < lo
    above = env_db >= hi
    knee = ~(below | above)

    gain_db[below] = -range_db
    gain_db[above] = 0.0
    if np.any(knee):
        frac = (env_db[knee] - lo) / max(knee_db, 1e-6)
        # Transizione a coseno
        smooth_frac = 0.5 - 0.5 * np.cos(np.pi * frac)
        gain_db[knee] = -range_db * (1.0 - smooth_frac)
    return gain_db


def apply_noise_gate(
    audio: np.ndarray,
    fs: int,
    threshold_db: float = -45.0,
    range_db: float = 18.0,
    knee_db: float = 6.0,
    attack_ms: float = 3.0,
    release_ms: float = 300.0,
    env_win_ms: float = 10.0,
    return_gain: bool = False,
):
    """
    Applica il gate/espansore a un segnale mono [N] o multicanale [N, C].
    threshold_db e' relativo al picco del segnale (non dBFS assoluto),
    cosi' i parametri si comportano in modo simile su file con livelli
    diversi.

    Se return_gain=True, ritorna anche la curva di guadagno applicata
    (in dB, canale 0).
    """
    x = np.asarray(audio, dtype=np.float64)
    single_channel = x.ndim == 1
    if single_channel:
        x = x[:, None]

    N, C = x.shape
    peak = np.max(np.abs(x)) + 1e-12
    out = np.zeros_like(x)
    gain_db_ch0 = None

    for c in range(C):
        env = _envelope_rms(x[:, c], fs, env_win_ms)
        env_db = 20 * np.log10(env / peak + 1e-12)

        gain_db_static = _gate_curve(env_db, threshold_db, range_db, knee_db)
        gain_db_smooth = _smooth_gain_attack_release(gain_db_static, fs, attack_ms, release_ms)

        if c == 0:
            gain_db_ch0 = gain_db_smooth

        gain_lin = 10 ** (gain_db_smooth / 20.0)
        out[:, c] = x[:, c] * gain_lin

    result = out[:, 0] if single_channel else out
    if return_gain:
        return result, gain_db_ch0
    return result
