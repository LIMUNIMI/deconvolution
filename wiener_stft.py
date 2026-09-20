"""
wiener_stft.py
==============
Implementazione del filtro di Wiener nel dominio STFT (frame per frame).

Differenza rispetto al Wiener stazionario (wiener_dereverb.py)
---------------------------------------------------------------
Il Wiener stazionario costruisce un unico filtro G(f) dalla RIR e lo
applica alla FFT dell'intero segnale:

    Y_out(f) = FFT(y_wet, N_totale) · G(f)

Il Wiener STFT divide il segnale in frame sovrapposti di durata breve
(default 42 ms), applica G a ciascun frame, e ricostruisce il segnale
via overlap-add (WOLA - Weighted Overlap-Add):

    per ogni frame k:
        Y_k(f) = FFT(y_wet[k·hop : k·hop + n_fft] · win_hann) · G(f)
    y_out = overlap_add(IFFT(Y_k) · win_hann) / sum(win_hann²)

Il filtro G(f) è lo stesso per tutti i frame (calcolato una sola volta
dalla RIR). Il miglioramento rispetto al caso stazionario non viene da
un G adattivo, ma dalla finestra temporale corta: un frame con una
transiente percussiva ha uno spettro diverso da un frame con una nota
sostenuta, e G agisce su quello spettro locale invece di mediare
sull'intero segnale.

Parametri STFT scelti
---------------------
n_fft = 2048 campioni = 42.7 ms a 48 kHz
    Risoluzione in frequenza: 23.4 Hz/bin.

hop = n_fft // 4 = 512 campioni = 10.7 ms (overlap 75%)
    Con finestra Hann soddisfa la condizione COLA (Constant Overlap-Add),
    garantendo ricostruzione perfetta senza artefatti ai bordi dei frame.

Finestra: Hann
    Lobi laterali a -31.5 dB. Compatibile con COLA a overlap 75%.
    Applicata sia in analisi che in sintesi (WOLA standard).
    La normalizzazione per sum(win²) compensa la doppia finestrazione.
"""

from __future__ import annotations

import numpy as np
from wiener_dereverb import build_wiener_filter, to_2d


def dereverb_wiener_stft(
    audio: np.ndarray,
    rir_proc: np.ndarray,
    fs: int,
    n_fft_stft: int = 2048,
    hop: int | None = None,
    **wiener_kwargs,
) -> tuple[np.ndarray, dict]:
    """
    Wiener STFT: applica il filtro G(f) frame per frame con WOLA.

    Parametri
    ---------
    audio       : segnale wet [N] o [N, C], float64
    rir_proc    : RIR pre-processata da prepare_rir
    fs          : sample rate in Hz
    n_fft_stft  : dimensione FFT per frame (default 2048 = 42.7ms a 48kHz)
    hop         : campioni tra frame (default n_fft_stft//4 = 75% overlap)
    **wiener_kwargs : parametri passati a build_wiener_filter

    Restituisce
    -----------
    (y, dbg) — y: segnale deRiverberato [N, C]; dbg: dizionario debug
    """
    x    = to_2d(np.asarray(audio, dtype=np.float64))
    N, C = x.shape

    if hop is None:
        hop = n_fft_stft // 4   # overlap 75%

    # Costruisce G con n_fft = n_fft_stft.
    # Il delay_ms viene mantenuto uguale al Wiener stazionario:
    # G non è causale (ha componenti prima di t=0) e il delay compensa
    # questa non-causalità anche applicato frame per frame.
    stft_kwargs = dict(wiener_kwargs)
    stft_kwargs["adaptive_beta"] = True

    dbg = build_wiener_filter(
        rir_proc = rir_proc,
        fs       = fs,
        n_fft    = n_fft_stft,
        **stft_kwargs,
    )
    G          = dbg["G"]           # shape: (n_fft_stft//2 + 1,)
    delay_samp = dbg["delay_samp"]

    # Finestra Hann — usata sia in analisi che in sintesi (WOLA)
    win = np.hanning(n_fft_stft)

    y = np.zeros((N + n_fft_stft, C))

    for c in range(C):
        x_pad    = np.pad(x[:, c], (0, n_fft_stft))
        out      = np.zeros(N + n_fft_stft)
        norm_acc = np.zeros(N + n_fft_stft)

        for start in range(0, N, hop):
            end   = start + n_fft_stft
            frame = x_pad[start:end]
            if len(frame) < n_fft_stft:
                frame = np.pad(frame, (0, n_fft_stft - len(frame)))

            # Analisi: finestra Hann + FFT
            Frame_f   = np.fft.rfft(frame * win)

            # Filtro di Wiener nel dominio della frequenza
            Out_f     = Frame_f * G

            # Sintesi: IFFT + finestra Hann (WOLA)
            out_frame = np.fft.irfft(Out_f, n=n_fft_stft)

            # Overlap-add con doppia finestrazione Hann
            # norm_acc accumula win² per normalizzare correttamente (WOLA)
            out[start:end]      += out_frame * win
            norm_acc[start:end] += win ** 2

        # Normalizzazione WOLA: con Hann e 75% overlap sum(win²) ≈ 1.5 costante
        norm_acc = np.maximum(norm_acc, 1e-8)
        out     /= norm_acc

        # Compensa il delay introdotto da G e ritaglia a N campioni
        start_crop = delay_samp
        end_crop   = start_crop + N
        if end_crop > len(out):
            out = np.pad(out, (0, end_crop - len(out)))
        y[:N, c] = out[start_crop:start_crop + N]

    return y[:N, :], dbg