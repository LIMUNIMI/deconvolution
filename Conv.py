import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from math import gcd


# === CARTELLE ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANECHOIC_DIR = os.path.join(BASE_DIR, "anechoic")
RIR_DIR = os.path.join(BASE_DIR, "rir")
OUTPUT_DIR = os.path.join(BASE_DIR, "reverberated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Frequenza di campionamento unica per TUTTI i file in uscita, indipendente
# dalla frequenza nativa dei singoli file dry/RIR di partenza (Cap.4 della
# tesi assume un dataset uniformemente a 48kHz).
TARGET_SR = 48000


def to_mono_dry(x):
    """Per i segnali anecoici (dry): media dei canali se multicanale.
    Adatto qui perche' i dry non sono Ambisonics, un'eventuale media
    tra due canali stereo non pone gli stessi problemi di fase
    distruttiva di un B-format a 9 canali (vedi to_mono_rir)."""
    if len(x.shape) > 1:
        return np.mean(x, axis=1)
    return x


def to_mono_rir(h):
    """Per le RIR Ambisonics B-format (multicanale): media dei canali.

    NOTA METODOLOGICA: la media aritmetica dei canali Ambisonics non e'
    tecnicamente il modo piu' rigoroso di ridurre un B-format a un
    singolo canale (i canali rappresentano componenti spaziali diverse,
    con possibili cancellazioni di fase tra loro). E' stata comunque
    preferita alla selezione del canale di ampiezza massima, verificata
    sperimentalmente, perche' quest'ultima produceva una colorazione
    metallica udibile sull'output de-riverberato (probabilmente perche'
    la media, mediando componenti in controfase, attenua naturalmente
    le risonanze piu' nette della stanza, che il canale singolo lascia
    invece intatte e piu' problematiche da invertire per il filtro di
    Wiener). Questa scelta e la sua motivazione sono documentate nel
    Capitolo 5 della tesi come limite noto della costruzione del
    dataset.

    IMPORTANTE: main.py deve usare la STESSA convenzione (media, non
    canale di ampiezza massima) quando carica la RIR per progettare il
    filtro, altrimenti la RIR "dentro" il wet e quella usata dal filtro
    tornano ad essere disallineate.
    """
    if len(h.shape) > 1:
        return np.mean(h, axis=1)
    return h


def normalize(x):
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * 0.9
    return x


def resample_audio(x, sr_in, sr_out):
    """Resampling con resample_poly"""
    if sr_in == sr_out:
        return x

    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g

    x_resampled = resample_poly(x, up, down)
    return x_resampled


def process():
    for audio_file in os.listdir(ANECHOIC_DIR):
        if not audio_file.endswith(".wav"):
            continue

        x, sr_x = sf.read(os.path.join(ANECHOIC_DIR, audio_file))
        x = to_mono_dry(x)

        # === RESAMPLING DRY -> TARGET_SR (sempre, indipendentemente
        #     dalla frequenza nativa del file) ===
        if sr_x != TARGET_SR:
            print(f"Resampling dry {audio_file}: {sr_x} → {TARGET_SR}")
            x = resample_audio(x, sr_x, TARGET_SR)

        for rir_file in os.listdir(RIR_DIR):
            if not rir_file.endswith(".wav"):
                continue

            h, sr_h = sf.read(os.path.join(RIR_DIR, rir_file))
            h = to_mono_rir(h)

            # === RESAMPLING RIR -> TARGET_SR (sempre) ===
            if sr_h != TARGET_SR:
                print(f"Resampling RIR {rir_file}: {sr_h} → {TARGET_SR}")
                h = resample_audio(h, sr_h, TARGET_SR)

            # normalizzazione RIR
            h = h / np.sqrt(np.sum(h**2))

            # allineamento direct path
            h = h[np.argmax(np.abs(h)):]

            # convoluzione
            y = fftconvolve(x, h, mode="full")

            # normalizzazione output
            y = normalize(y)

            out_name = f"{audio_file[:-4]}_{rir_file[:-4]}.wav"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            sf.write(out_path, y, TARGET_SR)

            print("Creato:", out_name)


if __name__ == "__main__":
    process()