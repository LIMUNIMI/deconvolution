# Single-channel dereverberation with Wiener filtering

Dereverberation system for monophonic musical signals, based on a
regularized Wiener filter with a known room impulse response (RIR).
Implements two filter variants (stationary and short-time, STFT) and an
optional post-processing stage (noise gate) to reduce residual background
noise in silent/decay portions of the signal.

Developed as part of a bachelor's thesis in Music Informatics.

## What it does

- **Dereverberates** an audio file (or a folder of files) given a single
  reference RIR, with automatic parameter estimation based on the
  reverberation time ($T_{60}$) and direct-to-reverberant ratio ($DRR$) of
  the RIR itself.
- Provides **two filter variants**: stationary (a single filter for the
  whole signal) and short-time (STFT, a filter applied frame by frame).
- Includes an **optional noise gate**, to attenuate residual background
  noise in silent/decaying portions without affecting sections with real
  signal.
- Provides a separate script (`conv.py`) to build your own test dataset
  with known ground truth, by convolving an anechoic signal with one or
  more RIRs.

## Installation

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## Usage

### Dereverberation (basic, interactive use)

```bash
python main.py
```

The script asks two things: the path to the audio file or folder to
process, and the path to the RIR of the environment. Everything else
(parameter estimation, temporal synchronization, post-processing) happens
automatically with sensible defaults.

### Dereverberation (advanced, command-line use)

```bash
python main.py -i input.wav -r rir.wav --out_dir output --gate
```

Main flags:

| Flag | Meaning |
|---|---|
| `-i`, `--input` | Audio file or folder to process |
| `-r`, `--rir` | RIR file to use |
| `--out_dir` | Output folder |
| `--gate` | Also apply the noise gate, in addition to the non-gated output |
| `--no_sync` | Disable automatic wet/RIR synchronization — use only if the file is already temporally coherent with the RIR (e.g. generated with `conv.py`) |
| `--no_auto` | Disable automatic parameter estimation from the RIR |

Run `python main.py --help` for the full list of parameters (regularization,
temporal windows, STFT parameters, gate parameters).

### Building a test dataset with ground truth

```bash
python conv.py -d anechoic.wav -r rir.wav -o wet.wav
```

Or, to convolve with multiple RIRs at once (useful for testing several
environments):

```bash
python conv.py -d anechoic.wav -r rir_folder/ -o output_folder/
```

The generated signal is already temporally coherent with the RIR used:
process it afterwards with `python main.py --no_sync`.

### Applying the noise gate afterwards

If you already have dereverberated files and want to apply (or re-apply
with different parameters) only the noise gate, without re-running the
whole dereverberation pipeline:

```bash
python apply_noise_gate.py --in_dir input_folder --out_dir output_folder
```

Add `--plots` to generate diagnostic plots (waveform, spectrograms, the
gain curve applied over time).

## Project structure

```
main.py               Entry point: dereverberation (stationary + STFT)
conv.py             Generates test signals via convolution
noise_gate.py           Noise gate / downward expander library
apply_noise_gate.py     Applies the gate in batch to already dereverberated files
io_utils.py             Support functions for audio reading/writing
rir_utils.py            RIR pre-processing, T60/DRR estimation, automatic parameters
wiener_dereverb.py      Stationary Wiener filter
wiener_stft.py          Wiener filter in the STFT domain
plot_utils.py           Diagnostic plot generation
requirements.txt        Python dependencies
```

## How it works (in brief)

The filter is designed in the frequency domain as:

```
G(f) = H*(f) / (|H(f)|² + β(f))
```

where `H(f)` is the Fourier transform of the RIR and `β(f)` a
regularization term, necessary because real RIRs are generally non-minimum
phase: their exact inverse cannot be realized in a stable, causal form. The
regularization parameters, RIR truncation duration, and the extent of the
filter's temporal window are estimated automatically from the RIR's own
estimated $T_{60}$ and $DRR$.

## Known limitations

- Single-channel method: does not exploit any multichannel information
  available in the RIR (e.g. Ambisonics formats).
- As with any Wiener filter using a known RIR, inversion of the late
  reverberant tail is structurally limited by the regularization required
  for filter stability — it is more effective on early reflections than on
  the diffuse late tail.
- Performance degrades in highly reverberant environments (long T60, very
  negative DRR).

## License

<!-- To be defined -->