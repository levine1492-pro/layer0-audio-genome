"""Axis 1: Signal decomposition (Stereo M/S, HPSS, and CPU htdemucs 4-stem split)."""
import os
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


def _bandpass(y: np.ndarray, sr: int, low: float = None, high: float = None) -> np.ndarray:
    nyq = 0.5 * sr
    if low and high:
        sos = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="band", output="sos")
    elif low:
        sos = butter(4, low / nyq, btype="high", output="sos")
    else:
        sos = butter(4, min(high / nyq, 0.99), btype="low", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _rms_db_1s(y: np.ndarray, sr: int) -> list:
    n_sec = max(1, int(np.floor(len(y) / sr)))
    vals = []
    for i in range(n_sec):
        seg = y[i * sr:(i + 1) * sr]
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        vals.append(round(20.0 * np.log10(rms + 1e-12), 2))
    return vals


def decompose_signal(stereo: np.ndarray, sr: int, use_demucs: bool = True) -> dict:
    """Decompose stereo (2, N) into M/S, HPSS, and 4 stems (drums/bass/vocals/other)."""
    left, right = stereo[0], stereo[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mono = mid.copy()

    harm_mix, perc_mix = librosa.effects.hpss(mono, margin=2.0)

    stems_mono = {}
    stems_stereo = {}
    method = "fast_dsp_fallback"

    if use_demucs:
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            torch.set_num_threads(max(1, os.cpu_count() or 4))
            model = get_model("htdemucs")
            model.eval()

            wav = torch.from_numpy(stereo.astype(np.float32))
            ref = wav.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std() + 1e-8
            wav_norm = (wav - ref_mean) / ref_std

            with torch.no_grad():
                sources = apply_model(
                    model, wav_norm[None],
                    shifts=1, split=True, overlap=0.25,
                    progress=False, device="cpu"
                )[0]
            sources = (sources * ref_std + ref_mean).cpu().numpy()

            for idx, name in enumerate(model.sources):
                st = sources[idx].astype(np.float32)
                stems_stereo[name] = st
                stems_mono[name] = (0.5 * (st[0] + st[1])).astype(np.float32)
            method = "htdemucs_4stem"
        except Exception as exc:
            print(f"[WARN] Demucs failed ({exc}); falling back to DSP split.")

    if not stems_mono:
        stems_mono = {
            "drums": perc_mix.astype(np.float32),
            "bass": _bandpass(harm_mix, sr, high=240.0),
            "vocals": _bandpass(harm_mix, sr, low=220.0, high=3200.0),
            "other": _bandpass(harm_mix, sr, low=140.0),
        }
        for k, v in stems_mono.items():
            stems_stereo[k] = np.stack([v, v], axis=0)

    # Unified lead/melodic carrier (140 Hz - 5.5 kHz captures cello/baritone/harmonica/guitar/vocals)
    other_harm, _ = librosa.effects.hpss(stems_mono["other"], margin=1.5)
    lead_raw = stems_mono["vocals"] + other_harm
    lead_carrier = _bandpass(lead_raw, sr, low=140.0, high=5500.0)

    # Stereo harmonic bed (other + vocals) for split-space ambience forensics
    harmonic_bed_stereo = stems_stereo["other"] + stems_stereo["vocals"]

    stem_db_1s = {k: _rms_db_1s(v, sr) for k, v in stems_mono.items()}
    stem_rms_global_db = {
        k: round(20.0 * np.log10(float(np.sqrt(np.mean(v ** 2) + 1e-12))), 2)
        for k, v in stems_mono.items()
    }

    return {
        "method": method,
        "stereo": stereo,
        "mono": mono,
        "mid": mid,
        "side": side,
        "harmonic_mix": harm_mix.astype(np.float32),
        "percussive_mix": perc_mix.astype(np.float32),
        "stems_mono": stems_mono,
        "stems_stereo": stems_stereo,
        "lead_carrier": lead_carrier,
        "harmonic_bed_stereo": harmonic_bed_stereo,
        "stem_db_1s": stem_db_1s,
        "stem_rms_global_db": stem_rms_global_db,
    }
