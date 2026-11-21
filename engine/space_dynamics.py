"""Axis 5: LUFS/LRA, Micro-Dynamics, Stereo Field & Split-Space Spatial Forensics."""
import numpy as np
import pyloudnorm as pyln
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly


def _band_filter(y: np.ndarray, sr: int, lo: float = None, hi: float = None) -> np.ndarray:
    nyq = 0.5 * sr
    if lo and hi:
        sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    elif lo:
        sos = butter(4, lo / nyq, btype="high", output="sos")
    else:
        sos = butter(4, min(hi / nyq, 0.99), btype="low", output="sos")
    return sosfiltfilt(sos, y)


def _crest_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    pk = float(np.max(np.abs(y)) + 1e-12)
    return round(20.0 * np.log10(pk / rms), 2)


def _stereo_width(left: np.ndarray, right: np.ndarray) -> float:
    m = 0.5 * (left + right)
    s = 0.5 * (left - right)
    rm = float(np.sqrt(np.mean(m ** 2) + 1e-12))
    rs = float(np.sqrt(np.mean(s ** 2) + 1e-12))
    return round(rs / (rm + rs), 3)


def _measure_decay_ms(y: np.ndarray, sr: int, max_win_s: float = 0.35) -> float:
    hop = 256
    rms_all = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms_all < 1e-4:
        return 120.0
    env = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    env_sr = sr / hop
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    pks = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop, wait=12)

    decay_ms = []
    win = int(max_win_s * env_sr)
    for p in pks:
        if p + win >= len(env):
            continue
        seg = env[p:p + win]
        pk = float(np.max(seg[:min(8, len(seg))]))
        if pk < 1e-4:
            continue
        below = np.where(seg < 0.316 * pk)[0]
        if len(below):
            decay_ms.append(float(below[0] / env_sr * 1000.0))
        else:
            decay_ms.append(max_win_s * 1000.0)
    return float(np.median(decay_ms)) if decay_ms else max_win_s * 1000.0


def _split_space_forensics(decomp: dict, sr: int, global_width: float) -> dict:
    """Separate Percussive Room Dryness from Harmonic/Melodic Ambience."""
    perc = decomp["percussive_mix"]
    harm_bed = decomp["harmonic_bed_stereo"]  # (2, N)
    harm_mono = 0.5 * (harm_bed[0] + harm_bed[1])

    perc_decay_ms = _measure_decay_ms(perc, sr, max_win_s=0.30)
    perc_dryness = float(np.clip(1.0 - (perc_decay_ms - 30.0) / 220.0, 0.0, 1.0))

    harm_decay_ms = _measure_decay_ms(harm_mono, sr, max_win_s=0.50)
    harm_width = _stereo_width(harm_bed[0], harm_bed[1])
    h_mid = 0.5 * (harm_bed[0] + harm_bed[1])
    h_side = 0.5 * (harm_bed[0] - harm_bed[1])
    h_side_mid = float(np.sqrt(np.mean(h_side ** 2) + 1e-12) / np.sqrt(np.mean(h_mid ** 2) + 1e-12))

    # Harmonic wetness blends long sustain decay + stereo ambient bloom
    harm_wetness = float(np.clip(0.65 * ((harm_decay_ms - 60.0) / 400.0) + 0.35 * (harm_width / 0.45), 0.0, 1.0))

    if global_width < 0.02:
        profile = "Pure Mono"
    elif perc_dryness >= 0.75 and (harm_wetness >= 0.48 or harm_width >= 0.28):
        profile = "Split-Space (Dry Drums + Ambient Bed)"
    elif perc_dryness >= 0.75 and harm_wetness < 0.48:
        profile = "Dry Intimate / Close-Mic"
    elif perc_dryness <= 0.45:
        profile = "Unified Ambient / Hall"
    else:
        profile = "Balanced Studio / Stage"

    s_rms = float(np.sqrt(np.mean(decomp["side"] ** 2) + 1e-12))
    m_rms = float(np.sqrt(np.mean(decomp["mid"] ** 2) + 1e-12))

    return {
        "space_profile": profile,
        "percussive_decay_10db_ms": round(perc_decay_ms, 1),
        "percussive_dryness": round(perc_dryness, 3),
        "harmonic_decay_10db_ms": round(harm_decay_ms, 1),
        "harmonic_stereo_width": harm_width,
        "harmonic_side_to_mid": round(h_side_mid, 3),
        "harmonic_wetness": round(harm_wetness, 3),
        "side_to_mid_ratio": round(s_rms / m_rms, 3),
    }


def analyze_space_dynamics(decomp: dict, sr: int) -> dict:
    stereo = decomp["stereo"]
    left, right = stereo[0], stereo[1]
    mono = decomp["mono"]
    total_sec = max(1, int(np.floor(len(mono) / sr)))

    meter = pyln.Meter(sr)
    audio_2d = stereo.T.astype(np.float64)
    if len(audio_2d) < sr:
        audio_2d = np.pad(audio_2d, ((0, sr - len(audio_2d)), (0, 0)))
    integrated_lufs = float(meter.integrated_loudness(audio_2d))

    st_lufs_1s = []
    win_n = int(3.0 * sr)
    for sec in range(total_sec):
        c = int((sec + 0.5) * sr)
        lo = max(0, c - win_n // 2)
        hi = min(len(audio_2d), lo + win_n)
        seg = audio_2d[lo:hi]
        if len(seg) < int(0.5 * sr):
            st_lufs_1s.append(-70.0)
            continue
        val = float(meter.integrated_loudness(seg))
        st_lufs_1s.append(round(val if np.isfinite(val) else -70.0, 2))

    st_arr = np.array(st_lufs_1s, dtype=np.float64)
    gate = max(-70.0, integrated_lufs - 20.0) if np.isfinite(integrated_lufs) else -70.0
    gated = st_arr[st_arr >= gate]
    lra_lu = float(np.percentile(gated, 95) - np.percentile(gated, 10)) if len(gated) >= 4 else 0.0

    os_l = resample_poly(left, 4, 1)
    os_r = resample_poly(right, 4, 1)
    true_peak = float(max(np.max(np.abs(os_l)), np.max(np.abs(os_r))) + 1e-12)
    true_peak_dbtp = 20.0 * np.log10(true_peak)

    near_clip = float(np.mean(np.abs(stereo) >= 0.985) * 100.0)
    stem_crest = {k: _crest_db(v) for k, v in decomp["stems_mono"].items()}

    width_global = _stereo_width(left, right)
    width_low = _stereo_width(_band_filter(left, sr, hi=250.0), _band_filter(right, sr, hi=250.0))
    width_mid = _stereo_width(_band_filter(left, sr, lo=250.0, hi=4000.0), _band_filter(right, sr, lo=250.0, hi=4000.0))
    width_high = _stereo_width(_band_filter(left, sr, lo=4000.0), _band_filter(right, sr, lo=4000.0))

    l_c = left - np.mean(left)
    r_c = right - np.mean(right)
    phase_corr = float(np.dot(l_c, r_c) / ((np.linalg.norm(l_c) * np.linalg.norm(r_c)) + 1e-12))
    lr_bal_db = 20.0 * np.log10(
        float(np.sqrt(np.mean(right ** 2) + 1e-12)) / float(np.sqrt(np.mean(left ** 2) + 1e-12))
    )

    width_1s = [_stereo_width(left[s * sr:min(len(left), (s + 1) * sr)], right[s * sr:min(len(right), (s + 1) * sr)]) for s in range(total_sec)]
    spatial = _split_space_forensics(decomp, sr, width_global)

    return {
        "summary": {
            "integrated_lufs": round(integrated_lufs, 2) if np.isfinite(integrated_lufs) else -70.0,
            "loudness_range_lra_lu": round(lra_lu, 2),
            "true_peak_dbtp": round(true_peak_dbtp, 2),
            "crest_factor_db": _crest_db(mono),
            "stem_crest_db": stem_crest,
            "clipping_ratio_pct": round(near_clip, 4),
            "stereo_width_global": width_global,
            "stereo_width_bands": {"low_lt250": width_low, "mid_250_4k": width_mid, "high_gt4k": width_high},
            "phase_correlation": round(phase_corr, 3),
            "lr_balance_db": round(float(lr_bal_db), 2),
            "reverb_proxy": spatial,
        },
        "timeseries": {
            "st_lufs": st_lufs_1s,
            "stereo_width": width_1s,
        },
    }
