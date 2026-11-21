"""Axis 5: ITU-R BS.1770 LUFS/LRA, Micro-Dynamics, Stereo Width & Reverb Decay Proxy."""
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


def _reverb_tail_proxy(perc: np.ndarray, mid: np.ndarray, side: np.ndarray, sr: int) -> dict:
    """Measure transient decay time inside percussive hits + Side/Mid tail ambience."""
    hop = 256
    env = librosa.feature.rms(y=perc, frame_length=1024, hop_length=hop)[0]
    env_sr = sr / hop
    o_env = librosa.onset.onset_strength(y=perc, sr=sr, hop_length=hop)
    pks = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop, wait=12)

    decay_times_ms = []
    win = int(0.30 * env_sr)
    for p in pks:
        if p + win >= len(env):
            continue
        seg = env[p:p + win]
        pk = float(np.max(seg[:min(8, len(seg))]))
        if pk < 1e-4:
            continue
        # Time for percussive energy to fall to -10 dB (0.316x amplitude)
        below = np.where(seg < 0.316 * pk)[0]
        if len(below):
            decay_times_ms.append(float(below[0] / env_sr * 1000.0))

    med_decay = float(np.median(decay_times_ms)) if decay_times_ms else 120.0
    # Dryness index: 1.0 = ultra-dry (<35ms decay), 0.0 = cathedral wash (>250ms)
    dryness = float(np.clip(1.0 - (med_decay - 30.0) / 220.0, 0.0, 1.0))

    # Ambient tail stereo ratio
    s_rms = float(np.sqrt(np.mean(side ** 2) + 1e-12))
    m_rms = float(np.sqrt(np.mean(mid ** 2) + 1e-12))
    return {
        "transient_decay_10db_ms": round(med_decay, 1),
        "dryness_index": round(dryness, 3),
        "side_to_mid_ratio": round(s_rms / m_rms, 3),
    }


def analyze_space_dynamics(decomp: dict, sr: int) -> dict:
    stereo = decomp["stereo"]  # (2, N)
    left, right = stereo[0], stereo[1]
    mono = decomp["mono"]
    total_sec = max(1, int(np.floor(len(mono) / sr)))

    # 1. ITU-R BS.1770 LUFS & Short-Term LUFS curve
    meter = pyln.Meter(sr)
    audio_2d = stereo.T.astype(np.float64)
    if len(audio_2d) < sr:
        audio_2d = np.pad(audio_2d, ((0, sr - len(audio_2d)), (0, 0)))
    integrated_lufs = float(meter.integrated_loudness(audio_2d))

    # 3-second sliding window short-term LUFS sampled every 1s
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

    # Loudness Range (LRA: P95 - P10 of gated short-term blocks > -70 and > int - 20)
    st_arr = np.array(st_lufs_1s, dtype=np.float64)
    gate = max(-70.0, integrated_lufs - 20.0) if np.isfinite(integrated_lufs) else -70.0
    gated = st_arr[st_arr >= gate]
    lra_lu = float(np.percentile(gated, 95) - np.percentile(gated, 10)) if len(gated) >= 4 else 0.0

    # True Peak (4x oversampled)
    os_l = resample_poly(left, 4, 1)
    os_r = resample_poly(right, 4, 1)
    true_peak = float(max(np.max(np.abs(os_l)), np.max(np.abs(os_r))) + 1e-12)
    true_peak_dbtp = 20.0 * np.log10(true_peak)

    # Clipping / brickwall limiting detector
    abs_s = np.abs(stereo)
    near_clip = np.mean(abs_s >= 0.985) * 100.0

    # Per-stem crest factors
    stem_crest = {k: _crest_db(v) for k, v in decomp["stems_mono"].items()}

    # Stereo width (global, by band, and 1-second curve)
    width_global = _stereo_width(left, right)
    width_low = _stereo_width(_band_filter(left, sr, hi=250.0), _band_filter(right, sr, hi=250.0))
    width_mid = _stereo_width(_band_filter(left, sr, lo=250.0, hi=4000.0), _band_filter(right, sr, lo=250.0, hi=4000.0))
    width_high = _stereo_width(_band_filter(left, sr, lo=4000.0), _band_filter(right, sr, lo=4000.0))

    # Phase correlation & L/R balance
    l_c = left - np.mean(left)
    r_c = right - np.mean(right)
    phase_corr = float(np.dot(l_c, r_c) / ((np.linalg.norm(l_c) * np.linalg.norm(r_c)) + 1e-12))
    lr_bal_db = 20.0 * np.log10(
        (float(np.sqrt(np.mean(right ** 2) + 1e-12))) / (float(np.sqrt(np.mean(left ** 2) + 1e-12)))
    )

    width_1s = []
    for sec in range(total_sec):
        s, e = sec * sr, min(len(left), (sec + 1) * sr)
        width_1s.append(_stereo_width(left[s:e], right[s:e]))

    reverb = _reverb_tail_proxy(decomp["percussive_mix"], decomp["mid"], decomp["side"], sr)

    return {
        "summary": {
            "integrated_lufs": round(integrated_lufs, 2) if np.isfinite(integrated_lufs) else -70.0,
            "loudness_range_lra_lu": round(lra_lu, 2),
            "true_peak_dbtp": round(true_peak_dbtp, 2),
            "crest_factor_db": _crest_db(mono),
            "stem_crest_db": stem_crest,
            "clipping_ratio_pct": round(float(near_clip), 4),
            "stereo_width_global": width_global,
            "stereo_width_bands": {"low_lt250": width_low, "mid_250_4k": width_mid, "high_gt4k": width_high},
            "phase_correlation": round(phase_corr, 3),
            "lr_balance_db": round(float(lr_bal_db), 2),
            "reverb_proxy": reverb,
        },
        "timeseries": {
            "st_lufs": st_lufs_1s,
            "stereo_width": width_1s,
        },
    }
