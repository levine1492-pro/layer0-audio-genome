"""Axis 6: Fused Multi-Modal Structural Segmentation & Per-Phase Matrix."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from engine.pitch_harmony import hz_to_note_cents


def _fmt_time(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu = np.mean(mat, axis=1, keepdims=True)
    sd = np.std(mat, axis=1, keepdims=True) + 1e-8
    return (mat - mu) / sd


def _checkerboard_novelty(feat: np.ndarray, L: int = 8) -> np.ndarray:
    """Foote checkerboard novelty on cosine self-similarity matrix."""
    T = feat.shape[1]
    if T < 2 * L + 2:
        return np.zeros(T)
    norms = np.linalg.norm(feat, axis=0, keepdims=True) + 1e-8
    F = feat / norms
    S = F.T @ F  # (T, T)

    # Gaussian-tapered checkerboard kernel of size (2L+1)
    idx = np.arange(-L, L + 1)
    g = np.exp(-0.5 * (idx / (0.5 * L)) ** 2)
    sign = np.sign(idx)
    K = np.outer(sign * g, sign * g)

    nov = np.zeros(T, dtype=np.float64)
    for t in range(L, T - L):
        patch = S[t - L:t + L + 1, t - L:t + L + 1]
        nov[t] = float(np.sum(patch * K))
    nov = np.maximum(0.0, nov)
    if np.max(nov) > 0:
        nov /= np.max(nov)
    return nov


def _stem_state(phase_db: float, track_max_db: float) -> str:
    if phase_db < -50.0:
        return "OFF"
    diff = track_max_db - phase_db
    if diff <= 5.0:
        return "HIGH"
    if diff <= 12.0:
        return "MID"
    if diff <= 20.0:
        return "LOW"
    return "OFF"


def analyze_structure(decomp: dict, pitch: dict, rhythm: dict, timbre: dict, space: dict, sr: int) -> list:
    mono = decomp["mono"]
    T = max(1, int(np.floor(len(mono) / sr)))

    # 1. Build multi-modal 1-second feature stack
    stem_names = ["drums", "bass", "vocals", "other"]
    stem_mat = np.array([decomp["stem_db_1s"][k][:T] for k in stem_names], dtype=np.float64)
    lufs_arr = np.array(space["timeseries"]["st_lufs"][:T], dtype=np.float64)[None, :]
    mfcc_mat = timbre["timeseries"]["mfcc_1s"][:, :T].astype(np.float64)
    chroma_mat = pitch["timeseries"]["chroma_1s"][:, :T].astype(np.float64)
    cent_arr = np.array(timbre["timeseries"]["centroid_hz"][:T], dtype=np.float64)[None, :]
    dnps_arr = np.array(rhythm["timeseries"]["drum_nps"][:T], dtype=np.float64)[None, :]
    mnps_arr = np.array(rhythm["timeseries"]["melodic_nps"][:T], dtype=np.float64)[None, :]

    fused = np.vstack([
        _zscore_rows(stem_mat) * 2.0,
        _zscore_rows(lufs_arr) * 1.6,
        _zscore_rows(mfcc_mat) * 1.1,
        _zscore_rows(cent_arr) * 0.9,
        _zscore_rows(chroma_mat) * 0.7,
        _zscore_rows(dnps_arr) * 0.8,
        _zscore_rows(mnps_arr) * 0.7,
    ])

    # 2. Multi-scale checkerboard novelty + abrupt stem/loudness cliff detector
    nov_short = _checkerboard_novelty(fused, L=6)
    nov_long = _checkerboard_novelty(fused, L=12)

    # Direct step-change cliff detector (catches short intros like 0:06 and breakdowns like 2:24)
    cliff = np.zeros(T, dtype=np.float64)
    for t in range(3, T - 3):
        pre_s = np.mean(stem_mat[:, max(0, t - 3):t], axis=1)
        post_s = np.mean(stem_mat[:, t:min(T, t + 3)], axis=1)
        cliff[t] = float(np.max(np.abs(post_s - pre_s)))
    cliff_norm = np.clip(cliff / 18.0, 0.0, 1.5)

    combined_nov = gaussian_filter1d(0.45 * nov_short + 0.35 * nov_long + 0.55 * cliff_norm, sigma=1.2)

    # Adaptive peak picking
    min_seg = 6
    thresh = float(np.mean(combined_nov) + 0.65 * np.std(combined_nov))
    pks, props = find_peaks(combined_nov, height=thresh, distance=min_seg, prominence=0.10)

    # Cap at 14 phases max (keep strongest structural transitions)
    if len(pks) > 13:
        top = np.argsort(props["peak_heights"])[::-1][:13]
        pks = np.sort(pks[top])

    # Filter boundaries too close to start/end
    bounds = [0] + [int(p) for p in pks if 4 <= p <= T - 4] + [T]

    # Precompute track-level max dB per stem (P95) for relative activity state
    stem_max_db = {k: float(np.percentile(decomp["stem_db_1s"][k][:T], 95)) for k in stem_names}

    # Raw pYIN contours for accurate per-phase pitch registers
    bass_f0 = pitch["raw_contours"]["bass_f0"]
    lead_f0 = pitch["raw_contours"]["lead_f0"]
    hop = pitch["raw_contours"]["hop"]
    a4_hz = pitch["raw_contours"]["a4_hz"]
    fps = sr / hop

    phases = []
    for i in range(len(bounds) - 1):
        s_sec, e_sec = bounds[i], bounds[i + 1]
        s_samp, e_samp = s_sec * sr, min(len(mono), e_sec * sr)
        seg_mono = mono[s_samp:e_samp]

        # Phase crest
        seg_rms = float(np.sqrt(np.mean(seg_mono ** 2) + 1e-12))
        seg_pk = float(np.max(np.abs(seg_mono)) + 1e-12)
        seg_crest = round(20.0 * np.log10(seg_pk / seg_rms), 2)

        # Phase pitch registers
        f_lo, f_hi = int(s_sec * fps), min(len(bass_f0), int(e_sec * fps))
        b_valid = bass_f0[f_lo:f_hi]
        b_valid = b_valid[np.isfinite(b_valid) & (b_valid > 0)]
        l_valid = lead_f0[f_lo:f_hi]
        l_valid = l_valid[np.isfinite(l_valid) & (l_valid > 0)]

        bass_center = hz_to_note_cents(float(np.median(b_valid)), a4_hz) if len(b_valid) >= 3 else "OFF"
        if len(l_valid) >= 3:
            lp10, lp50, lp90 = np.percentile(l_valid, [10, 50, 90])
            lead_range = f"{hz_to_note_cents(float(lp10), a4_hz)}..{hz_to_note_cents(float(lp90), a4_hz)}"
            lead_center = hz_to_note_cents(float(lp50), a4_hz)
        else:
            lead_range, lead_center = "OFF", "OFF"

        # Active stems
        stems_state = {}
        for k in stem_names:
            p_db = float(np.mean(decomp["stem_db_1s"][k][s_sec:e_sec]))
            stems_state[k] = _stem_state(p_db, stem_max_db[k])

        phases.append({
            "index": i + 1,
            "start_s": s_sec,
            "end_s": e_sec,
            "range": f"{_fmt_time(s_sec)}-{_fmt_time(e_sec)}",
            "duration_s": e_sec - s_sec,
            "lufs": round(float(np.mean(space["timeseries"]["st_lufs"][s_sec:e_sec])), 1),
            "crest_db": seg_crest,
            "stereo_width": round(float(np.mean(space["timeseries"]["stereo_width"][s_sec:e_sec])), 2),
            "stems": stems_state,
            "bass_center": bass_center,
            "lead_range": lead_range,
            "lead_center": lead_center,
            "drum_nps": round(float(np.mean(rhythm["timeseries"]["drum_nps"][s_sec:e_sec])), 2),
            "melodic_nps": round(float(np.mean(rhythm["timeseries"]["melodic_nps"][s_sec:e_sec])), 2),
            "centroid_hz": round(float(np.mean(timbre["timeseries"]["centroid_hz"][s_sec:e_sec])), 0),
            "flatness": round(float(np.mean(timbre["timeseries"]["flatness"][s_sec:e_sec])), 4),
        })

    return phases
