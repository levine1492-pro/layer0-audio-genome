"""Axis 6: Local Adaptive Multi-Modal Segmentation, >45s Sub-Split & OFF-Stem Gating."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from engine.pitch_harmony import hz_to_note_cents


def _fmt_time(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu = np.mean(mat, axis=1, keepdims=True)
    sd = np.std(mat, axis=1, keepdims=True) + 1e-8
    return (mat - mu) / sd


def _checkerboard_novelty(feat: np.ndarray, L: int = 8) -> np.ndarray:
    T = feat.shape[1]
    if T < 2 * L + 2:
        return np.zeros(T)
    norms = np.linalg.norm(feat, axis=0, keepdims=True) + 1e-8
    F = feat / norms
    S = F.T @ F

    idx = np.arange(-L, L + 1)
    g = np.exp(-0.5 * (idx / (0.5 * L)) ** 2)
    sign = np.sign(idx)
    K = np.outer(sign * g, sign * g)

    nov = np.zeros(T, dtype=np.float64)
    for t in range(L, T - L):
        nov[t] = float(np.sum(S[t - L:t + L + 1, t - L:t + L + 1] * K))
    nov = np.maximum(0.0, nov)
    if np.max(nov) > 0:
        nov /= np.max(nov)
    return nov


def _stem_state(phase_db: float, track_max_db: float, mix_global_db: float) -> str:
    # Gate silent stems cleanly (absolute floor OR > 26 dB quieter than overall mix)
    if phase_db < -50.0 or phase_db < mix_global_db - 26.0:
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
    mix_global_db = 20.0 * np.log10(float(np.sqrt(np.mean(mono ** 2) + 1e-12)))

    stem_names = ["drums", "bass", "vocals", "other"]
    stem_mat = np.array([decomp["stem_db_1s"][k][:T] for k in stem_names], dtype=np.float64)
    lufs_arr = np.array(space["timeseries"]["st_lufs"][:T], dtype=np.float64)[None, :]
    mfcc_mat = timbre["timeseries"]["mfcc_1s"][:, :T].astype(np.float64)
    chroma_mat = pitch["timeseries"]["chroma_1s"][:, :T].astype(np.float64)
    cent_arr = np.array(timbre["timeseries"]["centroid_hz"][:T], dtype=np.float64)[None, :]
    dnps_arr = np.array(rhythm["timeseries"]["drum_nps"][:T], dtype=np.float64)[None, :]
    mnps_arr = np.array(rhythm["timeseries"]["melodic_nps"][:T], dtype=np.float64)[None, :]

    # Lead register 1s curve (log Hz, forward-filled) to catch solo register leaps
    lead_1s_raw = pitch["timeseries"]["lead_f0_hz"][:T]
    lead_log = np.zeros((1, T), dtype=np.float64)
    last_val = 0.0
    for i, v in enumerate(lead_1s_raw):
        if v is not None and v > 0:
            last_val = float(np.log2(v))
        lead_log[0, i] = last_val

    fused = np.vstack([
        _zscore_rows(stem_mat) * 2.0,
        _zscore_rows(lufs_arr) * 1.5,
        _zscore_rows(mfcc_mat) * 1.1,
        _zscore_rows(cent_arr) * 0.9,
        _zscore_rows(lead_log) * 0.8,
        _zscore_rows(chroma_mat) * 0.7,
        _zscore_rows(dnps_arr) * 0.8,
        _zscore_rows(mnps_arr) * 0.7,
    ])

    nov_short = _checkerboard_novelty(fused, L=6)
    nov_long = _checkerboard_novelty(fused, L=12)

    cliff = np.zeros(T, dtype=np.float64)
    for t in range(3, T - 3):
        pre_s = np.mean(stem_mat[:, max(0, t - 3):t], axis=1)
        post_s = np.mean(stem_mat[:, t:min(T, t + 3)], axis=1)
        cliff[t] = float(np.max(np.abs(post_s - pre_s)))
    cliff_norm = np.clip(cliff / 18.0, 0.0, 1.5)

    combined_nov = gaussian_filter1d(0.45 * nov_short + 0.35 * nov_long + 0.50 * cliff_norm, sigma=1.2)

    # Local adaptive baseline (31s rolling window so late cliffs don't hide early turns)
    local_base = uniform_filter1d(combined_nov, size=max(3, min(T, 31)), mode="nearest")
    rel_nov = combined_nov - local_base
    rel_thresh = max(0.08, 0.55 * float(np.std(rel_nov)))
    pks, props = find_peaks(rel_nov, height=rel_thresh, distance=6, prominence=0.07)

    # Filter out ultra-weak absolute noise peaks
    valid_mask = combined_nov[pks] >= 0.16
    pks = pks[valid_mask]
    heights = combined_nov[pks]

    if len(pks) > 15:
        top = np.argsort(heights)[::-1][:15]
        pks = np.sort(pks[top])

    bounds = [0] + [int(p) for p in pks if 4 <= p <= T - 4] + [T]

    # Long-phase internal sub-split (>45s): inspect internal timbral/harmonic/register novelty
    refined = [bounds[0]]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if (e - s) > 45 and len(refined) < 16:
            int_feat = np.vstack([
                _zscore_rows(mfcc_mat[:, s:e]) * 1.2,
                _zscore_rows(chroma_mat[:, s:e]) * 1.0,
                _zscore_rows(lead_log[:, s:e]) * 1.2,
                _zscore_rows(cent_arr[:, s:e]) * 0.9,
                _zscore_rows(mnps_arr[:, s:e]) * 0.8,
            ])
            int_nov = _checkerboard_novelty(int_feat, L=6)
            if len(int_nov) > 22:
                int_nov[:10] = 0.0
                int_nov[-10:] = 0.0
                pk = int(np.argmax(int_nov))
                if int_nov[pk] >= 0.50:
                    refined.append(s + pk)
        refined.append(e)
    bounds = sorted(set(refined))

    stem_max_db = {k: float(np.percentile(decomp["stem_db_1s"][k][:T], 95)) for k in stem_names}
    bass_f0 = pitch["raw_contours"]["bass_f0"]
    lead_f0 = pitch["raw_contours"]["lead_f0"]
    hop = pitch["raw_contours"]["hop"]
    a4_hz = pitch["raw_contours"]["a4_hz"]
    fps = sr / hop

    phases = []
    for i in range(len(bounds) - 1):
        s_sec, e_sec = bounds[i], bounds[i + 1]
        seg_mono = mono[s_sec * sr:min(len(mono), e_sec * sr)]
        seg_rms = float(np.sqrt(np.mean(seg_mono ** 2) + 1e-12))
        seg_pk = float(np.max(np.abs(seg_mono)) + 1e-12)
        seg_crest = round(20.0 * np.log10(seg_pk / seg_rms), 2)

        stems_state = {}
        for k in stem_names:
            p_db = float(np.mean(decomp["stem_db_1s"][k][s_sec:e_sec]))
            stems_state[k] = _stem_state(p_db, stem_max_db[k], mix_global_db)

        f_lo, f_hi = int(s_sec * fps), min(len(bass_f0), int(e_sec * fps))
        b_valid = bass_f0[f_lo:f_hi]
        b_valid = b_valid[np.isfinite(b_valid) & (b_valid > 0)]
        l_valid = lead_f0[f_lo:f_hi]
        l_valid = l_valid[np.isfinite(l_valid) & (l_valid > 0)]

        # Clean OFF-stem pitch gating
        if stems_state["bass"] == "OFF" or len(b_valid) < 5:
            bass_center = "OFF"
        else:
            bass_center = hz_to_note_cents(float(np.median(b_valid)), a4_hz)

        if (stems_state["other"] == "OFF" and stems_state["vocals"] == "OFF") or len(l_valid) < 5:
            lead_range, lead_center = "OFF", "OFF"
        else:
            lp10, lp50, lp90 = np.percentile(l_valid, [10, 50, 90])
            lead_range = f"{hz_to_note_cents(float(lp10), a4_hz)}..{hz_to_note_cents(float(lp90), a4_hz)}"
            lead_center = hz_to_note_cents(float(lp50), a4_hz)

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
            "flatness_db": round(float(np.mean(timbre["timeseries"]["flatness_db"][s_sec:e_sec])), 1),
        })

    return phases
