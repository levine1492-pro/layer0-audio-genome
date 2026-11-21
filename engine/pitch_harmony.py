"""Axis 2: Pitch, 5-Cent Microtonality, 17-Scale Global Modal Bank & Sensory Roughness."""
import numpy as np
import librosa
from scipy.signal import find_peaks

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 17-Scale Global Bank: 7 Diatonic Modes + Non-Western, Blues, Pentatonic & Symmetric
MODE_TEMPLATES = {
    "Ionian (Major)":             [6.4, 1.0, 3.8, 1.0, 4.6, 4.1, 1.0, 5.3, 1.0, 3.8, 1.0, 3.5],
    "Aeolian (Minor)":            [6.4, 1.0, 3.8, 5.1, 1.0, 4.0, 1.0, 5.2, 4.2, 1.0, 3.9, 1.0],
    "Dorian":                     [6.4, 1.0, 3.9, 5.0, 1.0, 4.0, 1.0, 5.2, 1.0, 4.5, 3.9, 1.0],
    "Phrygian":                   [6.4, 4.8, 1.0, 4.9, 1.0, 4.0, 1.0, 5.1, 4.2, 1.0, 3.8, 1.0],
    "Lydian":                     [6.4, 1.0, 3.8, 1.0, 4.6, 1.0, 4.7, 5.2, 1.0, 3.8, 1.0, 3.6],
    "Mixolydian":                 [6.4, 1.0, 3.8, 1.0, 4.6, 4.0, 1.0, 5.2, 1.0, 3.7, 4.6, 1.0],
    "Locrian":                    [6.2, 4.6, 1.0, 4.8, 1.0, 4.0, 4.8, 1.0, 4.1, 1.0, 3.8, 1.0],
    "Phrygian Dominant (Hijaz)":  [6.5, 5.1, 1.0, 1.0, 5.0, 4.0, 1.0, 5.2, 4.2, 1.0, 4.1, 1.0],
    "Harmonic Minor":             [6.5, 1.0, 3.7, 5.0, 1.0, 4.0, 1.0, 5.2, 4.4, 1.0, 1.0, 4.6],
    "Double Harmonic (Byzantine)":[6.5, 5.0, 1.0, 1.0, 4.9, 4.0, 1.0, 5.2, 4.5, 1.0, 1.0, 4.7],
    "Hungarian Minor":            [6.5, 1.0, 3.7, 5.0, 1.0, 1.0, 4.9, 5.2, 4.4, 1.0, 1.0, 4.5],
    "Melodic Minor":              [6.4, 1.0, 3.8, 5.0, 1.0, 4.0, 1.0, 5.2, 1.0, 4.3, 1.0, 4.3],
    "Minor Blues":                [6.6, 0.8, 0.8, 5.4, 0.8, 4.6, 4.9, 5.3, 0.8, 0.8, 4.6, 0.8],
    "Major Blues":                [6.6, 0.8, 4.4, 4.8, 5.2, 0.8, 0.8, 5.3, 0.8, 4.6, 0.8, 0.8],
    "Minor Pentatonic":           [6.6, 0.8, 0.8, 5.4, 0.8, 4.8, 0.8, 5.4, 0.8, 0.8, 4.8, 0.8],
    "Major Pentatonic":           [6.6, 0.8, 4.6, 0.8, 5.2, 0.8, 0.8, 5.4, 0.8, 4.6, 0.8, 0.8],
    "Octatonic (Diminished)":     [6.0, 1.0, 4.4, 4.6, 1.0, 4.4, 4.6, 1.0, 4.4, 4.6, 1.0, 4.4],
}


def hz_to_note_cents(hz: float, a4_hz: float = 440.0) -> str:
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return "OFF"
    midi = 69.0 + 12.0 * np.log2(hz / a4_hz)
    nearest = int(np.round(midi))
    cents = int(np.round((midi - nearest) * 100.0))
    pc = PITCH_CLASSES[nearest % 12]
    octave = (nearest // 12) - 1
    if cents == 0:
        return f"{pc}{octave}"
    return f"{pc}{octave}{cents:+d}c"


def _empty_contour() -> dict:
    return {
        "voiced_pct": 0.0, "range_low": "OFF", "range_high": "OFF",
        "center_note": "OFF", "center_hz": None,
        "drift_gt15c_pct": 0.0, "drift_gt25c_pct": 0.0,
        "cent_mean": 0.0, "cent_std": 0.0, "cent_skew": 0.0,
        "cent_histogram_5c": [0.0] * 20,
        "pitch_class_gravity": {pc: 0.0 for pc in PITCH_CLASSES},
        "bend_count": 0, "bend_up_pct": 0.0, "bend_max_cents": 0.0,
        "vibrato_pct": 0.0, "vibrato_rate_hz": 0.0, "vibrato_depth_cents": 0.0,
        "intervals": {"step_0_2": 0.0, "skip_3_5": 0.0, "leap_6_12": 0.0, "wide_gt12": 0.0},
    }


def _run_pyin_contour(y: np.ndarray, sr: int, mix_rms: float, fmin: float, fmax: float, hop: int = 512):
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    n_frames = max(1, 1 + len(y) // hop)
    # Silent-stem guard (< 3% of mix RMS or absolute noise floor)
    if rms < 1e-4 or rms < 0.03 * mix_rms:
        return np.full(n_frames, np.nan), np.zeros(n_frames, dtype=bool)
    f0, voiced, probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop,
        resolution=0.05, fill_na=np.nan
    )
    valid = voiced & np.isfinite(f0) & (f0 > 0) & (probs > 0.15)
    f0_clean = np.where(valid, f0, np.nan)
    return f0_clean, valid


def _contour_stats(f0: np.ndarray, sr: int, hop: int, a4_hz: float) -> dict:
    valid_idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    total_frames = len(f0)
    if len(valid_idx) < 20:
        return _empty_contour()

    vals = f0[valid_idx]
    midi = 69.0 + 12.0 * np.log2(vals / a4_hz)
    nearest = np.round(midi).astype(int)
    cents = (midi - nearest) * 100.0

    # True 5-cent histogram (-50 to +50 in 20 bins)
    hist, _ = np.histogram(cents, bins=20, range=(-50.0, 50.0))
    hist_pct = [round(100.0 * float(h) / len(cents), 2) for h in hist]

    pc_counts = np.bincount(nearest % 12, minlength=12)
    pc_grav = {
        PITCH_CLASSES[i]: round(100.0 * float(pc_counts[i]) / len(nearest), 2)
        for i in range(12)
    }

    c_mean = float(np.mean(cents))
    c_std = float(np.std(cents) + 1e-8)
    c_skew = float(np.mean(((cents - c_mean) / c_std) ** 3))

    frame_dt = hop / sr
    midi_full = np.where(
        np.isfinite(f0) & (f0 > 0),
        69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / a4_hz),
        np.nan
    )

    bends_up, bends_down, max_bend = 0, 0, 0.0
    vib_frames, vib_rates, vib_depths = 0, [], []
    seg_medians = []

    is_v = np.isfinite(midi_full)
    edges = np.diff(is_v.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for s, e in zip(starts, ends):
        seg = midi_full[s:e]
        dur = (e - s) * frame_dt
        if len(seg) < 4:
            continue
        seg_medians.append(float(np.median(seg)))

        win = max(3, int(round(0.18 / frame_dt)))
        if len(seg) > win:
            deltas_cents = (seg[win:] - seg[:-win]) * 100.0
            pos_peaks, _ = find_peaks(deltas_cents, height=(45.0, 400.0), distance=win)
            neg_peaks, _ = find_peaks(-deltas_cents, height=(45.0, 400.0), distance=win)
            bends_up += len(pos_peaks)
            bends_down += len(neg_peaks)
            if len(pos_peaks):
                max_bend = max(max_bend, float(np.max(deltas_cents[pos_peaks])))
            if len(neg_peaks):
                max_bend = max(max_bend, float(np.max(-deltas_cents[neg_peaks])))

        if dur >= 0.35:
            x = np.arange(len(seg))
            poly = np.polyfit(x, seg, 1)
            detrended_c = (seg - np.polyval(poly, x)) * 100.0
            spec = np.abs(np.fft.rfft(detrended_c * np.hanning(len(detrended_c))))
            freqs = np.fft.rfftfreq(len(detrended_c), d=frame_dt)
            vib_mask = (freqs >= 3.5) & (freqs <= 9.0)
            if np.any(vib_mask):
                sub_spec = spec[vib_mask]
                sub_freqs = freqs[vib_mask]
                pk_idx = int(np.argmax(sub_spec))
                depth_est = float(np.std(detrended_c) * np.sqrt(2.0))
                ratio = float(sub_spec[pk_idx] / (np.sum(spec[1:]) + 1e-8))
                if 12.0 <= depth_est <= 150.0 and ratio > 0.12:
                    vib_frames += len(seg)
                    vib_rates.append(float(sub_freqs[pk_idx]))
                    vib_depths.append(depth_est)

    intervals = {"step_0_2": 0.0, "skip_3_5": 0.0, "leap_6_12": 0.0, "wide_gt12": 0.0}
    if len(seg_medians) >= 2:
        diffs = np.abs(np.diff(seg_medians))
        n_d = len(diffs)
        intervals = {
            "step_0_2": round(100.0 * float(np.sum(diffs <= 2.5)) / n_d, 2),
            "skip_3_5": round(100.0 * float(np.sum((diffs > 2.5) & (diffs <= 5.5))) / n_d, 2),
            "leap_6_12": round(100.0 * float(np.sum((diffs > 5.5) & (diffs <= 12.5))) / n_d, 2),
            "wide_gt12": round(100.0 * float(np.sum(diffs > 12.5)) / n_d, 2),
        }

    total_bends = bends_up + bends_down
    p05, p50, p95 = np.percentile(vals, [5, 50, 95])

    return {
        "voiced_pct": round(100.0 * len(valid_idx) / max(1, total_frames), 2),
        "range_low": hz_to_note_cents(float(p05), a4_hz),
        "range_high": hz_to_note_cents(float(p95), a4_hz),
        "center_note": hz_to_note_cents(float(p50), a4_hz),
        "center_hz": round(float(p50), 2),
        "drift_gt15c_pct": round(100.0 * float(np.mean(np.abs(cents) > 15.0)), 2),
        "drift_gt25c_pct": round(100.0 * float(np.mean(np.abs(cents) > 25.0)), 2),
        "cent_mean": round(c_mean, 2),
        "cent_std": round(float(np.std(cents)), 2),
        "cent_skew": round(c_skew, 3),
        "cent_histogram_5c": hist_pct,
        "pitch_class_gravity": pc_grav,
        "bend_count": int(total_bends),
        "bend_up_pct": round(100.0 * bends_up / total_bends, 1) if total_bends else 0.0,
        "bend_max_cents": round(max_bend, 1),
        "vibrato_pct": round(100.0 * vib_frames / max(1, len(valid_idx)), 2),
        "vibrato_rate_hz": round(float(np.mean(vib_rates)), 2) if vib_rates else 0.0,
        "vibrato_depth_cents": round(float(np.mean(vib_depths)), 1) if vib_depths else 0.0,
        "intervals": intervals,
    }


def _sensory_roughness(y: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 2048):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid_bins = (freqs >= 40.0) & (freqs <= 8000.0)
    S = S[valid_bins, :]
    f_sub = freqs[valid_bins]

    rough_frames = np.zeros(S.shape[1], dtype=np.float32)
    K = 12
    for t in range(S.shape[1]):
        col = S[:, t]
        if np.max(col) < 1e-6:
            continue
        pks, _ = find_peaks(col, distance=2)
        if len(pks) < 2:
            continue
        top = pks[np.argpartition(col[pks], -K)[-K:]] if len(pks) > K else pks
        fp = f_sub[top]
        ap = col[top]
        norm = float(np.sum(ap ** 2) + 1e-12)
        f_min = np.minimum(fp[:, None], fp[None, :])
        f_diff = np.abs(fp[:, None] - fp[None, :])
        s = 0.24 / (0.021 * f_min + 19.0)
        x = s * f_diff
        d = (ap[:, None] * ap[None, :]) * (np.exp(-3.5 * x) - np.exp(-5.75 * x))
        rough_frames[t] = float(np.sum(np.triu(d, k=1)) / norm)

    return rough_frames


def _estimate_modal_key(poly_vec: np.ndarray, lead_stats: dict) -> dict:
    # Fuse polyphonic chroma (70%) + lead melodic gravity (30%) when lead is active
    p_norm = poly_vec / (np.sum(poly_vec) + 1e-12)
    if lead_stats["voiced_pct"] >= 15.0:
        l_vec = np.array([lead_stats["pitch_class_gravity"][pc] for pc in PITCH_CLASSES], dtype=np.float64)
        l_norm = l_vec / (np.sum(l_vec) + 1e-12)
        blended = 0.65 * p_norm + 0.35 * l_norm
    else:
        blended = p_norm

    c_norm = blended - np.mean(blended)
    c_den = np.linalg.norm(c_norm) + 1e-12

    candidates = []
    for mode_name, tmpl in MODE_TEMPLATES.items():
        t_arr = np.array(tmpl, dtype=np.float64)
        for root in range(12):
            shifted = np.roll(t_arr, root)
            s_norm = shifted - np.mean(shifted)
            corr = float(np.dot(c_norm, s_norm) / (c_den * (np.linalg.norm(s_norm) + 1e-12)))
            candidates.append((corr, PITCH_CLASSES[root], mode_name))

    candidates.sort(reverse=True)
    best_corr, best_tonic, best_mode = candidates[0]
    runner_up = f"{candidates[1][1]} {candidates[1][2]} ({candidates[1][0]:.2f})"
    return {
        "tonic": best_tonic,
        "mode": best_mode,
        "confidence": round(best_corr, 3),
        "ambiguous": bool(best_corr < 0.70),
        "runner_up": runner_up,
    }


def _resample_1s(arr: np.ndarray, total_sec: int, agg: str = "mean") -> list:
    if len(arr) == 0:
        return [None] * total_sec
    idx_edges = np.linspace(0, len(arr), total_sec + 1).astype(int)
    out = []
    for i in range(total_sec):
        seg = arr[idx_edges[i]:max(idx_edges[i] + 1, idx_edges[i + 1])]
        valid = seg[np.isfinite(seg)]
        if len(valid) == 0:
            out.append(None)
        elif agg == "median":
            out.append(round(float(np.median(valid)), 2))
        else:
            out.append(round(float(np.mean(valid)), 4))
    return out


def analyze_pitch_harmony(decomp: dict, sr: int) -> dict:
    mono = decomp["mono"]
    harm = decomp["harmonic_mix"]
    bass_y = decomp["stems_mono"]["bass"]
    lead_y = decomp["lead_carrier"]
    mix_rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    total_sec = max(1, int(np.floor(len(mono) / sr)))
    hop = 512

    tuning_bins = float(librosa.estimate_tuning(y=harm, sr=sr, bins_per_octave=36))
    tuning_cents = tuning_bins * (100.0 / 3.0)
    a4_hz = 440.0 * (2.0 ** (tuning_cents / 1200.0))

    bass_f0, _ = _run_pyin_contour(bass_y, sr, mix_rms, fmin=28.0, fmax=350.0, hop=hop)
    lead_f0, _ = _run_pyin_contour(lead_y, sr, mix_rms, fmin=140.0, fmax=2400.0, hop=hop)

    bass_stats = _contour_stats(bass_f0, sr, hop, a4_hz)
    lead_stats = _contour_stats(lead_f0, sr, hop, a4_hz)

    chroma12 = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=hop, tuning=tuning_bins)
    chroma_mean = np.mean(chroma12, axis=1)
    chroma_pct = chroma_mean / (np.sum(chroma_mean) + 1e-12) * 100.0
    poly_gravity = {PITCH_CLASSES[i]: round(float(chroma_pct[i]), 2) for i in range(12)}

    chroma36 = librosa.feature.chroma_cqt(
        y=harm, sr=sr, hop_length=hop, n_chroma=36, bins_per_octave=36, tuning=tuning_bins
    )
    c36_mean = np.mean(chroma36, axis=1)
    off_grid_mask = np.array([i % 3 != 0 for i in range(36)])
    microtonal_cqt_ratio = float(np.sum(c36_mean[off_grid_mask]) / (np.sum(c36_mean) + 1e-12))

    modal_key = _estimate_modal_key(chroma_mean, lead_stats)

    tonnetz = librosa.feature.tonnetz(chroma=chroma12)
    hcdf_frames = np.linalg.norm(np.diff(tonnetz, axis=1, prepend=tonnetz[:, :1]), axis=0)
    hcdf_peaks, _ = find_peaks(
        hcdf_frames,
        height=float(np.mean(hcdf_frames) + np.std(hcdf_frames)),
        distance=int(0.5 * sr / hop)
    )
    hcdf_rate = float(len(hcdf_peaks) / max(1.0, len(mono) / sr))

    rough_frames = _sensory_roughness(mono, sr)

    chroma_1s = np.zeros((12, total_sec), dtype=np.float32)
    idx_edges = np.linspace(0, chroma12.shape[1], total_sec + 1).astype(int)
    for i in range(total_sec):
        chroma_1s[:, i] = np.mean(chroma12[:, idx_edges[i]:max(idx_edges[i] + 1, idx_edges[i + 1])], axis=1)

    return {
        "summary": {
            "a4_ref_hz": round(a4_hz, 2),
            "tuning_offset_cents": round(tuning_cents, 1),
            "modal_key": modal_key,
            "polyphonic_chroma_gravity": poly_gravity,
            "microtonal_cqt_offgrid_pct": round(microtonal_cqt_ratio * 100.0, 2),
            "sensory_roughness_mean": round(float(np.mean(rough_frames)), 4),
            "sensory_roughness_p90": round(float(np.percentile(rough_frames, 90)), 4),
            "harmonic_change_rate_hz": round(hcdf_rate, 3),
            "tonnetz_mean": [round(float(x), 4) for x in np.mean(tonnetz, axis=1)],
            "tonnetz_std": [round(float(x), 4) for x in np.std(tonnetz, axis=1)],
            "bass_contour": bass_stats,
            "lead_contour": lead_stats,
        },
        "timeseries": {
            "bass_f0_hz": _resample_1s(bass_f0, total_sec, agg="median"),
            "lead_f0_hz": _resample_1s(lead_f0, total_sec, agg="median"),
            "roughness": _resample_1s(rough_frames, total_sec, agg="mean"),
            "hcdf": _resample_1s(hcdf_frames, total_sec, agg="mean"),
            "chroma_1s": chroma_1s,
        },
        "raw_contours": {
            "bass_f0": bass_f0,
            "lead_f0": lead_f0,
            "hop": hop,
            "a4_hz": a4_hz,
        },
    }
