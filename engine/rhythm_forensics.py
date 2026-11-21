"""Axis 3: Macro Pulse, Meter & Dual-Layer Sovereign Timing Forensics."""
import numpy as np
import librosa
from scipy.signal import find_peaks


def _compute_dfa(series: np.ndarray) -> float:
    """Detrended Fluctuation Analysis (DFA-1) scaling exponent alpha."""
    n = len(series)
    if n < 24:
        return 0.5
    y = np.cumsum(series - np.mean(series))
    scales = np.unique(np.logspace(np.log10(4), np.log10(max(8, n // 4)), 10).astype(int))
    flucts, valid_s = [], []
    for s in scales:
        n_seg = n // s
        if n_seg < 2:
            continue
        x = np.arange(s)
        rms_vals = []
        for k in range(n_seg):
            seg = y[k * s:(k + 1) * s]
            coef = np.polyfit(x, seg, 1)
            rms_vals.append(np.mean((seg - np.polyval(coef, x)) ** 2))
        f_val = float(np.sqrt(np.mean(rms_vals)))
        if f_val > 1e-12:
            flucts.append(f_val)
            valid_s.append(s)
    if len(valid_s) < 3:
        return 0.5
    slope = float(np.polyfit(np.log10(valid_s), np.log10(flucts), 1)[0])
    return round(slope, 3)


def _compute_pink_slope(iois: np.ndarray) -> float:
    """1/f noise slope (beta) of detrended IOI sequence."""
    if len(iois) < 16:
        return 0.0
    x = iois - np.mean(iois)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    mask = (freqs > 0) & (spec > 1e-15)
    if np.sum(mask) < 4:
        return 0.0
    slope = float(np.polyfit(np.log10(freqs[mask]), np.log10(spec[mask]), 1)[0])
    return round(-slope, 3)


def _analyze_onset_layer(y: np.ndarray, sr: int, beat_times: np.ndarray, total_sec: int) -> dict:
    hop = 512
    rms_val = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms_val < 1e-4:
        return {
            "onset_count": 0, "nps": 0.0, "ioi_mean_ms": 0.0, "ioi_median_ms": 0.0,
            "ioi_iqr_ms": 0.0, "ioi_entropy": 0.0, "swing_ratio": 1.0,
            "hartzig_residual_ms": 0.0, "pink_noise_beta": 0.0, "dfa_alpha": 0.5,
            "attack_median_ms": 0.0, "attack_p90_ms": 0.0, "ghost_swell_pct": 0.0,
            "syncopation_index": 0.0, "nps_1s": [0.0] * total_sec,
        }

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False, wait=3
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    dur_s = len(y) / sr

    # 1-second NPS curve
    nps_1s = [0.0] * total_sec
    for t in onset_times:
        idx = min(total_sec - 1, max(0, int(t)))
        nps_1s[idx] += 1.0

    if len(onset_times) < 4:
        return {
            "onset_count": int(len(onset_times)), "nps": round(len(onset_times) / max(1.0, dur_s), 3),
            "ioi_mean_ms": 0.0, "ioi_median_ms": 0.0, "ioi_iqr_ms": 0.0, "ioi_entropy": 0.0,
            "swing_ratio": 1.0, "hartzig_residual_ms": 0.0, "pink_noise_beta": 0.0, "dfa_alpha": 0.5,
            "attack_median_ms": 0.0, "attack_p90_ms": 0.0, "ghost_swell_pct": 0.0,
            "syncopation_index": 0.0, "nps_1s": nps_1s,
        }

    iois_s = np.diff(onset_times)
    iois_ms = iois_s * 1000.0

    # Normalized Shannon entropy of log-IOI distribution (24 bins)
    log_ioi = np.log10(np.clip(iois_ms, 20.0, 4000.0))
    hist, _ = np.histogram(log_ioi, bins=24, range=(np.log10(20.0), np.log10(4000.0)))
    probs = hist / (np.sum(hist) + 1e-12)
    probs = probs[probs > 0]
    ioi_entropy = float(-np.sum(probs * np.log2(probs)) / np.log2(24))

    # Swing ratio on adjacent pairs
    pair_ratios = []
    for i in range(len(iois_ms) - 1):
        a, b = iois_ms[i], iois_ms[i + 1]
        if 80.0 < (a + b) < 900.0:
            r = max(a, b) / max(1.0, min(a, b))
            if r <= 2.5:
                pair_ratios.append(r)
    swing_ratio = float(np.median(pair_ratios)) if pair_ratios else 1.0

    # Local-grid Hartzig Residual (R_H) & Syncopation against local 16th-note subdivisions
    residuals_ms = []
    off_beat_strong = 0
    strong_total = 0
    if len(beat_times) >= 4:
        # Build 4x subdivision grid between consecutive beats
        grid = []
        for i in range(len(beat_times) - 1):
            b0, b1 = beat_times[i], beat_times[i + 1]
            for sub in range(4):
                grid.append(b0 + (b1 - b0) * (sub / 4.0))
        grid = np.array(grid)
        env_med = float(np.median(onset_env[onset_frames]))
        for f_idx, t in zip(onset_frames, onset_times):
            dist_grid = float(np.min(np.abs(grid - t))) * 1000.0
            residuals_ms.append(dist_grid)
            if onset_env[f_idx] >= env_med:
                strong_total += 1
                dist_beat = float(np.min(np.abs(beat_times - t)))
                local_bi = float(np.median(np.diff(beat_times)))
                if dist_beat > 0.18 * local_bi:
                    off_beat_strong += 1

    hartzig_ms = float(np.median(residuals_ms)) if residuals_ms else 0.0
    syncopation = float(off_beat_strong / max(1, strong_total))

    # Attack-time (ms) & Ghost Swell Percentage (G_p)
    rms_env = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_sr = sr / 256.0
    attack_times_ms = []
    swell_or_ghost = 0
    global_rms_p75 = float(np.percentile(rms_env, 75) + 1e-12)
    win_frames = int(0.25 * rms_sr)

    for t in onset_times:
        c = int(round(t * rms_sr))
        lo = max(0, c - 3)
        hi = min(len(rms_env), c + win_frames)
        if hi <= lo + 2:
            continue
        seg = rms_env[lo:hi]
        pk_rel = int(np.argmax(seg))
        val_rel = int(np.argmin(seg[:max(1, pk_rel + 1)]))
        rise_ms = max(0.0, (pk_rel - val_rel) / rms_sr * 1000.0)
        attack_times_ms.append(rise_ms)
        if rise_ms > 65.0 or float(seg[pk_rel]) < 0.35 * global_rms_p75:
            swell_or_ghost += 1

    q25, q50, q75 = np.percentile(iois_ms, [25, 50, 75])
    return {
        "onset_count": int(len(onset_times)),
        "nps": round(len(onset_times) / max(1.0, dur_s), 3),
        "ioi_mean_ms": round(float(np.mean(iois_ms)), 1),
        "ioi_median_ms": round(float(q50), 1),
        "ioi_iqr_ms": round(float(q75 - q25), 1),
        "ioi_entropy": round(ioi_entropy, 3),
        "swing_ratio": round(swing_ratio, 3),
        "hartzig_residual_ms": round(hartzig_ms, 1),
        "pink_noise_beta": _compute_pink_slope(iois_s),
        "dfa_alpha": _compute_dfa(iois_s),
        "attack_median_ms": round(float(np.median(attack_times_ms)), 1) if attack_times_ms else 0.0,
        "attack_p90_ms": round(float(np.percentile(attack_times_ms, 90)), 1) if attack_times_ms else 0.0,
        "ghost_swell_pct": round(100.0 * swell_or_ghost / max(1, len(attack_times_ms)), 1),
        "syncopation_index": round(syncopation, 3),
        "nps_1s": nps_1s,
    }


def analyze_rhythm_forensics(decomp: dict, sr: int) -> dict:
    mono = decomp["mono"]
    drums_y = decomp["stems_mono"]["drums"]
    melodic_y = decomp["lead_carrier"]
    total_sec = max(1, int(np.floor(len(mono) / sr)))
    hop = 512

    # Choose rhythmic reference signal (drums if active, else full percussive mix)
    drum_rms = float(np.sqrt(np.mean(drums_y ** 2) + 1e-12))
    mix_rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    rhythm_ref = drums_y if drum_rms > 0.05 * mix_rms else decomp["percussive_mix"]

    onset_env = librosa.onset.onset_strength(y=rhythm_ref, sr=sr, hop_length=hop)
    tempo_arr, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    primary_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # Multi-hypothesis tempogram peaks
    tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tgram_mean = np.mean(tgram, axis=1)
    bpm_grid = librosa.tempo_frequencies(tgram.shape[0], sr=sr, hop_length=hop)
    valid_bpm = np.isfinite(bpm_grid) & (bpm_grid >= 45.0) & (bpm_grid <= 220.0)
    sub_tg = tgram_mean[valid_bpm]
    sub_bpms = bpm_grid[valid_bpm]
    pks, _ = find_peaks(sub_tg, distance=3)
    hypotheses = []
    if len(pks):
        order = pks[np.argsort(sub_tg[pks])[::-1][:3]]
        mx = float(sub_tg[order[0]] + 1e-12)
        for idx in order:
            hypotheses.append({"bpm": round(float(sub_bpms[idx]), 1), "salience": round(float(sub_tg[idx] / mx), 2)})

    # Local tempo curve from beat intervals (rubato / drift detector)
    if len(beat_times) >= 5:
        inst_bpms = 60.0 / np.clip(np.diff(beat_times), 0.2, 2.0)
        tempo_std = float(np.std(inst_bpms))
        tempo_p10, tempo_p90 = np.percentile(inst_bpms, [10, 90])
    else:
        tempo_std, tempo_p10, tempo_p90 = 0.0, primary_bpm, primary_bpm

    # Meter grouping probe (duple vs triple autocorrelation of beat-synchronous energy)
    meter_hint = "Duple/Quad (4/4 or 2/4)"
    duple_score, triple_score = 0.0, 0.0
    if len(beat_frames) >= 12:
        b_sync = librosa.util.sync(onset_env[None, :], beat_frames, aggregate=np.mean)[0]
        b_sync = b_sync - np.mean(b_sync)
        ac = np.correlate(b_sync, b_sync, mode="full")[len(b_sync) - 1:]
        ac = ac / (ac[0] + 1e-12)
        if len(ac) > 6:
            duple_score = float(max(ac[2], ac[4]))
            triple_score = float(max(ac[3], ac[6]))
            if triple_score > duple_score + 0.08:
                meter_hint = "Triple/Compound (3/4 or 6/8)"
            elif abs(triple_score - duple_score) <= 0.08 and max(duple_score, triple_score) < 0.25:
                meter_hint = "Free / Odd / Unmetered"

    drums_forensics = _analyze_onset_layer(drums_y, sr, beat_times, total_sec)
    melodic_forensics = _analyze_onset_layer(melodic_y, sr, beat_times, total_sec)

    drum_nps_1s = drums_forensics.pop("nps_1s")
    melodic_nps_1s = melodic_forensics.pop("nps_1s")

    return {
        "summary": {
            "primary_bpm": round(primary_bpm, 2),
            "tempo_hypotheses": hypotheses,
            "local_tempo_std_bpm": round(tempo_std, 2),
            "local_tempo_range_bpm": [round(float(tempo_p10), 1), round(float(tempo_p90), 1)],
            "meter_hint": meter_hint,
            "meter_scores": {"duple": round(duple_score, 3), "triple": round(triple_score, 3)},
            "drums_layer": drums_forensics,
            "melodic_layer": melodic_forensics,
        },
        "timeseries": {
            "drum_nps": drum_nps_1s,
            "melodic_nps": melodic_nps_1s,
        },
    }
