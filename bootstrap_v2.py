#!/usr/bin/env python3
"""Self-unpacking installer for Layer 0 v2 Audio Genome Engine."""
from pathlib import Path

FILES = {
"requirements.txt": r'''numpy>=1.26,<2.2
scipy>=1.11
librosa>=0.10.2
soundfile>=0.12.1
pyloudnorm>=0.1.1
gdown>=5.2.0
demucs==4.0.1
''',

"engine/__init__.py": r'''"""Layer 0 v2 Physics-First Audio Genome Engine."""
SCHEMA_VERSION = "2.0.0"
''',

"engine/ingest.py": r'''"""Audio ingestion, queue parsing, and canonical 44.1kHz stereo float32 decoding."""
import hashlib
import re
import subprocess
import urllib.request
from pathlib import Path
import gdown
import soundfile as sf
import numpy as np

CANONICAL_SR = 44100


def parse_queue(queue_path: Path):
    """Parse links_queue.txt. Supports: URL | title | seed_id | tags | notes"""
    if not queue_path.exists():
        return []
    entries = []
    for idx, raw in enumerate(queue_path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        entries.append({
            "line_index": idx,
            "raw_line": raw,
            "source": parts[0],
            "title": parts[1] if len(parts) > 1 and parts[1] else "",
            "seed_id": parts[2] if len(parts) > 2 and parts[2] else "",
            "tags": parts[3] if len(parts) > 3 and parts[3] else "",
            "notes": parts[4] if len(parts) > 4 and parts[4] else "",
        })
    return entries


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("_")
    return s[:72] if s else "untitled_track"


def fetch_audio(source: str, work_dir: Path) -> Path:
    """Fetch audio from Google Drive, HTTP URL, or local path into work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local_candidate = Path(source)
    if local_candidate.exists():
        return local_candidate

    if "drive.google.com" in source or "docs.google.com" in source:
        out = gdown.download(url=source, output=str(work_dir) + "/", quiet=False, fuzzy=True)
        if not out or not Path(out).exists():
            raise RuntimeError(f"gdown failed to fetch Google Drive file: {source}")
        return Path(out)

    if source.startswith("http://") or source.startswith("https://"):
        fname = source.split("?")[0].rstrip("/").split("/")[-1] or "downloaded_audio"
        dest = work_dir / fname
        urllib.request.urlretrieve(source, dest)
        return dest

    raise FileNotFoundError(f"Audio source not found: {source}")


def decode_canonical(raw_path: Path, work_dir: Path):
    """Transcode any audio/video container to 44.1kHz stereo float32 WAV."""
    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()[:12]
    wav_path = work_dir / f"{sha}_canonical.wav"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(raw_path),
        "-vn", "-ac", "2", "-ar", str(CANONICAL_SR),
        "-c:a", "pcm_f32le", str(wav_path)
    ]
    subprocess.run(cmd, check=True)
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    stereo = data.T  # shape: (2, N)
    return stereo, sr, sha, raw_path.stem
''',

"engine/decompose.py": r'''"""Axis 1: Signal decomposition (Stereo M/S, HPSS, and CPU htdemucs 4-stem split)."""
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
                st = sources[idx].astype(np.float32)  # (2, N)
                stems_stereo[name] = st
                stems_mono[name] = (0.5 * (st[0] + st[1])).astype(np.float32)
            method = "htdemucs_4stem"
        except Exception as exc:
            print(f"[WARN] Demucs failed ({exc}); falling back to DSP split.")

    if not stems_mono:
        stems_mono = {
            "drums": perc_mix.astype(np.float32),
            "bass": _bandpass(harm_mix, sr, high=240.0),
            "vocals": _bandpass(harm_mix, sr, low=240.0, high=3200.0),
            "other": _bandpass(harm_mix, sr, low=180.0),
        }
        for k, v in stems_mono.items():
            stems_stereo[k] = np.stack([v, v], axis=0)

    # Unified lead/melodic carrier: captures vocals, lead guitars, solo instruments without bass bleed
    other_harm, _ = librosa.effects.hpss(stems_mono["other"], margin=1.5)
    lead_raw = stems_mono["vocals"] + other_harm
    lead_carrier = _bandpass(lead_raw, sr, low=200.0, high=5500.0)

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
        "stem_db_1s": stem_db_1s,
        "stem_rms_global_db": stem_rms_global_db,
    }
''',

"engine/pitch_harmony.py": r'''"""Axis 2: Pitch, Intonation, Microtonality, Polyphonic Harmony & Sensory Roughness."""
import numpy as np
import librosa
from scipy.signal import find_peaks

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Modal profiles (relative pitch-class weights anchored at tonic=0)
MODE_TEMPLATES = {
    "Ionian (Major)":    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "Aeolian (Minor)":   [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "Dorian":            [6.20, 2.20, 3.80, 5.10, 2.30, 3.70, 2.40, 4.90, 2.50, 4.20, 3.60, 2.40],
    "Phrygian":          [6.30, 4.60, 2.30, 4.90, 2.40, 3.80, 2.30, 4.80, 4.10, 2.30, 3.50, 2.40],
    "Lydian":            [6.20, 2.20, 3.70, 2.30, 4.60, 2.40, 4.30, 5.10, 2.30, 3.60, 2.30, 3.40],
    "Mixolydian":        [6.20, 2.20, 3.60, 2.30, 4.50, 3.90, 2.30, 5.10, 2.30, 3.50, 4.20, 2.30],
    "Locrian":           [6.00, 4.30, 2.30, 4.70, 2.30, 3.90, 4.50, 2.20, 3.90, 2.30, 3.70, 2.30],
}


def hz_to_note_cents(hz: float, a4_hz: float = 440.0) -> str:
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return "N/A"
    midi = 69.0 + 12.0 * np.log2(hz / a4_hz)
    nearest = int(np.round(midi))
    cents = int(np.round((midi - nearest) * 100.0))
    pc = PITCH_CLASSES[nearest % 12]
    octave = (nearest // 12) - 1
    if cents == 0:
        return f"{pc}{octave}"
    return f"{pc}{octave}{cents:+d}c"


def _run_pyin_contour(y: np.ndarray, sr: int, fmin: float, fmax: float, hop: int = 512):
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms < 1e-4:
        n_frames = max(1, 1 + len(y) // hop)
        return np.full(n_frames, np.nan), np.zeros(n_frames, dtype=bool)
    f0, voiced, probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop, fill_na=np.nan
    )
    valid = voiced & np.isfinite(f0) & (f0 > 0) & (probs > 0.25)
    f0_clean = np.where(valid, f0, np.nan)
    return f0_clean, valid


def _contour_stats(f0: np.ndarray, sr: int, hop: int, a4_hz: float) -> dict:
    valid_idx = np.where(np.isfinite(f0) & (f0 > 0))[0]
    total_frames = len(f0)
    if len(valid_idx) < 5:
        return {
            "voiced_pct": 0.0, "range_low": "N/A", "range_high": "N/A",
            "center_note": "N/A", "center_hz": None,
            "drift_gt15c_pct": 0.0, "drift_gt25c_pct": 0.0,
            "cent_mean": 0.0, "cent_std": 0.0, "cent_skew": 0.0,
            "cent_histogram_5c": [0] * 20,
            "pitch_class_gravity": {pc: 0.0 for pc in PITCH_CLASSES},
            "bend_count": 0, "bend_up_pct": 0.0, "bend_max_cents": 0.0,
            "vibrato_pct": 0.0, "vibrato_rate_hz": 0.0, "vibrato_depth_cents": 0.0,
            "intervals": {"step_0_2": 0.0, "skip_3_5": 0.0, "leap_6_12": 0.0, "wide_gt12": 0.0},
        }

    vals = f0[valid_idx]
    midi = 69.0 + 12.0 * np.log2(vals / a4_hz)
    nearest = np.round(midi).astype(int)
    cents = (midi - nearest) * 100.0

    # 5-cent histogram from -50 to +50 (20 bins)
    hist, _ = np.histogram(cents, bins=20, range=(-50.0, 50.0))
    hist_pct = [round(100.0 * float(h) / len(cents), 2) for h in hist]

    # Pitch class gravity
    pc_counts = np.bincount(nearest % 12, minlength=12)
    pc_grav = {
        PITCH_CLASSES[i]: round(100.0 * float(pc_counts[i]) / len(nearest), 2)
        for i in range(12)
    }

    # Skewness of cent deviations (+ skew = upward bends!)
    c_mean = float(np.mean(cents))
    c_std = float(np.std(cents) + 1e-8)
    c_skew = float(np.mean(((cents - c_mean) / c_std) ** 3))

    # Contiguous voiced segments for bends, vibrato, and melodic intervals
    frame_dt = hop / sr
    midi_full = np.where(np.isfinite(f0) & (f0 > 0), 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / a4_hz), np.nan)

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

        # Bend/glissando detection over ~180ms rolling window
        win = max(3, int(round(0.18 / frame_dt)))
        if len(seg) > win:
            deltas_cents = (seg[win:] - seg[:-win]) * 100.0
            # Peak excursions > 45c and < 400c (exclude abrupt octave leaps)
            pos_peaks, _ = find_peaks(deltas_cents, height=(45.0, 400.0), distance=win)
            neg_peaks, _ = find_peaks(-deltas_cents, height=(45.0, 400.0), distance=win)
            bends_up += len(pos_peaks)
            bends_down += len(neg_peaks)
            if len(pos_peaks):
                max_bend = max(max_bend, float(np.max(deltas_cents[pos_peaks])))
            if len(neg_peaks):
                max_bend = max(max_bend, float(np.max(-deltas_cents[neg_peaks])))

        # Vibrato detection on sustained notes >= 0.35s
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
                if depth_est >= 12.0 and depth_est <= 150.0 and ratio > 0.12:
                    vib_frames += len(seg)
                    vib_rates.append(float(sub_freqs[pk_idx]))
                    vib_depths.append(depth_est)

    # Melodic intervals between phrase/note segments
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
    """Sethares / Plomp-Levelt physics-based sensory roughness across top spectral peaks."""
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
        if len(pks) > K:
            top = pks[np.argpartition(col[pks], -K)[-K:]]
        else:
            top = pks
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


def _estimate_modal_key(chroma_vec: np.ndarray) -> dict:
    best_corr = -2.0
    best_tonic = "A"
    best_mode = "Aeolian (Minor)"
    c_norm = chroma_vec - np.mean(chroma_vec)
    c_den = np.linalg.norm(c_norm) + 1e-12

    candidates = []
    for mode_name, tmpl in MODE_TEMPLATES.items():
        t_arr = np.array(tmpl, dtype=np.float64)
        for root in range(12):
            shifted = np.roll(t_arr, root)
            s_norm = shifted - np.mean(shifted)
            corr = float(np.dot(c_norm, s_norm) / (c_den * (np.linalg.norm(s_norm) + 1e-12)))
            candidates.append((corr, PITCH_CLASSES[root], mode_name))
            if corr > best_corr:
                best_corr = corr
                best_tonic = PITCH_CLASSES[root]
                best_mode = mode_name

    candidates.sort(reverse=True)
    runner_up = f"{candidates[1][1]} {candidates[1][2]} ({candidates[1][0]:.2f})"
    return {
        "tonic": best_tonic,
        "mode": best_mode,
        "confidence": round(best_corr, 3),
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
    total_sec = max(1, int(np.floor(len(mono) / sr)))
    hop = 512

    # 1. Tuning calibration
    tuning_bins = float(librosa.estimate_tuning(y=harm, sr=sr, bins_per_octave=36))
    tuning_cents = tuning_bins * (100.0 / 3.0)
    a4_hz = 440.0 * (2.0 ** (tuning_cents / 1200.0))

    # 2. Stem-isolated pYIN contours
    bass_f0, _ = _run_pyin_contour(bass_y, sr, fmin=28.0, fmax=350.0, hop=hop)
    lead_f0, _ = _run_pyin_contour(lead_y, sr, fmin=180.0, fmax=2400.0, hop=hop)

    bass_stats = _contour_stats(bass_f0, sr, hop, a4_hz)
    lead_stats = _contour_stats(lead_f0, sr, hop, a4_hz)

    # 3. Polyphonic 12-bin Chroma + 36-bin Microtonal CQT
    chroma12 = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=hop, tuning=tuning_bins)
    chroma_mean = np.mean(chroma12, axis=1)
    chroma_pct = chroma_mean / (np.sum(chroma_mean) + 1e-12) * 100.0
    poly_gravity = {PITCH_CLASSES[i]: round(float(chroma_pct[i]), 2) for i in range(12)}

    chroma36 = librosa.feature.chroma_cqt(
        y=harm, sr=sr, hop_length=hop, n_chroma=36, bins_per_octave=36, tuning=tuning_bins
    )
    c36_mean = np.mean(chroma36, axis=1)
    # Off-grid third-semitone energy (indices not divisible by 3)
    off_grid_mask = np.array([i % 3 != 0 for i in range(36)])
    microtonal_cqt_ratio = float(np.sum(c36_mean[off_grid_mask]) / (np.sum(c36_mean) + 1e-12))

    modal_key = _estimate_modal_key(chroma_mean)

    # 4. Tonnetz & Harmonic Change Detection Function (HCDF)
    tonnetz = librosa.feature.tonnetz(chroma=chroma12)
    hcdf_frames = np.linalg.norm(np.diff(tonnetz, axis=1, prepend=tonnetz[:, :1]), axis=0)
    hcdf_peaks, _ = find_peaks(hcdf_frames, height=float(np.mean(hcdf_frames) + np.std(hcdf_frames)), distance=int(0.5 * sr / hop))
    hcdf_rate = float(len(hcdf_peaks) / max(1.0, len(mono) / sr))

    # 5. Physics-based Sensory Roughness (Plomp-Levelt)
    rough_frames = _sensory_roughness(mono, sr)

    # 1-second aligned arrays
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
''',

"engine/rhythm_forensics.py": r'''"""Axis 3: Macro Pulse, Meter & Dual-Layer Sovereign Timing Forensics."""
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
''',

"engine/timbre_physics.py": r'''"""Axis 4: 6-Band Somatic Psychoacoustics (Raw + A-Weighted), Saturation, HNR & MFCCs."""
import numpy as np
import librosa

SOMATIC_BANDS = [
    ("sub",      20.0,    60.0),
    ("bass",     60.0,   180.0),
    ("chest",   180.0,   500.0),
    ("presence", 500.0,  2500.0),
    ("edge",    2500.0,  6000.0),
    ("air",     6000.0, 16000.0),
]


def _a_weighting_linear(freqs: np.ndarray) -> np.ndarray:
    """Exact IEC 61672 A-weighting amplitude gain curve normalized to 1.0 at 1000 Hz."""
    f = np.maximum(freqs, 1.0)
    f2 = f ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    f1k = 1000.0 ** 2
    ra_1k = (12194.0 ** 2 * f1k ** 2) / (
        (f1k + 20.6 ** 2)
        * np.sqrt((f1k + 107.7 ** 2) * (f1k + 737.9 ** 2))
        * (f1k + 12194.0 ** 2)
    )
    return ra / ra_1k


def _stat_pack(arr: np.ndarray, ndigits: int = 2) -> dict:
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    return {
        "mean": round(float(np.mean(arr)), ndigits),
        "std": round(float(np.std(arr)), ndigits),
        "p10": round(float(p10), ndigits),
        "p50": round(float(p50), ndigits),
        "p90": round(float(p90), ndigits),
    }


def analyze_timbre_physics(decomp: dict, sr: int) -> dict:
    mono = decomp["mono"]
    harm = decomp["harmonic_mix"]
    perc = decomp["percussive_mix"]
    total_sec = max(1, int(np.floor(len(mono) / sr)))

    n_fft = 4096
    hop = 512
    S_mag = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    S_pow = S_mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # A-weighted power spectrum (what human ears actually perceive)
    a_gain = _a_weighting_linear(freqs)[:, None]
    S_pow_aw = (S_mag * a_gain) ** 2

    raw_energies = {}
    aw_energies = {}
    bands_db_frames = []

    for name, lo, hi in SOMATIC_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        raw_e = float(np.sum(S_pow[mask, :]))
        aw_e = float(np.sum(S_pow_aw[mask, :]))
        raw_energies[name] = raw_e
        aw_energies[name] = aw_e
        frame_e = np.mean(S_pow_aw[mask, :], axis=0)
        bands_db_frames.append(10.0 * np.log10(frame_e + 1e-12))

    raw_tot = sum(raw_energies.values()) + 1e-12
    aw_tot = sum(aw_energies.values()) + 1e-12
    aw_max = max(aw_energies.values()) + 1e-12

    somatic_profile = {
        "perceptual_a_weighted_pct": {k: round(100.0 * v / aw_tot, 2) for k, v in aw_energies.items()},
        "physical_raw_power_pct": {k: round(100.0 * v / raw_tot, 2) for k, v in raw_energies.items()},
        "perceptual_rel_db": {k: round(10.0 * np.log10((v + 1e-12) / aw_max), 2) for k, v in aw_energies.items()},
    }

    # Spectral geometry
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    rolloff_85 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85)[0]
    rolloff_95 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=S_mag)[0]
    zcr = librosa.feature.zero_crossing_rate(y=mono, frame_length=n_fft, hop_length=hop)[0]

    # Spectral flux (normalized frame-to-frame L2 change)
    S_norm = S_mag / (np.linalg.norm(S_mag, axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt(np.sum(np.maximum(0.0, np.diff(S_norm, axis=1, prepend=S_norm[:, :1])) ** 2, axis=0))

    # Spectral slope / tilt (dB per octave between 100 Hz and 10 kHz)
    slope_mask = (freqs >= 100.0) & (freqs <= 10000.0)
    mean_spec_db = 10.0 * np.log10(np.mean(S_pow[slope_mask, :], axis=1) + 1e-12)
    log2_f = np.log2(freqs[slope_mask])
    spectral_slope = float(np.polyfit(log2_f, mean_spec_db, 1)[0])

    # Harmonic-to-Noise Ratio (HNR) & Percussive Energy Ratio
    e_harm = float(np.mean(harm ** 2) + 1e-12)
    e_perc = float(np.mean(perc ** 2) + 1e-12)
    residual = mono - harm - perc
    e_noise = float(np.mean(residual ** 2) + 1e-12)
    hnr_db = 10.0 * np.log10(e_harm / (e_noise + 0.5 * e_perc))
    perc_ratio = e_perc / (e_harm + e_perc)

    # Saturation / Distortion Texture Index (0=pure/clean, 1=heavy wall-of-distortion)
    edge_share = (aw_energies["edge"] + aw_energies["presence"]) / aw_tot
    flat_term = min(1.0, float(np.mean(flatness)) / 0.06)
    slope_term = min(1.0, max(0.0, (spectral_slope + 12.0) / 10.0))
    saturation_index = round(float(0.45 * flat_term + 0.35 * slope_term + 0.20 * edge_share), 3)

    # 20 MFCCs + Deltas
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc_d = librosa.feature.delta(mfcc)

    # 1-second aligned timeseries
    idx_edges = np.linspace(0, S_mag.shape[1], total_sec + 1).astype(int)
    centroid_1s, flatness_1s, flux_1s = [], [], []
    mfcc_1s = np.zeros((12, total_sec), dtype=np.float32)
    bands_db_1s = []

    bands_arr = np.vstack(bands_db_frames)  # (6, frames)
    for i in range(total_sec):
        s, e = idx_edges[i], max(idx_edges[i] + 1, idx_edges[i + 1])
        centroid_1s.append(round(float(np.mean(centroid[s:e])), 1))
        flatness_1s.append(round(float(np.mean(flatness[s:e])), 5))
        flux_1s.append(round(float(np.mean(flux[s:e])), 4))
        mfcc_1s[:, i] = np.mean(mfcc[:12, s:e], axis=1)
        bands_db_1s.append([round(float(x), 1) for x in np.mean(bands_arr[:, s:e], axis=1)])

    return {
        "summary": {
            "somatic_6band": somatic_profile,
            "centroid_hz": _stat_pack(centroid, 1),
            "bandwidth_hz": _stat_pack(bandwidth, 1),
            "rolloff_85_hz": _stat_pack(rolloff_85, 1),
            "rolloff_95_hz": _stat_pack(rolloff_95, 1),
            "spectral_flatness": _stat_pack(flatness, 5),
            "spectral_flux": _stat_pack(flux, 4),
            "zcr": _stat_pack(zcr, 4),
            "spectral_slope_db_oct": round(spectral_slope, 2),
            "hnr_db": round(float(hnr_db), 2),
            "percussive_energy_ratio": round(float(perc_ratio), 3),
            "saturation_index": saturation_index,
            "mfcc_mean": [round(float(x), 3) for x in np.mean(mfcc, axis=1)],
            "mfcc_std": [round(float(x), 3) for x in np.std(mfcc, axis=1)],
            "mfcc_delta_std": [round(float(x), 3) for x in np.std(mfcc_d, axis=1)],
        },
        "timeseries": {
            "centroid_hz": centroid_1s,
            "flatness": flatness_1s,
            "flux": flux_1s,
            "bands_db": bands_db_1s,
            "mfcc_1s": mfcc_1s,
        },
    }
''',

"engine/space_dynamics.py": r'''"""Axis 5: ITU-R BS.1770 LUFS/LRA, Micro-Dynamics, Stereo Width & Reverb Decay Proxy."""
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
''',

"engine/structure.py": r'''"""Axis 6: Fused Multi-Modal Structural Segmentation & Per-Phase Matrix."""
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
''',

"engine/render.py": r'''"""Axis 7: Lossless Genome JSON, Compressed Timeseries, LLM Dossier & SQLite+FTS5 Catalog."""
import datetime
import gzip
import json
import math
import sqlite3
from pathlib import Path
import numpy as np
from engine import SCHEMA_VERSION
from engine.ingest import slugify


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def build_acoustic_vector(genome: dict) -> list:
    """Compact 64-D acoustic vector for future cosine-similarity associative search."""
    p = genome["pitch_harmony"]
    r = genome["rhythm_forensics"]
    t = genome["timbre_physics"]
    s = genome["space_dynamics"]

    vec = []
    # 6 perceptual bands + 6 raw bands (12)
    for k in ["sub", "bass", "chest", "presence", "edge", "air"]:
        vec.append(t["somatic_6band"]["perceptual_a_weighted_pct"][k] / 100.0)
    for k in ["sub", "bass", "chest", "presence", "edge", "air"]:
        vec.append(t["somatic_6band"]["physical_raw_power_pct"][k] / 100.0)
    # 12 polyphonic chroma (24)
    for pc in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
        vec.append(p["polyphonic_chroma_gravity"][pc] / 100.0)
    # 20 MFCC means normalized by 100 (44)
    vec.extend([x / 100.0 for x in t["mfcc_mean"][:20]])
    # 20 global physics/rhythm/pitch/space scalars (64)
    vec.extend([
        r["primary_bpm"] / 200.0,
        r["drums_layer"]["nps"] / 10.0,
        r["melodic_layer"]["nps"] / 10.0,
        r["drums_layer"]["ioi_entropy"],
        r["drums_layer"]["hartzig_residual_ms"] / 100.0,
        r["drums_layer"]["pink_noise_beta"] / 2.0,
        r["drums_layer"]["ghost_swell_pct"] / 100.0,
        p["lead_contour"]["drift_gt15c_pct"] / 100.0,
        p["lead_contour"]["vibrato_pct"] / 100.0,
        min(1.0, p["lead_contour"]["bend_count"] / 100.0),
        p["sensory_roughness_mean"] * 5.0,
        p["harmonic_change_rate_hz"],
        t["centroid_hz"]["mean"] / 8000.0,
        t["spectral_flatness"]["mean"] * 10.0,
        (t["spectral_slope_db_oct"] + 15.0) / 15.0,
        t["percussive_energy_ratio"],
        t["saturation_index"],
        (s["integrated_lufs"] + 40.0) / 40.0,
        s["loudness_range_lra_lu"] / 20.0,
        s["stereo_width_global"],
    ])
    return [round(float(x), 4) for x in vec]


def format_dossier_md(genome: dict) -> str:
    m = genome["meta"]
    d = genome["decomposition"]
    p = genome["pitch_harmony"]
    r = genome["rhythm_forensics"]
    t = genome["timbre_physics"]
    s = genome["space_dynamics"]
    phases = genome["structure_phases"]

    dur = int(m["duration_s"])
    dur_str = f"{dur // 60}:{dur % 60:02d}"

    # Sort pitch gravity descending (top 8 shown compactly, all 12 in JSON)
    poly_sorted = sorted(p["polyphonic_chroma_gravity"].items(), key=lambda x: x[1], reverse=True)
    bass_sorted = sorted(p["bass_contour"]["pitch_class_gravity"].items(), key=lambda x: x[1], reverse=True)
    poly_str = " | ".join([f"{k}:{v:.1f}%" for k, v in poly_sorted[:7]])
    bass_str = " | ".join([f"{k}:{v:.1f}%" for k, v in bass_sorted[:6]])

    bc = p["bass_contour"]
    lc = p["lead_contour"]
    dl = r["drums_layer"]
    ml = r["melodic_layer"]
    hyps = ", ".join([f"{h['bpm']} ({h['salience']:.2f})" for h in r["tempo_hypotheses"]])

    aw = t["somatic_6band"]["perceptual_a_weighted_pct"]
    rw = t["somatic_6band"]["physical_raw_power_pct"]
    stems_db = " | ".join([f"{k}:{v:+.1f}dB" for k, v in d["stem_rms_global_db"].items()])

    lines = [
        "=== LAYER 0 v2 GROUND-TRUTH AUDIO GENOME (DETERMINISTIC DSP — DO NOT CONTRADICT) ===",
        f"TITLE: {m['title']} | ID: {m['track_id']} | DURATION: {dur_str} | DECOMP: {d['method']}",
        f"SOURCE: {m['source']}",
        f"SEED_ID: {m['seed_id'] or 'N/A'} | TAGS: {m['tags'] or 'N/A'} | CAPTURED: {m['captured_utc']}",
        f"STEM LEVELS (Global RMS): {stems_db}",
        "",
        "[AXIS 2: PITCH, HARMONY & MICROTONALITY]",
        f"  Modal Key: {p['modal_key']['tonic']} {p['modal_key']['mode']} (conf {p['modal_key']['confidence']:.3f} | alt: {p['modal_key']['runner_up']})",
        f"  Tuning: A4={p['a4_ref_hz']} Hz ({p['tuning_offset_cents']:+.1f}c) | Off-Grid 36-CQT: {p['microtonal_cqt_offgrid_pct']}% | HCDF Rate: {p['harmonic_change_rate_hz']}/s",
        f"  Sensory Roughness (Plomp-Levelt): Mean {p['sensory_roughness_mean']} | P90 {p['sensory_roughness_p90']}",
        f"  Polyphonic Gravity: {poly_str}",
        f"  Bass Stem Gravity:  {bass_str}",
        f"  Bass F0 Contour:    Voiced {bc['voiced_pct']}% | Range {bc['range_low']}..{bc['range_high']} (Center {bc['center_note']}) | Drift>15c {bc['drift_gt15c_pct']}%",
        f"  Lead F0 Contour:    Voiced {lc['voiced_pct']}% | Range {lc['range_low']}..{lc['range_high']} (Center {lc['center_note']}) | Drift>15c {lc['drift_gt15c_pct']}% (>25c {lc['drift_gt25c_pct']}%)",
        f"  Lead Expressivity:  Cent Skew {lc['cent_skew']:+.3f} | Bends {lc['bend_count']} ({lc['bend_up_pct']}% up, max {lc['bend_max_cents']}c) | Vibrato {lc['vibrato_pct']}% ({lc['vibrato_rate_hz']}Hz, ±{lc['vibrato_depth_cents']}c)",
        f"  Lead Intervals:     Steps(0-2st) {lc['intervals']['step_0_2']}% | Skips(3-5st) {lc['intervals']['skip_3_5']}% | Leaps(6-12st) {lc['intervals']['leap_6_12']}% | Wide(>12st) {lc['intervals']['wide_gt12']}%",
        "",
        "[AXIS 3: RHYTHM, METER & SOVEREIGN MICRO-TIMING]",
        f"  Pulse & Meter: BPM {r['primary_bpm']} (Hypotheses: {hyps}) | Local Drift: {r['local_tempo_range_bpm'][0]}..{r['local_tempo_range_bpm'][1]} BPM (std {r['local_tempo_std_bpm']}) | Meter: {r['meter_hint']}",
        f"  DRUMS Layer:   NPS {dl['nps']} | IOI Mean {dl['ioi_mean_ms']}ms / Med {dl['ioi_median_ms']}ms (IQR {dl['ioi_iqr_ms']}ms) | Entropy {dl['ioi_entropy']} | Swing {dl['swing_ratio']}",
        f"                 Hartzig R_H {dl['hartzig_residual_ms']}ms | 1/f Beta {dl['pink_noise_beta']} | DFA {dl['dfa_alpha']} | Syncopation {dl['syncopation_index']} | Attack {dl['attack_median_ms']}ms (P90 {dl['attack_p90_ms']}ms) | Ghost {dl['ghost_swell_pct']}%",
        f"  MELODIC Layer: NPS {ml['nps']} | IOI Mean {ml['ioi_mean_ms']}ms / Med {ml['ioi_median_ms']}ms (IQR {ml['ioi_iqr_ms']}ms) | Entropy {ml['ioi_entropy']} | Swing {ml['swing_ratio']}",
        f"                 Hartzig R_H {ml['hartzig_residual_ms']}ms | 1/f Beta {ml['pink_noise_beta']} | DFA {ml['dfa_alpha']} | Syncopation {ml['syncopation_index']} | Attack {ml['attack_median_ms']}ms (P90 {ml['attack_p90_ms']}ms) | Ghost {ml['ghost_swell_pct']}%",
        "",
        "[AXIS 4: TIMBRE, SATURATION & 6-BAND SOMATIC PSYCHOACOUSTICS]",
        f"  Perceptual (A-Wt): Sub(20-60) {aw['sub']}% | Bass(60-180) {aw['bass']}% | Chest(180-500) {aw['chest']}% | Presence(500-2.5k) {aw['presence']}% | Edge(2.5k-6k) {aw['edge']}% | Air(6k-16k) {aw['air']}%",
        f"  Physical (Raw |X|²): Sub {rw['sub']}% | Bass {rw['bass']}% | Chest {rw['chest']}% | Presence {rw['presence']}% | Edge {rw['edge']}% | Air {rw['air']}%",
        f"  Spectral Shape:    Centroid {t['centroid_hz']['mean']}Hz (P10 {t['centroid_hz']['p10']} / P90 {t['centroid_hz']['p90']}) | RollOff85 {t['rolloff_85_hz']['mean']}Hz | RollOff95 {t['rolloff_95_hz']['mean']}Hz | Slope {t['spectral_slope_db_oct']} dB/oct",
        f"  Texture/Gain:      Flatness {t['spectral_flatness']['mean']} (P90 {t['spectral_flatness']['p90']}) | HNR {t['hnr_db']}dB | Percussive Ratio {t['percussive_energy_ratio']} | Saturation Index {t['saturation_index']} (0=clean, 1=wall)",
        "",
        "[AXIS 5: LOUDNESS, DYNAMICS & SPATIAL FIELD]",
        f"  Loudness (R128):   Integrated {s['integrated_lufs']} LUFS | Range (LRA) {s['loudness_range_lra_lu']} LU | True Peak {s['true_peak_dbtp']} dBTP | Clipping {s['clipping_ratio_pct']}%",
        f"  Micro-Dynamics:    Mix Crest {s['crest_factor_db']} dB | Stem Crests: drums {s['stem_crest_db']['drums']}dB, bass {s['stem_crest_db']['bass']}dB, vocals {s['stem_crest_db']['vocals']}dB, other {s['stem_crest_db']['other']}dB",
        f"  Stereo Field:      Global Width {s['stereo_width_global']} (Low {s['stereo_width_bands']['low_lt250']} | Mid {s['stereo_width_bands']['mid_250_4k']} | High {s['stereo_width_bands']['high_gt4k']}) | Phase Corr {s['phase_correlation']} | L/R Bal {s['lr_balance_db']:+.2f}dB",
        f"  Room / Decay:      Percussive -10dB Decay {s['reverb_proxy']['transient_decay_10db_ms']}ms | Dryness {s['reverb_proxy']['dryness_index']} | Side/Mid Ratio {s['reverb_proxy']['side_to_mid_ratio']}",
        "",
        "[AXIS 6: FUSED CHRONOLOGICAL STRUCTURE & PHASE TELEMETRY]",
    ]

    for ph in phases:
        st_str = f"D:{ph['stems']['drums']} B:{ph['stems']['bass']} V:{ph['stems']['vocals']} O:{ph['stems']['other']}"
        lines.append(
            f"  P{ph['index']:02d} [{ph['range']}] {ph['lufs']:>5.1f} LUFS (Cr {ph['crest_db']:>4.1f}dB, W {ph['stereo_width']:.2f}) | "
            f"Stems[{st_str}] | Bass:{ph['bass_center']} Lead:{ph['lead_range']} (ctr {ph['lead_center']}) | "
            f"NPS D:{ph['drum_nps']:.1f}/M:{ph['melodic_nps']:.1f} | Bright:{int(ph['centroid_hz'])}Hz Flat:{ph['flatness']:.4f}"
        )

    lines.extend([
        "",
        f"NOTES / CONTEXT: {m['notes'] or '(none)'}",
        "====================================================================================",
    ])
    return "\n".join(lines) + "\n"


def upsert_sqlite(db_path: Path, genome: dict, vector: list, dossier_text: str):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        track_id TEXT PRIMARY KEY,
        schema_version TEXT,
        title TEXT,
        source TEXT,
        seed_id TEXT,
        tags TEXT,
        notes TEXT,
        duration_s REAL,
        decomp_method TEXT,
        captured_utc TEXT,
        bpm REAL,
        modal_tonic TEXT,
        modal_mode TEXT,
        integrated_lufs REAL,
        lra_lu REAL,
        crest_db REAL,
        saturation_index REAL,
        lead_drift_gt15c_pct REAL,
        lead_bend_count INTEGER,
        stereo_width REAL,
        genome_json TEXT,
        acoustic_vector_json TEXT,
        dossier_md TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS phases (
        track_id TEXT,
        phase_index INTEGER,
        start_s INTEGER,
        end_s INTEGER,
        lufs REAL,
        crest_db REAL,
        stereo_width REAL,
        drums_state TEXT,
        bass_state TEXT,
        vocals_state TEXT,
        other_state TEXT,
        bass_center TEXT,
        lead_range TEXT,
        drum_nps REAL,
        melodic_nps REAL,
        centroid_hz REAL,
        flatness REAL,
        PRIMARY KEY (track_id, phase_index)
    )""")

    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS tracks_fts USING fts5(
        track_id UNINDEXED, title, seed_id, tags, notes, modal_key, dossier_md
    )""")

    m = genome["meta"]
    p = genome["pitch_harmony"]
    r = genome["rhythm_forensics"]
    t = genome["timbre_physics"]
    s = genome["space_dynamics"]

    cur.execute("""
    INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        m["track_id"], SCHEMA_VERSION, m["title"], m["source"], m["seed_id"], m["tags"], m["notes"],
        m["duration_s"], genome["decomposition"]["method"], m["captured_utc"],
        r["primary_bpm"], p["modal_key"]["tonic"], p["modal_key"]["mode"],
        s["integrated_lufs"], s["loudness_range_lra_lu"], s["crest_factor_db"],
        t["saturation_index"], p["lead_contour"]["drift_gt15c_pct"], p["lead_contour"]["bend_count"],
        s["stereo_width_global"], json.dumps(genome), json.dumps(vector), dossier_text
    ))

    cur.execute("DELETE FROM phases WHERE track_id = ?", (m["track_id"],))
    for ph in genome["structure_phases"]:
        cur.execute("""
        INSERT INTO phases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            m["track_id"], ph["index"], ph["start_s"], ph["end_s"],
            ph["lufs"], ph["crest_db"], ph["stereo_width"],
            ph["stems"]["drums"], ph["stems"]["bass"], ph["stems"]["vocals"], ph["stems"]["other"],
            ph["bass_center"], ph["lead_range"], ph["drum_nps"], ph["melodic_nps"],
            ph["centroid_hz"], ph["flatness"]
        ))

    cur.execute("DELETE FROM tracks_fts WHERE track_id = ?", (m["track_id"],))
    cur.execute("""
    INSERT INTO tracks_fts (track_id, title, seed_id, tags, notes, modal_key, dossier_md)
    VALUES (?,?,?,?,?,?,?)
    """, (
        m["track_id"], m["title"], m["seed_id"], m["tags"], m["notes"],
        f"{p['modal_key']['tonic']} {p['modal_key']['mode']}", dossier_text
    ))

    conn.commit()
    conn.close()


def save_outputs(repo_root: Path, genome: dict, timeseries: dict) -> Path:
    genome_clean = sanitize(genome)
    ts_clean = sanitize(timeseries)
    vector = build_acoustic_vector(genome_clean)
    genome_clean["acoustic_vector_64d"] = vector
    dossier_md = format_dossier_md(genome_clean)

    slug = f"{slugify(genome_clean['meta']['title'])}_{genome_clean['meta']['track_id']}"
    out_dir = repo_root / "archive" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "genome.json").write_text(json.dumps(genome_clean, indent=2), encoding="utf-8")
    with gzip.open(out_dir / "timeseries.json.gz", "wt", encoding="utf-8") as gz:
        json.dump(ts_clean, gz, separators=(",", ":"))
    (out_dir / "dossier.md").write_text(dossier_md, encoding="utf-8")
    (repo_root / "latest_dossier.md").write_text(dossier_md, encoding="utf-8")

    upsert_sqlite(repo_root / "db" / "layer0_catalog.db", genome_clean, vector, dossier_md)
    return out_dir
''',

"run_pipeline.py": r'''#!/usr/bin/env python3
"""Layer 0 v2 Audio Genome Pipeline Orchestrator."""
import argparse
import datetime
import shutil
import sqlite3
import tempfile
from pathlib import Path

from engine import SCHEMA_VERSION
from engine.ingest import parse_queue, fetch_audio, decode_canonical
from engine.decompose import decompose_signal
from engine.pitch_harmony import analyze_pitch_harmony
from engine.rhythm_forensics import analyze_rhythm_forensics
from engine.timbre_physics import analyze_timbre_physics
from engine.space_dynamics import analyze_space_dynamics
from engine.structure import analyze_structure
from engine.render import save_outputs


def is_already_in_db(db_path: Path, track_id: str) -> bool:
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    try:
        cur.execute("SELECT schema_version FROM tracks WHERE track_id = ?", (track_id,))
        row = cur.fetchone()
        return row is not None and row[0] == SCHEMA_VERSION
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def process_entry(entry: dict, repo_root: Path, use_demucs: bool, force: bool) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="layer0_"))
    try:
        print(f"\n>>> Fetching: {entry['source']}")
        raw_path = fetch_audio(entry["source"], work_dir)
        stereo, sr, track_id, default_title = decode_canonical(raw_path, work_dir)
        title = entry["title"] or default_title

        db_path = repo_root / "db" / "layer0_catalog.db"
        if not force and is_already_in_db(db_path, track_id):
            print(f"[SKIP] Track {track_id} ({title}) already captured at schema {SCHEMA_VERSION}.")
            return track_id

        dur_s = round(stereo.shape[1] / sr, 2)
        print(f">>> Analyzing '{title}' (ID={track_id}, {dur_s}s, demucs={use_demucs})...")

        decomp = decompose_signal(stereo, sr, use_demucs=use_demucs)
        print(f"    [Axis 1] Decomposed via {decomp['method']}")

        pitch = analyze_pitch_harmony(decomp, sr)
        print(f"    [Axis 2] Pitch/Harmony: {pitch['summary']['modal_key']['tonic']} {pitch['summary']['modal_key']['mode']}")

        rhythm = analyze_rhythm_forensics(decomp, sr)
        print(f"    [Axis 3] Rhythm: {rhythm['summary']['primary_bpm']} BPM")

        timbre = analyze_timbre_physics(decomp, sr)
        print(f"    [Axis 4] Timbre: Centroid {timbre['summary']['centroid_hz']['mean']}Hz | Sat {timbre['summary']['saturation_index']}")

        space = analyze_space_dynamics(decomp, sr)
        print(f"    [Axis 5] Dynamics/Space: {space['summary']['integrated_lufs']} LUFS | LRA {space['summary']['loudness_range_lra_lu']} LU")

        phases = analyze_structure(decomp, pitch, rhythm, timbre, space, sr)
        print(f"    [Axis 6] Structure: {len(phases)} fused phases detected")

        T = max(1, int(stereo.shape[1] // sr))
        genome = {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "track_id": track_id,
                "title": title,
                "source": entry["source"],
                "seed_id": entry["seed_id"],
                "tags": entry["tags"],
                "notes": entry["notes"],
                "duration_s": dur_s,
                "sample_rate": sr,
                "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "decomposition": {
                "method": decomp["method"],
                "stem_rms_global_db": decomp["stem_rms_global_db"],
            },
            "pitch_harmony": pitch["summary"],
            "rhythm_forensics": rhythm["summary"],
            "timbre_physics": timbre["summary"],
            "space_dynamics": space["summary"],
            "structure_phases": phases,
        }

        timeseries = {
            "t_sec": list(range(T)),
            "stem_db": {k: v[:T] for k, v in decomp["stem_db_1s"].items()},
            "st_lufs": space["timeseries"]["st_lufs"][:T],
            "stereo_width": space["timeseries"]["stereo_width"][:T],
            "bass_f0_hz": pitch["timeseries"]["bass_f0_hz"][:T],
            "lead_f0_hz": pitch["timeseries"]["lead_f0_hz"][:T],
            "roughness": pitch["timeseries"]["roughness"][:T],
            "hcdf": pitch["timeseries"]["hcdf"][:T],
            "drum_nps": rhythm["timeseries"]["drum_nps"][:T],
            "melodic_nps": rhythm["timeseries"]["melodic_nps"][:T],
            "centroid_hz": timbre["timeseries"]["centroid_hz"][:T],
            "flatness": timbre["timeseries"]["flatness"][:T],
            "flux": timbre["timeseries"]["flux"][:T],
            "bands_db": timbre["timeseries"]["bands_db"][:T],
        }

        out_dir = save_outputs(repo_root, genome, timeseries)
        print(f"    [Axis 7] Saved genome, timeseries.json.gz, dossier.md & SQLite -> {out_dir}")
        return track_id
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Layer 0 v2 Audio Genome Engine")
    parser.add_argument("--fast", action="store_true", help="Skip Demucs neural stems (pure DSP fallback)")
    parser.add_argument("--force", action="store_true", help="Reprocess tracks even if already in SQLite DB")
    parser.add_argument("--file", type=str, default="", help="Process a single local file or URL directly")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    use_demucs = not args.fast

    if args.file:
        process_entry({
            "source": args.file, "title": "", "seed_id": "", "tags": "", "notes": ""
        }, repo_root, use_demucs=use_demucs, force=args.force)
        return

    queue_path = repo_root / "links_queue.txt"
    entries = parse_queue(queue_path)
    if not entries:
        print("Queue is empty (no active links in links_queue.txt).")
        return

    raw_lines = queue_path.read_text(encoding="utf-8").splitlines()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    changed = False

    for entry in entries:
        try:
            tid = process_entry(entry, repo_root, use_demucs=use_demucs, force=args.force)
            raw_lines[entry["line_index"]] = f"# [DONE {stamp} | {tid}] {entry['raw_line']}"
            changed = True
        except Exception as exc:
            print(f"[ERROR] Failed on {entry['source']}: {exc}")

    if changed:
        queue_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
''',
}


def unpack():
    root = Path(__file__).resolve().parent
    for rel_path, content in FILES.items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print(f"[UNPACKED] {rel_path}")
    print("\nLayer 0 v2 modular engine unpacked successfully.")


if __name__ == "__main__":
    unpack()