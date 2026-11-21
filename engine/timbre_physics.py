"""Axis 4: 6-Band Somatic Psychoacoustics (Raw + A-Weighted), Log Flatness, HNR & MFCCs."""
import numpy as np
import librosa

SOMATIC_BANDS = [
    ("sub",       20.0,    60.0),
    ("bass",      60.0,   180.0),
    ("chest",    180.0,   500.0),
    ("presence", 500.0,  2500.0),
    ("edge",    2500.0,  6000.0),
    ("air",     6000.0, 16000.0),
]


def _a_weighting_linear(freqs: np.ndarray) -> np.ndarray:
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

    a_gain = _a_weighting_linear(freqs)[:, None]
    S_pow_aw = (S_mag * a_gain) ** 2

    raw_energies, aw_energies, bands_db_frames = {}, {}, []
    for name, lo, hi in SOMATIC_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        raw_energies[name] = float(np.sum(S_pow[mask, :]))
        aw_energies[name] = float(np.sum(S_pow_aw[mask, :]))
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

    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    rolloff_85 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85)[0]
    rolloff_95 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=S_mag)[0]
    flatness_db = 10.0 * np.log10(np.maximum(flatness, 1e-10))
    zcr = librosa.feature.zero_crossing_rate(y=mono, frame_length=n_fft, hop_length=hop)[0]

    S_norm = S_mag / (np.linalg.norm(S_mag, axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt(np.sum(np.maximum(0.0, np.diff(S_norm, axis=1, prepend=S_norm[:, :1])) ** 2, axis=0))

    slope_mask = (freqs >= 100.0) & (freqs <= 10000.0)
    mean_spec_db = 10.0 * np.log10(np.mean(S_pow[slope_mask, :], axis=1) + 1e-12)
    spectral_slope = float(np.polyfit(np.log2(freqs[slope_mask]), mean_spec_db, 1)[0])

    e_harm = float(np.mean(harm ** 2) + 1e-12)
    e_perc = float(np.mean(perc ** 2) + 1e-12)
    residual = mono - harm - perc
    e_noise = float(np.mean(residual ** 2) + 1e-12)
    hnr_db = 10.0 * np.log10(e_harm / (e_noise + 0.5 * e_perc))
    perc_ratio = e_perc / (e_harm + e_perc)

    edge_share = (aw_energies["edge"] + aw_energies["presence"]) / aw_tot
    flat_term = min(1.0, max(0.0, (float(np.mean(flatness_db)) + 50.0) / 30.0))
    slope_term = min(1.0, max(0.0, (spectral_slope + 12.0) / 10.0))
    saturation_index = round(float(0.45 * flat_term + 0.35 * slope_term + 0.20 * edge_share), 3)

    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=20, hop_length=hop)
    mfcc_d = librosa.feature.delta(mfcc)

    idx_edges = np.linspace(0, S_mag.shape[1], total_sec + 1).astype(int)
    centroid_1s, flatness_1s, flatness_db_1s, flux_1s = [], [], [], []
    mfcc_1s = np.zeros((12, total_sec), dtype=np.float32)
    bands_db_1s = []
    bands_arr = np.vstack(bands_db_frames)

    for i in range(total_sec):
        s, e = idx_edges[i], max(idx_edges[i] + 1, idx_edges[i + 1])
        centroid_1s.append(round(float(np.mean(centroid[s:e])), 1))
        flatness_1s.append(round(float(np.mean(flatness[s:e])), 6))
        flatness_db_1s.append(round(float(np.mean(flatness_db[s:e])), 1))
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
            "spectral_flatness": _stat_pack(flatness, 6),
            "spectral_flatness_db": _stat_pack(flatness_db, 1),
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
            "flatness_db": flatness_db_1s,
            "flux": flux_1s,
            "bands_db": bands_db_1s,
            "mfcc_1s": mfcc_1s,
        },
    }
