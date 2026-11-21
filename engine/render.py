"""Axis 7: Lossless Genome JSON, Compressed Timeseries, LLM Dossier & SQLite+FTS5 Catalog."""
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
