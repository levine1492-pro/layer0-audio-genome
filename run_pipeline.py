#!/usr/bin/env python3
"""Layer 0 v2.1 Audio Genome Pipeline Orchestrator."""
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
        mk = pitch["summary"]["modal_key"]
        print(f"    [Axis 2] Pitch/Harmony: {mk['tonic']} {mk['mode']} (conf {mk['confidence']}, amb={mk['ambiguous']})")

        rhythm = analyze_rhythm_forensics(decomp, sr)
        print(f"    [Axis 3] Rhythm: {rhythm['summary']['primary_bpm']} BPM | Drone {rhythm['summary']['drone_sustain_index']}")

        timbre = analyze_timbre_physics(decomp, sr)
        print(f"    [Axis 4] Timbre: Centroid {timbre['summary']['centroid_hz']['mean']}Hz | Flat {timbre['summary']['spectral_flatness_db']['mean']}dB")

        space = analyze_space_dynamics(decomp, sr)
        print(f"    [Axis 5] Space: {space['summary']['integrated_lufs']} LUFS | {space['summary']['reverb_proxy']['space_profile']}")

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
            "flatness_db": timbre["timeseries"]["flatness_db"][:T],
            "flux": timbre["timeseries"]["flux"][:T],
            "bands_db": timbre["timeseries"]["bands_db"][:T],
        }

        out_dir = save_outputs(repo_root, genome, timeseries)
        print(f"    [Axis 7] Saved v{SCHEMA_VERSION} genome, timeseries, dossier & SQLite -> {out_dir}")
        return track_id
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Layer 0 v2.1 Audio Genome Engine")
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
