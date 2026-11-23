#!/usr/bin/env python3
"""Export lightweight catalog_index.json from archive/*/genome.json for external viewers."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
DB_PATH = ROOT / "db" / "layer0_catalog.db"
OUT_PATH = ROOT / "catalog_index.json"


def load_db_meta():
    meta = {}
    if not DB_PATH.exists():
        return meta
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        for row in con.execute("SELECT * FROM tracks"):
            d = dict(row)
            tid = d.get("track_id")
            if tid:
                meta[tid] = d
        con.close()
    except Exception:
        pass
    return meta


def pick(d, *paths, default=None):
    for p in paths:
        cur = d
        ok = True
        for k in p.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def main():
    db_meta = load_db_meta()
    items = []

    if ARCHIVE.exists():
        for folder in sorted(ARCHIVE.iterdir(), reverse=True):
            gpath = folder / "genome.json"
            if not folder.is_dir() or not gpath.exists():
                continue
            try:
                g = json.loads(gpath.read_text(encoding="utf-8"))
            except Exception:
                continue

            tid = pick(g, "track_id", "meta.track_id", "identity.track_id") or folder.name.rsplit("_", 1)[-1]
            row = db_meta.get(tid, {})

            title = pick(g, "meta.title", "identity.title", "title") or row.get("title") or folder.name
            seed_id = pick(g, "meta.seed_id", "identity.seed_id", "seed_id") or row.get("seed_id") or ""
            tags = pick(g, "meta.tags", "identity.tags", "tags") or row.get("tags") or ""
            if isinstance(tags, list):
                tags = ", ".join(str(x) for x in tags)

            schema_ver = pick(g, "schema_version", "meta.schema_version") or row.get("schema_version") or ""
            key_mode = (
                pick(g, "pitch_harmony.modal.best_label", "pitch_harmony.key_mode", "axis2_pitch_harmony.best_scale")
                or row.get("key_mode")
                or ""
            )
            bpm_val = pick(g, "rhythm.bpm_primary", "rhythm_forensics.bpm_primary", "axis3_rhythm.bpm_primary") or row.get("bpm")
            lufs_val = pick(g, "space_dynamics.integrated_lufs", "dynamics.integrated_lufs", "axis5_space_dynamics.integrated_lufs") or row.get("lufs")
            space = pick(g, "space_dynamics.space_profile", "axis5_space_dynamics.space_profile") or row.get("space_profile") or ""
            created = pick(g, "meta.extracted_at", "extracted_at") or row.get("created_at") or ""

            items.append({
                "track_id": tid,
                "archive_dir": f"archive/{folder.name}",
                "title": title,
                "seed_id": seed_id,
                "tags": tags,
                "schema_version": f"v{schema_ver}" if schema_ver and not str(schema_ver).startswith("v") else str(schema_ver),
                "key_mode": str(key_mode),
                "bpm": f"{float(bpm_val):.1f} BPM" if isinstance(bpm_val, (int, float)) else (str(bpm_val) if bpm_val else ""),
                "lufs": f"{float(lufs_val):.1f} LUFS" if isinstance(lufs_val, (int, float)) else (str(lufs_val) if lufs_val else ""),
                "space_profile": str(space),
                "created_at": str(created),
            })

    items.sort(key=lambda x: x.get("created_at") or x["archive_dir"], reverse=True)
    OUT_PATH.write_text(json.dumps({"schema": "catalog_index.v1", "items": items}, indent=2), encoding="utf-8")
    print(f"Exported {len(items)} track(s) -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()