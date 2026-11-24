#!/usr/bin/env python3
"""Export lightweight catalog_index.json from archive/*/genome.json for external viewers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
OUT_PATH = ROOT / "catalog_index.json"


def get_in(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur if cur is not None else default


def main():
    items = []
    if ARCHIVE.exists():
        for folder in sorted(ARCHIVE.iterdir()):
            gpath = folder / "genome.json"
            if not folder.is_dir() or not gpath.exists():
                continue
            try:
                g = json.loads(gpath.read_text(encoding="utf-8"))
            except Exception:
                continue

            meta = g.get("meta") or {}
            tid = meta.get("track_id") or folder.name.rsplit("_", 1)[-1]
            schema_ver = str(g.get("schema_version") or "")

            mk = get_in(g, "pitch_harmony", "modal_key", default={}) or {}
            tonic = mk.get("tonic") or ""
            mode = mk.get("mode") or ""
            key_mode = f"{tonic} {mode}".strip()
            if key_mode and mk.get("ambiguous"):
                key_mode += " ?"

            bpm = get_in(g, "rhythm_forensics", "primary_bpm")
            lufs = get_in(g, "space_dynamics", "integrated_lufs")
            space = get_in(g, "space_dynamics", "reverb_proxy", "space_profile", default="")
            captured = str(meta.get("captured_utc") or "")

            tags = meta.get("tags") or ""
            if isinstance(tags, list):
                tags = ", ".join(str(x) for x in tags)

            items.append({
                "track_id": tid,
                "archive_dir": f"archive/{folder.name}",
                "title": meta.get("title") or folder.name,
                "seed_id": meta.get("seed_id") or "",
                "tags": str(tags),
                "duration_s": meta.get("duration_s"),
                "schema_version": f"v{schema_ver}" if schema_ver and not schema_ver.startswith("v") else schema_ver,
                "key_mode": key_mode,
                "bpm": f"{float(bpm):.1f} BPM" if isinstance(bpm, (int, float)) else "",
                "lufs": f"{float(lufs):.1f} LUFS" if isinstance(lufs, (int, float)) else "",
                "space_profile": str(space),
                "created_at": captured,
            })

    items.sort(key=lambda x: x.get("created_at") or x["archive_dir"], reverse=True)
    OUT_PATH.write_text(json.dumps({"schema": "catalog_index.v1", "items": items}, indent=2), encoding="utf-8")
    print(f"Exported {len(items)} track(s) -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
