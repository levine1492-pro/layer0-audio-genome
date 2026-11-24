#!/usr/bin/env python3
"""Bridge universal signal-dispatcher repository_dispatch payload directly into run_pipeline.process_entry."""
import json
import os
import sys
import urllib.request
from pathlib import Path

from run_pipeline import process_entry

REPO_ROOT = Path(__file__).resolve().parent


def main():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not Path(event_path).exists():
        print("No GITHUB_EVENT_PATH found.", file=sys.stderr)
        sys.exit(1)

    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    payload = event.get("client_payload") or {}
    items = payload.get("items") or []
    params = payload.get("params") or {}

    if not items:
        print("Empty dispatch items[]. Nothing to run.")
        return

    use_demucs = not bool(params.get("fast", False))
    force = bool(params.get("force", False))

    staged_keys = []
    succeeded = 0
    failed = 0

    try:
        for item in items:
            url = (item.get("source_url") or "").strip()
            if not url:
                continue
            if item.get("staged_key"):
                staged_keys.append(item["staged_key"])

            meta = item.get("meta") or {}
            default_title = Path(item.get("filename") or "").stem
            title = (meta.get("title") or default_title).strip()
            seed_id = (meta.get("seed_id") or "").strip()
            role = (meta.get("role") or "").strip()
            raw_tags = (meta.get("tags") or "").strip()

            tags_parts = [f"role:{role}"] if role and role != "standalone" else []
            if raw_tags:
                tags_parts.append(raw_tags)

            entry = {
                "source": url,
                "title": title,
                "seed_id": seed_id,
                "tags": ", ".join(tags_parts),
                "notes": (meta.get("notes") or "").strip(),
            }

            try:
                process_entry(entry, REPO_ROOT, use_demucs=use_demucs, force=force)
                succeeded += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] Dispatched item failed ({item.get('filename') or url}): {exc}", file=sys.stderr)
    finally:
        cb_url = payload.get("callback_url")
        cb_token = os.environ.get("DISPATCHER_TOKEN")
        if cb_url and cb_token and staged_keys:
            try:
                req = urllib.request.Request(
                    cb_url,
                    data=json.dumps({"job_id": payload.get("job_id"), "staged_keys": staged_keys}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {cb_token}",
                    },
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=15)
                print(f"Purged {len(staged_keys)} staged R2 object(s) via callback.")
            except Exception as exc:
                print(f"Callback purge skipped ({exc}); R2 24h lifecycle will clean up.")

    print(f"Dispatch complete: {succeeded} succeeded, {failed} failed.")
    if succeeded == 0 and failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
