#!/usr/bin/env python3
"""Bridge universal signal-dispatcher repository_dispatch payload into run_pipeline.py."""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

TMP_QUEUE = Path("/tmp/dispatch_queue.txt")


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

    lines = []
    staged_keys = []
    for item in items:
        url = (item.get("source_url") or "").strip()
        if not url:
            continue
        if item.get("staged_key"):
            staged_keys.append(item["staged_key"])

        meta = item.get("meta") or {}
        default_title = Path(item.get("filename") or "untitled").stem
        title = (meta.get("title") or default_title).replace("|", "-")
        seed_id = (meta.get("seed_id") or "").replace("|", "-")
        role = (meta.get("role") or "").strip()
        raw_tags = (meta.get("tags") or "").replace("|", ",")
        tags_parts = [f"role:{role}"] if role and role != "standalone" else []
        if raw_tags:
            tags_parts.append(raw_tags)
        tags = ", ".join(tags_parts)
        notes = (meta.get("notes") or "").replace("|", "-").replace("\n", " ")

        lines.append(f"{url} | {title} | {seed_id} | {tags} | {notes}")

    TMP_QUEUE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Prepared {len(lines)} dispatched track(s) in {TMP_QUEUE}")

    cmd = [sys.executable, "run_pipeline.py", "--file", str(TMP_QUEUE)]
    if params.get("fast"):
        cmd.append("--fast")
    if params.get("force"):
        cmd.append("--force")

    rc = subprocess.call(cmd)

    # Optional instant cleanup callback to purge R2 staging immediately
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
        except Exception as e:
            print(f"Callback purge skipped ({e}); R2 24h lifecycle will clean up.")

    sys.exit(rc)


if __name__ == "__main__":
    main()