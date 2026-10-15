"""Append-only, hash-chained decision log (schema: docs/decision_log_schema.json)."""
import hashlib
import json
from pathlib import Path


def append(record, path="outputs/decision_log.jsonl"):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    prev = "0" * 64
    if path.exists() and path.stat().st_size:
        prev = json.loads(path.read_text().splitlines()[-1])["hash"]
    record = {**record, "prev_hash": prev}
    record["hash"] = hashlib.sha256((json.dumps(record, sort_keys=True) + prev).encode()).hexdigest()
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
