"""
data/delta_writer.py
Delta Lake versioning for the ChromaDB knowledge base.

Keeps a versioned record of every chunk ever added.
Panel can see Version 0 → Version 1 in a table.
Supports rollback if bad data is ingested.
"""

import os
import datetime

DELTA_PATH = "data/delta_tables/knowledge"
_version_counter_file = "data/delta_tables/.version"


def _get_current_version() -> int:
    if os.path.exists(_version_counter_file):
        with open(_version_counter_file) as f:
            return int(f.read().strip())
    return 0


def _set_version(v: int):
    os.makedirs(os.path.dirname(_version_counter_file), exist_ok=True)
    with open(_version_counter_file, "w") as f:
        f.write(str(v))


def bump_version() -> int:
    v = _get_current_version() + 1
    _set_version(v)
    return v


def save_chunks_to_delta(chunks: list) -> int:
    """
    Appends chunks to a Delta-style JSONL log.
    Each call creates a new version entry with timestamp.

    Returns the new version number.
    """
    import json

    os.makedirs(DELTA_PATH, exist_ok=True)
    log_dir = os.path.join(DELTA_PATH, "_delta_log")
    os.makedirs(log_dir, exist_ok=True)

    version = bump_version()
    ts      = datetime.datetime.utcnow().isoformat()

    # Write the version log entry (simulates Delta Lake _delta_log)
    log_file = os.path.join(log_dir, f"{version:010d}.json")
    log_entry = {
        "version":    version,
        "timestamp":  ts,
        "operation":  "APPEND",
        "chunks_added": len(chunks),
        "chunk_ids":  [c["id"] for c in chunks],
    }
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)

    # Write the actual chunk data as JSONL (parquet simulation)
    data_file = os.path.join(DELTA_PATH, f"part-{version:05d}.jsonl")
    with open(data_file, "w") as f:
        for chunk in chunks:
            row = {
                "chunk_id":    chunk["id"],
                "text":        chunk["text"],
                "ingested_at": ts,
                "version":     version,
            }
            f.write(json.dumps(row) + "\n")

    print(f"[DELTA] ✅ Version {version} written: {len(chunks)} chunks at {ts}")
    return version


def show_history() -> list:
    """Returns all version entries for display in demo / Grafana."""
    import json

    log_dir = os.path.join(DELTA_PATH, "_delta_log")
    if not os.path.exists(log_dir):
        return []

    history = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(log_dir, fname)) as f:
                history.append(json.load(f))
    return history


def rollback_to_version(target_version: int):
    """
    Simulate a rollback by resetting the version counter.
    In a real Delta Lake this would run: RESTORE TABLE TIMESTAMP AS OF version.
    """
    current = _get_current_version()
    if target_version >= current:
        print(f"[DELTA] ⚠️  Target {target_version} >= current {current}. Nothing to rollback.")
        return

    _set_version(target_version)
    print(f"[DELTA] ↩️  Rolled back from version {current} → {target_version}")
    return target_version


if __name__ == "__main__":
    history = show_history()
    if history:
        print("\n── Delta Lake History ──")
        for entry in history:
            print(f"  v{entry['version']:>3}  |  {entry['timestamp']}  |  {entry['chunks_added']} chunks  |  {entry['operation']}")
    else:
        print("No Delta Lake history yet. Run the pipeline first.")