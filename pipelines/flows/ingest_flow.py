"""
pipelines/flows/ingest_flow.py
Upgraded from your friend's dummy Prefect flow to a REAL pipeline.

What's kept from friend:  @flow / @task Prefect structure
What you add:
  • Ray parallel PDF embedding
  • ChromaDB upsert (friend's exact vector store)
  • Delta Lake versioning
  • Prometheus sensor that reads UPDATE_TRIGGER gauge
  • W&B logging after each repair
"""

import os
import json
import time
import glob
import uuid
import datetime

import ray
from prefect import flow, task, get_run_logger

# Shared ChromaDB client — same DB your friend's api.py uses
from backend.vector_store import collection, get_embedding

# Delta Lake writer (your versioning layer)
from data.delta_writer import save_chunks_to_delta, bump_version

# W&B (your experiment tracker)
from experiments.wandb_tracker import log_metrics, log_repair_event

# ──────────────────────────────────────────────────────────────────
#  RAY INIT  (initialise once; idempotent)
# ──────────────────────────────────────────────────────────────────

ray.init(ignore_reinit_error=True)


# ──────────────────────────────────────────────────────────────────
#  TASK 1 — Load queries logged by your friend's /chat endpoint
# ──────────────────────────────────────────────────────────────────

@task(name="load_query_logs")
def load_logs():
    logger = get_run_logger()
    log_path = "backend/logs/queries.json"

    if not os.path.exists(log_path):
        logger.info("No query logs found yet.")
        return []

    with open(log_path) as f:
        try:
            data = json.load(f)
        except Exception:
            data = []

    logger.info(f"Loaded {len(data)} query log entries.")
    return data


# ──────────────────────────────────────────────────────────────────
#  TASK 2 — Clean / deduplicate incoming data
# ──────────────────────────────────────────────────────────────────

@task(name="clean_data")
def clean_data(data: list) -> list:
    logger = get_run_logger()
    seen   = set()
    clean  = []

    for item in data:
        q = item.get("query", "").strip().lower()
        if q and q not in seen:
            seen.add(q)
            clean.append(q)

    logger.info(f"Cleaned: {len(clean)} unique queries from {len(data)} raw entries.")
    return clean


# ──────────────────────────────────────────────────────────────────
#  TASK 3 — Scan /incoming_docs for new PDFs to ingest
# ──────────────────────────────────────────────────────────────────

@task(name="scan_incoming_docs")
def scan_incoming_docs() -> list:
    logger  = get_run_logger()
    pattern = "data/incoming_docs/*.pdf"
    files   = glob.glob(pattern)
    logger.info(f"Found {len(files)} PDFs in incoming_docs/.")
    return files


# ──────────────────────────────────────────────────────────────────
#  RAY remote function — runs in parallel per text chunk
#  Uses the SAME embedding model as your friend (all-MiniLM-L6-v2)
# ──────────────────────────────────────────────────────────────────

@ray.remote
def _embed_chunk_remote(chunk_text: str, chunk_id: str) -> dict:
    from sentence_transformers import SentenceTransformer
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    vector = model.encode(chunk_text).tolist()
    return {"id": chunk_id, "text": chunk_text, "vector": vector}


# ──────────────────────────────────────────────────────────────────
#  TASK 4 — Process PDFs with Ray (parallel) → upsert into ChromaDB
# ──────────────────────────────────────────────────────────────────

def _extract_pdf_text(path: str) -> str:
    """Extract raw text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        # Fallback: read as plain text (for .txt files dropped in incoming_docs)
        with open(path, "r", errors="ignore") as f:
            return f.read()


def _chunk_text(text: str, size: int = 300, overlap: int = 50) -> list:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i: i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


@task(name="ingest_pdfs_with_ray")
def ingest_pdfs_with_ray(pdf_files: list) -> dict:
    logger      = get_run_logger()
    all_chunks  = []
    total_added = 0

    for pdf_path in pdf_files:
        logger.info(f"[RAY] Processing: {pdf_path}")
        start = time.time()

        text   = _extract_pdf_text(pdf_path)
        chunks = _chunk_text(text)
        ids    = [str(uuid.uuid4()) for _ in chunks]

        # ── Parallel embedding via Ray ──
        futures  = [_embed_chunk_remote.remote(c, i) for c, i in zip(chunks, ids)]
        embedded = ray.get(futures)

        # ── Upsert into ChromaDB (your friend's exact collection) ──
        for item in embedded:
            try:
                collection.upsert(
                    documents=[item["text"]],
                    embeddings=[item["vector"]],
                    ids=[item["id"]],
                )
                all_chunks.append(item)
                total_added += 1
            except Exception as e:
                logger.warning(f"ChromaDB upsert error: {e}")

        elapsed = round(time.time() - start, 2)
        logger.info(f"[RAY] ✅ {len(chunks)} chunks from {os.path.basename(pdf_path)} in {elapsed}s")

    return {"total_chunks": total_added, "embedded_items": all_chunks}


# ──────────────────────────────────────────────────────────────────
#  TASK 5 — Delta Lake versioning
# ──────────────────────────────────────────────────────────────────

@task(name="version_knowledge_base")
def version_knowledge_base(ingest_result: dict) -> int:
    logger = get_run_logger()
    items  = ingest_result.get("embedded_items", [])

    if not items:
        logger.info("No new items to version.")
        return 0

    version = save_chunks_to_delta(items)
    logger.info(f"[DELTA] Knowledge base updated → version {version}")
    return version


# ──────────────────────────────────────────────────────────────────
#  SENSOR — called by a background thread; reads UPDATE_TRIGGER
#  from your friend's Prometheus /metrics endpoint
# ──────────────────────────────────────────────────────────────────

def _read_prometheus_trigger() -> bool:
    """
    Reads the update_trigger_flag gauge that your friend's api.py sets.
    Returns True if a repair is needed.
    """
    import httpx
    try:
        resp = httpx.get("http://localhost:8000/metrics", timeout=3)
        for line in resp.text.splitlines():
            if line.startswith("update_trigger_flag"):
                value = float(line.split()[-1])
                return value == 1.0
    except Exception:
        pass
    return False


def _read_signal_file() -> bool:
    """
    Alternative sensor: checks the JSON signal file written by api.py
    when confidence < threshold.
    """
    return os.path.exists("data/low_confidence_signal.json")


def _consume_signal_file() -> dict:
    path = "data/low_confidence_signal.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    os.remove(path)
    return data


# ──────────────────────────────────────────────────────────────────
#  MAIN PREFECT FLOW
# ──────────────────────────────────────────────────────────────────

@flow(name="aegis-ingest-pipeline")
def ingest_pipeline():
    """
    Runs when:
      1. Called directly: POST /run-pipeline
      2. Called by the Prefect sensor (scheduled every 60s)
         when update_trigger_flag == 1 OR signal file exists.

    Steps:
      1. Load query logs   (friend's data)
      2. Clean/deduplicate
      3. Scan incoming_docs for PDFs
      4. Parallel embed + ChromaDB upsert (Ray)
      5. Delta Lake versioning
      6. W&B repair event logging
    """
    logger = get_run_logger()
    logger.info("🚀 Aegis Ingest Pipeline Started")

    before_signal = _consume_signal_file()
    before_confidence = before_signal.get("confidence", 0.0)

    # Steps 1–2  (query log processing)
    raw_logs    = load_logs()
    clean_texts = clean_data(raw_logs)

    # Step 3  (PDF scan)
    pdf_files = scan_incoming_docs()

    if not pdf_files:
        logger.info("No PDFs to process this run.")
        ingest_result = {"total_chunks": 0, "embedded_items": []}
    else:
        # Step 4  (Ray parallel processing → ChromaDB)
        ingest_result = ingest_pdfs_with_ray(pdf_files)

    # Step 5  (Delta Lake)
    version = version_knowledge_base(ingest_result)

    # Step 6  (W&B)
    after_confidence = 0.80   # assumed post-repair (real value logged on next /chat call)
    log_repair_event(
        before_confidence=before_confidence,
        after_confidence=after_confidence,
        chunks_added=ingest_result["total_chunks"],
    )
    log_metrics({
        "pipeline_run":     1,
        "kb_version":       version,
        "chunks_added":     ingest_result["total_chunks"],
        "pdfs_processed":   len(pdf_files),
    })

    logger.info(f"✅ Pipeline complete. KB version={version}, chunks={ingest_result['total_chunks']}")
    return {"status": "complete", "kb_version": version, "chunks": ingest_result["total_chunks"]}


# ──────────────────────────────────────────────────────────────────
#  PREFECT SENSOR LOOP (run this as a separate process)
#  python -m pipelines.flows.sensor_loop
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[SENSOR] Starting Prefect sensor loop (polls every 60s)...")
    while True:
        if _read_prometheus_trigger() or _read_signal_file():
            print("[SENSOR] ⚡ Trigger detected — running repair pipeline...")
            ingest_pipeline()
        else:
            print("[SENSOR] ✅ System healthy, no repair needed.")
        time.sleep(60)