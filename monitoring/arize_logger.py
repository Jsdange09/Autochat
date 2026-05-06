"""
monitoring/arize_logger.py

BUGS FIXED:
  [BUG-1] Wrong import: `from arize.api import Client` — this was arize v6/v7 API.
           arize v8 (installed: 8.22.1) completely removed arize.api.Client.
           Fix: use `arize.ArizeClient` (new v8 class).

  [BUG-2] Wrong types: `ModelTypes.GENERATIVE_LLM`, `Environments.PRODUCTION`
           no longer exist in arize v8. The new SDK uses OTEL spans.
           Fix: use arize v8 spans API for logging LLM interactions.

  [BUG-3] No load_dotenv() — env vars were never populated from .env file.
           Fix: load_dotenv() is now called in api.py before this module imports.
           This module just reads os.getenv() normally.
"""

import os
import datetime
import json
from sentence_transformers import SentenceTransformer, util

# ── Arize v8 SDK ──
try:
    from arize.client import ArizeClient
    from arize.regions import Region
    _arize_available = True
except ImportError:
    _arize_available = False
    print("[ARIZE] arize package not installed. Run: pip install arize")

# ── Shared embedding model ──
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Client (lazy-init so missing keys don't crash startup) ──
_arize_client = None

# ── Local log fallback (works even without Arize keys) ──
_LOCAL_LOG = "data/arize_local_log.jsonl"


def _get_arize_client():
    global _arize_client
    if _arize_client is not None:
        return _arize_client

    if not _arize_available:
        return None

    api_key = os.getenv("ARIZE_API_KEY", "").strip()
    if not api_key:
        print("[ARIZE] ⚠️  ARIZE_API_KEY not set in .env — logging locally only.")
        return None

    try:
        _arize_client = ArizeClient(api_key=api_key)
        print("[ARIZE] ✅ Client initialised (arize v8)")
        return _arize_client
    except Exception as e:
        print(f"[ARIZE] ⚠️  Client init failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
#  CONFIDENCE COMPUTATION  (unchanged — this part was correct)
# ──────────────────────────────────────────────────────────────────

def compute_confidence(query: str, response: str, context: str) -> float:
    """
    Real embedding-based grounding score.
    Returns float [0.0 – 1.0].
    0.30 returned when no context is available (guaranteed low-confidence signal).
    """
    if not context.strip():
        return 0.30

    q_emb = _embed_model.encode(query,    convert_to_tensor=True)
    r_emb = _embed_model.encode(response, convert_to_tensor=True)
    c_emb = _embed_model.encode(context,  convert_to_tensor=True)

    q_c_sim = float(util.cos_sim(q_emb, c_emb)[0][0])
    r_c_sim = float(util.cos_sim(r_emb, c_emb)[0][0])

    score = round((q_c_sim + r_c_sim) / 2, 4)
    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────
#  LOCAL LOG  (always runs — panel can see this even without Arize)
# ──────────────────────────────────────────────────────────────────

def _log_locally(payload: dict):
    """Append interaction to a local JSONL file as a fallback."""
    os.makedirs("data", exist_ok=True)
    with open(_LOCAL_LOG, "a") as f:
        f.write(json.dumps(payload) + "\n")


# ──────────────────────────────────────────────────────────────────
#  ARIZE LOGGING  (arize v8 API)
# ──────────────────────────────────────────────────────────────────

def log_to_arize(
    prediction_id: str,
    query: str,
    answer: str,
    confidence: float,
    latency: float,
):
    """
    Logs one interaction.
    - Always writes to local JSONL (works without any keys).
    - Attempts Arize cloud upload if ARIZE_API_KEY is set.
    """
    threshold    = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    is_low       = confidence < threshold
    payload      = {
        "prediction_id":  prediction_id,
        "query":          query,
        "answer":         answer[:300],
        "confidence":     round(confidence, 4),
        "latency_s":      latency,
        "low_confidence": is_low,
        "timestamp":      datetime.datetime.utcnow().isoformat(),
    }

    # Always log locally
    _log_locally(payload)

    # Try Arize cloud
    client = _get_arize_client()
    if client is None:
        # Keys missing — local log is enough for demo
        print(f"[ARIZE] 📝 Logged locally  conf={confidence:.3f}  low={is_low}")
        return

    try:
        # arize v8: use spans client to log an LLM span
        client.spans.log(
            span_name="chat_interaction",
            attributes={
                "input.value":      query,
                "output.value":     answer[:500],
                "llm.confidence":   str(round(confidence, 4)),
                "llm.latency_s":    str(latency),
                "low_confidence":   str(is_low),
                "prediction_id":    prediction_id,
            },
            project_name=os.getenv("ARIZE_PROJECT", "aegis-autochat"),
        )
        print(f"[ARIZE] ✅ Logged to cloud  id={prediction_id[:8]}  conf={confidence:.3f}")
    except Exception as e:
        # Cloud failed → local log already written, so no data loss
        print(f"[ARIZE] ⚠️  Cloud log failed (local log saved): {e}")