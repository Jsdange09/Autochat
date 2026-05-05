"""
monitoring/arize_logger.py
Your observability layer — plugs into your friend's API.

🔑 PRIVATE DATA NEEDED:
   ARIZE_API_KEY   → from https://app.arize.com → Settings → API Keys
   ARIZE_SPACE_ID  → from https://app.arize.com → Settings → Space ID
   Set these in your .env file.
"""

import os
import datetime
from sentence_transformers import SentenceTransformer, util

# ── Arize SDK ──
try:
    from arize.api import Client
    from arize.utils.types import ModelTypes, Environments
    _arize_available = True
except ImportError:
    _arize_available = False
    print("[ARIZE] arize-sdk not installed. Run: pip install arize")

# ── Shared embedding model (same one your friend uses: all-MiniLM-L6-v2) ──
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Arize client (lazy-loaded so missing keys don't crash at import) ──
_arize_client = None


def _get_arize_client():
    global _arize_client
    if _arize_client is None and _arize_available:
        api_key  = os.getenv("ARIZE_API_KEY", "")
        space_id = os.getenv("ARIZE_SPACE_ID", "")
        if api_key and space_id:
            _arize_client = Client(api_key=api_key, space_id=space_id)
        else:
            print("[ARIZE] ⚠️  ARIZE_API_KEY or ARIZE_SPACE_ID not set in .env")
    return _arize_client


# ──────────────────────────────────────────────────────────────────
#  REAL CONFIDENCE COMPUTATION
#  Replaces your friend's `random.uniform(0.3, 0.9)` with a real
#  embedding-based grounding score.
# ──────────────────────────────────────────────────────────────────

def compute_confidence(query: str, response: str, context: str) -> float:
    """
    Returns a float [0.0 – 1.0] measuring how well the LLM response
    is grounded in the retrieved ChromaDB context.

    Formula:
      score = avg(
          cosine_sim(query_emb, context_emb),   ← "did we find relevant context?"
          cosine_sim(response_emb, context_emb) ← "did the answer use the context?"
      )

    < 0.65  → knowledge gap → triggers repair pipeline
    ≥ 0.65  → healthy response
    """
    if not context.strip():
        # No context retrieved → definitely low confidence
        return 0.30

    q_emb   = _embed_model.encode(query,    convert_to_tensor=True)
    r_emb   = _embed_model.encode(response, convert_to_tensor=True)
    c_emb   = _embed_model.encode(context,  convert_to_tensor=True)

    q_c_sim = float(util.cos_sim(q_emb, c_emb)[0][0])
    r_c_sim = float(util.cos_sim(r_emb, c_emb)[0][0])

    score = round((q_c_sim + r_c_sim) / 2, 4)
    return max(0.0, min(1.0, score))  # clamp to [0, 1]


# ──────────────────────────────────────────────────────────────────
#  ARIZE LOGGING
# ──────────────────────────────────────────────────────────────────

def log_to_arize(
    prediction_id: str,
    query: str,
    answer: str,
    confidence: float,
    latency: float,
):
    """
    Logs one interaction to Arize AI.
    Arize will surface this in its embedding map and confidence charts.
    If keys aren't set, silently skips (won't crash your friend's API).
    """
    client = _get_arize_client()
    if client is None:
        print(f"[ARIZE] Skipping log (no client). confidence={confidence:.3f}")
        return

    try:
        client.log(
            prediction_id=prediction_id,
            model_id="aegis-autochat",
            model_version="v1.0",
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.PRODUCTION,
            prediction_timestamp=datetime.datetime.now(),
            prediction_label=answer,
            features={
                "query":          query,
                "context_length": len(query),
            },
            tags={
                "confidence":    str(round(confidence, 3)),
                "latency_s":     str(latency),
                "low_confidence": str(confidence < float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))),
            },
        )
        print(f"[ARIZE] ✅ Logged  id={prediction_id[:8]}  confidence={confidence:.3f}")
    except Exception as e:
        print(f"[ARIZE] ⚠️  Log failed: {e}")