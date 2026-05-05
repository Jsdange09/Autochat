"""
backend/api.py  ─  Aegis-Query Integrated Backend
Combines your friend's Autochat with your MLOps layer.

Friend's original code kept intact.
Your additions are marked with:   # ── [YOUR LAYER] ──
"""

import os
import json
import time
import datetime
import uuid
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from llmware.models import ModelCatalog

from pipelines.flows.ingest_flow import ingest_pipeline
from backend.vector_store import search_memory, add_to_memory, collection

# ── [YOUR LAYER] ── Arize + W&B imports
from monitoring.arize_logger import compute_confidence, log_to_arize
from experiments.wandb_tracker import log_metrics, init_run

# ──────────────────────────────────────────────────────────────────
#  INIT
# ──────────────────────────────────────────────────────────────────

os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI(title="Aegis-Query (Autochat + MLOps)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model  (your friend's llmware model ─ kept exactly as-is)
model = ModelCatalog().load_model(
    "bling-phi-3-gguf",
    temperature=0.2,
    sample=False
)

# ── [YOUR LAYER] ── W&B run initialisation (once at startup)
init_run()

# ──────────────────────────────────────────────────────────────────
#  REQUEST / RESPONSE SCHEMAS  (your friend's originals)
# ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    feedback: str

# ──────────────────────────────────────────────────────────────────
#  PROMETHEUS METRICS  (your friend's originals + your additions)
# ──────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter("total_requests",    "Total API Requests",        ["endpoint"])
FEEDBACK_COUNT = Counter("feedback_count",   "Total Feedback Given",       ["type"])
RESPONSE_TIME  = Histogram("response_time_seconds", "Response time")
CONFIDENCE_SCORE = Gauge("confidence_score",  "Model confidence score")     # ← now REAL (your layer)
EMBEDDING_AGE    = Gauge("embedding_age_days","Simulated embedding age")
UPDATE_TRIGGER   = Gauge("update_trigger_flag", "1 if repair needed")       # ← your Prefect sensor reads this

# ── [YOUR LAYER] ── Additional MLOps gauges visible in Grafana
ARIZE_LOGGED     = Gauge("arize_logged_total",    "Interactions logged to Arize")
WANDB_LOGGED     = Gauge("wandb_logged_total",    "Interactions logged to W&B")
KB_VERSION       = Gauge("knowledge_base_version","Current Delta-Lake version of KB")

# ──────────────────────────────────────────────────────────────────
#  HELPERS  (your friend's originals)
# ──────────────────────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    query = query.strip().lower()
    if not query.endswith("?"):
        query += "?"
    return query


def log_query(query: str):
    """Append query to logs/queries.json  — friend's original code."""
    path = "backend/logs/queries.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)
    with open(path, "r+") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
        data.append({"query": query, "time": time.time()})
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()


def generate_improved_answer(query: str) -> str:
    """Friend's improved-answer helper — kept exactly as-is."""
    prompt = f"""
You are an expert AI assistant.

Give a clear, correct, and complete answer in 2-3 sentences.
Do not give one-word answers.
Do not repeat the question.

Question: {query}
Answer:
"""
    response = model.inference(prompt)
    answer = response["llm_response"] if isinstance(response, dict) else response
    return answer.strip()


# ── [YOUR LAYER] ── write low-confidence signal so Prefect sensor picks it up
def _write_low_confidence_signal(payload: dict):
    os.makedirs("data", exist_ok=True)
    with open("data/low_confidence_signal.json", "w") as f:
        json.dump(payload, f)


# ──────────────────────────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(request: QueryRequest):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="chat").inc()

    query = normalize_query(request.query)
    log_query(query)

    # ── Friend's RAG retrieval + LLM call ──
    try:
        context_list = search_memory(query)

        if context_list:
            context_text = "\n".join(context_list)
            prompt = f"""You are a smart AI assistant.

Use the context below if relevant.

Context:
{context_text}

Question: {query}

Give a clear, structured answer in 2-3 sentences.
"""
            response = model.inference(prompt)
            answer   = response["llm_response"] if isinstance(response, dict) else response
            source   = "rag"
        else:
            response = model.inference(query)
            answer   = response["llm_response"] if isinstance(response, dict) else response
            source   = "model"

    except Exception as e:
        return {"response": "Error generating response", "error": str(e)}

    latency = round(time.time() - start, 3)

    # ── [YOUR LAYER] ── Real confidence via embedding cosine similarity ──
    context_text_for_eval = "\n".join(context_list) if context_list else ""
    confidence = compute_confidence(query, answer, context_text_for_eval)

    # Friend's simulated embedding age (kept)
    import random
    age_days = random.randint(1, 15)

    # ── Prometheus gauges (real confidence replaces random value) ──
    CONFIDENCE_SCORE.set(confidence)
    EMBEDDING_AGE.set(age_days)

    needs_repair = confidence < float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")) or age_days > 7
    UPDATE_TRIGGER.set(1 if needs_repair else 0)

    RESPONSE_TIME.observe(latency)

    # ── [YOUR LAYER] ── Log to Arize AI ──
    prediction_id = str(uuid.uuid4())
    log_to_arize(
        prediction_id=prediction_id,
        query=query,
        answer=answer,
        confidence=confidence,
        latency=latency,
    )
    ARIZE_LOGGED.inc()

    # ── [YOUR LAYER] ── Log to Weights & Biases ──
    log_metrics({
        "confidence":   confidence,
        "latency_s":    latency,
        "source":       1 if source == "rag" else 0,
        "needs_repair": int(needs_repair),
    })
    WANDB_LOGGED.inc()

    # ── [YOUR LAYER] ── Write repair signal if low confidence ──
    if needs_repair:
        _write_low_confidence_signal({
            "prediction_id": prediction_id,
            "confidence":    confidence,
            "age_days":      age_days,
            "query":         query,
            "timestamp":     datetime.datetime.utcnow().isoformat(),
        })

    return {
        "response":           answer,
        "source":             source,
        "confidence":         round(confidence, 3),
        "embedding_age_days": age_days,
        "needs_repair":       needs_repair,
        "prediction_id":      prediction_id,
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Friend's feedback endpoint — kept exactly as-is."""
    start = time.time()

    if request.feedback.lower() not in ["no", "not useful"]:
        return {"status": "ignored"}

    FEEDBACK_COUNT.labels(type="negative").inc()

    query    = normalize_query(request.query)
    improved = generate_improved_answer(query)
    add_to_memory(query, improved)

    # ── [YOUR LAYER] ── Also log the improved answer to W&B ──
    log_metrics({"feedback_triggered_repair": 1})

    RESPONSE_TIME.observe(time.time() - start)
    return {"status": "stored"}


@app.get("/debug-memory")
def debug_memory():
    """Friend's debug endpoint — kept exactly as-is."""
    data = collection.get()
    return {
        "stored_queries":  data["ids"],
        "stored_answers":  data["documents"],
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint — friend's original."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/run-pipeline")
def run_pipeline(background_tasks: BackgroundTasks):
    """Friend's pipeline trigger — kept; now calls the real upgraded pipeline."""
    background_tasks.add_task(ingest_pipeline)
    return {"status": "Pipeline running in background"}


# ── [YOUR LAYER] ── New endpoint: system health summary for your React dashboard ──
@app.get("/system-health")
def system_health():
    signal_exists = os.path.exists("data/low_confidence_signal.json")
    signal_data   = {}
    if signal_exists:
        with open("data/low_confidence_signal.json") as f:
            signal_data = json.load(f)

    delta_version = 0
    delta_log = "data/delta_tables/knowledge/_delta_log"
    if os.path.exists(delta_log):
        delta_version = len([f for f in os.listdir(delta_log) if f.endswith(".json")]) - 1

    KB_VERSION.set(max(delta_version, 0))

    return {
        "status":              "repair_pending" if signal_exists else "healthy",
        "signal":              signal_data,
        "delta_kb_version":    max(delta_version, 0),
        "arize_dashboard":     "https://app.arize.com",
        "wandb_dashboard":     "https://wandb.ai",
        "prefect_dashboard":   "http://localhost:4200",
        "grafana_dashboard":   "http://localhost:3000",
        "ray_dashboard":       "http://localhost:8265",
    }