"""
experiments/wandb_tracker.py
Experiment tracking layer — logs every API call to Weights & Biases.

🔑 PRIVATE DATA NEEDED:
   WANDB_API_KEY  → from https://wandb.ai/authorize
   WANDB_PROJECT  → whatever you want, e.g. "aegis-autochat"
   Set these in your .env file.
"""

import os
import wandb

_run = None
_logged_count = 0


def init_run(prompt_version: str = "v1"):
    """
    Call once at server startup (already called in api.py).
    Creates (or resumes) a W&B run.
    """
    global _run

    api_key = os.getenv("WANDB_API_KEY", "")
    project = os.getenv("WANDB_PROJECT", "aegis-autochat")

    if not api_key:
        print("[W&B] ⚠️  WANDB_API_KEY not set. Metrics won't be logged.")
        return

    wandb.login(key=api_key, relogin=False)

    _run = wandb.init(
        project=project,
        name=f"autochat-run-{prompt_version}",
        config={
            "model":               "bling-phi-3-gguf",   # your friend's model
            "embedding_model":     "all-MiniLM-L6-v2",  # shared embedding
            "vector_store":        "chromadb",           # your friend's DB
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")),
            "prompt_version":      prompt_version,
        },
        resume="allow",
    )
    print(f"[W&B] ✅ Run started: {_run.url if _run else 'local'}")


def log_metrics(metrics: dict):
    """Log a dict of metrics. Safe to call even if W&B isn't initialised."""
    global _run, _logged_count
    if _run is None:
        return
    try:
        wandb.log(metrics)
        _logged_count += 1
    except Exception as e:
        print(f"[W&B] ⚠️  Log error: {e}")


def log_repair_event(before_confidence: float, after_confidence: float, chunks_added: int):
    """
    Call after each successful repair pipeline run.
    Creates a W&B Table row showing before/after improvement.
    """
    global _run
    if _run is None:
        return
    table = wandb.Table(columns=["Event", "Before Confidence", "After Confidence", "Chunks Added"])
    table.add_data("KB Repair", round(before_confidence, 3), round(after_confidence, 3), chunks_added)
    wandb.log({"repair_events": table})
    print(f"[W&B] ✅ Repair event logged. Δconfidence = {after_confidence - before_confidence:+.3f}")


def finish():
    global _run
    if _run:
        _run.finish()