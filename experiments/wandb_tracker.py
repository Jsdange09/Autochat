"""
experiments/wandb_tracker.py

BUGS FIXED:
  [BUG-1] WANDB_API_KEY was read via os.getenv() BEFORE load_dotenv() was
           ever called. So os.getenv always returned "" even when .env had the key.
           Fix: load_dotenv() now called in api.py at the very top before imports.

  [BUG-2] wandb.login(key="") with an empty string raises UsageError in wandb 0.26.x.
           Fix: guard against empty string — only call wandb.login() when key is truthy.

  [BUG-3] No offline fallback — if key missing, module was entirely silent with no
           local metrics. Fix: WANDB_MODE=offline runs without any key and saves
           locally. Can be synced to cloud later with `wandb sync`.
"""

import os
import wandb

_run = None


def init_run(prompt_version: str = "v1"):
    """
    Called once at server startup from api.py.
    Behaviour:
      - Key set in .env  → online run → metrics visible at wandb.ai
      - Key not set      → offline run → metrics saved locally
    """
    global _run

    api_key = os.getenv("WANDB_API_KEY", "").strip()
    project = os.getenv("WANDB_PROJECT", "aegis-autochat").strip()
    mode    = "online" if api_key else "offline"

    if not api_key:
        print("[W&B] ⚠️  WANDB_API_KEY not set — running in OFFLINE mode.")
        print("[W&B]     Metrics saved locally. Sync later with: wandb sync ./wandb")
        os.environ["WANDB_MODE"] = "offline"
    else:
        # Only call wandb.login() when we actually have a key
        try:
            wandb.login(key=api_key, relogin=False)
        except Exception as e:
            print(f"[W&B] ⚠️  Login failed: {e}")
            os.environ["WANDB_MODE"] = "offline"
            mode = "offline"

    try:
        _run = wandb.init(
            project=project,
            name=f"autochat-{prompt_version}",
            mode=mode,
            config={
                "model":                "bling-phi-3-gguf",
                "embedding_model":      "all-MiniLM-L6-v2",
                "vector_store":         "chromadb",
                "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")),
                "prompt_version":       prompt_version,
            },
            resume="allow",
        )
        if mode == "online":
            print(f"[W&B] ✅ Online run started: {_run.url}")
        else:
            print(f"[W&B] ✅ Offline run started (id: {_run.id})")
    except Exception as e:
        print(f"[W&B] ⚠️  init failed: {e}")
        _run = None


def log_metrics(metrics: dict):
    """Log metrics. Works in both online and offline mode."""
    global _run
    if _run is None:
        return
    try:
        wandb.log(metrics)
    except Exception as e:
        print(f"[W&B] ⚠️  log error: {e}")


def log_repair_event(before_confidence: float, after_confidence: float, chunks_added: int):
    """Creates a W&B Table row for before/after repair comparison."""
    global _run
    if _run is None:
        return
    try:
        table = wandb.Table(columns=["Event", "Before", "After", "Chunks"])
        table.add_data("KB Repair",
                       round(before_confidence, 3),
                       round(after_confidence, 3),
                       chunks_added)
        wandb.log({"repair_events": table})
        delta = after_confidence - before_confidence
        print(f"[W&B] ✅ Repair event logged. Δconfidence={delta:+.3f}")
    except Exception as e:
        print(f"[W&B] ⚠️  repair log error: {e}")


def finish():
    global _run
    if _run:
        _run.finish()
        _run = None