// frontend/chatbot-ui/src/App.jsx
// Your friend's chat UI + your MLOps system health panel
// All of your friend's original chat logic is kept 100% intact.
// Your additions are in the SystemHealth component at the bottom.

import { useState, useEffect } from "react";

const API = "http://localhost:8000";

// ── colour helpers ──
const confident  = c => c >= 0.65 ? "#22c55e" : c >= 0.45 ? "#f59e0b" : "#ef4444";
const healthText = s => s === "healthy" ? "✅ Healthy" : "⚡ Repair Pending";
const healthCol  = s => s === "healthy" ? "#22c55e" : "#f59e0b";

// ──────────────────────────────────────────────────────────────────
//  YOUR LAYER — System Health panel (reads /system-health endpoint)
// ──────────────────────────────────────────────────────────────────

function SystemHealth() {
  const [health, setHealth] = useState(null);

  const fetchHealth = async () => {
    try {
      const res  = await fetch(`${API}/system-health`);
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  };

  useEffect(() => {
    fetchHealth();
    const id = setInterval(fetchHealth, 10_000);  // refresh every 10s
    return () => clearInterval(id);
  }, []);

  if (!health) return null;

  return (
    <div style={{
      marginTop: "24px",
      padding: "16px",
      borderRadius: "12px",
      background: "#1e1e2e",
      color: "#cdd6f4",
      fontFamily: "monospace",
      fontSize: "13px",
    }}>
      <div style={{ fontWeight: 700, marginBottom: "10px", fontSize: "14px" }}>
        🛡️ Aegis MLOps Dashboard
      </div>

      {/* System Status */}
      <div style={{ marginBottom: "8px" }}>
        Status:{" "}
        <span style={{ color: healthCol(health.status), fontWeight: 600 }}>
          {healthText(health.status)}
        </span>
      </div>

      {/* KB Version */}
      <div style={{ marginBottom: "8px" }}>
        Knowledge Base:{" "}
        <span style={{ color: "#89b4fa" }}>
          Delta Lake v{health.delta_kb_version}
        </span>
      </div>

      {/* Repair Signal Details */}
      {health.signal?.confidence && (
        <div style={{ marginBottom: "8px" }}>
          Last trigger:{" "}
          <span style={{ color: "#f38ba8" }}>
            confidence={health.signal.confidence.toFixed(3)} @ {health.signal.timestamp?.slice(11, 19)}
          </span>
        </div>
      )}

      {/* Dashboard Links */}
      <div style={{ marginTop: "12px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {[
          ["Arize AI",  health.arize_dashboard],
          ["W&B",       health.wandb_dashboard],
          ["Prefect",   health.prefect_dashboard],
          ["Grafana",   health.grafana_dashboard],
          ["Ray",       health.ray_dashboard],
        ].map(([label, url]) => (
          <a key={label} href={url} target="_blank" rel="noreferrer" style={{
            padding: "4px 10px",
            background: "#313244",
            borderRadius: "6px",
            color: "#89dceb",
            textDecoration: "none",
            fontSize: "11px",
          }}>
            {label} ↗
          </a>
        ))}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
//  LOG PANEL — shows the self-healing action log
// ──────────────────────────────────────────────────────────────────

function ActionLog({ logs }) {
  if (!logs.length) return null;
  return (
    <div style={{
      marginTop: "16px",
      padding: "12px",
      borderRadius: "10px",
      background: "#0d1117",
      fontFamily: "monospace",
      fontSize: "12px",
      maxHeight: "140px",
      overflowY: "auto",
    }}>
      {logs.map((log, i) => (
        <div key={i} style={{ color: log.color, marginBottom: "2px" }}>
          <span style={{ color: "#4a5568" }}>[{log.time}] </span>
          {log.text}
        </div>
      ))}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
//  MAIN APP  — your friend's chat UI (100% intact) + your additions
// ──────────────────────────────────────────────────────────────────

function App() {
  // ── Your friend's state ──
  const [messages, setMessages] = useState([]);
  const [query,    setQuery]    = useState("");
  const [loading,  setLoading]  = useState(false);

  // ── Your additions ──
  const [actionLogs, setActionLogs] = useState([]);

  const pushLog = (text, color = "#7a86a8") => {
    const time = new Date().toTimeString().slice(0, 8);
    setActionLogs(prev => [...prev.slice(-20), { text, time, color }]);
  };

  // ── Your friend's sendQuery (kept, + your log additions) ──
  const sendQuery = async () => {
    if (!query.trim()) return;

    const userMessage = { role: "user", text: query };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    const res  = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await res.json();

    const botMessage = {
      role:       "bot",
      text:       data.response,
      query:      query,
      confidence: data.confidence,        // your addition
      source:     data.source,            // your addition
      repair:     data.needs_repair,      // your addition
    };

    setMessages(prev => [...prev, botMessage]);
    setQuery("");
    setLoading(false);

    // ── Your action log entries ──
    if (data.needs_repair) {
      pushLog(`[ALERT]   Low Confidence Detected → score: ${data.confidence?.toFixed(3)}`, "#ef4444");
      pushLog("[ACTION]  Prefect sensor will trigger Knowledge Repair...", "#f59e0b");
    } else {
      pushLog(`[INFO]    Response served (${data.source?.toUpperCase()}) conf=${data.confidence?.toFixed(3)}`, "#6b7280");
    }
  };

  // ── Your friend's sendFeedback (kept exactly) ──
  const sendFeedback = async (query) => {
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, feedback: "no" }),
    });
    pushLog("[ACTION]  Feedback stored — improved answer added to ChromaDB", "#a78bfa");
    alert("Saved for improvement ✅");
  };

  return (
    <div style={{ maxWidth: "700px", margin: "auto", padding: "20px" }}>
      <h1>💬 Aegis-Autochat</h1>
      <p style={{ fontSize: "13px", color: "#6b7280" }}>
        Self-Healing AI · llmware (bling-phi-3) · ChromaDB · Prefect · Arize AI
      </p>

      {/* ── Your friend's message list (kept, + confidence badge) ── */}
      <div style={{ minHeight: "400px" }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: "12px" }}>
            <div style={{ textAlign: msg.role === "user" ? "right" : "left" }}>
              <span style={{
                display: "inline-block",
                padding: "10px",
                borderRadius: "10px",
                background: msg.role === "user" ? "#007bff" : "#e5e5ea",
                color: msg.role === "user" ? "white" : "black",
                maxWidth: "80%",
              }}>
                {msg.text}
              </span>
            </div>

            {/* Your addition: confidence badge */}
            {msg.role === "bot" && msg.confidence !== undefined && (
              <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                conf:{" "}
                <span style={{ color: confident(msg.confidence), fontWeight: 600 }}>
                  {msg.confidence.toFixed(3)}
                </span>
                {" "}· src: {msg.source}
                {msg.repair && " · ⚡ repair triggered"}
              </div>
            )}

            {/* Your friend's feedback button (kept exactly) */}
            {msg.role === "bot" && (
              <button
                onClick={() => sendFeedback(msg.query)}
                style={{ fontSize: "12px", marginTop: "4px", cursor: "pointer" }}
              >
                ❌ Not Useful
              </button>
            )}
          </div>
        ))}

        {loading && <p>Thinking...</p>}
      </div>

      {/* ── Your friend's input (kept exactly) ── */}
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        onKeyDown={e => e.key === "Enter" && sendQuery()}
        placeholder="Ask something..."
        style={{ width: "75%", padding: "10px" }}
      />
      <button onClick={sendQuery} style={{ padding: "10px" }}>
        Send
      </button>

      {/* ── Your additions below the chat ── */}
      <ActionLog logs={actionLogs} />
      <SystemHealth />
    </div>
  );
}

export default App;