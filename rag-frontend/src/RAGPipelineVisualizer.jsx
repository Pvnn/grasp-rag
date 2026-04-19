import { useState, useEffect, useRef, useMemo, Component } from "react";
// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";

// ─── ERROR BOUNDARY ───────────────────────────────────────────────────────────
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { crashed: false, msg: "" }; }
  static getDerivedStateFromError(err) { return { crashed: true, msg: String(err) }; }
  render() {
    if (this.state.crashed) {
      return (
        <div style={{ padding: 32, fontFamily: "monospace", color: "#b91c1c", background: "#fee2e2", borderRadius: 12, margin: 16 }}>
          <strong>Render error:</strong>
          <pre style={{ marginTop: 8, whiteSpace: "pre-wrap", fontSize: 12 }}>{this.state.msg}</pre>
          <button onClick={() => { this.setState({ crashed: false }); this.props.onReset?.(); }}
            style={{ marginTop: 12, padding: "8px 20px", background: "#b91c1c", color: "white", border: "none", borderRadius: 8, cursor: "pointer" }}>
            Reset
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const API_BASE = "http://localhost:8000";

async function fetchDatasets() {
  try {
    const res = await fetch(`${API_BASE}/datasets`);
    if (!res.ok) throw new Error("Failed to fetch datasets");
    return await res.json();
  } catch (err) {
    console.error(err);
    return null;
  }
}

async function runPipelineAPI(query, enableFilter, enableComp, topK) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: topK,
      compare_original: true,
      use_coarse: enableFilter,
      use_fine: enableComp
    })
  });
  if (!res.ok) throw new Error("Pipeline execution failed");
  return await res.json();
}

// ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
const C = {
  bg: "#fcfcfc", paper: "#ffffff", paperAlt: "#f9f9f9",
  ink: "#111111", inkSub: "#555555", inkMuted: "#888888",
  grid: "rgba(0,0,0,0.03)",
  blue: "#2563eb", blueL: "#eff6ff", blueB: "#bfdbfe",
  green: "#059669", greenL: "#ecfdf5", greenB: "#a7f3d0",
  red: "#dc2626", redL: "#fef2f2", redB: "#fecaca",
  amber: "#d97706", amberL: "#fffbeb", amberB: "#fde68a",
  violet: "#7c3aed", violetL: "#f5f3ff", violetB: "#ddd6fe",
  border: "#eaeaea", borderMed: "#e0e0e0",
};
const PAL = {
  blue: { a: C.blue, l: C.blueL, b: C.blueB },
  violet: { a: C.violet, l: C.violetL, b: C.violetB },
  green: { a: C.green, l: C.greenL, b: C.greenB },
  amber: { a: C.amber, l: C.amberL, b: C.amberB },
  red: { a: C.red, l: C.redL, b: C.redB },
};

const STAGES = [
  { id: "query", num: "01", label: "Query Input", color: "blue", icon: "search", toggleable: false },
  { id: "retrieve", num: "02", label: "Hybrid Retriever", color: "blue", icon: "list", toggleable: false },
  { id: "filter", num: "03", label: "QUITO-X Coarse Filter", color: "violet", icon: "filter", toggleable: true },
  { id: "compress", num: "04", label: "EP-EXIT Compression", color: "green", icon: "compress", toggleable: true },
  { id: "llm", num: "05", label: "Reader LLM", color: "amber", icon: "cpu", toggleable: false },
];

// ─── ICONS ────────────────────────────────────────────────────────────────────
function Ico({ n, sz = 18, c = "currentColor" }) {
  const s = { width: sz, height: sz, display: "block" };
  const m = {
    search: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" /></svg>,
    list: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6" /><line x1="4" y1="10" x2="20" y2="10" /><line x1="4" y1="14" x2="14" y2="14" /><line x1="4" y1="18" x2="11" y2="18" /></svg>,
    filter: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>,
    compress: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><polyline points="4 14 10 14 10 20" /><polyline points="20 10 14 10 14 4" /><line x1="10" y1="14" x2="21" y2="3" /><line x1="3" y1="21" x2="14" y2="10" /></svg>,
    cpu: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" /><line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" /><line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" /></svg>,
    check: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>,
    x: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2.2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>,
    down: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><polyline points="19 12 12 19 5 12" /></svg>,
    play: <svg style={s} viewBox="0 0 24 24" fill={c}><polygon points="5 3 19 12 5 21 5 3" /></svg>,
    chevron: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9" /></svg>,
    skip: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12" /><line x1="5" y1="7" x2="5" y2="17" strokeDasharray="2 2" /><line x1="19" y1="7" x2="19" y2="17" strokeDasharray="2 2" /></svg>,
  };
  return m[n] || null;
}

function Spin({ sz = 14, color = C.blue }) {
  return <span style={{ display: "inline-block", width: sz, height: sz, flexShrink: 0, border: "2px solid transparent", borderTopColor: color, borderRightColor: color, borderRadius: "50%", animation: "spin360 0.75s linear infinite" }} />;
}

function ScoreBar({ score }) {
  const v = Math.min(Math.max(+score || 0, 0), 1);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
      <div style={{ flex: 1, height: 4, background: C.border, borderRadius: 99, overflow: "hidden" }}>
        <motion.div initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          style={{ height: "100%", borderRadius: 99, background: v > 0.85 ? C.blue : v > 0.7 ? C.inkSub : C.inkMuted }} />
      </div>
      <span style={{ fontSize: 12, fontFamily: "monospace", color: C.inkMuted, minWidth: 32, textAlign: "right" }}>{(v * 100).toFixed(0)}%</span>
    </div>
  );
}

function Pill({ children, color = "blue" }) {
  const p = PAL[color] || PAL.blue;
  return <span style={{ display: "inline-block", background: p.l, color: p.a, border: `1px solid ${p.b}`, fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", padding: "2px 8px", borderRadius: 4, fontFamily: "monospace", whiteSpace: "nowrap" }}>{children}</span>;
}

function Toggle({ on, onChange, disabled }) {
  return (
    <button onClick={e => { e.stopPropagation(); if (!disabled) onChange(!on); }}
      style={{ display: "flex", alignItems: "center", gap: 7, padding: "5px 12px 5px 7px", background: on ? C.blueL : C.paperAlt, border: `1.5px solid ${on ? C.blueB : C.border}`, borderRadius: 20, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1, transition: "all 0.2s", outline: "none" }}>
      <div style={{ width: 32, height: 18, borderRadius: 99, position: "relative", flexShrink: 0, background: on ? C.blue : "#c0bfca", transition: "background 0.2s" }}>
        <div style={{ position: "absolute", top: 3, left: on ? 15 : 3, width: 12, height: 12, borderRadius: "50%", background: "white", boxShadow: "0 1px 3px rgba(0,0,0,0.25)", transition: "left 0.2s" }} />
      </div>
      <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: on ? C.blue : C.inkMuted }}>{on ? "Enabled" : "Disabled"}</span>
    </button>
  );
}

// ─── PIPELINE NODE ────────────────────────────────────────────────────────────
function PipelineNode({ stage, status, enabled, onToggle, running }) {
  const col = PAL[stage.color] || PAL.blue;
  const isActive = status === "active";
  const isDone = status === "done";
  const isIdle = status === "idle";
  const isSkipped = isDone && stage.toggleable && !enabled;
  const isOk = isDone && !isSkipped;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12, padding: "14px 16px", minHeight: 120, background: C.paper,
      border: `1px solid ${isActive ? col.a : isOk ? C.borderMed : C.border}`, borderRadius: 8,
      opacity: isIdle ? 0.5 : 1, filter: isSkipped ? "grayscale(1) opacity(0.5)" : "none",
      boxShadow: isActive ? `0 0 0 2px ${col.l}, 0 4px 12px rgba(0,0,0,0.05)` : "0 1px 3px rgba(0,0,0,0.02)",
      transition: "all 0.2s ease"
    }}>
      <div style={{
        width: 36, height: 36, borderRadius: 8, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
        background: (isOk || isActive) ? col.l : C.paperAlt, border: `1px solid ${(isOk || isActive) ? col.b : C.border}`,
        color: (isOk || isActive) ? col.a : C.inkMuted, transition: "all 0.2s"
      }}>
        <Ico n={stage.icon} sz={16} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: isSkipped ? C.inkMuted : col.a, marginBottom: 2 }}>
          Stage {stage.num}{isSkipped ? " · SKIPPED" : ""}
        </div>
        <div style={{ fontSize: 14, fontWeight: 600, color: isSkipped ? C.inkMuted : C.ink, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", lineHeight: 1.3 }}>
          {stage.label}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
        {stage.toggleable && <Toggle on={enabled} onChange={onToggle} disabled={running} />}
        <div style={{ width: 20, height: 20, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
          {isActive && <Spin color={col.a} sz={16} />}
          {isOk && <Ico n="check" sz={16} c={C.green} />}
          {isSkipped && <span style={{ fontSize: 14, color: C.inkMuted, fontWeight: 700 }}>—</span>}
          {isIdle && <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.border }} />}
        </div>
      </div>
    </div>
  );
}

// ─── OUTPUT PANEL ─────────────────────────────────────────────────────────────
function OutputPanel({ status, enabled, stageId, children }) {
  if (stageId === "query") {
    return (
      <div style={{ background: C.paper, border: `1px solid ${C.borderMed}`, borderRadius: 8, minHeight: 120, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.03)" }}>
        <div style={{ padding: "18px 20px" }}>{children}</div>
      </div>
    );
  }
  if (status === "idle") return (
    <div style={{ minHeight: 70, background: "transparent", border: `1px dashed ${C.border}`, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span style={{ fontSize: 13, color: C.inkMuted, padding: "16px 18px" }}>Awaiting pipeline run…</span>
    </div>
  );
  if (status === "active") return (
    <div style={{ minHeight: 70, background: C.paper, border: `1px solid ${C.borderMed}`, borderRadius: 8, display: "flex", alignItems: "center", gap: 12, padding: "16px 18px", boxShadow: "0 1px 4px rgba(0,0,0,0.03)" }}>
      <Spin color={C.inkMuted} sz={15} /><span style={{ fontSize: 14, color: C.inkSub }}>Processing against Backend…</span>
    </div>
  );
  if (!enabled) return (
    <div style={{ minHeight: 70, background: C.paperAlt, border: `1px dashed ${C.border}`, borderRadius: 8, display: "flex", alignItems: "center", gap: 14, padding: "16px 18px", opacity: 0.7 }}>
      <div style={{ width: 32, height: 32, borderRadius: "50%", background: C.border, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}><Ico n="skip" sz={14} c={C.inkMuted} /></div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: C.inkSub, textTransform: "uppercase", letterSpacing: "0.05em" }}>Stage Disabled</div>
        <div style={{ fontSize: 13, color: C.inkMuted, marginTop: 2 }}>This stage was skipped during the pipeline run.</div>
      </div>
    </div>
  );
  if (!children) return (
    <div style={{ minHeight: 70, background: C.paper, border: `1px solid ${C.border}`, borderRadius: 8, display: "flex", alignItems: "center", gap: 12, padding: "16px 18px" }}>
      <Spin color={C.inkMuted} sz={14} /><span style={{ fontSize: 14, color: C.inkMuted }}>Loading…</span>
    </div>
  );
  return (
    <div style={{ background: C.paper, border: `1px solid ${C.borderMed}`, borderRadius: 8, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.03)" }}>
      <div style={{ padding: "18px 20px" }}>{children}</div>
    </div>
  );
}

// ─── STAGE CONTENT ────────────────────────────────────────────────────────────
function QueryContent({ query }) {
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
      <div style={{ flex: "1 1 200px", padding: "14px 17px", background: C.paperAlt, border: `1px solid ${C.borderMed}`, borderRadius: 6 }}>
        <div style={{ fontSize: 11, color: C.inkSub, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 8 }}>Input Query</div>
        <div style={{ fontSize: 18, color: C.ink, fontWeight: 500, lineHeight: 1.5 }}>"{query}"</div>
      </div>
      <div style={{ flex: "0 0 auto", padding: "14px 17px", background: C.paperAlt, border: `1px solid ${C.borderMed}`, borderRadius: 6, minWidth: 180 }}>
        <div style={{ fontSize: 11, color: C.inkSub, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 8 }}>Retriever Engine</div>
        <div style={{ fontSize: 14, color: C.ink, lineHeight: 1.6 }}>Hybrid Vector<br />Retriever</div>
      </div>
    </div>
  );
}

function RetrieverContent({ docs }) {
  if (!docs?.length) return null;
  return (
    <div>
      <div style={{ fontSize: 12, color: C.inkSub, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 14 }}>{docs.length} Documents Retrieved — Ranked by Similarity</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(220px,1fr))", gap: 10 }}>
        {docs.map((doc, i) => (
          <div key={i} style={{ padding: "13px 15px", background: C.paperAlt, border: `1px solid ${C.borderMed}`, borderRadius: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: C.ink, lineHeight: 1.3 }}>Document {doc.doc_index}</span>
              <span style={{ fontSize: 11, color: C.inkSub, background: C.border, padding: "2px 6px", borderRadius: 4 }}>Top {doc.doc_index}</span>
            </div>
            <p style={{ fontSize: 13, color: C.inkSub, lineHeight: 1.65, margin: 0 }}>
              {doc.text.substring(0, 150)}...
            </p>
            <ScoreBar score={doc.score} />
          </div>
        ))}
      </div>
    </div>
  );
}

function FilterContent({ quitoxDetails }) {
  if (!quitoxDetails || quitoxDetails.length === 0) return null;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: C.violet }}>QUITO-X T5 Analysis</div>
        <span style={{ fontSize: 13, color: C.inkMuted }}>— Cross Attention Based Coarse Filter</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {quitoxDetails.map((q, i) => {
          const detailsList = Array.isArray(q.details) ? q.details : [];
          const kept = detailsList.filter(d => d.retained);
          const removed = detailsList.filter(d => !d.retained);
          return (
            <div key={i} style={{ padding: "14px", borderRadius: 6, background: C.paperAlt, border: `1px solid ${C.borderMed}` }}>
              <div style={{ fontSize: 12, color: C.ink, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                Document {q.doc_index}
                <span style={{ color: C.inkMuted, fontWeight: 400, marginLeft: 8, textTransform: "none", letterSpacing: "normal" }}>
                  ({kept.length} Kept · {removed.length} Removed)
                </span>
              </div>
              {kept.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  {kept.map((unit, j) => (
                    <div key={`kept-${j}`} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "8px 10px", background: C.violetL, borderLeft: `3px solid ${C.violet}`, marginBottom: 6, borderRadius: 4 }}>
                      <div style={{ marginTop: 2 }}><Ico n="check" sz={14} c={C.violet} /></div>
                      <div style={{ flex: 1, fontSize: 13, color: C.ink, lineHeight: 1.5 }}>{unit.text}</div>
                      {unit.score !== undefined && <div style={{ fontSize: 11, fontFamily: "monospace", color: C.violet, fontWeight: 600 }}>[{unit.score.toFixed(2)}]</div>}
                    </div>
                  ))}
                </div>
              )}
              {removed.length > 0 && (
                <div>
                  {removed.map((unit, j) => (
                    <div key={`rem-${j}`} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "8px 10px", background: C.redL, borderLeft: `3px solid ${C.red}`, marginBottom: 6, borderRadius: 4 }}>
                      <div style={{ marginTop: 2 }}><Ico n="x" sz={14} c={C.red} /></div>
                      <div style={{ flex: 1, fontSize: 13, color: C.inkSub, lineHeight: 1.5, textDecoration: "line-through", opacity: 0.8 }}>{unit.text}</div>
                      {unit.score !== undefined && <div style={{ fontSize: 11, fontFamily: "monospace", color: C.red, fontWeight: 600 }}>[{unit.score.toFixed(2)}]</div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CompressContent({ epExitDetails, ratio }) {
  if (!epExitDetails) return null;
  const pct = Math.round((+ratio || 0));
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: C.green }}>EP-EXIT Evidence Grouping</div>
        <span style={{ fontSize: 13, color: C.inkMuted }}>— Gemma-2B Binary Relevance Classification &nbsp;·&nbsp; <strong style={{ color: C.green }}>{pct}% overall reduction</strong></span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {epExitDetails.map((doc, idx) => (
          <div key={idx} style={{ padding: "14px", border: `1px solid ${C.borderMed}`, borderRadius: 6, background: C.paperAlt }}>
            <div style={{ fontSize: 12, color: C.ink, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>Document {doc.doc_index}</div>
            {doc.kept_units?.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                {doc.kept_units.map((unit, j) => {
                  const txt = typeof unit === 'object' ? unit.text : unit;
                  const score = typeof unit === 'object' ? unit.score : null;
                  return (
                    <div key={`kept-${j}`} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "8px 10px", background: C.greenL, borderLeft: `3px solid ${C.green}`, marginBottom: 6, borderRadius: 4 }}>
                      <div style={{ marginTop: 2 }}><Ico n="check" sz={14} c={C.green} /></div>
                      <div style={{ flex: 1, fontSize: 13, color: C.ink, lineHeight: 1.5 }}>{txt}</div>
                      {score && <div style={{ fontSize: 11, fontFamily: "monospace", color: C.green, fontWeight: 600 }}>[{score.toFixed(2)}]</div>}
                    </div>
                  );
                })}
              </div>
            )}
            {doc.removed_units?.length > 0 && (
              <div>
                {doc.removed_units.map((unit, j) => {
                  const txt = typeof unit === 'object' ? unit.text : unit;
                  const score = typeof unit === 'object' ? unit.score : null;
                  return (
                    <div key={`rem-${j}`} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "8px 10px", background: C.redL, borderLeft: `3px solid ${C.red}`, marginBottom: 6, borderRadius: 4 }}>
                      <div style={{ marginTop: 2 }}><Ico n="x" sz={14} c={C.red} /></div>
                      <div style={{ flex: 1, fontSize: 13, color: C.inkSub, lineHeight: 1.5, textDecoration: "line-through", opacity: 0.8 }}>{txt}</div>
                      {score && <div style={{ fontSize: 11, fontFamily: "monospace", color: C.red, fontWeight: 600 }}>[{score.toFixed(2)}]</div>}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function LLMContent({ metrics }) {
  if (!metrics) return null;
  const tokens = metrics.usage?.compressed_api_tokens || 0;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: C.amber }}>Reader Engine Input</div>
        <span style={{ fontSize: 13, color: C.inkMuted }}>— Generated via Gemma-3-27b-it</span>
      </div>
      <div style={{ padding: "15px 17px", background: C.paperAlt, border: `1px solid ${C.borderMed}`, borderRadius: 6 }}>
        <div style={{ fontSize: 11, color: C.inkSub, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>Context Window Tokens</div>
        <div style={{ fontSize: 24, fontWeight: 600, color: C.ink, lineHeight: 1.75 }}>{tokens}</div>
        <div style={{ fontSize: 13, color: C.inkSub, marginTop: 4 }}>Total tokens sent to the LLM (including prompt structure).</div>
      </div>
    </div>
  );
}

function AnswerContent({ compressedAns, originalAns, metrics }) {
  const m = metrics || {};
  const cRatio = m.compression?.ratio_chars ? m.compression.ratio_chars.toFixed(1) : 0;
  const tRatio = m.compression?.ratio_tokens ? m.compression.ratio_tokens.toFixed(1) : 0;
  const tSaved = m.times?.net_time_saved ? m.times.net_time_saved.toFixed(2) : 0;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={{ padding: "18px 20px", background: C.blueL, border: `1px solid ${C.blueB}`, borderRadius: 8 }}>
          <div style={{ fontSize: 11, color: C.blue, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 9 }}>Compressed Answer</div>
          <div style={{ fontSize: 15, color: C.ink, lineHeight: 1.6 }}>{compressedAns || "—"}</div>
        </div>
        <div style={{ padding: "18px 20px", background: C.paperAlt, border: `1px solid ${C.borderMed}`, borderRadius: 8 }}>
          <div style={{ fontSize: 11, color: C.inkSub, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 9 }}>Original Baseline Answer</div>
          <div style={{ fontSize: 15, color: C.inkSub, lineHeight: 1.6, fontStyle: "italic" }}>{originalAns || "—"}</div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
        {[
          { label: "Context Reduction", value: `${cRatio}%`, sub: "characters removed" },
          { label: "Token Savings", value: `${tRatio}%`, sub: "API cost reduction" },
          { label: "Time Differential", value: `${tSaved}s`, sub: "net latency impact" }
        ].map(x => (
          <div key={x.label} style={{ padding: 16, background: C.paperAlt, border: `1px solid ${C.borderMed}`, borderRadius: 6, textAlign: "center" }}>
            <div style={{ fontSize: 24, fontWeight: 600, color: C.ink, lineHeight: 1 }}>{x.value}</div>
            <div style={{ fontSize: 13, fontWeight: 500, color: C.inkSub, marginTop: 5 }}>{x.label}</div>
            <div style={{ fontSize: 12, color: C.inkMuted, marginTop: 3 }}>{x.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
function PipelineApp() {
  const [queries, setQueries] = useState([]);
  // query = confirmed selection used by the pipeline
  // inputValue = what is typed in the search box (filter only)
  const [query, setQuery] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [topK, setTopK] = useState(10);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState(-1);
  const [result, setResult] = useState(null);
  const [dropOpen, setDropOpen] = useState(false);
  const [enableFilter, setEnableFilter] = useState(true);
  const [enableComp, setEnableComp] = useState(true);

  const dropRef = useRef(null);
  const abortRef = useRef(false);

  // Memoized filter — only recomputes when queries list or inputValue changes
  const filteredQueries = useMemo(
    () => queries.filter(q => q.toLowerCase().includes(inputValue.toLowerCase())),
    [queries, inputValue]
  );

  useEffect(() => {
    fetchDatasets().then(data => {
      if (data) {
        const allQueries = [];
        Object.values(data).forEach(ds => {
          if (ds.queries) allQueries.push(...ds.queries);
        });
        if (allQueries.length > 0) {
          setQueries(allQueries);
          setQuery(allQueries[0]);
          setInputValue(allQueries[0]);
        }
      }
    });

    const fn = e => { if (dropRef.current && !dropRef.current.contains(e.target)) setDropOpen(false); };
    document.addEventListener("mousedown", fn);
    return () => document.removeEventListener("mousedown", fn);
  }, []);

  function stageEnabled(id) {
    if (id === "filter") return enableFilter;
    if (id === "compress") return enableComp;
    return true;
  }

  function statusOf(i) {
    if (phase < 0) return "idle";
    if (running && phase === i) return "active";
    if (phase > i) return "done";
    return "idle";
  }

  async function handleRun() {
    if (running || !query.trim()) return;
    abortRef.current = false;
    setRunning(true);
    setResult(null);

    const step = async (i, ms) => { setPhase(i); await sleep(ms); return abortRef.current; };

    if (await step(0, 420)) return;
    setPhase(1);

    try {
      const data = await runPipelineAPI(query, enableFilter, enableComp, topK);
      if (abortRef.current) return;
      setResult(data);

      if (await step(2, enableFilter ? 600 : 200)) return;
      if (await step(3, enableComp ? 800 : 200)) return;
      if (await step(4, 400)) return;

      setPhase(99);
    } catch (err) {
      console.error(err);
      alert("Pipeline failed. Ensure backend is running at localhost:8000.");
    } finally {
      setRunning(false);
    }
  }

  function handleReset() {
    abortRef.current = true;
    setResult(null);
    setPhase(-1);
    setRunning(false);
  }

  const isComplete = phase === 99 && !running && !!result;

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "'Inter', 'Segoe UI', sans-serif", color: C.ink }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
        html,body,#root { width:100%; min-height:100vh; }
        @keyframes spin360 { to { transform:rotate(360deg); } }
        .outer-wrap { width:100%; max-width:1400px; margin:0 auto; padding:0 32px; }
        .rsp-qopt:hover { background:${C.paperAlt}!important; color:${C.ink}!important; }
        .rsp-run { transition:all 0.15s; cursor:pointer; }
        .rsp-run:not([disabled]):hover { filter:brightness(1.05); }
        ::-webkit-scrollbar { width:6px; }
        ::-webkit-scrollbar-thumb { background:${C.borderMed}; border-radius:99px; }
      `}</style>

      {/* ── HEADER ── */}
      <header style={{ position: "sticky", top: 0, zIndex: 50, background: "rgba(255,255,255,0.85)", backdropFilter: "blur(12px)", borderBottom: `1px solid ${C.border}` }}>
        <div className="outer-wrap">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
              <div style={{ fontSize: 16, fontWeight: 600, color: C.ink, letterSpacing: "-0.01em" }}>Query-Aware RAG</div>
              <div style={{ fontSize: 13, color: C.inkMuted, fontWeight: 400 }}>Pipeline Visualizer</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ display: "flex", gap: 12 }}>
                {!enableFilter && <span style={{ fontSize: 12, color: C.inkMuted, textDecoration: "line-through" }}>QUITO-X</span>}
                {!enableComp && <span style={{ fontSize: 12, color: C.inkMuted, textDecoration: "line-through" }}>EP-EXIT</span>}
              </div>
              <div style={{ width: 1, height: 16, background: C.borderMed }} />
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                {isComplete ? (
                  <><div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green }} /><span style={{ fontSize: 13, color: C.ink, fontWeight: 500 }}>Complete</span></>
                ) : running ? (
                  <><Spin sz={10} color={C.blue} /><span style={{ fontSize: 13, color: C.ink, fontWeight: 500 }}>Running...</span></>
                ) : (
                  <><div style={{ width: 6, height: 6, borderRadius: "50%", background: C.borderMed }} /><span style={{ fontSize: 13, color: C.inkSub, fontWeight: 500 }}>Ready</span></>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ── BODY ── */}
      <div className="outer-wrap" style={{ position: "relative", zIndex: 1, paddingTop: 32, paddingBottom: 130 }}>
        <div style={{ display: "flex", gap: 0, marginBottom: 16, alignItems: "center" }}>
          <div style={{ width: 380, flexShrink: 0, display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, color: C.inkMuted, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>Pipeline Flow</span>
            <div style={{ height: 1, flex: 1, background: C.border }} />
          </div>
          <div style={{ width: 1, background: "transparent", margin: "0 24px", flexShrink: 0 }} />
          <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, color: C.inkMuted, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>Stage Output</span>
            <div style={{ height: 1, flex: 1, background: C.border }} />
          </div>
        </div>

        <div style={{ display: "flex", gap: 0, alignItems: "stretch" }}>
          {/* LEFT COLUMN */}
          <div style={{ width: 380, flexShrink: 0, display: "flex", flexDirection: "column" }}>
            {STAGES.map((stage, i) => {
              const st = statusOf(i);
              const enabled = stageEnabled(stage.id);
              const lit = phase > i;
              return (
                <div key={stage.id} style={{ display: "flex", flexDirection: "column" }}>
                  <PipelineNode
                    stage={stage} status={st} enabled={enabled}
                    onToggle={stage.id === "filter" ? setEnableFilter : setEnableComp}
                    running={running}
                  />
                  {i < STAGES.length - 1 && (
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", height: 32, paddingLeft: 18 }}>
                      <div style={{ width: 1.5, height: 20, background: lit ? C.blueB : C.border, transition: "background 0.35s" }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ width: 1, background: C.border, margin: "0 24px", flexShrink: 0 }} />

          {/* RIGHT COLUMN */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            {STAGES.map((stage, i) => {
              const st = statusOf(i);
              const enabled = stageEnabled(stage.id);
              let content = null;
              try {
                if (stage.id === "query") {
                  content = <QueryContent query={query} />;
                } else if (stage.id === "retrieve" && st === "done" && result) {
                  content = <RetrieverContent docs={result.retrieved_docs} />;
                } else if (stage.id === "filter" && st === "done" && result && enabled) {
                  content = <FilterContent quitoxDetails={result.quitox_details} />;
                } else if (stage.id === "compress" && st === "done" && result && enabled) {
                  content = <CompressContent epExitDetails={result.ep_exit_details} ratio={enabled ? result.metrics?.compression?.ratio_chars : 0} />;
                } else if (stage.id === "llm" && st === "done" && result) {
                  content = <LLMContent metrics={result.metrics} />;
                }
              } catch (e) {
                content = <div style={{ fontSize: 13, color: C.red, fontFamily: "monospace" }}>Error: {String(e)}</div>;
              }
              const isLast = i === STAGES.length - 1;
              return (
                <div key={stage.id} style={{ display: "flex", flexDirection: "column", marginBottom: isLast ? 0 : 32 }}>
                  <OutputPanel status={st} enabled={enabled} stageId={stage.id}>
                    {content}
                  </OutputPanel>
                </div>
              );
            })}
          </div>
        </div>

        {/* Final Answer */}
        {isComplete && result && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }} style={{ marginTop: 32 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
              <span style={{ fontSize: 11, color: C.inkMuted, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>Final Metrics</span>
              <div style={{ height: 1, flex: 1, background: C.border }} />
            </div>
            <div style={{ background: C.paper, border: `1px solid ${C.borderMed}`, borderRadius: 8, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.03)" }}>
              <div style={{ padding: "22px 26px" }}><AnswerContent compressedAns={result.answer} originalAns={result.original_docs_answer} metrics={result.metrics} /></div>
            </div>
          </motion.div>
        )}
      </div>

      {/* ── STICKY CONTROL BAR ── */}
      <div style={{ position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 60, background: "rgba(255,255,255,0.9)", backdropFilter: "blur(12px)", borderTop: `1px solid ${C.border}` }}>
        <div className="outer-wrap">
          <div style={{ display: "flex", gap: 12, alignItems: "center", padding: "16px 0" }}>

            {/* Combobox — input filters list, selection sets query */}
            <div ref={dropRef} style={{ flex: 1, position: "relative", maxWidth: 800 }}>

              <div style={{ display: "flex", alignItems: "center", background: C.paper, border: `1.5px solid ${dropOpen ? C.blue : C.borderMed}`, borderRadius: 6, transition: "border-color 0.15s" }}>
                <input
                  type="text"
                  value={inputValue}
                  onChange={e => {
                    setInputValue(e.target.value);
                    setDropOpen(true);
                  }}
                  onFocus={() => setDropOpen(true)}
                  disabled={running}
                  placeholder="Search queries…"
                  style={{ flex: 1, padding: "12px 18px", border: "none", background: "transparent", color: C.ink, fontSize: 15, fontWeight: 500, outline: "none", width: "100%" }}
                />
                {/* Clear search button — only clears the filter, not the selection */}
                {inputValue && (
                  <button
                    onClick={() => { setInputValue(""); setDropOpen(true); }}
                    style={{ background: "none", border: "none", cursor: "pointer", padding: "0 8px", display: "flex", color: C.inkMuted }}
                  >
                    <Ico n="x" sz={14} />
                  </button>
                )}
                <div style={{ padding: "0 14px", cursor: "pointer", display: "flex" }} onClick={() => setDropOpen(d => !d)}>
                  <div style={{ transform: dropOpen ? "rotate(180deg)" : "none", transition: "transform 0.2s", display: "flex" }}>
                    <Ico n="chevron" sz={15} c={C.inkMuted} />
                  </div>
                </div>
              </div>

              {dropOpen && (
                <div style={{ position: "absolute", bottom: "calc(100% + 8px)", left: 0, right: 0, zIndex: 200, background: C.paper, border: `1px solid ${C.borderMed}`, borderRadius: 8, boxShadow: "0 4px 20px rgba(0,0,0,0.08)", maxHeight: "40vh", overflowY: "auto" }}>
                  {filteredQueries.length === 0 ? (
                    <div style={{ padding: "12px 16px", color: C.inkMuted, fontSize: 14 }}>
                      No queries match "{inputValue}"
                    </div>
                  ) : (
                    // Cap render at 50 to keep DOM small
                    filteredQueries.slice(0, 50).map((q, idx) => (
                      <button
                        key={idx}
                        className="rsp-qopt"
                        onClick={() => {
                          setQuery(q);
                          setInputValue(q);
                          setDropOpen(false);
                          handleReset();
                        }}
                        style={{
                          display: "block", width: "100%", padding: "11px 16px",
                          cursor: "pointer", border: "none",
                          background: q === query ? C.blueL : "transparent",
                          color: q === query ? C.blue : C.ink,
                          fontSize: 14, textAlign: "left",
                          fontWeight: q === query ? 600 : 400,
                          borderLeft: q === query ? `3px solid ${C.blue}` : "3px solid transparent",
                          transition: "background 0.1s"
                        }}
                      >
                        {q}
                      </button>
                    ))
                  )}
                  {filteredQueries.length > 50 && (
                    <div style={{ padding: "8px 16px", color: C.inkMuted, fontSize: 12, borderTop: `1px solid ${C.border}` }}>
                      Showing 50 of {filteredQueries.length} — type to narrow results
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Top-K */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, background: C.paper, border: `1px solid ${C.borderMed}`, borderRadius: 6, padding: "0 12px" }}>
              <span style={{ fontSize: 13, color: C.inkMuted, fontWeight: 600, letterSpacing: "0.05em" }}>TOP-K</span>
              <input
                type="number" min={1} max={50} value={topK}
                onChange={e => setTopK(parseInt(e.target.value) || 10)}
                disabled={running}
                style={{ width: 45, padding: "10px 0", border: "none", background: "transparent", color: C.ink, fontSize: 15, fontWeight: 600, outline: "none", textAlign: "center" }}
              />
            </div>

            <button className="rsp-run" disabled={running || !query.trim()} onClick={handleRun}
              style={{ padding: "12px 28px", borderRadius: 6, border: "none", background: running ? C.inkMuted : C.ink, color: "white", fontSize: 14, fontWeight: 600, display: "flex", alignItems: "center", gap: 10, whiteSpace: "nowrap", minWidth: 160, justifyContent: "center" }}>
              {running ? <><Spin color="white" sz={14} />&nbsp;Processing…</> : "Run Pipeline"}
            </button>

            {(isComplete || phase > -1) && (
              <button onClick={handleReset} style={{ background: "none", border: "none", cursor: "pointer", color: C.inkSub, fontSize: 14, padding: "0 8px", fontWeight: 500 }}>
                Reset
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RAGPipelineVisualizer() {
  const [key, setKey] = useState(0);
  return (
    <ErrorBoundary onReset={() => setKey(k => k + 1)}>
      <PipelineApp key={key} />
    </ErrorBoundary>
  );
}
