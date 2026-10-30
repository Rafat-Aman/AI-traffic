/* Natural-language query page (R-3). Stage A shows what was understood (editable); Stage B separates what was
   computed from the index from what a vision-language model said, with the evidence and its limits. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const S = { me: null, cams: [], last: null, busy: false };
  const CLASSES = ["Bicycle", "Bus", "Bhotbhoti", "Car", "CNG", "Easybike", "Leguna", "Motorbike", "MPV", "Pedestrian", "Pickup", "PowerTiller", "Rickshaw", "Van", "Truck", "Wheelbarrow"];
  const COLOURS = ["white", "black", "grey", "red", "orange", "yellow", "green", "blue", "purple"];
  const CODES = { ANY: "any violation", V1: "V1 bus outside stop", V2: "V2 speeding", V3: "V3 parked", V4: "V4 wrong way", V5: "V5 lane line", V6: "V6 no helmet", V7: "V7 blocking lane", V8: "V8 banned vehicle", V9: "V9 bus lane", V10: "V10 erratic", V11: "V11 near-collision" };
  const BEHAVIOURS = ["stopped", "slow", "congestion"];
  const EXAMPLES = [
    "how many buses stopped at camera 2 between 5 and 6 pm?",
    "show white cars going wrong way yesterday",
    "why did traffic back up at 5:40?",
    "CNGs over 40 km/h in the last hour",
    "bus with plate 11-2233",
    "what time did vehicle 5 pass camera 1?",
  ];

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.body !== undefined) { init.headers["content-type"] = "application/json"; init.body = JSON.stringify(opts.body); }
    const res = await fetch(path, init);
    if (res.status === 401) { location.href = "/login"; throw new Error("Sign in to continue"); }
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    return res.json();
  }
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
    clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 4500);
  }
  const camName = (id) => (S.cams.find((c) => c.id === id) || {}).name || `camera ${id}`;

  // ------------------------------------------------------------------ the parsed query
  function chips(p) {
    const items = [
      ["vehicle", (p.classes || []).join(" / ")],
      ["colour", (p.colours || []).join(" / ")],
      ["behaviour", [...(p.codes || []).map((c) => CODES[c] || c), ...(p.behaviours || [])].join(", ")],
      ["camera", (p.camera_ids || []).map(camName).join(", ")],
      ["when", p.t0_local ? `${p.t0_local.replace("T", " ")} to ${p.t1_local.replace("T", " ")}` : ""],
      ["plate", p.plate || ""],
      ["vehicle id", p.global_id ? `#${p.global_id}` : ""],
      ["speed", p.min_kph != null ? `≥ ${p.min_kph} km/h` : (p.max_kph != null ? `≤ ${p.max_kph} km/h` : "")],
      ["direction", p.direction ? (p.direction === "incoming" ? "toward the camera" : "away from the camera") : ""],
      ["words", (p.keywords || []).join(" ")],
    ];
    return items.map(([k, v]) => `<span class="q-chip${v ? "" : " none"}"><i>${k}</i> <b>${v ? esc(v) : "any"}</b></span>`).join("");
  }
  function renderParsed(p) {
    $("#parsed-card").classList.remove("hidden");
    $("#understood").innerHTML = chips(p);
    const notes = [...(p.notes || []).map((n) => `<li>${esc(n)}</li>`),
      ...(p.llm_rejected || []).map((n) => `<li class="llm">language model: ${esc(n)}</li>`)];
    if (p.llm_used) notes.push(`<li class="llm">the language model filled in fields the rules left empty (each one validated)</li>`);
    $("#notes").innerHTML = notes.join("");
    fillForm(p);
  }
  function multi(id, label, options, chosen, labels = {}) {
    return `<label>${label}<select id="${id}" multiple>${options.map((o) => `<option value="${esc(o)}"${chosen.includes(o) ? " selected" : ""}>${esc(labels[o] || o)}</option>`).join("")}</select></label>`;
  }
  function fillForm(p) {
    const camOpts = S.cams.map((c) => c.id);
    $("#form").innerHTML =
      multi("f-classes", "Vehicle classes", CLASSES, p.classes || []) +
      multi("f-colours", "Colours (estimated)", COLOURS, p.colours || []) +
      multi("f-codes", "Violation codes", Object.keys(CODES), p.codes || [], CODES) +
      multi("f-beh", "Behaviours", BEHAVIOURS, p.behaviours || []) +
      multi("f-cams", "Cameras", camOpts, p.camera_ids || [], Object.fromEntries(S.cams.map((c) => [c.id, c.name]))) +
      `<label>From (Dhaka time)<input id="f-t0" type="datetime-local" value="${esc(p.t0_local || "")}"></label>
       <label>To (Dhaka time)<input id="f-t1" type="datetime-local" value="${esc(p.t1_local || "")}"></label>
       <label>Plate contains<input id="f-plate" value="${esc(p.plate || "")}" maxlength="24"></label>
       <label>Vehicle id (global)<input id="f-gid" type="number" min="1" value="${p.global_id || ""}"></label>
       <label>Speed at least (km/h)<input id="f-min" type="number" min="0" max="250" value="${p.min_kph ?? ""}"></label>
       <label>Speed at most (km/h)<input id="f-max" type="number" min="0" max="250" value="${p.max_kph ?? ""}"></label>
       <label>Direction<select id="f-dir"><option value="">any</option><option value="incoming"${p.direction === "incoming" ? " selected" : ""}>toward the camera</option><option value="outgoing"${p.direction === "outgoing" ? " selected" : ""}>away from the camera</option></select></label>
       <label>Answer as<select id="f-agg">${["list", "count", "speed", "when"].map((a) => `<option${p.agg === a ? " selected" : ""}>${a}</option>`).join("")}</select></label>`;
    S.pointKept = p.point; S.t0Shown = p.t0_local; S.t1Shown = p.t1_local;
  }
  const picked = (id) => [...$(id).selectedOptions].map((o) => o.value);
  function editedParse() {
    const p = { question: $("#question").value.trim() };
    p.classes = picked("#f-classes"); p.colours = picked("#f-colours"); p.codes = picked("#f-codes"); p.behaviours = picked("#f-beh");
    p.camera_ids = picked("#f-cams").map(Number);
    const t0 = $("#f-t0").value, t1 = $("#f-t1").value;
    if (t0 && t1) { p.t0 = t0; p.t1 = t1; if (t0 === S.t0Shown && t1 === S.t1Shown && S.pointKept) p.point = S.pointKept; }
    p.plate = $("#f-plate").value.trim() || null;
    p.global_id = $("#f-gid").value ? Number($("#f-gid").value) : null;
    p.min_kph = $("#f-min").value !== "" ? Number($("#f-min").value) : null;
    p.max_kph = $("#f-max").value !== "" ? Number($("#f-max").value) : null;
    p.direction = $("#f-dir").value || null;
    p.agg = $("#f-agg").value;
    return p;
  }

  // ------------------------------------------------------------------ answer
  const fmtKph = (v) => (v == null ? "" : `${Number(v).toFixed(0)} km/h`);
  function links(row) {
    const out = [];
    if (row.violation_id) out.push(`<a href="/#/review/${row.violation_id}">open in review</a>`);
    if (row.global_id) out.push(`<a href="/#/vehicles/${row.global_id}">vehicle #${row.global_id} journey</a>`);
    return out.join(" · ");
  }
  function renderAnswer(r) {
    const a = r.answer, card = $("#answer-card");
    card.classList.remove("hidden");
    const route = `<div class="route"><span class="tag ${a.mode === "vlm" ? "vlm" : a.mode}">${a.mode === "index" ? "computed from the index" : a.mode === "vlm" ? "VLM reasoning" : "insufficient evidence"}</span>
      <span>Routed as <b>${esc(r.route.intent)}</b>: ${esc((r.route.why || []).join("; "))} &rarr; ${esc(r.route.stage_b)}.</span></div>`;
    let body = "";
    if (a.mode === "index") {
      const bd = a.breakdown || {};
      const parts = Object.entries(bd.by_class || {}).map(([k, v]) => `${esc(k)} ${v}`).join(", ");
      body = `<div class="ans-box index"><h3>Computed from the index</h3>
        <div class="q-ans">${a.value != null && a.stats == null ? `<div class="big">${a.value}</div>` : ""}<div><p>${esc(a.text)}</p>
        ${parts ? `<p class="small muted">${parts}</p>` : ""}</div></div>
        ${(a.rows || []).length ? `<ul class="rows">${a.rows.map((x) => `<li>${esc(x.text)} <span class="muted">(${esc(camName(x.camera_id))})</span> <a href="#seg-${x.window_id}" data-seg="${x.window_id}">segment</a>${links(x) ? " · " + links(x) : ""}</li>`).join("")}</ul>` : ""}
        ${(a.caveats || []).length ? `<ul class="caveats">${a.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}</div>`;
    } else if (a.mode === "vlm") {
      body = `<div class="ans-box vlm"><h3>VLM reasoning over ${a.evidence.length} retrieved segment${a.evidence.length === 1 ? "" : "s"}</h3>
        <p>${esc(a.text)}</p>
        <p class="small secondary">Cites ${a.citations.map((c) => `<a href="#seg-${c}" data-seg="${c}">segment ${c}</a>`).join(", ")}${a.confidence != null ? ` · model confidence ${Math.round(a.confidence * 100)}%` : ""}${a.seconds ? ` · ${a.seconds} s` : ""}</p>
        <div class="q-evidence">${a.evidence.map((e) => `<a href="#seg-${e.segment_id}" data-seg="${e.segment_id}"><img alt="frame from segment ${e.segment_id}" src="/api/query/segments/${e.segment_id}/thumb" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'thumb-missing',textContent:'no frame'}))">segment ${e.segment_id}: ${e.frames} frame${e.frames === 1 ? "" : "s"} (${esc(e.frame_source)})</a>`).join("")}</div>
        <ul class="caveats">${(a.limits || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div>`;
    } else {
      body = `<div class="ans-box insufficient"><h3>Insufficient evidence</h3><p>${esc(a.text)}</p>
        ${(a.evidence || []).length ? `<p class="small muted">${a.evidence.map((e) => `segment ${e.segment_id}: ${esc(e.note || e.frame_source)}`).join(" · ")}</p>` : ""}
        ${(a.limits || []).length ? `<ul class="caveats">${a.limits.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}</div>`;
    }
    card.innerHTML = route + body;
    card.querySelectorAll("[data-seg]").forEach((el) => el.onclick = (e) => { e.preventDefault(); const t = document.getElementById("seg-" + el.dataset.seg); if (t) { t.scrollIntoView({ behavior: "smooth", block: "center" }); t.style.outline = "2px solid var(--accent)"; setTimeout(() => t.style.outline = "", 2500); } });
  }

  function renderSegments(r) {
    const segs = r.segments || [], wrap = $("#results-wrap");
    wrap.classList.remove("hidden");
    const rt = r.retrieval || {};
    $("#results-note").textContent = `${segs.length} shown · ${rt.mode} retrieval · ${rt.total_rows} candidate rows` + ((rt.notes || []).length ? " · " + rt.notes.join("; ") : "");
    if (!segs.length) { $("#results").innerHTML = `<div class="card muted">No segments matched. Loosen the structured query above, or check the cameras and time window.</div>`; return; }
    const max = Math.max(...segs.map((s) => s.score), 0.01);
    $("#results").innerHTML = segs.map((s) => `
      <article class="q-seg${s.cited ? " cited" : ""}" id="seg-${s.id}">
        <div class="thumbbox"><img class="thumb" alt="segment ${s.id}" loading="lazy" src="/api/query/segments/${s.id}/thumb" data-seg="${s.id}"></div>
        <div>
          <h3>${esc(s.camera)} <span class="mono">${esc(s.start)} &ndash; ${esc(s.end)}</span>
            <span class="score" style="width:${Math.max(6, Math.round(70 * s.score / max))}px" title="score ${s.score}"></span>
            <span class="muted small">score ${s.score}</span>
            ${s.cited ? `<span class="tag vlm">cited</span>` : (s.given_to_reasoner ? `<span class="tag">seen by the reasoner</span>` : "")}</h3>
          <div class="why">${(s.reasons || []).map((x) => `<span>${esc(x)}</span>`).join("")}</div>
          <div class="small muted">${esc(s.summary || "")}</div>
          ${(s.matches || []).slice(0, 4).map((m) => `<div class="veh"><span>${esc(m.text)}</span>${links(m) ? `<span>${links(m)}</span>` : ""}</div>`).join("")}
          ${(s.n_matches || 0) > 4 ? `<div class="small muted">+ ${s.n_matches - 4} more in this segment</div>` : ""}
          <button class="btn small ghost" type="button" data-frames="${s.id}">Show frames</button>
          <div class="frames" id="frames-${s.id}"></div>
        </div>
      </article>`).join("");
    $("#results").querySelectorAll("img.thumb").forEach((img) => img.onerror = () => {
      img.replaceWith(Object.assign(document.createElement("div"), { className: "thumb-missing", textContent: "no frame: the stored video is not available" }));
    });
    $("#results").querySelectorAll("[data-frames]").forEach((b) => b.onclick = () => {
      const id = b.dataset.frames, box = $("#frames-" + id);
      if (box.dataset.open) { box.innerHTML = ""; delete box.dataset.open; b.textContent = "Show frames"; return; }
      box.dataset.open = "1"; b.textContent = "Hide frames";
      box.innerHTML = [0, 1, 2].map((i) => `<img alt="frame ${i + 1}" src="/api/query/segments/${id}/frame?i=${i}&n=3" onerror="this.remove()">`).join("");
    });
  }

  // ------------------------------------------------------------------ asking
  async function ask(edited) {
    const q = $("#question").value.trim();
    if (!q || S.busy) return;
    S.busy = true; $("#go").disabled = true; $("#rerun").disabled = true; $("#go").textContent = "Asking…";
    try {
      const body = { question: q, use_llm: $("#use-llm").checked };
      if ($("#intent").value) body.intent = $("#intent").value;
      if (edited) body.parsed = editedParse();
      const r = await api("/api/query", { method: "POST", body });
      S.last = r;
      renderParsed(r.parsed); renderAnswer(r); renderSegments(r);
      history.replaceState(null, "", "?q=" + encodeURIComponent(q));
    } catch (e) { toast(e.message); }
    finally { S.busy = false; $("#go").disabled = false; $("#rerun").disabled = false; $("#go").textContent = "Ask"; }
  }

  async function loadStatus() {
    try {
      const s = await api("/api/query/index/status");
      const when = s.built_at ? new Date(s.built_at * 1000).toLocaleString() : "never";
      $("#index-note").textContent = s.windows
        ? `Index: ${s.windows} windows of ${s.window_s} s · ${s.vehicle_rows} vehicle passes · built ${when} · ${s.fts ? "full-text search on" : "plain-text search"}${s.unindexed_tracklets ? ` · ${s.unindexed_tracklets} tracklets not indexed yet` : ""}`
        : "The index is empty: no analysed video yet" + (S.me && S.me.role === "admin" ? " (use Rebuild index)." : ".");
      if (!s.reasoner) $("#index-note").textContent += " · visual reasoner not running: causal questions will say so";
      if (s.job && s.job.state === "running") setTimeout(loadStatus, 1500);
      else if (s.job && s.job.state === "failed") toast("Rebuild failed: " + s.job.error);
    } catch (e) { $("#index-note").textContent = e.message; }
  }

  async function start() {
    try { S.me = await api("/api/me"); } catch (e) { return; }
    if (!S.me.signed_in) { location.href = "/login"; return; }
    if (S.me.role === "driver") { location.href = "/portal"; return; }
    $("#who").textContent = S.me.name || S.me.username || "";
    try { S.cams = await api("/api/cameras"); } catch (e) { S.cams = []; }
    $("#examples").innerHTML = EXAMPLES.map((x) => `<button type="button" class="q-ex">${esc(x)}</button>`).join("");
    $("#examples").querySelectorAll(".q-ex").forEach((b) => b.onclick = () => { $("#question").value = b.textContent; ask(false); });
    $("#ask").onsubmit = (e) => { e.preventDefault(); ask(false); };
    $("#rerun").onclick = () => ask(true);
    if (S.me.role === "admin") {
      const rb = $("#rebuild"); rb.classList.remove("hidden");
      rb.onclick = async () => {
        try { await api("/api/query/index/rebuild", { method: "POST", body: {} }); toast("Rebuilding the index in the background"); setTimeout(loadStatus, 800); }
        catch (e) { toast(e.message); }
      };
    }
    await loadStatus();
    const q = new URLSearchParams(location.search).get("q");
    if (q) { $("#question").value = q; ask(false); } else $("#question").focus();
  }
  document.addEventListener("DOMContentLoaded", () => { try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ } start(); });
})();
