/* Storage and ingest page (WP13, R-5). Staff see everything; only administrators can change the policy,
   preview/apply a retention plan, and retry/cancel ingest jobs. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const S = { data: null, plan: null, ingest: null };
  const LABELS = {
    raw_video: "Raw videos", evidence_pending: "Evidence, pending cases", evidence_confirmed: "Evidence, confirmed",
    evidence_cited: "Evidence, cited (sealed)", evidence_dismissed: "Evidence, dismissed", delivery_clip: "Delivery clips",
    partial_clip: "Unfinished clips (.part)", evidence_orphan: "Evidence, no case", autozone: "Auto-zone previews", zone_refs: "Zone references",
  };
  const POL = [
    ["raw_video_days", "Raw video kept (days)", "A week covers re-analysis after a fix; every case keeps its own 10 s clip and stills. Held while the camera has a pending case or a citation."],
    ["dismissed_days", "Dismissed cases' media (days)", "False alarms have no legal value; a month leaves room for an appeal."],
    ["legal_retention_days", "Confirmed cases' evidence (days)", "3 years by default. This is the authority's decision, not the system's."],
    ["legal_retention_confirmed", "Authority has confirmed that period", "Until ticked, nothing in confirmed cases is deleted automatically."],
    ["delivery_clip_days", "Driver copies (days)", "Regenerable from the sealed original. Never removed while a citation stands."],
    ["embed_days", "Keep appearance vectors (days)", "Only needed to link vehicles across cameras. The tracklet rows stay."],
    ["stale_part_minutes", "Stale unfinished clip (minutes)", "A .part.mp4 this old is a crashed encode and blocks citing."],
    ["orphan_days", "Files no case refers to (days, 0 = never)", "Reported, deleted only if you set a number."],
    ["warn_pct", "Disk warning (%)", ""], ["pause_pct", "Pause new ingest (%)", "Queued videos wait until the disk drops below this."],
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
  function bytes(n) {
    n = Number(n) || 0; const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i ? n.toFixed(1) : n.toFixed(0)) + " " + u[i];
  }
  const when = (t) => t ? new Date(t * 1000).toLocaleString() : "";

  function drawDisk(d) {
    const cls = d.state === "ok" ? "" : d.state;
    $("#disk").innerHTML = `<div class="card-head"><h2>Disk</h2></div>
      <div class="st-big">${d.pct}% full</div>
      <div class="meter"><i class="${cls}" style="width:${d.pct}%"></i></div>
      <div class="small secondary">${bytes(d.used)} used of ${bytes(d.total)}; ${bytes(d.free)} free. Warning at ${d.warn_pct}%, ingest pauses at ${d.pause_pct}%.</div>
      ${d.message ? `<div class="banner ${esc(d.state)}">${esc(d.message)}</div>` : ""}`;
  }
  function drawUsage(inv) {
    const cats = Object.entries(inv.categories).sort((a, b) => b[1].bytes - a[1].bytes);
    const max = Math.max(1, ...cats.map(([, c]) => c.bytes), inv.db_bytes);
    const row = (name, b, n) => `<div class="cat-row"><span>${esc(name)}</span><span class="bar"><i style="width:${(100 * b / max).toFixed(1)}%"></i></span><span class="num">${bytes(b)}</span><span class="num muted">${n ?? ""}</span></div>`;
    const t = inv.tracklets;
    $("#usage").innerHTML = cats.map(([k, c]) => row(LABELS[k] || k, c.bytes, c.files + " files")).join("") +
      row("Database", inv.db_bytes, "") +
      `<div class="small secondary" style="margin-top:6px">Tracklets: ${t.rows.toLocaleString()} rows; appearance vectors ${bytes(t.embed_bytes + t.reid_bytes)}, colour summaries ${bytes(t.hist_lab_bytes)}.</div>`;
  }
  function drawGrowth(g) {
    const daysText = (d) => d > 3650 ? "more than 10 years" : d >= 730 ? `about ${Math.round(d / 365)} years` : d >= 60 ? `about ${Math.round(d / 30)} months` : `${Math.max(1, Math.round(d))} day${Math.round(d) === 1 ? "" : "s"}`;
    const max = Math.max(1, ...g.series.map((s) => s.bytes));
    $("#growth").innerHTML = `<div class="growth-bars" role="img" aria-label="Bytes added per day, last 14 days">${g.series.map((s) => `<i title="${esc(s.day)}: ${bytes(s.bytes)}" style="height:${(100 * s.bytes / max).toFixed(0)}%"></i>`).join("")}</div>
      <div class="small secondary">Adding about ${bytes(g.bytes_per_day)} a day.
      ${g.days_to_full == null ? "No growth measured yet." : `At that rate the disk is full in <b>${daysText(g.days_to_full)}</b>.`}</div>
      <div class="small muted">${esc(g.note)}</div>`;
  }
  function drawPolicy(pol, bounds, edit) {
    $("#policy").innerHTML = `<div class="pol-grid">${POL.map(([k, label, why]) => {
      const v = pol[k];
      const input = typeof v === "boolean" ? `<input type="checkbox" data-k="${k}" ${v ? "checked" : ""} ${edit ? "" : "disabled"}>`
        : `<input type="number" data-k="${k}" value="${esc(v)}" min="${bounds[k][0]}" max="${bounds[k][1]}" ${edit ? "" : "disabled"}>`;
      return `<label for="">${esc(label)}</label>${input}${why ? `<div class="why">${esc(why)}</div>` : ""}`;
    }).join("")}</div>${edit ? `<p><button class="btn small primary" id="save-pol">Save policy</button> <span class="err" id="pol-err"></span></p>` : `<p class="muted small">Only an administrator can change the policy.</p>`}`;
    const save = $("#save-pol");
    if (save) save.onclick = async () => {
      const body = {};
      document.querySelectorAll("#policy [data-k]").forEach((el) => { body[el.dataset.k] = el.type === "checkbox" ? el.checked : Number(el.value); });
      try { await api("/api/storage/policy", { method: "PUT", body }); $("#pol-err").textContent = ""; toast("Policy saved"); S.plan = null; drawPlan(); load(); }
      catch (e) { $("#pol-err").textContent = e.message; }
    };
  }
  function drawIngest(q) {
    if (!q) { $("#ingest").innerHTML = `<p class="muted small">The ingest queue is not running on this server.</p>`; return; }
    const c = q.stats.counts;
    $("#ingest-note").textContent = `${c.running} running (max ${q.stats.max_parallel}), ${c.queued} queued, ${c.failed} failed` + (q.stats.paused ? ", PAUSED: disk full" : "");
    if (!q.jobs.length) { $("#ingest").innerHTML = `<p class="muted small">No jobs yet.</p>`; return; }
    $("#ingest").innerHTML = `<table class="list"><thead><tr><th>Camera</th><th>State</th><th>Tries</th><th>Queued</th><th>Error</th><th></th></tr></thead><tbody>` +
      q.jobs.slice(0, 50).map((j) => `<tr><td>${esc(j.camera_name)}</td><td><span class="tag ${esc(j.state)}">${esc(j.state)}</span></td><td>${j.attempts}/${j.max_attempts}</td><td class="small">${esc(when(j.created_at))}</td><td class="small">${esc(j.error)}</td><td>${q.can_edit ? (["failed", "cancelled"].includes(j.state) ? `<button class="btn small" data-retry="${j.id}">Retry</button>` : ["queued", "running"].includes(j.state) ? `<button class="btn small danger" data-cancel="${j.id}">Cancel</button>` : "") : ""}</td></tr>`).join("") + `</tbody></table>`;
    document.querySelectorAll("[data-retry],[data-cancel]").forEach((b) => b.onclick = async () => {
      const id = b.dataset.retry || b.dataset.cancel;
      try { await api(`/api/ingest/${id}/${b.dataset.retry ? "retry" : "cancel"}`, { method: "POST" }); loadIngest(); } catch (e) { toast(e.message); }
    });
  }
  function drawPlan() {
    const p = S.plan, box = $("#plan");
    $("#apply-plan").disabled = !p || p.status !== "planned" || !p.items_total;
    if (!p) { box.innerHTML = ""; return; }
    const rules = Object.entries(p.by_rule).map(([r, v]) => `<tr><td>${esc(r)}</td><td>${v.items}</td><td>${bytes(v.bytes)}</td></tr>`).join("");
    const kept = Object.entries(p.kept).sort((a, b) => b[1].bytes - a[1].bytes).map(([r, v]) => `<tr><td>${esc(r)}</td><td>${v.files}</td><td>${bytes(v.bytes)}</td><td class="small">${esc(v.example)}</td></tr>`).join("");
    box.innerHTML = `<div class="plan-sum"><div><b>${p.totals.files}</b><span>files to delete</span></div><div><b>${bytes(p.totals.bytes)}</b><span>to free</span></div><div><b>#${p.id}</b><span>${esc(p.status)}${p.dry_run ? " (preview)" : ""}</span></div></div>
      ${p.result ? `<div class="banner">Applied: ${p.result.deleted} items, ${bytes(p.result.freed_bytes)} freed, ${p.result.skipped.length} skipped.</div>` : ""}
      <h3 class="small">Would delete, by rule</h3><table class="list"><tbody>${rules || `<tr><td class="muted">Nothing to delete.</td></tr>`}</tbody></table>
      <h3 class="small" style="margin-top:12px">Kept, and why</h3><table class="list"><thead><tr><th>Rule</th><th>Files</th><th>Size</th><th>Example reason</th></tr></thead><tbody>${kept}</tbody></table>
      <h3 class="small" style="margin-top:12px">Items (${Math.min(p.items.length, p.items_total)} of ${p.items_total})</h3>
      <div class="plan-items"><table class="list"><thead><tr><th>What</th><th>Size</th><th>Why</th></tr></thead><tbody>${p.items.map((i) => `<tr><td class="path">${esc(i.path)}</td><td>${bytes(i.bytes)}</td><td>${esc(i.reason)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  async function loadIngest() { try { S.ingest = await api("/api/ingest"); } catch (e) { S.ingest = null; } drawIngest(S.ingest); }
  async function load() {
    const d = S.data = await api("/api/storage");
    $("#who").textContent = d.can_edit ? "administrator" : "view only";
    drawDisk(d.disk); drawUsage(d.inventory); drawGrowth(d.growth); drawPolicy(d.policy, d.bounds, d.can_edit);
    $("#make-plan").disabled = !d.can_edit;
    await loadIngest();
  }
  $("#make-plan").onclick = async () => {
    try { S.plan = await api("/api/storage/plan", { method: "POST" }); drawPlan(); } catch (e) { toast(e.message); }
  };
  $("#apply-plan").onclick = async () => {
    if (!S.plan || !confirm(`Delete ${S.plan.totals.files} files (${bytes(S.plan.totals.bytes)})? This cannot be undone.`)) return;
    try {
      const r = await api("/api/storage/apply", { method: "POST", body: { plan_id: S.plan.id } });
      S.plan = { ...S.plan, status: "applied", dry_run: false, result: r }; drawPlan(); toast(`Freed ${bytes(r.freed_bytes)}`); load();
    } catch (e) { toast(e.message); }
  };
  load().catch((e) => toast(e.message));
  setInterval(() => loadIngest().catch(() => {}), 10000);
})();
