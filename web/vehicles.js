/* Passed vehicles (WP17, P5): every vehicle that crossed a camera, with its plate where it could be read, a colour
   name, a search box, a CSV button, "find anywhere" and look-alike suggestions. Staff only; the server refuses drivers. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const PAGE = 50;
  const S = { camera: "", run: "", q: "", status: "", offset: 0, total: 0, items: [], sel: null, resolvedRun: null };
  const SWATCH = { white: "#f4f4f4", silver: "#c4c7cc", grey: "#8a8d93", black: "#1a1a1a", red: "#d33", orange: "#ee8a1c",
                   yellow: "#f0d020", green: "#2f9e44", blue: "#2f6fdb", purple: "#8450c8", brown: "#7a4a24", mixed: "linear-gradient(90deg,#d33,#2f9e44,#2f6fdb)" };
  const STATUS_HELP = { read: "confirmed", partial: "number only", unreadable: "seen but not readable", "no plate seen": "no plate in any view",
                        pending: "waiting to be read", "skipped: busy": "reader was busy", "skipped: small": "vehicle too small",
                        "skipped: pedal": "pedal vehicle", "not read": "plate reading is off for this camera" };

  async function api(path) {
    const res = await fetch(path);
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
  const qs = (o) => Object.entries(o).filter(([, v]) => v !== "" && v != null).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
  const tagClass = (s) => (s || "").startsWith("skipped") ? "skipped" : (s || "").replace(/\W+/g, "-");
  const colour = (c) => c ? `<span class="sw" style="background:${SWATCH[c] || "transparent"}"></span>${esc(c)}` : "";
  const plateCell = (r) => (r.plate ? `<div class="v-plate">${esc(r.plate)}</div>` : "") +
    `<span class="tag ${tagClass(r.plate_status)}" title="${esc(STATUS_HELP[r.plate_status] || "")}">${esc(r.plate_status || "-")}</span>`;
  const seen = (wall, video) => wall ? esc(wall) : (video != null ? `video ${Math.floor(video / 60)}:${String(Math.round(video % 60)).padStart(2, "0")}` : "");
  const gidLink = (g) => g ? `<a class="gid" data-gid="${esc(g)}">G${esc(g)}</a>` : '<span class="muted">-</span>';

  function filterParams(extra = {}) {
    return { camera_id: S.camera, run_id: S.run, q: S.q, status: S.status, ...extra };
  }

  async function loadCameras() {
    const d = await api("/api/passed/cameras");
    $("#camera").innerHTML = '<option value="">All cameras</option>' +
      d.items.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}${c.location ? " - " + esc(c.location) : ""} (${esc(c.vehicles)})</option>`).join("");
    S.retention = d.retention_days;
  }
  async function loadRuns() {
    const sel = $("#run");
    if (!S.camera) { sel.innerHTML = '<option value="">Newest</option>'; sel.disabled = true; return; }
    const d = await api("/api/passed/runs?" + qs({ camera_id: S.camera }));
    sel.innerHTML = '<option value="">Newest run</option><option value="all">All runs</option>' +
      d.items.map((r) => `<option value="${esc(r.run_id)}">Run ${esc(r.run_id)} - ${esc(r.vehicles)} vehicles${r.first_seen ? " - " + esc(r.first_seen) : ""}</option>`).join("");
    sel.disabled = false;
    sel.value = S.run;
  }

  async function load() {
    $("#note").textContent = "Loading...";
    try {
      const d = await api("/api/passed?" + qs(filterParams({ limit: PAGE, offset: S.offset })));
      S.items = d.items; S.total = d.total; S.resolvedRun = d.run_id;
      render(d);
    } catch (e) { $("#note").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("#csv").href = "/api/passed/export.csv?" + qs(filterParams());
  }

  function render(d) {
    $("#rows").innerHTML = S.items.map((r) => `<tr data-id="${esc(r.id)}" class="${S.sel === r.id ? "sel" : ""}">
      <td class="c-photo">${r.thumb_url ? `<img class="th" loading="lazy" alt="" src="${esc(r.thumb_url)}">` : '<div class="th"></div>'}</td>
      <td class="num" data-label="Vehicle">#${esc(r.track_id)}${S.camera ? "" : ` <span class="muted small">${esc(r.camera_name || "")}</span>`}</td>
      <td data-label="Class">${esc(r.cls)}</td>
      <td data-label="Colour">${colour(r.colour)}</td>
      <td data-label="Plate">${plateCell(r)}</td>
      <td class="num" data-label="First seen">${seen(r.first_wall, r.first_video_t)}</td>
      <td class="num" data-label="Last seen">${seen(r.last_wall, r.last_video_t)}</td>
      <td data-label="Global id">${gidLink(r.global_id)}</td></tr>`).join("") ||
      '<tr><td colspan="8" class="muted" style="padding:16px">No vehicles match.</td></tr>';
    const a = S.total ? S.offset + 1 : 0, b = Math.min(S.total, S.offset + PAGE);
    $("#page-note").textContent = `${a}-${b} of ${S.total}`;
    $("#prev").disabled = S.offset === 0;
    $("#next").disabled = b >= S.total;
    const run = d.run_id != null ? ` Showing run ${d.run_id}.` : "";
    $("#note").textContent = `${S.total} passed vehicle${S.total === 1 ? "" : "s"}.${run} Records expire after ${S.retention || 30} days: these drivers broke no rule.`;
    const counts = d.plate_status_counts || {};
    $("#status-chips").innerHTML = Object.entries(counts).sort((x, y) => y[1] - x[1]).map(([k, n]) =>
      `<button type="button" data-status="${esc(k)}" class="${S.status === k ? "on" : ""}" title="${esc(STATUS_HELP[k] || "")}">${esc(k || "-")} ${esc(n)}</button>`).join("");
    const sel = $("#status");
    const keep = S.status;
    sel.innerHTML = '<option value="">Any</option>' + Object.keys(counts).map((k) => `<option value="${esc(k)}">${esc(k)}</option>`).join("") +
      (keep && !(keep in counts) ? `<option value="${esc(keep)}">${esc(keep)}</option>` : "");
    sel.value = keep;
  }

  async function openDetail(id) {
    S.sel = id;
    document.querySelectorAll("#rows tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.id === String(id)));
    const box = $("#detail");
    box.classList.remove("hidden"); $(".v-cols").classList.add("open");
    box.innerHTML = '<p class="muted small">Loading...</p>';
    try {
      const r = await api(`/api/passed/${id}`);
      box.innerHTML = `<div class="card-head"><h2>Vehicle #${esc(r.track_id)}</h2><span class="spacer"></span>
          <button class="btn small ghost" id="d-close" type="button">Close</button></div>
        ${r.thumb_url ? `<img class="v-detail-img" alt="" src="${esc(r.thumb_url)}">` : ""}
        <table class="kv"><tbody>
          <tr><th>Camera</th><td>${esc(r.camera_name || "")}${r.camera_location ? " - " + esc(r.camera_location) : ""}</td></tr>
          <tr><th>Run</th><td>${esc(r.run_id ?? "")}</td></tr>
          <tr><th>Class / colour</th><td>${esc(r.cls)} / ${colour(r.colour) || "-"}${r.look_vlm ? ` / ${esc(r.look_vlm)}` : ""}</td></tr>
          <tr><th>Plate</th><td>${plateCell(r)}${r.plate_px ? `<div class="muted small">widest plate ${esc(r.plate_px)} px, confidence ${esc(r.plate_conf ?? "")}</div>` : ""}</td></tr>
          <tr><th>First seen</th><td>${seen(r.first_wall, r.first_video_t)}</td></tr>
          <tr><th>Last seen</th><td>${seen(r.last_wall, r.last_video_t)}</td></tr>
          <tr><th>Global id</th><td>${gidLink(r.global_id)}</td></tr></tbody></table>
        ${r.journey && r.journey.length ? `<h3 class="small" style="margin:12px 0 4px">Journey</h3>` + r.journey.map((j) =>
          `<div class="small">${esc(j.camera_name || "camera " + j.camera_id)} - ${j.t_cross ? esc(new Date(j.t_cross * 1000).toLocaleString()) : ""}</div>`).join("") : ""}
        <div class="card-head" style="margin-top:14px"><h3 style="font-size:13px">Look-alikes</h3><span class="spacer"></span>
          <span class="muted small">suggestions only - nothing is linked</span></div>
        <div id="look" class="muted small">Searching...</div>`;
      $("#d-close").onclick = closeDetail;
      loadLookalikes(id);
    } catch (e) { box.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
  function closeDetail() {
    S.sel = null; $("#detail").classList.add("hidden"); $(".v-cols").classList.remove("open");
    document.querySelectorAll("#rows tr.sel").forEach((tr) => tr.classList.remove("sel"));
  }
  async function loadLookalikes(id) {
    const el = $("#look");
    try {
      const d = await api(`/api/passed/${id}/similar?k=8`);
      if (S.sel !== id) return;
      if (!d.items.length) { el.textContent = d.reason || "Nothing similar is stored."; return; }
      el.className = "";
      el.innerHTML = `<div class="v-look">` + d.items.map((i) => `<figure>
        ${i.thumb_url ? `<img loading="lazy" alt="" src="${esc(i.thumb_url)}">` : '<div style="height:66px"></div>'}
        <figcaption><b>${Math.round(i.score * 100)}%</b> ${esc(i.cls)}${i.colour ? ", " + esc(i.colour) : ""}<br>${esc(i.camera_name || "")}
        ${i.global_id ? `<br>${gidLink(i.global_id)}${i.same_global_id ? " (same vehicle)" : ""}` : ""}</figcaption></figure>`).join("") + `</div>
        <p class="muted small">Searched ${esc(d.searched)} vectors in ${esc(d.ms)} ms${d.left_out_by_cap ? `; ${esc(d.left_out_by_cap)} older ones are over the cap` : ""}.</p>`;
    } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  }

  async function findAnywhere(extra = {}) {
    const params = { q: S.q, ...extra };
    if (!params.q && !params.gid) { toast("Type a plate, a class, a colour or g12 to look for"); return; }
    const card = $("#found-card"), out = $("#found");
    card.classList.remove("hidden"); out.innerHTML = '<p class="muted small">Searching every camera...</p>';
    try {
      const d = await api("/api/passed/find?" + qs(params));
      out.innerHTML = d.items.length ? d.items.map((i) => `<div class="v-found">
        ${i.thumb_url ? `<img alt="" loading="lazy" src="${esc(i.thumb_url)}">` : '<div style="width:72px;height:54px"></div>'}
        <div><div><b>${esc(i.cls || "vehicle")}</b> ${esc(i.colour)} ${i.plate ? `<span class="v-plate">${esc(i.plate)}</span>` : ""} ${i.plate_status ? `<span class="tag ${tagClass(i.plate_status)}">${esc(i.plate_status)}</span>` : ""} ${i.global_id ? gidLink(i.global_id) : ""}</div>
        <div class="where">Last seen at <b>${esc(i.last_seen.camera_name || "camera " + i.last_seen.camera_id)}</b>${i.last_seen.camera_location ? " (" + esc(i.last_seen.camera_location) + ")" : ""}
          ${i.last_seen.wall ? " at " + esc(i.last_seen.wall) : ""}. ${esc(i.sightings)} sighting${i.sightings === 1 ? "" : "s"} on ${esc(i.cameras)} camera${i.cameras === 1 ? "" : "s"}.</div></div></div>`).join("")
        : '<p class="muted">Nothing found. A vehicle is only listed here if it was recorded and its record has not expired.</p>';
    } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  // ---- wiring
  $("#camera").onchange = async (e) => { S.camera = e.target.value; S.run = ""; S.offset = 0; closeDetail(); await loadRuns(); load(); };
  $("#run").onchange = (e) => { S.run = e.target.value; S.offset = 0; closeDetail(); load(); };
  $("#status").onchange = (e) => { S.status = e.target.value; S.offset = 0; load(); };
  $("#search").onsubmit = (e) => { e.preventDefault(); S.q = $("#q").value.trim(); S.offset = 0; load(); };
  $("#find").onclick = () => { S.q = $("#q").value.trim(); findAnywhere(); };
  $("#found-close").onclick = () => $("#found-card").classList.add("hidden");
  $("#prev").onclick = () => { S.offset = Math.max(0, S.offset - PAGE); load(); };
  $("#next").onclick = () => { S.offset += PAGE; load(); };
  $("#rows").onclick = (e) => {
    const g = e.target.closest("a.gid");
    if (g) { e.stopPropagation(); findAnywhere({ gid: g.dataset.gid, q: "" }); return; }
    const tr = e.target.closest("tr[data-id]");
    if (tr) openDetail(+tr.dataset.id);
  };
  document.addEventListener("click", (e) => {
    const g = e.target.closest("#detail a.gid, #found a.gid");
    if (g) findAnywhere({ gid: g.dataset.gid, q: "" });
    const chip = e.target.closest("#status-chips button");
    if (chip) { S.status = S.status === chip.dataset.status ? "" : chip.dataset.status; S.offset = 0; load(); }
  });

  (async () => {
    try { await loadCameras(); await load(); }
    catch (e) { $("#note").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  })();
})();
