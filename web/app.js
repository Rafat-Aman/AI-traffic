/* Traffic Surveillance dashboard: routing, live updates, cameras, review queue and statistics.
   The zone editor is in editor.js. All user / database text goes through esc() or textContent. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.body !== undefined) {
      init.headers["content-type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }
    const res = await fetch(path, init);
    if (res.status === 401) {          // the session ended or never existed
      location.href = "/login";
      throw new Error("Sign in to continue");
    }
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res.blob();
  }

  const ICON = {
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z"/></svg>',
    stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>',
    zones: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20L9 4h6l5 16z"/><path d="M12 6v3M12 12v3M12 18v2"/></svg>',
    plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M12 5v14M5 12h14"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M5 12l5 5 9-10"/></svg>',
    cross: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    cam: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 8h12l4-3v14l-4-3H3z"/></svg>',
    print: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9V3h12v6M6 18H4v-7h16v7h-2M6 14h12v7H6z"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12M7 10l5 5 5-5M4 20h16"/></svg>',
    edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/></svg>',
    auto: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/></svg>',
    pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21s-6-5.5-6-11a6 6 0 0112 0c0 5.5-6 11-6 11z"/><circle cx="12" cy="10" r="2"/></svg>',
    expand: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
    brain: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 4a3 3 0 00-3 3 3 3 0 00-1 5.8V15a3 3 0 003 3 2 2 0 004 0V6a2 2 0 00-3-2z"/><path d="M15 4a3 3 0 013 3 3 3 0 011 5.8V15a3 3 0 01-3 3 2 2 0 01-4 0"/></svg>',
  };
  // Short names for the violation types (the full titles come from the server).
  const VSHORT = { V1: "Bus stopping", V2: "Speeding", V3: "Kerb parking", V4: "Wrong side", V5: "Solid line", V6: "No helmet", V7: "Lane blocking", V8: "Banned vehicle" };
  const vcode = (c) => `<span class="code" data-v="${esc(c)}">${esc(c)}</span>`;
  const STATUS_ICON = { confirmed: ICON.check, rejected: ICON.cross, pending: ICON.clock };
  const STATE_LABEL = { idle: "Idle", starting: "Starting", running: "Running", reconnecting: "Reconnecting", done: "Finished", stopped: "Stopped", error: "Error" };

  // Warnings that change while a camera runs (refreshed on every status message, see updateCamViews)
  function camFlags(st) {
    return (st && st.state === "running" && st.speed_paused ? '<span class="chip reconnecting">Speed check paused</span>' : "") + viewChip(st);
  }
  // Camera movement relative to the view the zones were drawn on (surveillance/camera_motion.py)
  function viewChip(st) {
    const v = st && st.state === "running" && st.view;
    if (!v || v.state === "stable") return "";
    if (v.state === "compensated") return `<span class="chip" title="The picture moved ${esc(v.shift_px)} px (zoom ${esc(v.zoom)}); vehicle positions are mapped back onto the zones">Camera moved: zones re-aligned</span>`;
    if (v.state === "lost") return '<span class="chip reconnecting" title="The view changed so much that the zones no longer match it. Point the camera back or redraw the zones.">View changed: enforcement paused</span>';
    return '<span class="chip" title="The view could not be checked just now (dark / low contrast); the camera is assumed not to have moved">View not verified</span>';
  }
  function statusBadge(s) {
    return `<span class="status-badge ${esc(s)}">${STATUS_ICON[s] || ""}${esc(s[0].toUpperCase() + s.slice(1))}</span>`;
  }
  function stateChip(st) {
    const s = (st && st.state) || "idle";
    return `<span class="chip ${esc(s)}"><span class="dot"></span>${esc(STATE_LABEL[s] || s)}</span>`;
  }
  function when(v) {
    if (v.wall_time) return v.wall_time;
    const t = v.video_time_s || 0;
    return `video ${Math.floor(t / 60)}:${String((t % 60).toFixed(1)).padStart(4, "0")}`;
  }
  function plateChip(v) {
    const st = v.plate_status || "";
    if (st === "read" || st === "partial") return `<span class="plate-chip" title="${esc(v.plate_text_en)}">${esc(v.plate_text)}</span>`;
    if (st === "pending") return '<span class="plate-chip muted">plate…</span>';
    if (st === "unreadable") return '<span class="plate-chip muted" title="Plate not readable in the video">no plate</span>';
    return "";
  }
  function media(p) { return p ? "/media/" + p.split("/").map(encodeURIComponent).join("/") : ""; }
  function summary(v) {
    const d = v.details || {};
    if (v.code === "V2") return `${Math.round(d.kph ?? v.kph)} km/h in a ${d.limit_kph} km/h zone`;
    if (v.code === "V1") return `stood ${Math.round(d.dwell_s)} s · ${d.where || "no bus stop"}`;
    if (v.code === "V3") return `parked ${Math.round(d.dwell_s)} s at the kerb${d.enforced_because ? " · " + d.enforced_because : ""}`;
    if (v.code === "V7") return `stood ${Math.round(d.dwell_s)} s in a lane · ${d.vehicles_passed} vehicles went round`;
    if (v.code === "V4") return `${d.wrong_way_m} m against traffic`;
    if (v.code === "V5") return `crossed ${d.line}`;
    if (v.code === "V6") return `${d.bare_head_sightings} bare-head sightings`;
    if (v.code === "V8") return `banned on ${v.zone}`;
    return v.zone || "";
  }

  function toast(msg, kind = "", onclick = null, ms = 5000) {
    const t = document.createElement("div");
    t.className = "toast " + kind;
    t.innerHTML = msg;
    if (onclick) t.addEventListener("click", () => { onclick(); t.remove(); });
    $("#toasts").appendChild(t);
    setTimeout(() => t.remove(), ms);
  }
  const fail = (e) => toast(esc(e.message || e), "error");

  function modal(html, cls = "", onClose = null) {
    const back = document.createElement("div");
    back.className = "modal-back";
    back.innerHTML = `<div class="modal ${cls}" role="dialog" aria-modal="true">${html}</div>`;
    document.body.appendChild(back);
    const close = () => { back.remove(); document.removeEventListener("keydown", onKey); if (onClose) onClose(); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);
    back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
    $$("[data-close]", back).forEach((b) => b.addEventListener("click", close));
    return { el: back.firstElementChild, close };
  }

  function confirmBox(text, okLabel = "Delete") {
    return new Promise((resolve) => {
      let answer = false;
      const m = modal(`<div class="modal-head"><h2>Are you sure?</h2></div><p class="secondary">${text}</p>
        <div class="row" style="justify-content:flex-end"><button class="btn" data-close>Cancel</button><button class="btn bad" id="ok">${esc(okLabel)}</button></div>`,
        "", () => resolve(answer));
      $("#ok", m.el).onclick = () => { answer = true; m.close(); };
    });
  }

  // ------------------------------------------------------------------ app state, live events
  const state = { meta: null, status: {}, pending: 0, sse: null, page: null, me: null };
  // Hiding a control is a courtesy to the user, not a security measure: the server refuses the request
  // regardless of what the page chose to draw.
  const isAdmin = () => state.me && state.me.role === "admin";
  const listeners = new Set(); // page-level handlers for live messages

  function connectEvents() {
    const es = new EventSource("/api/events");
    state.sse = es;
    es.onopen = () => { $("#sse-dot").classList.add("on"); $("#sse-text").textContent = "Live updates on"; };
    es.onerror = () => { $("#sse-dot").classList.remove("on"); $("#sse-text").textContent = "Reconnecting..."; };
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "violation") {
        const v = msg.violation;
        state.pending += 1;
        renderPendingCount();
        toast(`<b>${esc(v.code)} ${esc(v.title)}</b><br><span class="muted">${esc(v.cls)} · ${esc(v.camera_name)} · ${esc(summary(v))}</span>`,
          "violation", () => go(`#/review/${v.id}`));
      }
      if (msg.type === "autozone" && ["done", "error"].includes(msg.status.state)) {
        const ok = msg.status.state === "done";
        toast(ok ? `<b>Zones learned automatically</b><br><span class="muted">${esc(msg.status.message)}</span>`
          : `<b>Automatic zone setup failed</b><br><span class="muted">${esc(msg.status.message)}</span>`, ok ? "ok" : "error",
          () => go(`#/camera/${msg.camera_id}`), 9000);
      }
      if (msg.type === "plate") {  // a feed item on any page gets its plate as soon as it is read
        $$(`.feed-item[data-vid="${msg.violation.id}"]`).forEach((el) => { el.outerHTML = feedItem(msg.violation); });
      }
      listeners.forEach((fn) => fn(msg));
    };
  }

  async function refreshPending() {
    try {
      const r = await api("/api/violations?status=pending&limit=1");
      state.pending = r.total;
      renderPendingCount();
    } catch (e) { /* server restarting */ }
  }
  function renderPendingCount() {
    const el = $("#pending-count");
    el.textContent = state.pending;
    el.classList.toggle("hidden", !state.pending);
  }

  // Poll camera status while a page shows cameras; pages register an update callback.
  let statusTimer = null;
  function pollStatus(fn) {
    clearInterval(statusTimer);
    const tick = async () => {
      try { state.status = await api("/api/status"); fn(state.status); } catch (e) { /* ignore */ }
    };
    tick();
    statusTimer = setInterval(tick, 1500);
  }

  // ------------------------------------------------------------------ router
  let cleanups = [];
  let leaveGuard = null, currentHash = location.hash;
  function onLeave(fn) { cleanups.push(fn); }
  function setLeaveGuard(fn) { leaveGuard = fn; }
  function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }

  async function route() {
    if (leaveGuard && location.hash !== currentHash && !leaveGuard()) {
      history.replaceState(null, "", currentHash); // stay on the page
      return;
    }
    leaveGuard = null;
    currentHash = location.hash;
    cleanups.forEach((fn) => { try { fn(); } catch (e) { /* ignore */ } });
    cleanups = [];
    listeners.clear();
    clearInterval(statusTimer);
    $$("img.stream").forEach((img) => { img.src = ""; }); // close MJPEG connections
    const parts = location.hash.replace(/^#\/?/, "").split("?")[0].split("/").filter(Boolean);  // "?…" = page options
    const main = $("#main");
    main.innerHTML = "";
    const name = parts[0] || "dashboard";
    $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === (name === "camera" ? "cameras" : name)));
    try {
      if (name === "dashboard") await pageDashboard(main);
      else if (name === "cameras") await pageCameras(main);
      else if (name === "camera" && parts[2] === "zones") await window.Editor.render(main, +parts[1], App);
      else if (name === "camera") await pageCamera(main, +parts[1]);
      else if (name === "review") await pageReview(main, parts[1] ? +parts[1] : null);
      else if (name === "vehicles") await pageVehicles(main, parts[1] ? +parts[1] : null);
      else if (name === "admin") await pageAdmin(main);
      else if (name === "stats") await pageStats(main);
      else main.innerHTML = `<div class="empty">Page not found</div>`;
    } catch (e) {
      main.innerHTML = `<div class="card errors">Could not load this page: ${esc(e.message)}</div>`;
    }
  }

  // ------------------------------------------------------------------ camera tile (dashboard + camera page)
  const LEARNING = ["starting", "observing", "analysing"];
  const isLearning = (st) => !!(st && st.learning && LEARNING.includes(st.learning.state));
  const zonesReady = (cam, st) => cam.zones_ready || !!(st && st.learning && st.learning.state === "done" && st.learning.version);
  function camView(cam, st, big = false) {
    const live = st && ["running", "starting", "reconnecting"].includes(st.state);
    const learn = isLearning(st);
    const L = learn ? st.learning : null;
    const idleText = learn ? `Learning the road layout from traffic…<br><span class="small">${esc(L.message || "")}</span>`
      : cam.zones_ready ? (st && st.state === "done" ? "Analysis finished" : "Not running")
      : "Set up the road zones to start: learn them automatically or draw them";
    const inner = live
      ? `<img class="stream" alt="Live view of ${esc(cam.name)}" src="/api/cameras/${cam.id}/stream?ts=${Date.now()}">`
      : `${cam.kind === "file" ? `<img class="poster" alt="" src="/api/cameras/${cam.id}/frame?t=5" loading="lazy" onerror="this.remove()">` : ""}
        <div class="placeholder">${ICON.cam.replace("<svg", '<svg width="40" height="40"')}<br>${idleText}</div>`;
    const p = learn ? L.progress : st && st.progress;
    const prog = p != null ? `<div class="progress${learn ? " learn" : ""}"><div style="width:${(p * 100).toFixed(1)}%"></div></div>` : "";
    return `<div class="cam-view${big ? " big" : ""}" data-cam-view="${cam.id}" data-live="${live ? 1 : 0}" data-learn="${learn ? 1 : 0}">${inner}
      <div class="overlay-top">${learn ? `<span class="chip learning"><span class="dot"></span>Learning zones ${Math.round((L.progress || 0) * 100)}%</span>` : stateChip(st)}
        <span class="spacer"></span>${cam.zone_mode === "auto" ? '<span class="chip">Auto zones</span>' : ""}<span class="cam-flags">${camFlags(st)}</span></div></div>${prog}`;
  }
  function camStatsText(cam, st) {
    if (isLearning(st)) return st.learning.message || "Learning the zones";
    if (st && st.learning && st.learning.state === "error" && !cam.zones_ready) return "Automatic zones failed: " + st.learning.message;
    if (!st || st.state === "idle") return cam.kind === "file" ? "Video file" : "Live camera";
    const parts = [];
    if (st.fps) parts.push(`${st.fps} FPS`);
    if (st.speed_x) parts.push(`${st.speed_x}× real time`);
    if (cam.kind === "live" && st.lag != null && st.state === "running") parts.push(`lag ${st.lag}s`);
    if (st.tracks != null && st.state === "running") parts.push(`${st.tracks} vehicles`);
    if (st.progress != null && st.state === "running") parts.push(`${Math.round(st.progress * 100)}%`);
    if (st.message && st.state !== "running") parts.push(st.message);
    return parts.join(" · ");
  }
  function camButtons(cam, st) {
    const running = st && ["running", "starting", "reconnecting"].includes(st.state);
    if (isLearning(st)) return `<button class="btn small" data-act="cancel-learn" data-id="${cam.id}">${ICON.stop}Cancel learning</button>`;
    const startLabel = cam.kind === "file" ? (st && st.state === "done" ? "Analyse again" : "Analyse video") : "Start live";
    if (!running && !zonesReady(cam, st)) {
      return `<button class="btn small primary" data-act="learn" data-id="${cam.id}">${ICON.auto}Learn zones automatically</button>
        <a class="btn small" href="#/camera/${cam.id}/zones">${ICON.zones}Draw zones</a>`;
    }
    return `${running
      ? `<button class="btn small" data-act="stop" data-id="${cam.id}">${ICON.stop}Stop</button>`
      : `<button class="btn small primary" data-act="start" data-id="${cam.id}">${ICON.play}${startLabel}</button>`}
      <a class="btn small" href="#/camera/${cam.id}/zones">${ICON.zones}Zones</a>`;
  }
  async function camAction(act, id) {
    try {
      if (act === "start") { await api(`/api/cameras/${id}/start`, { method: "POST" }); toast("Starting camera...", "ok", null, 2500); }
      if (act === "stop") { await api(`/api/cameras/${id}/stop`, { method: "POST" }); }
      if (act === "learn") {
        await api(`/api/cameras/${id}/auto-zones`, { method: "POST", body: { apply: true, start: true } });
        toast("Learning the road layout from traffic. The camera starts by itself when the zones are ready.", "ok", null, 5000);
      }
      if (act === "cancel-learn") { await api(`/api/cameras/${id}/auto-zones`, { method: "DELETE" }); }
    } catch (e) { fail(e); }
  }
  function bindCamButtons(root) {
    root.addEventListener("click", (e) => {
      const b = e.target.closest("[data-act]");
      if (b && ["start", "stop", "learn", "cancel-learn"].includes(b.dataset.act)) camAction(b.dataset.act, +b.dataset.id);
    });
  }
  // Re-render a camera's view when its running state changes; otherwise only update the text.
  function updateCamViews(cams, status) {
    cams.forEach((cam) => {
      const st = status[cam.id];
      $$(`[data-cam-view="${cam.id}"]`).forEach((view) => {
        const live = st && ["running", "starting", "reconnecting"].includes(st.state) ? "1" : "0";
        const learn = isLearning(st) ? "1" : "0";
        if (learn === "1") {
          const chip = view.querySelector(".chip.learning");
          if (chip) chip.lastChild.textContent = `Learning zones ${Math.round((st.learning.progress || 0) * 100)}%`;
          const sm = view.querySelector(".placeholder .small");
          if (sm) sm.textContent = st.learning.message || "";
          const bar = view.nextElementSibling && view.nextElementSibling.querySelector(":scope > div");
          if (bar) bar.style.width = ((st.learning.progress || 0) * 100).toFixed(1) + "%";
        }
        if (view.dataset.learn !== learn || (learn === "0" && (view.dataset.live !== live || !view.querySelector(".chip." + (st ? st.state : "idle"))))) {
          const old = view.querySelector("img.stream");
          if (old) old.src = "";
          const tmp = document.createElement("div");
          tmp.innerHTML = camView(cam, st, view.classList.contains("big"));
          const oldProg = view.nextElementSibling && view.nextElementSibling.classList.contains("progress") ? view.nextElementSibling : null;
          if (oldProg) oldProg.remove();
          view.replaceWith(...tmp.childNodes);
        } else {
          const flags = view.querySelector(".cam-flags"), html = camFlags(st);
          if (flags && flags.innerHTML !== html) flags.innerHTML = html;
          const bar = view.nextElementSibling && view.nextElementSibling.querySelector(":scope > div");
          if (bar && st.progress != null) bar.style.width = (st.progress * 100).toFixed(1) + "%";
          else if (!bar && st && st.progress != null) view.insertAdjacentHTML("afterend", `<div class="progress"><div style="width:${(st.progress * 100).toFixed(1)}%"></div></div>`);
        }
      });
      // status text and buttons can sit outside the view's card (camera page header): find them by camera id
      $$(`[data-cam-stats="${cam.id}"]`).forEach((meta) => { meta.textContent = camStatsText(cam, st); });
      const key = (st ? st.state : "idle") + (isLearning(st) ? "-learn" : "") + (zonesReady(cam, st) ? "-z" : "");
      $$(`[data-cam-btns="${cam.id}"]`).forEach((btns) => {
        if (btns.dataset.state !== key) { btns.innerHTML = camButtons(cam, st); btns.dataset.state = key; }
      });
    });
  }

  function feedItem(v, isNew) {
    const n = 1 + (v.crop ? 1 : 0) + (v.evidence || []).length + (v.clip ? 1 : 0) + (v.plate_img ? 1 : 0);
    return `<div class="feed-item${isNew ? " new" : ""}" data-vid="${v.id}" data-code="${esc(v.code)}">
      <img src="${media(v.crop || v.snapshot)}" alt="" loading="lazy">
      <div class="feed-body"><div class="t">${vcode(v.code)} ${esc(v.title)}${statusBadge(v.status)}</div>
      <div class="m">${esc(v.cls)} · ${esc(summary(v))} ${plateChip(v)}</div>
      <div class="m">${ICON.pin.replace("<svg", '<svg class="i"')}${esc(v.camera_name)} · ${esc(when(v))} · ${n} evidence</div></div></div>`;
  }

  // ------------------------------------------------------------------ dashboard
  async function pageDashboard(main) {
    const [cams, recent, today, all] = await Promise.all([
      api("/api/cameras"), api("/api/violations?limit=40"), api("/api/stats?days=1&basis=detected"), api("/api/stats?days=36500&include_rejected=true")]);
    const byStatus = Object.fromEntries(all.by_status.map((r) => [r.status, r.n]));
    const reviewed = (byStatus.confirmed || 0) + (byStatus.rejected || 0);
    const precision = reviewed ? Math.round((100 * (byStatus.confirmed || 0)) / reviewed) + "%" : "–";
    const todayN = today.by_code.reduce((a, r) => a + r.n, 0);
    const allBy = Object.fromEntries(all.by_code.map((r) => [r.code, r.n]));
    const todayBy = Object.fromEntries(today.by_code.map((r) => [r.code, r.n]));
    const rules = Object.values(state.meta.rules).sort((a, b) => a.code.localeCompare(b.code));
    const coverage = (code) => {
      const withZones = cams.filter((c) => c.zones_ready);
      const on = withZones.filter((c) => c.rules && !c.rules[code]);
      if (!withZones.length) return { on: 0, text: "no camera set up" };
      if (on.length) return { on: on.length, text: `on ${on.length} of ${withZones.length} camera${withZones.length === 1 ? "" : "s"}` };
      const why = withZones[0].rules ? withZones[0].rules[code] : "";
      return { on: 0, text: "off: " + why };
    };
    const plates = recent.items.filter((v) => v.plate_status === "read" || v.plate_status === "partial").length;
    main.innerHTML = `
      <div class="page-head"><div><h1>Control room</h1><p>Live cameras, recorded videos and violations as they are detected</p></div>
        <span class="spacer"></span><a class="btn" href="#/review">${ICON.check}Review queue</a><a class="btn primary" href="#/cameras?add=1">${ICON.plus}Add camera</a></div>
      <div class="tiles">
        <div class="tile"><div class="label">Detected in the last 24 h</div><div class="value" id="k-today">${todayN}</div><div class="sub">excluding rejected</div></div>
        <div class="tile accent-warn"><div class="label">Waiting for review</div><div class="value" id="k-pending">${byStatus.pending || 0}</div><div class="sub"><a href="#/review">Open review queue</a></div></div>
        <div class="tile"><div class="label">Cameras online</div><div class="value"><span id="k-running">0</span><span class="of"> / ${cams.length}</span></div><div class="sub" id="k-run-sub">none running</div></div>
        <div class="tile"><div class="label">Plates read (latest 40)</div><div class="value">${plates}</div><div class="sub">others: identify from images</div></div>
        <div class="tile"><div class="label">Confirmed by reviewers</div><div class="value">${precision}</div><div class="sub">${reviewed} reviewed cases</div></div>
      </div>
      <div class="vtypes" id="vtypes">${rules.map((r) => {
        const cv = coverage(r.code);
        return `<a class="vtype${cv.on ? "" : " off"}" href="#/review?code=${r.code}" data-vt="${r.code}" title="${esc(r.title)} – ${esc(cv.text)}">
          <div class="vt-head">${vcode(r.code)}<span class="vt-name">${esc(VSHORT[r.code] || r.title)}</span></div>
          <div class="vt-num"><b data-vt-all="${r.code}">${allBy[r.code] || 0}</b><span class="muted small"> · <span data-vt-today="${r.code}">${todayBy[r.code] || 0}</span> today</span></div>
          <div class="vt-cov small ${cv.on ? "muted" : "warn"}">${esc(cv.text)}</div></a>`;
      }).join("")}</div>
      <div class="dash">
        <div class="cams" id="cams">${cams.length ? cams.map((c) => `
          <div class="cam-tile">${camView(c, c.status)}
            <div class="cam-meta"><div class="cam-title"><a class="name" href="#/camera/${c.id}" data-cam-name="${c.id}">${esc(c.name)}</a>
                <button class="btn ghost small" data-edit="${c.id}" title="Change the camera's name and location">${ICON.edit}Edit name &amp; location</button>
                <div class="small muted" data-cam-loc="${c.id}">${ICON.pin.replace("<svg", '<svg class="i"')}${esc(c.location || "No location set")}</div></div>
              <span class="spacer"></span>
              <span class="row" data-cam-btns="${c.id}" data-state="${esc(c.status.state)}">${camButtons(c, c.status)}</span>
              <div class="stats" data-cam-stats="${c.id}">${esc(camStatsText(c, c.status))}</div></div></div>`).join("")
          : `<div class="card empty">${ICON.cam}<br>No cameras yet.<br><br><a class="btn primary" href="#/cameras?add=1">${ICON.plus}Add a camera or video</a></div>`}
        </div>
        <div class="card feed-card"><div class="card-head"><h2>Live alerts</h2><span class="live-dot"></span><span class="spacer"></span><a class="small" href="#/review">Review all</a></div>
          <div class="chips" id="feed-filter"><button class="on" data-f="">All</button>${rules.map((r) => `<button data-f="${r.code}">${r.code}</button>`).join("")}</div>
          <div class="feed" id="feed">${recent.items.length ? recent.items.map((v) => feedItem(v)).join("") : `<div class="empty">Violations appear here the moment they are detected.</div>`}</div></div>
      </div>`;
    bindCamButtons($("#cams"));
    $("#cams").addEventListener("click", (e) => {
      const b = e.target.closest("[data-edit]");
      if (!b) return;
      const cam = cams.find((c) => c.id === +b.dataset.edit);
      editCameraModal(cam, (upd) => {
        Object.assign(cam, { name: upd.name, location: upd.location, source: upd.source });
        $(`[data-cam-name="${cam.id}"]`).textContent = upd.name;
        $(`[data-cam-loc="${cam.id}"]`).innerHTML = ICON.pin.replace("<svg", '<svg class="i"') + esc(upd.location || "No location set");
      });
    });
    let filter = "";
    const applyFilter = () => $$("#feed .feed-item").forEach((it) => it.classList.toggle("hidden", !!filter && it.dataset.code !== filter));
    $("#feed-filter").onclick = (e) => {
      const b = e.target.closest("[data-f]");
      if (!b) return;
      filter = b.dataset.f;
      $$("#feed-filter button").forEach((x) => x.classList.toggle("on", x === b));
      applyFilter();
    };
    $("#feed").addEventListener("click", (e) => { const it = e.target.closest("[data-vid]"); if (it) go(`#/review/${it.dataset.vid}`); });
    pollStatus((status) => {
      updateCamViews(cams, status);
      const running = cams.filter((c) => status[c.id] && ["running", "starting", "reconnecting"].includes(status[c.id].state));
      $("#k-running").textContent = running.length;
      $("#k-run-sub").textContent = running.length ? running.map((c) => c.name).join(", ") : "none running";
    });
    listeners.add((msg) => {
      if (msg.type !== "violation") return;
      const feed = $("#feed");
      if (!feed) return;
      if (!feed.querySelector("[data-vid]")) feed.innerHTML = "";
      feed.insertAdjacentHTML("afterbegin", feedItem(msg.violation, true));
      applyFilter();
      const c = msg.violation.code;
      $("#k-today").textContent = +$("#k-today").textContent + 1;
      $("#k-pending").textContent = +$("#k-pending").textContent + 1;
      const a = $(`[data-vt-all="${c}"]`), t = $(`[data-vt-today="${c}"]`);
      if (a) a.textContent = +a.textContent + 1;
      if (t) t.textContent = +t.textContent + 1;
      const card = $(`[data-vt="${c}"]`);
      if (card) { card.classList.remove("pulse"); void card.offsetWidth; card.classList.add("pulse"); }
    });
  }

  // ------------------------------------------------------------------ cameras list + add
  async function pageCameras(main) {
    const cams = await api("/api/cameras");
    main.innerHTML = `
      <div class="page-head"><div><h1>Cameras</h1><p>Each camera position has its own road zones. Recorded videos are analysed completely; live cameras in real time.</p></div>
        <span class="spacer"></span><button class="btn primary" id="add">${ICON.plus}Add camera</button></div>
      <div class="card flush">${cams.length ? `<table class="list"><tr><th>Camera</th><th>Source</th><th>Zones</th><th>Status</th><th>Violations</th><th></th></tr>
        ${cams.map((c) => `<tr>
          <td><a href="#/camera/${c.id}"><b>${esc(c.name)}</b></a><div class="small muted">${esc(c.location || "No location set")}</div></td>
          <td><span class="chip">${c.kind === "file" ? "Video file" : "Live"}</span><div class="small muted" title="${esc(c.source)}">${esc(c.source.length > 46 ? "…" + c.source.slice(-45) : c.source)}</div></td>
          <td>${c.zones_ready ? `v${c.zone_version} <span class="chip">${c.zone_mode === "auto" ? "automatic" : "manual"}</span>` : isLearning(c.status) ? '<span class="chip learning">learning…</span>' : `<a href="#/camera/${c.id}/zones">Set up zones</a>`}</td>
          <td data-state-of="${c.id}">${stateChip(c.status)}</td>
          <td class="mono">${c.total}${c.pending ? ` <span class="small muted">(${c.pending} pending)</span>` : ""}</td>
          <td><div class="row" style="justify-content:flex-end"><span class="row" data-cam-btns="${c.id}">${camButtons(c, c.status)}</span>
            <button class="btn small ghost" data-edit="${c.id}" title="Rename / edit camera position">${ICON.edit}Edit</button>
            <button class="btn small ghost danger" data-del="${c.id}" title="Delete camera">${ICON.trash}</button></div></td></tr>`).join("")}</table>`
        : `<div class="empty">${ICON.cam}<br>No cameras yet. Add a recorded video or a live phone camera to begin.</div>`}</div>`;
    $("#add").onclick = () => addCameraModal();
    bindCamButtons(main);
    main.addEventListener("click", async (e) => {
      const ed = e.target.closest("[data-edit]");
      if (ed) { editCameraModal(cams.find((c) => c.id === +ed.dataset.edit), () => route()); return; }
      const d = e.target.closest("[data-del]");
      if (!d) return;
      const cam = cams.find((c) => c.id === +d.dataset.del);
      if (await confirmBox(`Delete <b>${esc(cam.name)}</b> with its zones, ${cam.total} violations and all evidence files?`)) {
        try { await api(`/api/cameras/${cam.id}`, { method: "DELETE" }); route(); } catch (err) { fail(err); }
      }
    });
    pollStatus((status) => cams.forEach((c) => {
      const st = status[c.id];
      const cell = $(`[data-state-of="${c.id}"]`);
      if (cell) cell.innerHTML = stateChip(st);
      const btns = $(`[data-cam-btns="${c.id}"]`);
      const key = (st ? st.state : "idle") + (isLearning(st) ? "-learn" : "") + (zonesReady(c, st) ? "-z" : "");
      if (btns && btns.dataset.state !== key) { btns.innerHTML = camButtons(c, st); btns.dataset.state = key; }
    }));
    if (location.hash.includes("add=1")) { history.replaceState(null, "", "#/cameras"); addCameraModal(); }
  }

  function addCameraModal() {
    const m = modal(`
      <div class="modal-head"><h2>Add camera</h2><span class="spacer"></span><button class="btn ghost small" data-close>${ICON.cross}</button></div>
      <div class="tabs"><button class="on" data-tab="file">Recorded video</button><button data-tab="phone">Phone camera (Wi-Fi)</button><button data-tab="device">iPhone / Mac camera</button></div>
      <div class="grid2" style="margin-bottom:12px">
        <label class="field">Name<input id="c-name" placeholder="e.g. Banani Kakoli footbridge – north view" maxlength="80"></label>
        <label class="field">Location<input id="c-loc" placeholder="e.g. Kakoli, Banani, Dhaka" maxlength="120"></label>
      </div>
      <div data-pane="file" class="stack">
        <label class="field">Video file on this Mac<input id="c-path" placeholder="/Users/you/Desktop/traffic.mp4">
          <span class="hint">Fastest for big CCTV exports: the file is read in place. Tip: drag the file onto Terminal to copy its path.</span></label>
        <div class="muted small" style="text-align:center">or</div>
        <label class="field">Upload a video<input type="file" id="c-file" accept="video/*,.mp4,.mov,.mkv,.avi,.ts,.webm,.ogv,.mpg,.wmv,.3gp"></label>
        <label class="field">Recording start time (optional)<input type="datetime-local" step="1" id="c-start">
          <span class="hint">Leave empty to read it from the file name or video metadata. Needed for real times on notices and peak-hour parking rules.</span></label>
        <div id="up-prog" class="progress hidden"><div style="width:0"></div></div>
      </div>
      <div data-pane="phone" class="stack hidden">
        <div class="howto"><b>Android</b> – install <i>IP Webcam</i>, tap <i>Start server</i>, then enter the address it shows plus <code>/video</code>,
          e.g. <code>http://192.168.0.105:8080/video</code>. <i>DroidCam</i>: <code>http://PHONE-IP:4747/video</code>.<br>
          <b>iPhone</b> – an RTSP camera app (e.g. <i>IP Camera Lite</i>) gives an <code>rtsp://…</code> address, or use the <i>iPhone / Mac camera</i> tab (Continuity Camera).<br>
          Phone and Mac must be on the same Wi-Fi. Mount the phone high and steady, looking along the road.</div>
        <label class="field">Stream address<input id="c-url" placeholder="http://192.168.0.105:8080/video"></label>
        <div class="row"><button class="btn" id="test-url">Test connection</button><span class="small muted" id="test-msg"></span></div>
        <div class="preview hidden" id="url-prev"><img alt="Camera preview"></div>
      </div>
      <div data-pane="device" class="stack hidden">
        <div class="howto"><b>iPhone as a webcam (Continuity Camera)</b><ol><li>iPhone and Mac signed in to the same Apple ID, Wi-Fi and Bluetooth on.</li>
          <li>Place the iPhone in landscape on a stand near the Mac, screen off.</li><li>Click <i>Find cameras</i>; macOS may ask to allow camera access for Terminal / Python.</li></ol></div>
        <div class="row"><button class="btn" id="find-dev">Find cameras</button><span class="small muted" id="dev-msg"></span></div>
        <div id="dev-list" class="stack"></div>
      </div>
      <div class="field" style="margin-top:14px">Road zones
        <div class="modes" id="c-mode">
          <label class="mode on"><input type="radio" name="zmode" value="auto" checked>
            <b>Automatic</b><span>The AI watches 2–5 minutes of traffic and sets up the zones itself: carriageways and
            directions, kerb no-parking strips, junctions, bus stops, signal queues. Then it starts.</span></label>
          <label class="mode"><input type="radio" name="zmode" value="manual">
            <b>Manual</b><span>Draw the zones yourself in the zone editor and type the measured road sizes. Most exact for
            speed.</span></label>
        </div>
        <span class="hint">Either way you can change the zones later in the editor, or switch to the other method.</span></div>
      <div class="row" style="justify-content:flex-end;margin-top:16px"><span class="errors" id="c-err"></span><span class="spacer"></span>
        <button class="btn" data-close>Cancel</button><button class="btn primary" id="c-save">Add and learn zones</button></div>`, "wide");
    const el = m.el;
    let tab = "file", deviceSource = null, zoneMode = "auto";
    $$("input[name=zmode]", el).forEach((r) => r.onchange = () => {
      zoneMode = r.value;
      $$(".mode", el).forEach((x) => x.classList.toggle("on", x.contains(r)));
      $("#c-save", el).textContent = zoneMode === "auto" ? "Add and learn zones" : "Add and draw zones";
    });
    $$(".tabs button", el).forEach((b) => b.onclick = () => {
      tab = b.dataset.tab;
      $$(".tabs button", el).forEach((x) => x.classList.toggle("on", x === b));
      $$("[data-pane]", el).forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== tab));
    });
    const preview = async (source, imgBox, msg) => {
      msg.textContent = "Connecting (up to 10 s)...";
      try {
        const blob = await api("/api/test-source", { method: "POST", body: { name: "test", source } });
        imgBox.classList.remove("hidden");
        imgBox.querySelector("img").src = URL.createObjectURL(blob);
        msg.textContent = "Connected";
        return true;
      } catch (e) { msg.textContent = e.message; return false; }
    };
    $("#test-url", el).onclick = () => preview($("#c-url", el).value.trim(), $("#url-prev", el), $("#test-msg", el));
    $("#find-dev", el).onclick = async () => {
      $("#dev-msg", el).textContent = "Looking for cameras...";
      try {
        const devs = await api("/api/devices");
        $("#dev-msg", el).textContent = devs.length ? `${devs.length} found – pick the iPhone (usually not camera 0, the built-in one)` : "No cameras found";
        $("#dev-list", el).innerHTML = devs.map((d) => `<label class="row"><input type="radio" name="dev" value="${d.index}"> Camera ${d.index} <span class="muted small">${d.width}×${d.height}</span>
          <button class="btn small" data-prev="${d.index}">Preview</button></label><div class="preview hidden" data-prevbox="${d.index}"><img alt=""></div>`).join("");
        $$("input[name=dev]", el).forEach((r) => r.onchange = () => { deviceSource = r.value; });
        $$("[data-prev]", el).forEach((b) => b.onclick = (e) => { e.preventDefault(); preview(b.dataset.prev, $(`[data-prevbox="${b.dataset.prev}"]`, el), $("#dev-msg", el)); });
      } catch (e) { $("#dev-msg", el).textContent = e.message; }
    };
    $("#c-save", el).onclick = async () => {
      const name = $("#c-name", el).value.trim() || "Camera";
      const location = $("#c-loc", el).value.trim();
      const err = $("#c-err", el);
      err.textContent = "";
      const btn = $("#c-save", el);
      btn.disabled = true;
      try {
        let cam;
        if (tab === "file") {
          const file = $("#c-file", el).files[0];
          const start = $("#c-start", el).value || null;
          if (file) cam = await upload(file, name, location, start, $("#up-prog", el), zoneMode);
          else {
            const path = $("#c-path", el).value.trim().replace(/^['"]|['"]$/g, "");
            if (!path) throw new Error("Enter a file path or choose a file to upload");
            cam = await api("/api/cameras", { method: "POST", body: { name, location, source: path, start_time: start, zone_mode: zoneMode } });
          }
        } else if (tab === "phone") {
          const url = $("#c-url", el).value.trim();
          if (!url) throw new Error("Enter the stream address");
          cam = await api("/api/cameras", { method: "POST", body: { name, location, source: url, zone_mode: zoneMode } });
        } else {
          if (deviceSource == null) throw new Error("Find and select a camera first");
          cam = await api("/api/cameras", { method: "POST", body: { name, location, source: String(deviceSource), zone_mode: zoneMode } });
        }
        m.close();
        if (zoneMode === "auto") {
          toast(`Added <b>${esc(cam.name)}</b>. Learning the road layout from traffic…`, "ok");
          go(`#/camera/${cam.id}`);
        } else {
          toast(`Added <b>${esc(cam.name)}</b>. Now mark the road zones.`, "ok");
          go(`#/camera/${cam.id}/zones`);
        }
      } catch (e) { err.textContent = e.message; btn.disabled = false; }
    };
  }

  function editCameraModal(cam, onSaved) {
    const m = modal(`
      <div class="modal-head"><h2>Camera name &amp; location</h2><span class="spacer"></span><button class="btn ghost small" data-close>${ICON.cross}</button></div>
      <div class="stack">
        <label class="field">Camera name<input id="e-name" value="${esc(cam.name)}" maxlength="80" placeholder="e.g. Banani Kakoli footbridge – north view"></label>
        <label class="field">Position / location<input id="e-loc" value="${esc(cam.location || "")}" maxlength="120" placeholder="e.g. Kakoli, Banani, Dhaka">
          <span class="hint">Printed on violation notices and shown on every case from this camera.</span></label>
        <label class="field">${cam.kind === "file" ? "Video file" : "Stream address"}<input id="e-src" value="${esc(cam.source)}">
          <span class="hint">${cam.kind === "file" ? "Point the camera at a different recording of the same position." : "Change if the phone's IP address changed."} The camera must be stopped.</span></label>
        <div class="row" style="justify-content:flex-end"><span class="errors" id="e-err"></span><span class="spacer"></span>
          <button class="btn" data-close>Cancel</button><button class="btn primary" id="e-save">Save</button></div>
      </div>`);
    const el = m.el;
    setTimeout(() => $("#e-name", el).select(), 30);
    const save = async () => {
      try {
        const upd = await api(`/api/cameras/${cam.id}`, { method: "PATCH", body: { name: $("#e-name", el).value, location: $("#e-loc", el).value, source: $("#e-src", el).value } });
        m.close();
        toast(`Saved <b>${esc(upd.name)}</b>`, "ok", null, 2500);
        if (onSaved) onSaved(upd);
      } catch (e) { $("#e-err", el).textContent = e.message; }
    };
    $("#e-save", el).onclick = save;
    $$("input", el).forEach((i) => i.addEventListener("keydown", (e) => { if (e.key === "Enter") save(); }));
  }

  function lightbox(src, caption) {
    modal(`<div class="modal-head"><h2>${esc(caption)}</h2><span class="spacer"></span><a class="btn small" href="${src}" target="_blank" rel="noopener">Open full size</a>
      <button class="btn ghost small" data-close>${ICON.cross}</button></div><img class="lightbox-img" src="${src}" alt="${esc(caption)}">`, "wide");
  }
  document.addEventListener("click", (e) => {
    const im = e.target.closest("[data-zoom]");
    if (im) { e.preventDefault(); lightbox(im.dataset.zoom, im.dataset.caption || "Evidence"); }
  });

  // Form-data POST with upload progress, for anything that is not the single-camera form.
  function postForm(url, fd, onPct) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onPct) onPct(Math.round(100 * e.loaded / e.total));
      };
      xhr.onload = () => {
        if (xhr.status < 300) resolve(xhr.responseText ? JSON.parse(xhr.responseText) : {});
        else { try { reject(new Error(JSON.parse(xhr.responseText).detail)); } catch (e) { reject(new Error(xhr.statusText)); } }
      };
      xhr.onerror = () => reject(new Error("Upload failed"));
      xhr.send(fd);
    });
  }

  function upload(file, name, location, start, prog, zoneMode = "manual") {
    return new Promise((resolve, reject) => {
      const fd = new FormData();
      fd.append("name", name); fd.append("location", location); fd.append("start_time", start || ""); fd.append("zone_mode", zoneMode); fd.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/cameras/upload");
      prog.classList.remove("hidden");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) prog.firstElementChild.style.width = (100 * e.loaded / e.total).toFixed(1) + "%"; };
      xhr.onload = () => {
        if (xhr.status < 300) resolve(JSON.parse(xhr.responseText));
        else { try { reject(new Error(JSON.parse(xhr.responseText).detail)); } catch (e) { reject(new Error(xhr.statusText)); } }
      };
      xhr.onerror = () => reject(new Error("Upload failed"));
      xhr.send(fd);
    });
  }

  // ------------------------------------------------------------------ automatic zone report
  function autoReport(rep, compact = false) {
    if (!rep) return "";
    const c = rep.calibration || {};
    const unc = c.speed_uncertainty_pct;
    const flag60 = unc ? Math.round(60 * (1 + Math.max(5, 1.5 * unc) / 100)) : null;
    const cws = (rep.carriageways || []).map((cw) => `<li>${esc({ toward_camera: "Toward the camera", away_from_camera: "Away from the camera", both: "Two-way" }[cw.direction] || cw.direction)}
      – ${cw.width_m} m wide, ${cw.lanes || "?"} lane${cw.lanes === 1 ? "" : "s"}, ${cw.vehicles} vehicles (${Math.round(cw.one_way_share * 100)}% one way)</li>`).join("");
    const f = rep.found || {};
    const found = [["intersections", "junction"], ["bus_stops", "bus stop"], ["signal_queues", "signal queue"], ["uturns", "U-turn opening"]]
      .map(([k, l]) => `${f[k] || 0} ${l}${(f[k] || 0) === 1 ? "" : "s"}`).join(" · ");
    return `<div class="auto-report">
      <div><b>Camera calibration</b> <span class="muted small">(${esc(c.lens || "")})</span><br>
        Field of view ${c.hfov_deg}°, camera ${c.camera_height_m} m above the road, tilted ${c.tilt_deg}° down, roll ${c.roll_deg}°.<br>
        Speeds accurate to about <b>±${unc}%</b>${flag60 ? ` – so in a 60 km/h zone only vehicles above <b>${flag60} km/h</b> are reported` : ""}.
        ${c.warning ? `<div class="warnline">${esc(c.warning)}</div>` : ""}</div>
      ${compact ? "" : `<div><b>Carriageways</b><ul>${cws}</ul></div>`}
      <div><b>Found from traffic:</b> ${found}</div>
      ${(rep.notes || []).map((n) => `<div class="warnline">${esc(n)}</div>`).join("")}
      <div class="muted small">Learned from ${rep.moving_vehicles} moving vehicles in ${Math.round(rep.observed_s)} s of video. Legal facts – banned vehicle classes, legal parking bays, lower local speed limits – cannot be seen: set them in the zone editor.</div>
    </div>`;
  }

  function zonesCard(cam, st) {
    const L = st && st.learning;
    const learn = isLearning(st);
    let body;
    if (learn) {
      body = `<div class="small">${esc(L.message || "Learning…")}</div>
        <div class="progress learn" style="margin:8px 0"><div style="width:${((L.progress || 0) * 100).toFixed(1)}%"></div></div>
        <div class="small muted">${L.moving_vehicles || 0} moving vehicles seen in ${Math.round(L.seconds || 0)} s (needs about 300 and 90 s).</div>
        <button class="btn small" data-zact="cancel">${ICON.stop}Cancel</button>`;
    } else {
      const err = L && L.state === "error" ? `<div class="errors">Automatic setup failed: ${esc(L.message)}</div>` : "";
      const mode = cam.zones_ready ? `<span class="chip">${cam.zone_mode === "auto" ? "Automatic" : "Manual"}</span> version ${cam.zone_version}` : "Not set up yet";
      body = `<div class="row small">${mode}</div>${err}
        ${cam.zone_mode === "auto" && cam.auto_preview ? `<img class="zone-preview" src="${media(cam.auto_preview)}" data-zoom="${media(cam.auto_preview)}" data-caption="Automatically learned zones" alt="Learned zones">` : ""}
        ${cam.zone_mode === "auto" ? autoReport(cam.auto_report, true) : ""}
        <div class="row"><button class="btn small ${cam.zones_ready ? "" : "primary"}" data-zact="learn">${ICON.auto}${cam.zones_ready ? "Learn again automatically" : "Learn zones automatically"}</button>
          <a class="btn small" href="#/camera/${cam.id}/zones">${ICON.zones}${cam.zones_ready ? "Open zone editor" : "Draw zones manually"}</a></div>`;
    }
    return `<div class="card-head"><h2>Road zones</h2></div><div class="stack" style="gap:10px">${body}</div>`;
  }

  // ------------------------------------------------------------------ camera page
  async function pageCamera(main, id) {
    const cam = await api(`/api/cameras/${id}`);
    const recent = await api(`/api/violations?camera_id=${id}&limit=20`);
    const s = cam.settings;
    const ruleNames = Object.fromEntries(Object.values(state.meta.rules).map((r) => [r.code, r.title]));
    main.innerHTML = `
      <div class="page-head"><div><h1 class="row" style="gap:6px">${esc(cam.name)}<button class="btn ghost small" id="edit-cam" title="Change the camera's name and location">${ICON.edit}Edit name &amp; location</button></h1>
        <p>${ICON.pin.replace("<svg", '<svg class="i"')}${esc(cam.location || "No location set")} ${cam.kind === "file" ? "· recorded video" : "· live camera"}</p></div>
        <span class="spacer"></span><span class="row" data-cam-btns="${id}" data-state="${esc(cam.status.state)}">${camButtons(cam, cam.status)}</span>
        <button class="btn danger" id="clear">${ICON.trash}Clear results</button></div>
      <div class="cam-page">
        <div class="stack">
          <div class="card flush"><div>${camView(cam, cam.status, true)}</div>
            <div class="cam-meta"><span class="stats" data-cam-stats="${id}">${esc(camStatsText(cam, cam.status))}</span></div></div>
          <div class="card"><div class="card-head"><h2>Violations from this camera</h2><span class="spacer"></span><a class="small" href="#/review?camera=${id}">Review</a></div>
            <div class="feed" id="feed" style="max-height:none">${recent.items.length ? recent.items.map((v) => feedItem(v)).join("") : '<div class="empty">None yet</div>'}</div></div>
        </div>
        <div class="stack">
          <div class="card" id="zones-card"></div>
          <div class="card"><div class="card-head"><h2>Status</h2></div><table class="kv" id="kv"></table></div>
          <div class="card"><div class="card-head"><h2>Rules</h2></div><div id="rules" class="small"></div></div>
          <div class="card"><div class="card-head"><h2>Settings</h2></div>
            <div class="stack">
              <label class="field">Detector input size
                <span class="seg" id="imgsz">${[640, 960].map((v) => `<button data-v="${v}" class="${s.imgsz === v ? "on" : ""}">${v} px${v === 640 ? " · fast" : " · accurate"}</button>`).join("")}</span>
                <span class="hint">960 px sees smaller, farther vehicles and measures more of each zone reliably (~30 FPS on an M5). 640 px runs ~60 FPS: best for live cameras.</span></label>
              <label class="field">Analysis resolution<select id="maxw">${[[0, "Full resolution (needed for helmets and plates)"], [1920, "1920 px (faster on 4K video)"], [1280, "1280 px (fastest)"]].map(([v, l]) => `<option value="${v}" ${s.max_width === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>
              <label class="row small"><input type="checkbox" id="anpr" ${s.anpr ? "checked" : ""} ${state.meta.features.anpr ? "" : "disabled"}> Read number plates of violators${state.meta.features.anpr ? "" : " (plate models not installed)"}</label>
              <label class="row small"><input type="checkbox" id="helmet" ${s.helmet ? "checked" : ""} ${state.meta.features.helmet ? "" : "disabled"}> Detect riders without helmet (V6)${state.meta.features.helmet ? "" : " (helmet model not installed)"}</label>
              ${cam.kind === "file" ? `
              <label class="field">Recording start time<input type="datetime-local" step="1" id="start" value="${esc(cam.start_time || "")}">
                <span class="hint">${cam.start_time ? "Found from: " + esc(cam.start_time_source) + ". Check it against the clock shown in the video." : "Unknown: notices will show video time instead of real time."}</span></label>
              <label class="field">Frames analysed per second<select id="afps">${[15, 20, 25, 30].map((v) => `<option ${s.analysis_fps === v ? "selected" : ""}>${v}</option>`).join("")}</select>
                <span class="hint">30 recommended. Lower is faster; below 15 speeds become less precise.</span></label>
              <label class="row small"><input type="checkbox" id="realtime" ${s.realtime ? "checked" : ""}> Play at real speed (behave like a live camera)</label>` : ""}
              <div class="row"><button class="btn primary small" id="save-set">Save settings</button><span class="small muted" id="set-msg"></span></div>
            </div></div>
          <div class="card"><div class="card-head"><h2>Analysis runs</h2></div>${cam.runs.length ? `<table class="list small"><tr><th>Started</th><th>Mode</th><th>Frames</th><th>Result</th></tr>
            ${cam.runs.map((r) => `<tr><td>${esc(r.started_at.replace("T", " "))}</td><td>${esc(r.mode)}</td><td class="mono">${r.frames}</td>
              <td>${esc(r.status)} · <a href="#/review?run=${r.id}">${r.violations}</a></td></tr>`).join("")}</table>` : '<div class="muted small">Not run yet</div>'}</div>
        </div>
      </div>`;
    bindCamButtons(main);
    $("#edit-cam").onclick = () => editCameraModal(cam, () => route());
    let imgsz = s.imgsz;
    $$("#imgsz button").forEach((b) => b.onclick = () => { imgsz = +b.dataset.v; $$("#imgsz button").forEach((x) => x.classList.toggle("on", x === b)); });
    $("#save-set").onclick = async () => {
      const body = { settings: { imgsz, max_width: +$("#maxw").value, anpr: $("#anpr").checked, helmet: $("#helmet").checked } };
      if (cam.kind === "file") {
        body.settings.analysis_fps = +$("#afps").value;
        body.settings.realtime = $("#realtime").checked;
        body.start_time = $("#start").value || "";
      }
      try { await api(`/api/cameras/${id}`, { method: "PATCH", body }); $("#set-msg").textContent = "Saved – used from the next start"; }
      catch (e) { $("#set-msg").textContent = e.message; }
    };
    $("#clear").onclick = async () => {
      if (await confirmBox(`Delete all ${cam.total} violations, runs and evidence files of <b>${esc(cam.name)}</b>? Zones are kept.`, "Clear")) {
        try { await api(`/api/cameras/${id}/violations`, { method: "DELETE" }); route(); } catch (e) { fail(e); }
      }
    };
    $("#feed").addEventListener("click", (e) => { const it = e.target.closest("[data-vid]"); if (it) go(`#/review/${it.dataset.vid}`); });
    const renderStatus = (st) => {
      const rows = [["State", STATE_LABEL[st.state] || st.state], ["Detector", st.backend || "–"],
        ["Processing", st.fps ? `${st.fps} FPS${st.speed_x ? ` (${st.speed_x}× real time)` : ""}` : "–"],
        [cam.kind === "live" ? "Delay" : "Video time", cam.kind === "live" ? (st.lag != null ? st.lag + " s" : "–") : (st.video_time != null ? st.video_time + " s" : "–")],
        ["Vehicles in view", st.tracks ?? "–"], ["Violations this run", st.violations ?? "–"]];
      if (st.message) rows.push(["Message", st.message]);
      $("#kv").innerHTML = rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
      const inactive = st.state === "idle" || !st.inactive ? (cam.rules || {}) : st.inactive;
      const codes = Object.keys(ruleNames).sort();
      $("#rules").innerHTML = `<ul class="reasons">${codes.map((c) => inactive[c]
          ? `<li class="off">${vcode(c)} ${esc(ruleNames[c])}<div class="small muted">off: ${esc(inactive[c])}</div></li>`
          : `<li>${vcode(c)} ${esc(ruleNames[c])}${c === "V2" && st.speed_paused ? ' <b style="color:var(--warning-ink)">paused: frame rate too low</b>' : ""}</li>`).join("")}</ul>
        <a class="small" href="#/camera/${id}/zones">Change zones to switch rules on</a>`;
    };
    renderStatus(cam.status);
    let lastLearn = JSON.stringify(cam.status.learning || {});
    const renderZones = (st) => { $("#zones-card").innerHTML = zonesCard(cam, st); };
    renderZones(cam.status);
    $("#zones-card").addEventListener("click", async (e) => {
      const b = e.target.closest("[data-zact]");
      if (!b) return;
      if (b.dataset.zact === "learn") {
        if (cam.zones_ready && !(await confirmBox(`Learn the zones of <b>${esc(cam.name)}</b> again from traffic? The result is saved as a new zone version (the current one stays in the history).`, "Learn again"))) return;
        try {
          await api(`/api/cameras/${id}/auto-zones`, { method: "POST", body: { apply: true, start: cam.kind === "live" } });
          toast("Learning the road layout from traffic…", "ok", null, 3000);
        } catch (err) { fail(err); }
      }
      if (b.dataset.zact === "cancel") { try { await api(`/api/cameras/${id}/auto-zones`, { method: "DELETE" }); } catch (err) { fail(err); } }
    });
    pollStatus((status) => {
      const st = status[id] || { state: "idle" };
      updateCamViews([cam], status);
      renderStatus(st);
      const key = JSON.stringify(st.learning || {});
      if (key !== lastLearn) {
        const wasLearning = (JSON.parse(lastLearn).state || "") && LEARNING.includes(JSON.parse(lastLearn).state);
        lastLearn = key;
        if (wasLearning && st.learning && st.learning.state === "done") { route(); return; }  // new zones: reload page
        renderZones(st);
      }
    });
    listeners.add((msg) => {
      if (msg.type === "violation" && msg.camera_id === id) {
        const feed = $("#feed");
        if (!feed.querySelector("[data-vid]")) feed.innerHTML = "";
        feed.insertAdjacentHTML("afterbegin", feedItem(msg.violation, true));
      }
    });
  }

  // ------------------------------------------------------------------ review queue
  async function pageReview(main, selectId) {
    const q = new URLSearchParams(location.hash.split("?")[1] || "");
    const cams = await api("/api/cameras");
    const f = { camera_id: q.get("camera") || "", code: q.get("code") || "", status: q.get("run") || q.get("code") || q.get("plate") || selectId ? "" : "pending", run_id: q.get("run") || "", plate: q.get("plate") || "" };
    const codes = Object.values(state.meta.rules).map((r) => [r.code, r.title]).sort();
    main.innerHTML = `
      <div class="page-head"><div><h1>Review</h1><p>Every detection is checked by a person before it becomes a notice. Keys: <kbd>C</kbd> confirm · <kbd>R</kbd> reject · <kbd>↑</kbd><kbd>↓</kbd> move</p></div>
        <span class="spacer"></span><a class="btn" id="export" href="/api/export.csv">${ICON.download}Export CSV</a></div>
      <div class="row" style="margin-bottom:12px">
        <select id="f-cam"><option value="">All cameras</option>${cams.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("")}</select>
        <select id="f-code"><option value="">All violation types</option>${codes.map(([c, t]) => `<option value="${c}">${c} ${esc(t)}</option>`).join("")}</select>
        <span class="seg" id="f-status">${["pending", "confirmed", "rejected", ""].map((s) => `<button data-v="${s}">${s ? s[0].toUpperCase() + s.slice(1) : "All"}</button>`).join("")}</span>
        <input id="f-plate" type="search" placeholder="Search plate (Bangla or e.g. 12-3456)" style="width:250px">
        <span class="muted small" id="f-count"></span>
      </div>
      <div class="review"><div class="rv-list" id="list"></div><div id="detail"></div></div>`;
    $("#f-cam").value = f.camera_id;
    $("#f-code").value = f.code;
    $("#f-plate").value = f.plate;
    let items = [], sel = null;
    const setSeg = () => $$("#f-status button").forEach((b) => b.classList.toggle("on", b.dataset.v === f.status));
    setSeg();

    async function load(keepId) {
      const p = new URLSearchParams({ limit: 300 });
      for (const k of ["camera_id", "code", "status", "run_id", "plate"]) if (f[k]) p.set(k, f[k]);
      const r = await api("/api/violations?" + p);
      items = r.items;
      $("#f-count").textContent = `${r.total} case${r.total === 1 ? "" : "s"}`;
      $("#export").href = "/api/export.csv?" + new URLSearchParams(Object.fromEntries(Object.entries({ camera_id: f.camera_id, status: f.status }).filter(([, v]) => v)));
      $("#list").innerHTML = items.length ? items.map((v) => rvItem(v)).join("")
        : `<div class="card empty">Nothing ${f.status ? esc(f.status) : ""} here.</div>`;
      let target = items.find((v) => v.id === keepId) || items[0];
      if (keepId && !items.find((v) => v.id === keepId)) {
        try { const v = await api(`/api/violations/${keepId}`); items.unshift(v); target = v; } catch (e) { /* deleted */ }
      }
      select(target ? target.id : null);
    }

    function rvItem(v) {
      return `<div class="rv-item" data-vid="${v.id}"><img src="${media(v.crop || v.snapshot)}" alt="" loading="lazy">
        <div><div class="small">${vcode(v.code)} <b>${esc(v.cls)}</b> ${statusBadge(v.status)}</div>
        <div class="small muted">${esc(summary(v))}</div><div class="small muted">${esc(v.camera_name)} · ${esc(when(v))}</div>
        <div class="small" data-plate-of="${v.id}">${plateChip(v)}</div></div></div>`;
    }

    function plateBlock(v) {
      const st = v.plate_status || "";
      if (!st) return "";
      const img = v.plate_img ? `<img class="plate-img" src="${media(v.plate_img)}" alt="Number plate">` : "";
      let body;
      if (st === "pending") body = `<div class="muted">Reading the plate from the best frames of this vehicle…</div>`;
      else if (st === "unreadable") body = `<div><b>Not readable in the video</b></div><div class="small muted">The plate was too small or blurred to read reliably${v.plate_img ? " – best view shown" : ""}. Identify the vehicle from the images.</div>`;
      else body = `<div class="plate-text">${esc(v.plate_text)}</div><div class="small muted">${esc(v.plate_text_en)} · confidence ${Math.round((v.plate_conf || 0) * 100)}%</div>
        <div class="small muted">The six-digit number is read automatically (checked on several frames).
        ${st === "partial" ? "City and class letter could not be read" : "City and class letter are a suggestion"} – confirm them on the plate image.</div>`;
      return `<div class="plate-box">${img}<div>${body}</div></div>`;
    }

    // The vision-language model's reading of the case: what happened, why, and whether the photos back the
    // rule up. "Insufficient evidence" is a real answer and is shown as such, never as agreement.
    const VERDICT = {
      supported: { label: "Evidence supports the finding", cls: "ok" },
      insufficient: { label: "Not enough evidence to judge", cls: "warn" },
      contradicted: { label: "Evidence contradicts the finding", cls: "bad" },
    };

    function vlmBlock(v) {
      const st = v.vlm_status || "";
      if (st === "pending") {
        return `<div class="vlm-box pending"><div class="vlm-head">${ICON.brain}<b>Reading the evidence…</b></div>
          <div class="small muted">The model is looking at the photographs. On the local CPU model this takes
          roughly 20–60 seconds; the answer appears here by itself.</div></div>`;
      }
      const x = v.vlm;
      if (!x) return "";
      if (!x.ok) {
        return `<div class="vlm-box bad"><div class="vlm-head">${ICON.brain}<b>Could not explain this case</b></div>
          <div class="small muted">${esc(x.error || "unknown error")}</div></div>`;
      }
      const verdict = VERDICT[x.assessment] || VERDICT.insufficient;
      const part = (label, text) => text ? `<div class="vlm-part"><span>${label}</span><p>${esc(text)}</p></div>` : "";
      return `<div class="vlm-box ${verdict.cls}">
        <div class="vlm-head">${ICON.brain}<b>${esc(verdict.label)}</b>
          <span class="vlm-conf">confidence ${Math.round((x.confidence || 0) * 100)}%</span>
          <span class="spacer"></span>
          <button class="btn small" id="vlm-again" title="Ask again">${ICON.auto}Again</button></div>
        ${x.downgraded ? `<div class="vlm-overruled">${esc(x.downgraded)}</div>` : ""}
        <div class="vlm-summary">${esc(x.summary || "")}</div>
        ${part("In the photos", x.visible)}
        ${x.other_cameras && !/^(nothing|not applicable|n\/a)\.?$/i.test(x.other_cameras.trim())
            ? part("Other cameras", x.other_cameras) : ""}
        ${part("Before", x.pre_event)}${part("The event", x.event)}
        ${part("After", x.aftermath)}${part("Probable cause", x.cause)}
        <div class="small muted vlm-foot">${esc(x.model || "model")} · ${esc(x.backend || "")} ·
          ${esc(String(x.seconds || 0))} s · ${esc(String(x.images || 0))} photo(s).
          Written by a language model from the photographs and the measurements above. Check it against the
          evidence before relying on it; it is not itself evidence.</div></div>`;
    }

    function select(id) {
      sel = items.find((v) => v.id === id) || null;
      $$(".rv-item").forEach((el) => el.classList.toggle("sel", +el.dataset.vid === id));
      const cur = $(`.rv-item[data-vid="${id}"]`);
      if (cur) cur.scrollIntoView({ block: "nearest" });
      if (id) history.replaceState(null, "", `#/review/${id}`);
      renderDetail();
    }

    function evidenceItems(v) {
      const at = (e) => e.wall_time || `video ${(+e.time_s).toFixed(1)} s`;
      const out = [];
      (v.evidence || []).forEach((e) => out.push({ type: "img", src: media(e.path), label: e.label === "stopped" ? "Stopped" : e.label, sub: at(e) }));
      out.push({ type: "img", src: media(v.snapshot), label: "Scene", sub: (v.code === "V6" ? "best view · " : "reported · ") + when(v) });
      if (v.crop) out.push({ type: "img", src: media(v.crop), label: "Close-up", sub: "full resolution" });
      if (v.clip) out.push({ type: "video", src: media(v.clip), label: "Video clip", sub: "10 s around the event" });
      if (v.plate_img) out.push({ type: "img", src: media(v.plate_img), label: "Plate", sub: "best plate view" });
      return out;
    }

    function timeline(v) {
      const d = v.details || {};
      if (!["V1", "V3", "V7"].includes(v.code) || !d.started) return "";
      const stopped = (v.evidence || []).find((e) => e.label === "stopped");
      return `<div class="timeline">
        <div><span class="tl-dot"></span><b>Stopped</b><span class="muted small">${esc(d.started)}</span></div>
        ${stopped ? `<div><span class="tl-dot"></span><b>Photo 1</b><span class="muted small">${esc(stopped.wall_time || "video " + stopped.time_s + " s")}</span></div>` : ""}
        <div><span class="tl-dot"></span><b>${esc(d.vehicles_passed)} vehicles passed it</b><span class="muted small">traffic moving ${esc(d.traffic_moving_s)} s · queued ${esc(d.queued_s)} s</span></div>
        <div><span class="tl-dot hot"></span><b>Reported after ${Math.round(d.dwell_s)} s</b><span class="muted small">${esc(when(v))}</span></div></div>`;
    }

    function renderDetail() {
      const d = $("#detail");
      if (!sel) { d.innerHTML = `<div class="card empty">${ICON.check.replace("<svg", '<svg width="34" height="34"')}<br>All caught up.</div>`; return; }
      const v = sel;
      const ev = evidenceItems(v);
      const main = Math.max(0, ev.findIndex((x) => x.label === "Scene"));
      const rows = Object.entries(v.details).filter(([k]) => k !== "started").map(([k, val]) => [k.replace(/_/g, " "), val]);
      d.innerHTML = `<div class="card stack">
        <div class="row">${vcode(v.code)}<h2>${esc(v.title)}</h2>${statusBadge(v.status)}<span class="spacer"></span>
          <span class="small muted">Case TV-${v.id}</span>
          ${!v.vlm && v.vlm_status !== "pending"
            ? `<button class="btn small" id="explain">${ICON.brain}Explain</button>` : ""}
          <a class="btn small" href="/notice/${v.id}" target="_blank" rel="noopener">${ICON.print}Notice</a></div>
        <div class="small secondary">${esc(summary(v))}</div>
        <div class="evidence">
          <div class="ev-main" id="ev-main"></div>
          <div class="ev-thumbs">${ev.map((x, i) => `<button class="ev-thumb" data-ev="${i}">
            ${x.type === "video" ? `<span class="ev-vid">▶</span>` : `<img src="${x.src}" alt="" loading="lazy">`}
            <span class="ev-label">${esc(x.label)}</span></button>`).join("")}</div>
          <div class="small muted" id="clip-note"></div>
        </div>
        ${timeline(v)}
        <div id="vlm-block">${vlmBlock(v)}</div>
        <div id="plate-block">${plateBlock(v)}</div>
        <div class="grid2"><table class="kv">
            <tr><th>Vehicle</th><td>${esc(v.cls)} #${v.track_id} <span class="muted">(class ${Math.round(v.class_conf * 100)}% consistent)</span></td></tr>
            <tr><th>Time</th><td>${esc(when(v))}</td></tr><tr><th>Camera</th><td>${esc(v.camera_name)}</td></tr>
            <tr><th>Position</th><td>${esc(v.camera_location || "–")}</td></tr>
            <tr><th>Zone</th><td>${esc(v.zone)}${v.areas ? ` · ${esc(v.areas.replace(/_/g, " "))}` : ""}</td></tr></table>
          <table class="kv">${rows.map(([k, val]) => `<tr><th>${esc(k)}</th><td>${esc(val)}</td></tr>`).join("")}</table></div>
        <label class="field">Reviewer note<input id="note" value="${esc(v.note || "")}" placeholder="optional, e.g. plate number or reason for rejecting"></label>
        <div class="rv-actions"><button class="btn good" id="ok">${ICON.check}Confirm <kbd>C</kbd></button>
          <button class="btn bad" id="no">${ICON.cross}Reject <kbd>R</kbd></button>
          ${v.status !== "pending" ? '<button class="btn" id="undo">Back to pending</button>' : ""}
          <span class="spacer"></span><span class="small muted">Rejected cases are kept out of statistics and show how accurate each rule is.</span></div></div>`;
      const show = (i) => {
        const x = ev[i];
        $$(".ev-thumb", d).forEach((b) => b.classList.toggle("on", +b.dataset.ev === i));
        $("#ev-main").innerHTML = x.type === "video"
          ? `<video src="${x.src}" controls muted autoplay loop playsinline preload="auto"></video>`
          : `<img src="${x.src}" alt="${esc(x.label)}" data-zoom="${x.src}" data-caption="${esc(x.label + " · " + x.sub)}" title="Click to enlarge">`;
        $("#ev-main").insertAdjacentHTML("beforeend", `<div class="ev-caption"><b>${esc(x.label)}</b> · ${esc(x.sub)}${x.type === "img" ? ` <span class="muted">· click to enlarge</span>` : ""}</div>`);
        const vid = $("video", $("#ev-main"));
        $("#clip-note").textContent = "";
        if (vid) vid.onerror = () => { $("#clip-note").textContent = "Clip is still being prepared (a few seconds after the event) or unavailable."; };
      };
      $$(".ev-thumb", d).forEach((b) => b.onclick = () => show(+b.dataset.ev));
      show(main);
      $("#ok").onclick = () => decide("confirmed");
      $("#no").onclick = () => decide("rejected");
      if ($("#undo")) $("#undo").onclick = () => decide("pending");
      if ($("#explain")) $("#explain").onclick = () => explain(false);
      if ($("#vlm-again")) $("#vlm-again").onclick = () => explain(true);
    }

    async function explain(force) {
      if (!sel) return;
      const id = sel.id;
      try {
        const r = await api(`/api/violations/${id}/explain`, { method: "POST", body: { force } });
        patch(r.violation);
      } catch (e) { fail(e); }
    }

    // Merge a server copy of a case into the list and the open detail pane, without losing the reviewer's
    // unsaved note or overwriting a review decision made while the model was thinking.
    function patch(v) {
      const i = items.findIndex((x) => x.id === v.id);
      if (i < 0) return;
      items[i] = { ...items[i], ...v, status: items[i].status, note: items[i].note };
      if (sel && sel.id === v.id) {
        sel = items[i];
        const box = $("#vlm-block");
        if (box) box.innerHTML = vlmBlock(sel);
        const btn = $("#explain");
        if (btn && (sel.vlm || sel.vlm_status === "pending")) btn.remove();
        if ($("#vlm-again")) $("#vlm-again").onclick = () => explain(true);
      }
    }

    async function decide(status) {
      if (!sel) return;
      try {
        const v = await api(`/api/violations/${sel.id}`, { method: "PATCH", body: { status, note: $("#note").value } });
        const idx = items.findIndex((x) => x.id === sel.id);
        if (sel.status === "pending" && status !== "pending") { state.pending = Math.max(0, state.pending - 1); renderPendingCount(); }
        if (f.status && f.status !== status) {
          items.splice(idx, 1);
          $(`.rv-item[data-vid="${v.id}"]`).remove();
          const next = items[idx] || items[idx - 1];
          select(next ? next.id : null);
        } else {
          items[idx] = v;
          const el = $(`.rv-item[data-vid="${v.id}"] .status-badge`);
          if (el) el.outerHTML = statusBadge(v.status);
          const next = items[idx + 1];
          select(next ? next.id : v.id);
        }
      } catch (e) { fail(e); }
    }

    $("#list").addEventListener("click", (e) => { const it = e.target.closest("[data-vid]"); if (it) select(+it.dataset.vid); });
    $("#f-cam").onchange = () => { f.camera_id = $("#f-cam").value; f.run_id = ""; load(); };
    $("#f-code").onchange = () => { f.code = $("#f-code").value; load(); };
    let pt = null;
    $("#f-plate").oninput = () => { clearTimeout(pt); pt = setTimeout(() => { f.plate = $("#f-plate").value.trim(); if (f.plate) f.status = ""; setSeg(); load(); }, 300); };
    $$("#f-status button").forEach((b) => b.onclick = () => { f.status = b.dataset.v; setSeg(); load(); });
    const onKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey) return;
      const idx = items.findIndex((x) => sel && x.id === sel.id);
      if (e.key === "c" || e.key === "C") decide("confirmed");
      else if (e.key === "r" || e.key === "R") decide("rejected");
      else if (e.key === "ArrowDown" || e.key === "j") { if (items[idx + 1]) select(items[idx + 1].id); e.preventDefault(); }
      else if (e.key === "ArrowUp" || e.key === "k") { if (items[idx - 1]) select(items[idx - 1].id); e.preventDefault(); }
    };
    document.addEventListener("keydown", onKey);
    onLeave(() => document.removeEventListener("keydown", onKey));
    listeners.add((msg) => {
      if (msg.type === "plate") {
        const v = msg.violation, i = items.findIndex((x) => x.id === v.id);
        if (i < 0) return;
        items[i] = { ...items[i], ...v, status: items[i].status, note: items[i].note };
        const chip = $(`[data-plate-of="${v.id}"]`);
        if (chip) chip.innerHTML = plateChip(v);
        if (sel && sel.id === v.id) { sel = items[i]; $("#plate-block").innerHTML = plateBlock(sel); }
      }
      if (msg.type === "explanation") patch(msg.violation);
    });
    listeners.add((msg) => {
      if (msg.type === "violation" && (!f.status || f.status === "pending") && (!f.camera_id || +f.camera_id === msg.camera_id)) {
        const v = msg.violation;
        items.unshift(v);
        $("#list").querySelector(".empty")?.remove();
        $("#list").insertAdjacentHTML("afterbegin", rvItem(v));
        if (!sel) select(v.id);
      }
    });
    await load(selectId);
  }

  // ------------------------------------------------------------------ statistics
  // ---------------------------------------------------------------- vehicles across cameras
  async function pageVehicles(root, selected) {
    const fmtTime = (t) => t ? new Date(t * 1000).toLocaleString() : "-";
    root.innerHTML = `<div class="row page-head"><h1>Vehicles across cameras</h1><span class="spacer"></span>
        <span class="small muted" id="v-routes"></span>
        <button class="btn" id="v-add">${ICON.plus}Add camera views</button>
        <button class="btn primary" id="v-link">${ICON.auto}Match vehicles across cameras</button></div>
      <div class="card" id="v-upload" hidden>
        <div class="card-head"><h2>Several views of one place</h2><span class="spacer"></span>
          <button class="btn small" id="v-cancel">${ICON.cross}Close</button></div>
        <div class="small secondary">Choose every video of the same junction or stretch of road. Each becomes
          its own camera, they are analysed one after another, and the vehicles are matched across them at the
          end. Recording times are read from the file names where possible - matching is most accurate when
          the clips overlap in real time.</div>
        <label class="field">Place<input id="v-site" placeholder="e.g. Kakoli footbridge"></label>
        <label class="field">Location note<input id="v-loc" placeholder="optional, e.g. Banani"></label>
        <label class="field">Videos
          <input id="v-files" type="file" multiple accept="video/*,.mp4,.mov,.mkv,.avi,.ts,.webm"></label>
        <div id="v-chosen" class="small muted"></div>
        <div class="row"><button class="btn primary" id="v-go">${ICON.play}Upload and analyse</button>
          <span class="small muted" id="v-prog"></span></div>
      </div>
      <div class="card" id="v-site-status" hidden></div>
      <div class="card" id="v-status" hidden></div>
      <div class="card"><div class="small secondary">A vehicle is recognised on a second camera from where
        and when it crossed, how fast it was going, how wide it is, its class, and the colour of its body -
        or of the rider's clothing on a two-wheeler. Matching happens in metres on the road, so it is the
        same test at every camera. Where two vehicles cannot be told apart, nothing is linked.</div></div>
      <div id="v-body"></div>`;

    async function refreshStatus() {
      const st = await api("/api/link");
      $("#v-routes").textContent = st.routes_known ? `${st.routes_known} route(s) learned` : "";
      const box = $("#v-status");
      if (st.state === "running") {
        box.hidden = false;
        box.innerHTML = `<div class="row"><b>Matching…</b><span class="muted small">${esc(st.message || "")}</span></div>
          <pre class="small muted vlog">${esc((st.log || []).slice(-8).join("\n"))}</pre>`;
        setTimeout(refreshStatus, 1200);
      } else if (st.result) {
        const r = st.result;
        box.hidden = false;
        box.innerHTML = `<div class="small"><b>${esc(st.message || "")}</b></div>
          <div class="small muted">${r.tracklets} vehicle passes · ${r.linked} links · ${r.multi_camera || 0} seen by more than one camera${st.seconds ? ` · ${st.seconds}s` : ""}</div>
          ${(r.flows || []).map((f) => `<div class="small muted">route ${esc(f.name)}: camera ${f.from_camera} (${esc(f.from_direction)}) → camera ${f.to_camera} (${esc(f.to_direction)}), ${f.gap_m} m apart${f.refused && f.refused.length ? ` · ${esc(f.refused.join(", "))} too alike to link` : ""}</div>`).join("")}`;
      } else { box.hidden = true; }
    }

    async function load() {
      const r = await api("/api/vehicles?min_cameras=2&limit=100");
      const body = $("#v-body");
      if (!r.items.length) {
        body.innerHTML = `<div class="card empty">No vehicle has been matched across cameras yet.<br>
          <span class="small muted">${r.tracklets} vehicle pass(es) recorded. Analyse at least two cameras,
          then press <b>Match vehicles across cameras</b>.</span></div>`;
        return;
      }
      body.innerHTML = `<div class="card"><table class="kv wide"><tr><th>Vehicle</th><th>Class</th>
        <th>Cameras</th><th>Passes</th><th>First seen</th><th>Last seen</th></tr>
        ${r.items.map((v) => `<tr class="vrow" data-gid="${v.global_id}"><td><b>V-${v.global_id}</b></td>
          <td>${esc((v.classes || "").split(",")[0])}</td><td>${v.cameras}</td><td>${v.passes}</td>
          <td class="small">${esc(fmtTime(v.first_seen))}</td><td class="small">${esc(fmtTime(v.last_seen))}</td></tr>`).join("")}
        </table></div><div id="v-journey"></div>`;
      $$(".vrow").forEach((el) => el.onclick = () => go(`#/vehicles/${el.dataset.gid}`));
      if (selected) showJourney(selected);
    }

    async function showJourney(gid) {
      const j = await api(`/api/vehicles/${gid}`);
      $("#v-journey").innerHTML = `<div class="card"><div class="card-head"><h2>Vehicle V-${gid}</h2>
        <span class="spacer"></span><span class="small muted">${j.passes.length} camera(s), in order</span></div>
        <div class="timeline">${j.passes.map((p, i) => `<div><span class="tl-dot${i ? "" : " hot"}"></span>
          <b>${esc(p.camera_name)}</b>
          <span class="muted small">${esc(p.camera_location || "")} · ${esc(fmtTime(p.t_cross))} ·
          ${Math.round((p.speed || 0) * 3.6)} km/h · ${esc(p.cls || "")}${p.seconds_from_previous !== undefined
            ? ` · ${p.seconds_from_previous}s after the previous camera` : ""}
          ${p.link_source ? ` · linked ${esc(p.link_source)}` : ""}</span></div>`).join("")}</div></div>`;
      $("#v-journey").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    $("#v-add").onclick = () => { $("#v-upload").hidden = !$("#v-upload").hidden; };
    $("#v-cancel").onclick = () => { $("#v-upload").hidden = true; };
    $("#v-files").onchange = () => {
      const f = [...($("#v-files").files || [])];
      $("#v-chosen").textContent = f.length
        ? `${f.length} video(s): ` + f.map((x) => `${x.name} (${(x.size / 1e6).toFixed(0)} MB)`).join(", ")
        : "";
    };
    $("#v-go").onclick = async () => {
      const files = [...($("#v-files").files || [])];
      if (!files.length) return fail(new Error("Choose at least one video"));
      const fd = new FormData();
      fd.append("name", $("#v-site").value);
      fd.append("location", $("#v-loc").value);
      fd.append("zone_mode", "auto");
      files.forEach((f) => fd.append("files", f));
      $("#v-go").disabled = true;
      try {
        await postForm("/api/sites/upload", fd, (pct) => { $("#v-prog").textContent = `uploading ${pct}%`; });
        $("#v-prog").textContent = "";
        $("#v-upload").hidden = true;
        siteStatus();
      } catch (e) { fail(e); } finally { $("#v-go").disabled = false; }
    };

    async function siteStatus() {
      let sites;
      try { sites = await api("/api/sites"); } catch (e) { return; }
      const names = Object.keys(sites);
      const box = $("#v-site-status");
      if (!names.length) { box.hidden = true; return; }
      box.hidden = false;
      box.innerHTML = names.map((n) => {
        const j = sites[n];
        const pct = Math.round(100 * (j.done || []).length / Math.max(j.total || 1, 1));
        return `<div><div class="row"><b>${esc(n)}</b>
          <span class="muted small">${esc(j.message || "")}</span><span class="spacer"></span>
          <span class="small muted">${(j.done || []).length}/${j.total} camera(s)</span></div>
          <div class="bar"><div style="width:${pct}%"></div></div></div>`;
      }).join("");
      if (names.some((n) => !["done", "error"].includes(sites[n].state))) setTimeout(siteStatus, 2000);
      else { refreshStatus(); load(); }
    }

    $("#v-link").onclick = async () => {
      try { await api("/api/link", { method: "POST", body: {} }); refreshStatus();
            setTimeout(load, 2500); } catch (e) { fail(e); }
    };
    listeners.add((msg) => {
      if (msg.type === "link") { refreshStatus(); load(); }
      if (msg.type === "site") siteStatus();
      if (msg.type === "vehicle_linked") load();
    });
    await refreshStatus();
    await siteStatus();
    await load();
  }

  // ---------------------------------------------------------------- accounts, registry, audit (admin)
  async function pageAdmin(root) {
    if (!isAdmin()) { root.innerHTML = `<div class="card empty">Administrators only.</div>`; return; }
    const fmt = (t) => t ? new Date(t).toLocaleString() : "-";
    root.innerHTML = `<div class="row page-head"><h1>Accounts</h1><span class="spacer"></span>
        <span class="seg" id="a-tabs">${["people", "vehicles", "activity"].map((k, i) =>
          `<button data-v="${k}"${i ? "" : ' class="on"'}>${k[0].toUpperCase() + k.slice(1)}</button>`).join("")}</span></div>
      <div id="a-body"></div>`;
    let tab = "people";
    $$("#a-tabs button").forEach((b) => b.onclick = () => {
      tab = b.dataset.v;
      $$("#a-tabs button").forEach((x) => x.classList.toggle("on", x === b));
      render();
    });

    async function render() {
      const body = $("#a-body");
      if (tab === "people") {
        const r = await api("/api/users");
        body.innerHTML = `<div class="card">
          <div class="card-head"><h2>People</h2></div>
          <table class="kv wide"><tr><th>Name</th><th>Role</th><th>Signs in with</th><th>Last seen</th><th></th></tr>
          ${r.items.map((u) => `<tr><td>${esc(u.name || "-")}</td><td>${esc(u.role)}</td>
            <td class="small">${esc(u.username || u.licence_no || "-")}${u.phone ? " · " + esc(u.phone) : ""}</td>
            <td class="small muted">${esc(fmt(u.last_login))}</td>
            <td><button class="btn small" data-toggle="${u.id}" data-active="${u.active}">${
              u.active ? "Disable" : "Enable"}</button></td></tr>`).join("")}</table></div>
          <div class="grid2">
            <div class="card"><div class="card-head"><h2>Add a driver</h2></div>
              <label class="field">Name<input id="d-name"></label>
              <label class="field">Driving licence number<input id="d-lic" placeholder="DK-1234567"></label>
              <label class="field">Mobile (receives the sign-in code)<input id="d-phone" placeholder="+8801..."></label>
              <button class="btn primary" id="d-add">Add driver</button></div>
            <div class="card"><div class="card-head"><h2>Add staff</h2></div>
              <label class="field">Name<input id="s-name"></label>
              <label class="field">Username<input id="s-user"></label>
              <label class="field">Password<input id="s-pass" type="password"></label>
              <label class="field">Role<select id="s-role"><option value="controller">Traffic controller</option>
                <option value="admin">Administrator</option></select></label>
              <button class="btn primary" id="s-add">Add staff</button></div></div>`;
        $$("[data-toggle]").forEach((b) => b.onclick = async () => {
          try { await api(`/api/users/${b.dataset.toggle}`, { method: "PATCH",
                  body: { active: b.dataset.active !== "1" } }); render(); } catch (e) { fail(e); }
        });
        $("#d-add").onclick = async () => {
          try { await api("/api/users", { method: "POST", body: { role: "driver", name: $("#d-name").value,
                  licence_no: $("#d-lic").value, phone: $("#d-phone").value } }); render(); } catch (e) { fail(e); }
        };
        $("#s-add").onclick = async () => {
          try { await api("/api/users", { method: "POST", body: { role: $("#s-role").value,
                  name: $("#s-name").value, username: $("#s-user").value, password: $("#s-pass").value } });
                render(); } catch (e) { fail(e); }
        };
      } else if (tab === "vehicles") {
        const r = await api("/api/registry");
        const pending = r.items.filter((v) => !v.verified).length;
        body.innerHTML = `<div class="card">
          <div class="card-head"><h2>Registered vehicles</h2><span class="spacer"></span>
            <span class="small muted">${pending} awaiting verification</span></div>
          <div class="small secondary">A driver only sees violations for a plate once it is verified here.
            Verifying a plate someone claimed wrongly hands them another person's record, so check it
            against the registration document.</div>
          <table class="kv wide"><tr><th>Plate</th><th>Keeper</th><th>Added</th><th>Status</th><th></th></tr>
          ${r.items.map((v) => `<tr><td><b>${esc(v.plate_en || v.plate_key)}</b></td>
            <td>${esc(v.owner_name || "-")}<span class="small muted"> ${esc(v.owner_licence || "")}</span></td>
            <td class="small muted">${esc(v.source)}</td>
            <td>${v.verified ? '<span class="small">verified</span>' : '<span class="small" style="color:var(--warning-ink)">claimed, unverified</span>'}</td>
            <td>${v.verified ? "" : `<button class="btn small good" data-ok="${v.id}">Verify</button>`}
              <button class="btn small danger" data-rm="${v.id}">Remove</button></td></tr>`).join("")
            || '<tr><td colspan="5" class="muted small">Nothing registered yet.</td></tr>'}</table></div>
          <div class="card"><div class="card-head"><h2>Link a plate to a licence</h2></div>
            <div class="row"><input id="r-plate" placeholder="DHAKA METRO-GA 12-3456" style="flex:1">
              <input id="r-lic" placeholder="Licence number" style="flex:1">
              <button class="btn primary" id="r-add">Link</button></div></div>`;
        $$("[data-ok]").forEach((b) => b.onclick = async () => {
          try { await api(`/api/registry/${b.dataset.ok}/verify`, { method: "POST" }); render(); } catch (e) { fail(e); }
        });
        $$("[data-rm]").forEach((b) => b.onclick = async () => {
          try { await api(`/api/registry/${b.dataset.rm}`, { method: "DELETE" }); render(); } catch (e) { fail(e); }
        });
        $("#r-add").onclick = async () => {
          try { await api("/api/registry", { method: "POST", body: { plate: $("#r-plate").value,
                  licence_no: $("#r-lic").value } }); render(); } catch (e) { fail(e); }
        };
      } else {
        const r = await api("/api/audit?limit=300");
        body.innerHTML = `<div class="card"><div class="card-head"><h2>Activity</h2></div>
          <div class="small secondary">Every decision recorded against the person who made it. A case with
            nobody's name on it is not evidence of anything, so this is written on each confirm, reject,
            registration and sign-in.</div>
          <table class="kv wide"><tr><th>When</th><th>Who</th><th>Did what</th><th>To</th><th>Detail</th></tr>
          ${r.items.map((a) => `<tr><td class="small muted">${esc(fmt(a.at))}</td>
            <td>${esc(a.who)}<span class="small muted"> ${esc(a.role)}</span></td>
            <td><code class="small">${esc(a.action)}</code></td>
            <td class="small">${esc(a.object_type)}${a.object_id ? " " + esc(a.object_id) : ""}</td>
            <td class="small muted">${esc(a.detail || "")}</td></tr>`).join("")}</table></div>`;
      }
    }
    await render();
  }

  async function pageStats(main) {
    const cams = await api("/api/cameras");
    const f = { camera_id: "", days: 30, include_rejected: false };
    main.innerHTML = `
      <div class="page-head"><div><h1>Statistics</h1><p>Where, when and which vehicles break the rules, by the time they happened. Rejected cases are excluded unless you include them.</p></div></div>
      <div class="row" style="margin-bottom:14px">
        <span class="seg" id="days">${[[1, "Today"], [7, "7 days"], [30, "30 days"], [365, "1 year"]].map(([d, l]) => `<button data-v="${d}" class="${d === 30 ? "on" : ""}">${l}</button>`).join("")}</span>
        <select id="s-cam"><option value="">All cameras</option>${cams.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("")}</select>
        <label class="row small"><input type="checkbox" id="rej"> Include rejected</label>
      </div>
      <div class="tiles" id="tiles"></div>
      <div class="charts" id="charts"></div>`;
    const chartCard = (id, title, sub) => `<div class="chart-card"><div class="card-head"><div><h3>${esc(title)}</h3><div class="small muted">${esc(sub)}</div></div>
      <button class="btn small ghost" data-table="${id}">Table</button></div><div class="chart" id="${id}"></div></div>`;
    $("#charts").innerHTML = chartCard("c-type", "By violation type", "cases") + chartCard("c-hour", "By hour of day", "cases, real time when known") +
      chartCard("c-class", "By vehicle class", "cases") + chartCard("c-day", "Per day", "cases") + chartCard("c-cam", "By camera", "cases");
    const data = {};
    const draw = {
      "c-type": (el) => Charts.bars(el, data.type, { labelWidth: 230 }),
      "c-hour": (el) => Charts.columns(el, data.hour, { labelEvery: 3 }),
      "c-class": (el) => Charts.bars(el, data.cls, { labelWidth: 110 }),
      "c-day": (el) => Charts.columns(el, data.day),
      "c-cam": (el) => Charts.bars(el, data.cam, { labelWidth: 170 }),
    };
    const tables = new Set();
    async function load() {
      const p = new URLSearchParams({ days: f.days, include_rejected: f.include_rejected });
      if (f.camera_id) p.set("camera_id", f.camera_id);
      $("#charts").style.opacity = 0.6;
      const s = await api("/api/stats?" + p);
      $("#charts").style.opacity = 1;
      data.type = s.by_code.map((r) => ({ label: `${r.code} ${r.title}`, value: r.n }));
      const hours = Object.fromEntries(s.by_hour.map((r) => [r.hour, r.n]));
      data.hour = Array.from({ length: 24 }, (_, h) => ({ label: String(h).padStart(2, "0"), value: hours[h] || 0, tip: `${String(h).padStart(2, "0")}:00–${String(h + 1).padStart(2, "0")}:00` }));
      data.cls = s.by_class.map((r) => ({ label: r.cls, value: r.n }));
      const days = {};
      s.by_day.forEach((r) => { days[r.day] = (days[r.day] || 0) + r.n; });
      data.day = Object.keys(days).sort().map((d) => ({ label: d.slice(5), value: days[d], tip: d }));
      data.cam = s.by_camera.map((r) => ({ label: r.camera, value: r.n }));
      const st = Object.fromEntries(s.by_status.map((r) => [r.status, r.n]));
      const total = data.type.reduce((a, d) => a + d.value, 0);
      const reviewed = (st.confirmed || 0) + (st.rejected || 0);
      $("#tiles").innerHTML = `
        <div class="tile"><div class="label">Cases in period</div><div class="value">${total}</div></div>
        <div class="tile"><div class="label">Confirmed</div><div class="value">${st.confirmed || 0}</div><div class="sub">${statusBadge("confirmed")}</div></div>
        <div class="tile"><div class="label">Waiting for review</div><div class="value">${st.pending || 0}</div><div class="sub">${statusBadge("pending")}</div></div>
        <div class="tile"><div class="label">Detection precision</div><div class="value">${reviewed ? Math.round(100 * (st.confirmed || 0) / reviewed) + "%" : "–"}</div>
          <div class="sub">confirmed ÷ reviewed (${reviewed})</div></div>`;
      for (const id in draw) {
        const el = $("#" + id);
        if (tables.has(id)) Charts.table(el, { "c-type": data.type, "c-hour": data.hour, "c-class": data.cls, "c-day": data.day, "c-cam": data.cam }[id], ["Category", "Cases"]);
        else draw[id](el);
      }
    }
    let rt = null;
    const onResize = () => { clearTimeout(rt); rt = setTimeout(() => { for (const id in draw) if (!tables.has(id) && data.type) draw[id]($("#" + id)); }, 150); };
    window.addEventListener("resize", onResize);
    onLeave(() => window.removeEventListener("resize", onResize));
    $$("#days button").forEach((b) => b.onclick = () => { f.days = +b.dataset.v; $$("#days button").forEach((x) => x.classList.toggle("on", x === b)); load(); });
    $("#s-cam").onchange = () => { f.camera_id = $("#s-cam").value; load(); };
    $("#rej").onchange = () => { f.include_rejected = $("#rej").checked; load(); };
    $("#charts").addEventListener("click", (e) => {
      const b = e.target.closest("[data-table]");
      if (!b) return;
      const id = b.dataset.table;
      if (tables.has(id)) { tables.delete(id); b.textContent = "Table"; } else { tables.add(id); b.textContent = "Chart"; }
      load();
    });
    await load();
  }

  // ------------------------------------------------------------------ boot
  const App = { api, esc, toast, fail, modal, confirmBox, go, onLeave, setLeaveGuard, pollStatus, ICON, state, media, editCameraModal, autoReport };
  window.App = App;

  $("#theme-btn").onclick = () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
  };
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }

  // Control-room clock in Bangladesh time, whatever the Mac's time zone.
  const clockFmt = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Dhaka", weekday: "short", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const tickClock = () => { const el = $("#clock"); if (el) el.textContent = clockFmt.format(new Date()) + " BST"; };
  tickClock();
  setInterval(tickClock, 1000);
  const tickOnline = async () => {
    try {
      const st = await api("/api/status");
      const n = Object.values(st).filter((x) => ["running", "starting", "reconnecting"].includes(x.state)).length;
      $("#online").innerHTML = `<span class="dot${n ? " on" : ""}"></span>${n} of ${Object.keys(st).length} camera${Object.keys(st).length === 1 ? "" : "s"} running`;
    } catch (e) { /* server restarting */ }
  };
  tickOnline();
  setInterval(tickOnline, 3000);
  $("#plate-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.value.trim()) { go("#/review?plate=" + encodeURIComponent(e.target.value.trim())); e.target.value = ""; }
  });

  window.addEventListener("hashchange", route);
  window.addEventListener("DOMContentLoaded", async () => {
    try { state.me = await api("/api/me"); } catch (e) { return; }          // 401 already redirected
    if (!state.me.signed_in) { location.href = "/login"; return; }
    if (state.me.role === "driver") { location.href = "/portal"; return; }
    $("#who-box").hidden = false;
    $("#who-name").textContent = state.me.name || state.me.username || "";
    $("#who-role").textContent = state.me.role === "admin" ? "Administrator" : "Traffic controller";
    $("#nav-admin").hidden = !isAdmin();
    $("#signout").onclick = async () => { await api("/api/logout", { method: "POST" }); location.href = "/login"; };
    try { state.meta = await api("/api/meta"); } catch (e) { $("#main").innerHTML = `<div class="card errors">Server not reachable: ${esc(e.message)}</div>`; return; }
    if (state.meta.lan_url) $("#lan-url").innerHTML = `Phone / LAN: <a href="${esc(state.meta.lan_url)}">${esc(state.meta.lan_url.replace("http://", ""))}</a>`;
    connectEvents();
    refreshPending();
    route();
  });
})();
