/* Camera map (R-2, R-8). Provider-pluggable: Google Maps when a browser key is configured, otherwise Leaflet +
   OpenStreetMap. Google traffic is drawn only on the Google map (its terms forbid it on other maps); the OSM view
   shows only this system's own measurements. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
  const R = 6371000;

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
    clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 4000);
  }

  // ---------------------------------------------------------------- geometry (same plane approximation as geo.py)
  const rad = (d) => d * Math.PI / 180;
  function dest(lat, lon, bearing, m) {
    const b = rad(bearing);
    return [lat + (m * Math.cos(b) / R) * 180 / Math.PI, lon + (m * Math.sin(b) / (R * Math.cos(rad(lat)))) * 180 / Math.PI];
  }
  function bearing(lat1, lon1, lat2, lon2) {
    const x = rad(lon2 - lon1) * Math.cos(rad(lat1)), y = rad(lat2 - lat1);
    return (Math.atan2(x, y) * 180 / Math.PI + 360) % 360;
  }
  function wedge(g) {
    const pts = [[g.lat, g.lon]], n = 12;
    for (let i = 0; i <= n; i++) pts.push(dest(g.lat, g.lon, g.heading_deg - g.fov_deg / 2 + g.fov_deg * i / n, g.range_m));
    return pts;
  }
  function offsetLine(a, b, metres) {            // shift a->b to the right of travel so both directions show
    const dx = (b[1] - a[1]) * Math.cos(rad(a[0])), dy = b[0] - a[0], len = Math.hypot(dx, dy) || 1;
    const dLat = (dx / len) * metres / R * 180 / Math.PI, dLon = (-dy / len) * metres / (R * Math.cos(rad(a[0]))) * 180 / Math.PI;
    return [[a[0] + dLat, a[1] + dLon], [b[0] + dLat, b[1] + dLon]];
  }

  // ---------------------------------------------------------------- provider adapters
  // Both expose: init, onClick, fit, marker, polygon, polyline, label, setTraffic. Handles have remove() and,
  // where it makes sense, setPath/setPos. Positions are [lat, lon].
  function loadTag(tag, attrs) {
    return new Promise((resolve, reject) => {
      const el = document.createElement(tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      el.onload = resolve; el.onerror = () => reject(new Error("could not load " + (attrs.src || attrs.href)));
      document.head.appendChild(el);
    });
  }

  const Leaflet = {
    name: "osm",
    async load() {
      await loadTag("link", { rel: "stylesheet", href: "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
        integrity: "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=", crossorigin: "anonymous" });
      await loadTag("script", { src: "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
        integrity: "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=", crossorigin: "anonymous" });
    },
    init(el, centre, zoom, cfg) {
      this.map = L.map(el, { zoomControl: true }).setView(centre, zoom);
      L.tileLayer(cfg.osm.url, { maxZoom: cfg.osm.max_zoom, attribution: cfg.osm.attribution }).addTo(this.map);
    },
    onClick(fn) { this.map.on("click", (e) => fn([e.latlng.lat, e.latlng.lng])); },
    centre() { const c = this.map.getCenter(); return [c.lat, c.lng]; },
    fit(pts) { if (pts.length) this.map.fitBounds(pts, { padding: [40, 40], maxZoom: 16 }); },
    panTo(p) { this.map.panTo(p); },
    marker(o) {
      const icon = L.divIcon({ className: "leaflet-div-icon-clean", html: o.html, iconSize: [o.size, o.size], iconAnchor: [o.size / 2, o.size / 2] });
      const m = L.marker(o.pos, { icon, draggable: !!o.draggable, title: o.title || "", zIndexOffset: o.z || 0 }).addTo(this.map);
      if (o.onClick) m.on("click", (e) => { L.DomEvent.stopPropagation(e); o.onClick(); });
      if (o.onDrag) m.on("drag", () => { const p = m.getLatLng(); o.onDrag([p.lat, p.lng]); });
      if (o.onDragEnd) m.on("dragend", () => { const p = m.getLatLng(); o.onDragEnd([p.lat, p.lng]); });
      return { remove: () => m.remove() };
    },
    polygon(path, st) { const p = L.polygon(path, { color: st.color, weight: 1, fillColor: st.color, fillOpacity: st.fill, interactive: false }).addTo(this.map);
      return { remove: () => p.remove(), setPath: (x) => p.setLatLngs(x) }; },
    polyline(path, st, onClick) { const p = L.polyline(path, { color: st.color, weight: st.weight, opacity: 0.9, dashArray: st.dash || null }).addTo(this.map);
      if (onClick) p.on("click", (e) => { L.DomEvent.stopPropagation(e); onClick(); });
      return { remove: () => p.remove() }; },
    label(pos, text, grey, onClick) {
      const icon = L.divIcon({ className: "leaflet-div-icon-clean", html: `<div class="pair-label ${grey ? "grey" : ""}">${esc(text)}</div>`, iconSize: [0, 0] });
      const m = L.marker(pos, { icon }).addTo(this.map);
      if (onClick) m.on("click", (e) => { L.DomEvent.stopPropagation(e); onClick(); });
      return { remove: () => m.remove() };
    },
    setTraffic() { /* Google traffic is never drawn on a non-Google map */ },
  };

  const Google = {
    name: "google",
    load(cfg) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("Google Maps did not load")), 15000);
        window.gm_authFailure = () => { clearTimeout(timer); reject(new Error("Google rejected the Maps key")); };
        window.__gmReady = () => { clearTimeout(timer); resolve(); };
        const s = document.createElement("script");
        s.src = "https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(cfg.google_browser_key) + "&v=weekly&callback=__gmReady";
        s.async = true; s.onerror = () => { clearTimeout(timer); reject(new Error("could not load Google Maps")); };
        document.head.appendChild(s);
      });
    },
    init(el, centre, zoom) {
      this.map = new google.maps.Map(el, { center: { lat: centre[0], lng: centre[1] }, zoom, clickableIcons: false, streetViewControl: false, mapTypeControl: false });
      this.traffic = new google.maps.TrafficLayer();
    },
    ll: (p) => ({ lat: p[0], lng: p[1] }),
    onClick(fn) { this.map.addListener("click", (e) => fn([e.latLng.lat(), e.latLng.lng()])); },
    centre() { const c = this.map.getCenter(); return [c.lat(), c.lng()]; },
    fit(pts) { if (!pts.length) return; const b = new google.maps.LatLngBounds(); pts.forEach((p) => b.extend(this.ll(p))); this.map.fitBounds(b, 40); },
    panTo(p) { this.map.panTo(this.ll(p)); },
    marker(o) {
      const m = new google.maps.Marker({ map: this.map, position: this.ll(o.pos), draggable: !!o.draggable, title: o.title || "", zIndex: o.z || 1,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: o.size / 2 - 2, fillColor: o.color, fillOpacity: 1, strokeColor: "#fff", strokeWeight: o.selected ? 4 : 2 } });
      if (o.onClick) m.addListener("click", o.onClick);
      const pos = () => { const p = m.getPosition(); return [p.lat(), p.lng()]; };
      if (o.onDrag) m.addListener("drag", () => o.onDrag(pos()));
      if (o.onDragEnd) m.addListener("dragend", () => o.onDragEnd(pos()));
      return { remove: () => m.setMap(null) };
    },
    polygon(path, st) { const p = new google.maps.Polygon({ map: this.map, paths: path.map(this.ll), strokeColor: st.color, strokeWeight: 1, fillColor: st.color, fillOpacity: st.fill, clickable: false });
      return { remove: () => p.setMap(null), setPath: (x) => p.setPaths(x.map(this.ll)) }; },
    polyline(path, st, onClick) { const p = new google.maps.Polyline({ map: this.map, path: path.map(this.ll), strokeColor: st.color, strokeWeight: st.weight, strokeOpacity: 0.9 });
      if (onClick) p.addListener("click", onClick);
      return { remove: () => p.setMap(null) }; },
    label(pos, text, grey, onClick) {
      const m = new google.maps.Marker({ map: this.map, position: this.ll(pos), zIndex: 5,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 0 }, label: { text, color: grey ? "#6f6d68" : "#0b0b0b", fontSize: "12px", fontWeight: "600" } });
      if (onClick) m.addListener("click", onClick);
      return { remove: () => m.setMap(null) };
    },
    setTraffic(on) { this.traffic.setMap(on ? this.map : null); },
  };

  // ---------------------------------------------------------------- state
  const S = { me: null, cfg: null, P: null, cams: [], draft: {}, sel: null, pairs: null, pair: null, handles: [], busy: false, win: "all", err: "" };
  const isAdmin = () => S.me && S.me.role === "admin";
  const geoOf = (c) => S.draft[c.id] || c.geo;

  function speedColour(p) {
    if (!p.enough) return css("--text-muted");
    const v = p.speed.median_kph;
    return v < 10 ? css("--critical") : v < 20 ? css("--serious") : v < 30 ? css("--warning") : css("--good");
  }

  // ---------------------------------------------------------------- map drawing
  function clearMap() { S.handles.forEach((h) => h.remove()); S.handles = []; }

  function draw() {
    clearMap();
    const pos = {};
    S.cams.forEach((c) => {
      const g = geoOf(c);
      if (!g) return;
      pos[c.id] = [g.lat, g.lon];
      const sel = S.sel === c.id, draft = !!S.draft[c.id];
      const live = c.state || "idle";
      const colour = live === "running" ? css("--good") : (live === "error" || live === "stopped") ? css("--critical") : css("--text-muted");
      const w = S.P.polygon(wedge(g), { color: sel ? css("--accent") : colour, fill: sel ? 0.28 : 0.14 });
      S.handles.push(w);
      const m = S.P.marker({ pos: [g.lat, g.lon], size: 22, color: colour, selected: sel, title: c.name,
        html: `<div class="cam-pin ${esc(live)} ${sel ? "sel" : ""} ${draft ? "draft" : ""}"></div>`, z: sel ? 1000 : 500,
        draggable: isAdmin(), onClick: () => select(c.id),
        onDrag: (p) => { S.drag = Date.now(); w.setPath(wedge({ ...g, lat: p[0], lon: p[1] })); },
        onDragEnd: (p) => { edit(c.id, { lat: round(p[0], 7), lon: round(p[1], 7) }); } });
      S.handles.push(m);
      if (sel && isAdmin()) {                                    // the aim handle sits at the end of the view axis
        const tip = dest(g.lat, g.lon, g.heading_deg, g.range_m);
        S.handles.push(S.P.marker({ pos: tip, size: 16, color: css("--accent"), title: "Drag to aim", z: 1100, draggable: true,
          html: '<div class="aim-pin"></div>',
          onDrag: (p) => { S.drag = Date.now(); const h = round(bearing(g.lat, g.lon, p[0], p[1]), 1); w.setPath(wedge({ ...g, heading_deg: h })); const el = $("#f-heading"); if (el) el.value = h; },
          onDragEnd: (p) => { edit(c.id, { heading_deg: round(bearing(g.lat, g.lon, p[0], p[1]), 1) }); } }));
      }
    });
    if (S.pairs) S.pairs.pairs.forEach((p) => {
      const a = pos[p.from], b = pos[p.to];
      if (!a || !b) return;
      const [oa, ob] = offsetLine(a, b, 6);
      const col = speedColour(p), pick = () => { S.pair = p; renderPair(); };
      S.handles.push(S.P.polyline([oa, ob], { color: col, weight: p.enough ? 5 : 3 }, pick));
      const text = p.enough ? `n=${p.n_vehicles} · ${p.speed.median_kph} km/h` : `n=${p.n_vehicles} · too few`;
      S.handles.push(S.P.label([(oa[0] + ob[0]) / 2, (oa[1] + ob[1]) / 2], text, !p.enough, pick));
    });
  }

  const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

  // ---------------------------------------------------------------- panel
  function renderList() {
    $("#cam-list").innerHTML = S.cams.length ? S.cams.map((c) => {
      const g = geoOf(c);
      return `<div class="cam-row ${S.sel === c.id ? "sel" : ""}" data-id="${c.id}">
        <span class="sdot ${esc(c.state)}" title="${esc(c.state)}"></span><span class="nm">${esc(c.name)}</span>
        ${g ? (S.draft[c.id] ? '<span class="chip small">unsaved</span>' : '<span class="muted small">placed</span>')
            : (isAdmin() ? `<button class="btn small" data-place="${c.id}">Place</button>` : '<span class="muted small">not placed</span>')}</div>`;
    }).join("") : '<p class="muted">No cameras yet. Add one on the dashboard.</p>';
    document.querySelectorAll(".cam-row").forEach((r) => r.onclick = () => select(+r.dataset.id));
    document.querySelectorAll("[data-place]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); place(+b.dataset.place); });
    $("#edit-hint").textContent = isAdmin() ? "Drag a camera to move it. With a camera selected, drag the blue diamond to aim it."
      : "View only. An administrator places and aims the cameras.";
  }

  function renderEditor() {
    const el = $("#editor"), c = S.cams.find((x) => x.id === S.sel), g = c && geoOf(c);
    if (!c || !g) { el.classList.add("hidden"); return; }
    el.classList.remove("hidden");
    const ro = isAdmin() ? "" : "readonly";
    const f = (id, label, v, step, extra = "") => `<label>${label}<input id="f-${id}" type="number" step="${step}" value="${esc(v)}" ${ro} ${extra}></label>`;
    el.innerHTML = `<div class="card-head"><h2>${esc(c.name)}</h2><span class="spacer"></span><span class="muted small">${esc(c.state)}</span></div>
      <div class="form-grid">${f("lat", "Latitude", g.lat, "0.000001")}${f("lon", "Longitude", g.lon, "0.000001")}
        ${f("heading", "Heading (°, 0 = north, clockwise)", g.heading_deg, "1", 'min="0" max="360"')}
        ${f("fov", "Field of view (°)", g.fov_deg, "1", 'min="5" max="180"')}
        ${f("range", "View range (m)", g.range_m, "5", 'min="5" max="1000"')}
        ${f("offset", "Clock offset (s)", g.clock_offset_s, "1")}</div>
      <p class="muted small">Clock offset is added to this camera's times when comparing cameras: set it only if you know its clock is wrong.</p>
      <div id="ed-msg"></div>
      ${isAdmin() ? `<div class="row"><button class="btn primary small" id="ed-save">Save</button>
        <button class="btn small" id="ed-revert">Revert</button>
        ${c.placed ? '<button class="btn small danger" id="ed-remove">Remove from map</button>' : ""}</div>` : ""}`;
    if (!isAdmin()) return;
    const read = () => ({ lat: +$("#f-lat").value, lon: +$("#f-lon").value, heading_deg: +$("#f-heading").value,
      fov_deg: +$("#f-fov").value, range_m: +$("#f-range").value, clock_offset_s: +$("#f-offset").value });
    ["lat", "lon", "heading", "fov", "range", "offset"].forEach((k) => $("#f-" + k).onchange = () => edit(c.id, read()));
    $("#ed-save").onclick = () => save(c.id, read());
    $("#ed-revert").onclick = () => { delete S.draft[c.id]; refreshAll(); };
    const rm = $("#ed-remove");
    if (rm) rm.onclick = async () => {
      if (!confirm(`Take "${c.name}" off the map?`)) return;
      try { await api(`/api/map/cameras/${c.id}`, { method: "DELETE" }); delete S.draft[c.id]; S.sel = null; await loadCams(); refreshAll(); }
      catch (e) { toast(e.message); }
    };
  }

  function select(id) {
    S.sel = id; S.pair = null;
    const c = S.cams.find((x) => x.id === id), g = c && geoOf(c);
    if (g) S.P.panTo([g.lat, g.lon]);
    refreshAll();
  }

  function place(id) {
    const [lat, lon] = S.P.centre();
    S.draft[id] = { lat: round(lat, 7), lon: round(lon, 7), heading_deg: 0, fov_deg: 60, range_m: 80, clock_offset_s: 0 };
    S.sel = id; refreshAll();
  }

  function edit(id, patch) {
    const c = S.cams.find((x) => x.id === id);
    S.draft[id] = { ...geoOf(c), ...patch };
    refreshAll();
  }

  async function save(id, body) {
    const msg = $("#ed-msg");
    try {
      const out = await api(`/api/map/cameras/${id}`, { method: "PUT", body });
      delete S.draft[id];
      await loadCams(); refreshAll();
      if (out.warnings && out.warnings.length) $("#ed-msg").innerHTML = out.warnings.map((w) => `<p class="warn">${esc(w)}</p>`).join("");
      toast("Saved");
    } catch (e) { if (msg) msg.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }

  function histogram(speeds, median) {
    if (!speeds || !speeds.length) return "";
    const top = Math.max(40, Math.ceil(Math.max(...speeds) / 10) * 10), bw = 5, n = top / bw, bins = new Array(n).fill(0);
    speeds.forEach((s) => { bins[Math.min(n - 1, Math.floor(s / bw))]++; });
    const mx = Math.max(...bins), W = 300, H = 70, w = W / n;
    const bars = bins.map((b, i) => `<rect x="${(i * w + 1).toFixed(1)}" y="${(H - (b / mx) * H).toFixed(1)}" width="${(w - 2).toFixed(1)}" height="${((b / mx) * H).toFixed(1)}"><title>${i * bw}-${(i + 1) * bw} km/h: ${b}</title></rect>`).join("");
    const mx2 = (median / top * W).toFixed(1);
    return `<svg class="hist" viewBox="0 0 ${W} ${H + 14}" preserveAspectRatio="none" role="img" aria-label="Distribution of section speeds">
      ${bars}<line x1="${mx2}" x2="${mx2}" y1="0" y2="${H}"/><text x="0" y="${H + 11}">0</text><text x="${W - 30}" y="${H + 11}">${top} km/h</text></svg>`;
  }

  function renderPair() {
    const el = $("#pair"), p = S.pair;
    if (!p) { el.classList.add("hidden"); return; }
    el.classList.remove("hidden");
    const basis = p.distance_basis === "road" ? "road distance (optimiser network)" : `straight line × ${S.cfg.bounds.circuity} (assumed road factor)`;
    const rej = Object.entries(p.rejected).filter(([, n]) => n).map(([k, n]) => `${esc(k.replace(/_/g, " "))}: ${n}`).join(", ");
    el.innerHTML = `<div class="card-head"><h2>${esc(p.from_name)} → ${esc(p.to_name)}</h2><span class="spacer"></span>
      <button class="btn small ghost" id="pair-x" aria-label="Close">×</button></div>
      ${p.enough ? `<dl class="kv"><dt>Median</dt><dd><b>${p.speed.median_kph} km/h</b></dd><dt>Mean</dt><dd>${p.speed.mean_kph} km/h</dd>
        <dt>Space-mean</dt><dd>${p.speed.space_mean_kph} km/h</dd><dt>15th – 85th percentile</dt><dd>${p.speed.p15_kph} – ${p.speed.p85_kph} km/h</dd>
        <dt>Vehicles / segments</dt><dd>${p.n_vehicles} / ${p.n}</dd><dt>Distance</dt><dd>${Math.round(p.distance_m)} m</dd></dl>
        ${histogram(p.speeds_kph, p.speed.median_kph)}`
        : `<p><b>No speed shown.</b> ${esc(p.reason)}.</p><dl class="kv"><dt>Vehicles matched</dt><dd>${p.n_vehicles}</dd><dt>Distance</dt><dd>${Math.round(p.distance_m)} m</dd></dl>`}
      <p class="muted small">Distance: ${esc(basis)}.</p>
      <p class="muted small">${rej ? "Rejected as implausible — " + rej + "." : "No segment rejected."}
        Accepted: ${S.cfg.bounds.min_speed_kph}–${S.cfg.bounds.max_speed_kph} km/h, under ${S.cfg.bounds.max_travel_s / 60} min between the cameras.</p>`;
    $("#pair-x").onclick = () => { S.pair = null; renderPair(); };
  }

  function renderLegend() {
    const b = S.cfg.bounds;
    const row = (c, t) => `<div class="legend-row"><span class="swatch" style="background:${c}"></span>${t}</div>`;
    $("#legend").innerHTML = `<div class="card-head"><h2>Section speed (median)</h2></div>
      ${row(css("--critical"), "under 10 km/h")}${row(css("--serious"), "10–20 km/h")}${row(css("--warning"), "20–30 km/h")}${row(css("--good"), "30 km/h and over")}
      ${row(css("--text-muted"), `fewer than ${b.min_vehicles} matched vehicles — no number`)}
      <p class="muted small">Speed of the same vehicle between two cameras, matched by appearance. Colour bands are for reading the map, not a standard. ${esc(S.cfg.clock_note)}</p>`;
  }

  function renderSettings(st) {
    const el = $("#settings");
    if (!isAdmin()) { el.classList.add("hidden"); return; }
    el.classList.remove("hidden");
    const line = (label, k) => `<label class="wide">${label}<input id="k-${k}" type="password" autocomplete="off" placeholder="${st[k].set ? "leave blank to keep" : "paste key"}"></label>
      <p class="muted small wide">${st[k].set ? esc(st[k].hint) + (st[k].source === "env" ? " (from the environment, which overrides this page)" : "") : "not set"}
      ${st[k].set && st[k].source === "stored" ? `<button class="btn small ghost" data-clear="${k}">Clear</button>` : ""}</p>`;
    el.innerHTML = `<div class="card-head"><h2>Google Maps keys</h2></div>
      <p class="muted small">Without a browser key the map uses OpenStreetMap and shows only this system's measurements. Restrict the browser key by HTTP referrer; the server key (live-traffic route times) is stored on the server and is never shown again.</p>
      <div class="form-grid">${line("Browser key (Maps JavaScript API)", "browser_key")}${line("Server key (Routes API)", "server_key")}</div>
      <div id="set-msg"></div><button class="btn primary small" id="set-save">Save keys</button>`;
    const send = async (body) => {
      try { const out = await api("/api/settings/maps", { method: "PUT", body }); toast("Saved. Reload the page to switch map provider."); renderSettings(out); }
      catch (e) { $("#set-msg").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
    };
    $("#set-save").onclick = () => {
      const body = {};
      ["browser_key", "server_key"].forEach((k) => { const v = $("#k-" + k).value.trim(); if (v) body[k] = v; });
      if (Object.keys(body).length) send(body);
    };
    el.querySelectorAll("[data-clear]").forEach((b) => b.onclick = () => send({ [b.dataset.clear]: "" }));
  }

  // Optimiser network overlay. GOOGLE RULE (R-8): this asks the server for Google-supplemented travel times, which
  // Google's terms allow only on a Google map, so it is offered ONLY when the provider is Google (never on OSM/Leaflet,
  // and never on the MapLibre 3D view at /city, which does not ask for Google values at all).
  async function drawNetOverlay(on) {
    (S.netHandles || []).forEach((h) => h.remove()); S.netHandles = [];
    if (!on || S.P !== Google) return;
    let live;
    try { live = await api("/api/optimize/live?use_google=true"); } catch (e) { toast("Optimiser network: " + e.message); $("#net-overlay").checked = false; return; }
    const nodes = {}; live.nodes.forEach((n) => { nodes[n.id] = [n.lat, n.lon]; });
    live.links.forEach((l) => {
      if (!nodes[l.from] || !nodes[l.to]) return;
      const [a, b] = offsetLine(nodes[l.from], nodes[l.to], 4);
      const col = l.source === "free_flow" ? css("--text-muted") : l.ratio < 1.3 ? css("--good") : l.ratio < 2 ? css("--warning") : l.ratio < 3 ? css("--serious") : css("--critical");
      S.netHandles.push(S.P.polyline([a, b], { color: col, weight: l.source === "google" ? 4 : 6, dash: l.source === "google" ? [4, 4] : null }));
    });
    toast(`Optimiser network: ${live.links.filter((l) => l.source === "camera").length} links from cameras, ${live.links.filter((l) => l.source === "google").length} filled by Google (dashed), the rest free flow (grey)` + (live.synthetic ? ". Synthetic demo layout." : ""));
  }

  function refreshAll() { if (!S.P) return; renderList(); renderEditor(); renderPair(); draw(); }

  // ---------------------------------------------------------------- data
  async function loadCams() { S.cams = (await api("/api/map/cameras")).cameras; }
  async function loadPairs() {
    const q = S.win === "all" ? "" : `?since=${(Date.now() / 1000 - +S.win).toFixed(0)}`;
    S.pairs = await api("/api/map/section-speeds" + q);
    if (S.pair) S.pair = S.pairs.pairs.find((p) => p.from === S.pair.from && p.to === S.pair.to) || null;
  }

  async function tick() {
    if (S.busy || Date.now() - (S.drag || 0) < 5000) return;
    S.busy = true;
    try { await loadCams(); await loadPairs(); refreshAll(); } catch (e) { /* keep the last picture */ } finally { S.busy = false; }
  }

  async function start() {
    try { S.me = await api("/api/me"); } catch (e) { return; }
    if (!S.me.signed_in) { location.href = "/login"; return; }
    if (S.me.role === "driver") { location.href = "/portal"; return; }
    $("#who").textContent = (S.me.name || S.me.username || "") + (isAdmin() ? " (administrator)" : " (view only)");
    S.cfg = await api("/api/map/config");
    let provider = S.cfg.provider === "google" ? Google : Leaflet, note = "";
    try { await provider.load(S.cfg); }
    catch (e) {
      if (provider === Google) { note = e.message + "; showing OpenStreetMap instead. "; provider = Leaflet; await Leaflet.load(); }
      else { $("#map").innerHTML = `<p class="err" style="padding:16px">${esc(e.message)}</p>`; throw e; }
    }
    S.P = provider; provider.init($("#map"), S.cfg.centre, S.cfg.zoom, S.cfg);
    $("#provider-note").textContent = note + (provider === Google ? "Google Maps" : "OpenStreetMap — own measurements only");
    if (provider === Google) {
      $("#traffic-wrap").classList.remove("hidden");
      $("#traffic").onchange = (e) => Google.setTraffic(e.target.checked);
      $("#net-wrap").classList.remove("hidden");
      $("#net-overlay").onchange = (e) => drawNetOverlay(e.target.checked);
    }
    provider.onClick(() => { S.pair = null; renderPair(); });
    $("#window").onchange = (e) => { S.win = e.target.value; tick(); };
    renderLegend();
    if (isAdmin()) { try { renderSettings(await api("/api/settings/maps")); } catch (e) { /* shown as absent */ } } else renderSettings(null);
    await tick();
    provider.fit(S.cams.map(geoOf).filter(Boolean).map((g) => [g.lat, g.lon]));
    setInterval(tick, 15000);
  }

  document.addEventListener("DOMContentLoaded", () => { try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ } start(); });
})();
