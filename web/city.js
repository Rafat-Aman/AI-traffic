/* 3D city view (R-6): MapLibre GL + OpenFreeMap vector tiles with extruded buildings.
 *
 * GOOGLE RULE (R-8): Google Maps Platform terms forbid showing Google Maps content on a non-Google map. This page
 * is a non-Google map, so it NEVER asks the server for Google-supplemented values (no use_google parameter is ever
 * sent) and it defensively draws any link whose source is "google" as "no camera data". Google-supplemented times
 * appear only on the Google-provider view of /map and in text outside any map.
 *
 * Data: everything drawn comes from /api/optimize/* (live camera-measured state, recommendation, simulator, router).
 */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888";
  const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : (+x).toFixed(d);
  const STYLE = "https://tiles.openfreemap.org/styles/liberty";
  const SPACING_M = 7;
  const GREY = "#8a8a8a";

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.body !== undefined) { init.headers["content-type"] = "application/json"; init.body = JSON.stringify(opts.body); }
    const res = await fetch(path, init);
    if (res.status === 401) { location.href = "/login"; throw new Error("Sign in to continue"); }
    if (!res.ok) {
      let msg = res.statusText, problems = null;
      try {
        const j = await res.json(), d = j.detail;
        if (d && typeof d === "object" && d.problems) { msg = d.message; problems = d.problems; }
        else msg = typeof d === "string" ? d : JSON.stringify(d);
      } catch (e) { /* not json */ }
      const err = new Error(msg); err.problems = problems; throw err;
    }
    return res.json();
  }
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
    clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 4500);
  }

  // ---------------------------------------------------------------- state
  const S = { me: null, map: null, ready: false, tab: "live", net: null, live: null, rec: null, metric: "density",
    sim: null, simRun: 0, simIdx: 0, playing: false, route: null, routeSel: 0, pick: null, sel: null,
    draft: null, tool: "select", selNode: null, selRoad: null, roadFrom: null, dirty: false, busy: false };
  const isAdmin = () => S.me && S.me.role === "admin";

  // ---------------------------------------------------------------- geometry
  const R = 111195;
  const nodeMap = () => { const m = {}; ((S.live && S.live.nodes) || []).forEach((n) => { m[n.id] = n; }); return m; };
  const dOf = () => S.draft;                        // the document being edited, or null
  function square(lat, lon, m) {
    const dLat = m / R, dLon = m / (R * Math.cos(lat * Math.PI / 180));
    return [[lon - dLon, lat - dLat], [lon + dLon, lat - dLat], [lon + dLon, lat + dLat], [lon - dLon, lat + dLat], [lon - dLon, lat - dLat]];
  }
  function along(a, b, f) { return [a.lon + (b.lon - a.lon) * f, a.lat + (b.lat - a.lat) * f]; }

  // ---------------------------------------------------------------- colour scales (bands are for reading, not a standard)
  function band(metric, v) {
    if (v === null || v === undefined) return GREY;
    const c = [css("--good"), css("--warning"), css("--serious"), css("--critical")];
    if (metric === "density") return v < 15 ? c[0] : v < 30 ? c[1] : v < 50 ? c[2] : c[3];
    if (metric === "speed") return v >= 30 ? c[0] : v >= 20 ? c[1] : v >= 10 ? c[2] : c[3];
    return v < 1.3 ? c[0] : v < 2 ? c[1] : v < 3 ? c[2] : c[3];
  }
  function linkValue(l) {
    if (!l.measured) return null;
    if (S.metric === "density") return l.measured.density;
    if (S.metric === "speed") return l.measured.speed_kph;
    return l.ratio;
  }
  // Google rule: a link from Google is "no camera data" here, whatever the server sent.
  const camData = (l) => l.source === "camera";

  // ---------------------------------------------------------------- map layers
  const EMPTY = { type: "FeatureCollection", features: [] };
  const src = (id) => S.map.getSource(id);
  function setData(id, feats) { const s = S.ready && src(id); if (s) s.setData({ type: "FeatureCollection", features: feats }); }
  const pt = (c, props = {}) => ({ type: "Feature", properties: props, geometry: { type: "Point", coordinates: c } });
  const ln = (cs, props = {}) => ({ type: "Feature", properties: props, geometry: { type: "LineString", coordinates: cs } });
  const poly = (ring, props = {}) => ({ type: "Feature", properties: props, geometry: { type: "Polygon", coordinates: [ring] } });

  // The OpenFreeMap "liberty" basemap compares tile attributes that are null on some features
  // (poi rank, road ref_length, admin_level), which MapLibre logs as "Expected value to be of type number, but
  // found null instead" three times a load. Wrap those reads in coalesce with a fallback that makes the
  // comparison false, which is what a missing value should mean. Our own layers never use such a filter.
  function numSafe(f) {
    if (!Array.isArray(f)) return f;
    if (["<", "<=", ">", ">="].includes(f[0]) && Array.isArray(f[1]) && f[1][0] === "get" && typeof f[1][1] === "string") {
      const lowIsFalse = f[0] === ">" || f[0] === ">=";
      return [f[0], ["coalesce", f[1], lowIsFalse ? -1e9 : 1e9], ...f.slice(2).map(numSafe)];
    }
    return f.map(numSafe);
  }
  // Patch the style document before the map is built: tiles are parsed with whatever filters exist at that
  // moment, so patching the loaded map would be too late. Any failure falls back to the plain style URL.
  async function basemapStyle() {
    try {
      const st = await (await fetch(STYLE)).json();
      st.layers.forEach((l) => {
        if (l.filter) l.filter = numSafe(l.filter);
        if (l.type === "fill-extrusion" && l.paint) {
          const num = (k) => ["coalesce", ["get", k], 0];
          l.paint["fill-extrusion-height"] = num("render_height");
          l.paint["fill-extrusion-base"] = num("render_min_height");
        }
      });
      return st;
    } catch (e) { return STYLE; }
  }

  function initLayers() {
    const m = S.map;
    ["links", "wedges", "cols", "nodes", "lights", "veh", "routes", "labels", "draft-roads", "draft-nodes"].forEach((id) => m.addSource(id, { type: "geojson", data: EMPTY }));
    m.addLayer({ id: "wedges", type: "fill", source: "wedges", paint: { "fill-color": ["get", "color"], "fill-opacity": 0.22 } });
    m.addLayer({ id: "links-solid", type: "line", source: "links", filter: ["!=", ["get", "stale"], true],
      layout: { "line-cap": "round" }, paint: { "line-color": ["get", "color"], "line-width": ["get", "w"], "line-offset": 3, "line-opacity": 0.95 } });
    m.addLayer({ id: "links-stale", type: "line", source: "links", filter: ["==", ["get", "stale"], true],
      paint: { "line-color": ["get", "color"], "line-width": ["get", "w"], "line-offset": 3, "line-opacity": 0.5, "line-dasharray": [1.5, 1.5] } });
    m.addLayer({ id: "routes", type: "line", source: "routes", layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": ["get", "color"], "line-width": ["get", "w"], "line-opacity": 0.9 } });
    // fill-extrusion-opacity cannot be data-driven: an expression there is evaluated with no feature, logs
    // "Expected value to be of type number, but found null" and the layer is dropped. So stale cameras get their
    // own layer with a constant lower opacity.
    m.addLayer({ id: "cols", type: "fill-extrusion", source: "cols", filter: ["!=", ["get", "stale"], true],
      paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-height": ["get", "h"], "fill-extrusion-opacity": 0.9 } });
    m.addLayer({ id: "cols-stale", type: "fill-extrusion", source: "cols", filter: ["==", ["get", "stale"], true],
      paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-height": ["get", "h"], "fill-extrusion-opacity": 0.45 } });
    m.addLayer({ id: "nodes", type: "circle", source: "nodes",
      paint: { "circle-radius": ["get", "r"], "circle-color": ["get", "fill"], "circle-stroke-color": ["get", "stroke"], "circle-stroke-width": 2 } });
    m.addLayer({ id: "lights", type: "circle", source: "lights",
      paint: { "circle-radius": 5, "circle-color": ["get", "color"], "circle-stroke-color": "#111", "circle-stroke-width": 1 } });
    m.addLayer({ id: "veh", type: "circle", source: "veh", paint: { "circle-radius": ["get", "r"], "circle-color": ["get", "color"], "circle-stroke-color": "#111", "circle-stroke-width": 0.5 } });
    m.addLayer({ id: "labels", type: "symbol", source: "labels",
      layout: { "text-field": ["get", "t"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-offset": [0, 1.4], "text-anchor": "top", "text-allow-overlap": false },
      paint: { "text-color": "#111", "text-halo-color": "#fff", "text-halo-width": 1.4 } });
    m.addLayer({ id: "draft-roads", type: "line", source: "draft-roads", paint: { "line-color": ["get", "color"], "line-width": ["get", "w"], "line-opacity": 0.95 } });
    m.addLayer({ id: "draft-nodes", type: "circle", source: "draft-nodes",
      paint: { "circle-radius": ["get", "r"], "circle-color": ["get", "fill"], "circle-stroke-color": ["get", "stroke"], "circle-stroke-width": 2 } });
    m.on("click", "links-solid", (e) => onLink(e)); m.on("click", "links-stale", (e) => onLink(e));
    m.on("click", "cols", (e) => onCamera(e.features[0].properties.cid));
    m.on("click", "cols-stale", (e) => onCamera(e.features[0].properties.cid));
    m.on("click", "wedges", (e) => onCamera(e.features[0].properties.cid));
    m.on("click", "nodes", (e) => onNode(e.features[0].properties.id));
    ["links-solid", "links-stale", "cols", "cols-stale", "nodes", "draft-nodes", "draft-roads"].forEach((l) => {
      m.on("mouseenter", l, () => { m.getCanvas().style.cursor = "pointer"; });
      m.on("mouseleave", l, () => { m.getCanvas().style.cursor = ""; });
    });
    m.on("click", (e) => onMapClick(e));
    editorDrag();
  }

  function onLink(e) { if (S.tab === "edit") return; S.sel = { type: "link", id: e.features[0].properties.id }; renderInfo(); }
  function onCamera(cid) { S.sel = { type: "camera", id: +cid }; renderInfo(); }
  function onNode(id) {
    if (S.pick) { const k = S.pick; S.pick = null; $("#r-" + k).value = id; toast(`${k === "o" ? "Start" : "End"} set to ${id}`); return; }
    if (S.tab === "edit") return;
    S.sel = { type: "node", id }; renderInfo();
  }
  function onMapClick(e) {
    if (S.tab !== "edit" || !S.draft) return;
    const hit = S.map.queryRenderedFeatures(e.point, { layers: ["draft-nodes"] });
    const road = S.map.queryRenderedFeatures(e.point, { layers: ["draft-roads"] });
    if (S.tool === "node" && !hit.length) { addNode(e.lngLat.lat, e.lngLat.lng); return; }
    if (hit.length) {
      const id = hit[0].properties.id;
      if (S.tool === "road") {
        if (!S.roadFrom) { S.roadFrom = id; renderEdit(); }
        else if (S.roadFrom !== id) { addRoad(S.roadFrom, id); S.roadFrom = null; }
      } else if (S.tool === "delete") delNode(id);
      else { S.selNode = id; S.selRoad = null; renderEdit(); }
    } else if (road.length) {
      const i = +road[0].properties.idx;
      if (S.tool === "delete") { S.draft.roads.splice(i, 1); touch(); } else { S.selRoad = i; S.selNode = null; renderEdit(); }
    } else { S.selNode = S.selRoad = null; renderEdit(); }
  }

  // ---------------------------------------------------------------- drawing: live
  function drawLive() {
    if (!S.ready) return;
    const inEdit = S.tab === "edit" && S.draft;
    if (inEdit) { drawDraft(); ["links", "wedges", "cols", "nodes", "lights", "veh", "labels", "routes"].forEach((id) => setData(id, [])); drawCameras(true); return; }
    setData("draft-roads", []); setData("draft-nodes", []);
    const L = S.live; if (!L) { ["links", "wedges", "cols", "nodes", "lights", "veh", "labels"].forEach((id) => setData(id, [])); return; }
    const nm = nodeMap(), frame = simFrame(), feats = [];
    L.links.forEach((l) => {
      const a = nm[l.from], b = nm[l.to]; if (!a || !b) return;
      let color = GREY, w = 4, stale = false;
      if (frame) {
        const f = frame.links[l.id], storage = Math.max(1, l.lanes * l.length_m / SPACING_M), q = f ? f[1] : 0, n = f ? f[0] + f[1] : 0;
        color = n === 0 ? GREY : band("ratio", 1 + 4 * q / storage + n / storage * 0.3); w = 3 + Math.min(6, 8 * n / storage);
      } else if (camData(l)) { color = band(S.metric, linkValue(l)); stale = l.status === "stale"; w = 6; }
      feats.push(ln([[a.lon, a.lat], [b.lon, b.lat]], { id: l.id, color, w, stale }));
    });
    setData("links", feats);
    drawCameras(false);
    const nf = [];
    L.nodes.forEach((n) => {
      const sig = n.kind === "signal", bnd = n.kind === "boundary";
      nf.push(pt([n.lon, n.lat], { id: n.id, r: sig ? 7 : bnd ? 5 : 5, fill: sig ? css("--accent") : bnd ? "#ffffff" : GREY, stroke: "#111" }));
    });
    setData("nodes", nf);
    setData("labels", [...L.nodes.filter((n) => n.name && n.kind !== "boundary").map((n) => pt([n.lon, n.lat], { t: n.name })),
      ...L.cameras.filter((c) => c.geo).map((c) => pt([c.geo.lon, c.geo.lat], { t: c.name }))].slice(0, 80));
    drawLights(frame);
    drawVehicles(frame);
    drawRoutes();
  }

  function drawCameras(flat) {
    const cams = (S.live && S.live.cameras) || (S.net && S.net.cameras) || [];
    const wf = [], cf = [];
    cams.forEach((c) => {
      if (!c.geo) return;
      const g = c.geo, stale = c.stale, none = c.density === null || c.density === undefined;
      const color = stale || none ? GREY : band("density", c.density);
      wf.push(poly(g.wedge.map((p) => [p[1], p[0]]).concat([[g.wedge[0][1], g.wedge[0][0]]]), { color: css("--accent"), cid: c.id }));
      if (!flat) cf.push(poly(square(g.lat, g.lon, 7), { color, cid: c.id, h: none ? 3 : 6 + Math.min(150, c.density * 3.5), stale: !!stale }));
      else cf.push(poly(square(g.lat, g.lon, 7), { color: css("--accent"), cid: c.id, h: 8, stale: false }));
    });
    setData("wedges", wf); setData("cols", cf);
  }

  function phaseLinks(node) {          // from the recommendation: which in-links each phase serves
    const s = S.rec && S.rec.signals.find((x) => x.node === node); return s ? s.phases : [];
  }
  function drawLights(frame) {
    const L = S.live, nm = nodeMap(), feats = [];
    if (!L) return;
    L.nodes.filter((n) => n.kind === "signal").forEach((n) => {
      const phases = phaseLinks(n.id), sig = S.rec && S.rec.signals.find((x) => x.node === n.id);
      L.links.filter((l) => l.to === n.id).forEach((l) => {
        const a = nm[l.from]; if (!a) return;
        let color = GREY;
        if (frame) {
          const p = frame.signals[n.id];
          color = p === undefined ? GREY : p < 0 ? css("--warning") : (phases[p] && phases[p].links.includes(l.id)) ? css("--good") : css("--critical");
        } else if (sig && sig.favoured_phase !== null && sig.favoured_phase !== undefined) {
          color = phases[sig.favoured_phase] && phases[sig.favoured_phase].links.includes(l.id) ? css("--good") : css("--critical");
        }
        feats.push(pt(along(a, n, 0.88), { color }));
      });
    });
    setData("lights", feats);
  }
  function drawVehicles(frame) {
    if (!frame || !S.live) return setData("veh", []);
    const nm = nodeMap(), feats = [];
    S.live.links.forEach((l) => {
      const f = frame.links[l.id], a = nm[l.from], b = nm[l.to]; if (!f || !a || !b) return;
      const moving = Math.min(f[0], 14), queued = Math.min(f[1], 24), len = Math.max(l.length_m, 1);
      for (let i = 0; i < queued; i++) feats.push(pt(along(a, b, Math.max(0.05, 0.97 - i * SPACING_M / len)), { r: 3, color: css("--critical") }));
      for (let i = 0; i < moving; i++) feats.push(pt(along(a, b, ((i + 0.5) / moving) * Math.max(0.05, 0.9 - queued * SPACING_M / len)), { r: 3, color: css("--good") }));
    });
    (frame.evs || []).forEach((e) => {
      const l = S.live.links.find((x) => x.id === e.link), a = l && nm[l.from], b = l && nm[l.to];
      if (a && b) feats.push(pt(along(a, b, e.frac), { r: 7, color: "#e91e63" }));
    });
    setData("veh", feats);
  }
  function drawRoutes() {
    const r = S.route; if (!r || !S.live) return setData("routes", []);
    const nm = nodeMap(), cols = [css("--accent"), "#a15bd6", "#e07b00", "#2a9d8f"], feats = [];
    r.routes.forEach((rt, i) => {
      const cs = rt.nodes.map((id) => nm[id]).filter(Boolean).map((n) => [n.lon, n.lat]);
      if (cs.length > 1) feats.push(ln(cs, { color: cols[i % 4], w: i === S.routeSel ? 9 : 4 }));
    });
    (r.incidents || []).forEach((inc) => {
      const l = S.live.links.find((x) => x.id === inc.link), a = l && nm[l.from], b = l && nm[l.to];
      if (a && b) feats.push(ln([[a.lon, a.lat], [b.lon, b.lat]], { color: css("--critical"), w: 12 }));
    });
    setData("routes", feats);
  }

  // ---------------------------------------------------------------- panels: live / info
  function sourceNote() {
    const n = S.net;
    if (!n) return "";
    if (n.source === "demo") return "No network has been drawn: this is the SYNTHETIC demo network around the placed cameras. Only the camera positions are real.";
    if (n.synthetic) return "This saved network is marked SYNTHETIC (generated as a demo): only the camera positions are real.";
    return "";
  }
  function renderBanner() {
    const L = S.live, notes = [];
    const s = sourceNote();
    $("#synthetic-chip").classList.toggle("hidden", !s);
    if (s) notes.push(`<div class="note warn-note"><b>Synthetic.</b> ${esc(s)}</div>`);
    if (S.simFrame) notes.push('<div class="note"><b>Simulation playback</b> - synthetic traffic, not measured.</div>');
    notes.push('<div class="note">Colours show <b>our cameras\' measurements only</b>. Grey = no camera data. No Google data is shown on this map.</div>');
    $("#banner").innerHTML = notes.join("");
  }
  function renderLegend() {
    const row = (c, t) => `<div class="legend-row"><span class="swatch" style="background:${c}"></span>${t}</div>`;
    const tx = { density: ["under 15", "15-30", "30-50", "50 and over", "veh/km/lane"], speed: ["30 and over", "20-30", "10-20", "under 10", "km/h"],
      ratio: ["under 1.3x", "1.3-2x", "2-3x", "3x and over", "travel time / free flow"] }[S.metric];
    $("#legend").innerHTML = `<b>Links: camera ${esc(tx[4])}</b>
      ${row(css("--good"), tx[0])}${row(css("--warning"), tx[1])}${row(css("--serious"), tx[2])}${row(css("--critical"), tx[3])}
      ${row(GREY, "no camera data (free flow assumed)")}<div class="legend-row"><span class="swatch dash"></span>camera data older than ${S.live ? S.live.fresh_s : 120} s</div>
      <div class="legend-row"><span class="dotsw" style="background:${css("--accent")}"></span>signal &nbsp;<span class="dotsw" style="background:#fff;border:2px solid #111"></span>entry/exit</div>
      <div class="muted">Columns at cameras: height = measured density. Bands are for reading the map, not a standard.</div>`;
  }
  function renderHonest() {
    $("#p-honest").innerHTML = `<b>Read this first</b><ul style="margin:6px 0 0 16px;padding:0">
      <li>Simulator results are <b>synthetic</b> (a queue model written for this project, not SUMO, not calibrated to Dhaka).</li>
      <li><b>Max pressure</b> is an established adaptive rule (Varaiya 2013), not a trained model.</li>
      <li>1200 vehicles/h/lane saturation flow is an <b>assumption</b>.</li>
      <li>Signal timings are <b>suggestions</b>. Nothing is applied to any real signal, and the real signal state is not read.</li>
      <li>Camera data covers a short stretch of a road; its value is assumed to hold along the whole link.</li></ul>`;
  }
  function renderLive() {
    const L = S.live, el = $("#p-live");
    if (!L) { el.innerHTML = '<div class="card-head"><h2>Live traffic</h2></div><p class="muted">' + esc(S.liveErr || "Loading...") + "</p>"; return; }
    const cams = L.cameras.filter((c) => c.placed);
    const nlive = L.links.filter((l) => l.status === "live").length, nstale = L.links.filter((l) => l.status === "stale").length;
    el.innerHTML = `<div class="card-head"><h2>Live traffic</h2></div>
      <p class="small">${L.links.length} directed links: <b>${nlive}</b> measured now, <b>${nstale}</b> stale, ${L.links.length - nlive - nstale} with no camera data.</p>
      <table class="t"><tr><th>Camera</th><th>Density</th><th>Age</th><th></th></tr>
      ${cams.map((c) => `<tr data-cam="${c.id}" style="cursor:pointer"><td>${esc(c.name)}</td><td>${c.density === null ? "no data" : fmt(c.density, 1)}</td>
        <td>${c.zones.filter((z) => z.age_s !== null && z.age_s !== undefined).length ? fmt(Math.min(...c.zones.filter((z) => z.age_s != null).map((z) => z.age_s)), 0) + " s" : "-"}</td>
        <td>${c.stale ? '<span class="chip small">stale</span>' : c.fresh ? "live" : ""}</td></tr>`).join("") || '<tr><td colspan="4" class="muted">No camera is placed on the map.</td></tr>'}</table>
      ${L.notes.length ? `<details class="small muted"><summary>${L.notes.length} note(s) from the optimiser</summary>${L.notes.map((n) => `<div>${esc(n)}</div>`).join("")}</details>` : ""}
      <p class="muted small">Updated ${new Date(L.generated_at * 1000).toLocaleTimeString()}. Density is vehicles per km per lane from the camera's road zone.</p>`;
    el.querySelectorAll("[data-cam]").forEach((r) => r.onclick = () => { onCamera(r.dataset.cam); const c = cams.find((x) => x.id === +r.dataset.cam); if (c && c.geo) S.map.easeTo({ center: [c.geo.lon, c.geo.lat], zoom: 17, pitch: 65, duration: 800 }); });
  }
  function renderInfo() {
    const el = $("#p-info"), s = S.sel, L = S.live;
    if (!s || !L) { el.classList.add("hidden"); return; }
    el.classList.remove("hidden");
    let h = "";
    if (s.type === "link") {
      const l = L.links.find((x) => x.id === s.id); if (!l) return el.classList.add("hidden");
      const m = l.measured;
      h = `<div class="card-head"><h2>Link ${esc(l.id)}</h2><span class="spacer"></span><button class="btn small ghost" id="info-x" aria-label="Close">x</button></div>
        <dl class="kv"><dt>Length / lanes</dt><dd>${fmt(l.length_m, 0)} m / ${l.lanes}</dd><dt>Free-flow time</dt><dd>${fmt(l.free_flow_s, 0)} s</dd>
        <dt>Travel time now</dt><dd>${camData(l) ? fmt(l.travel_time_s, 0) + " s (" + fmt(l.delay_s, 0) + " s delay)" : "no camera data"}</dd>
        <dt>Data</dt><dd>${camData(l) ? (l.status === "stale" ? "camera, STALE " : "camera, ") + fmt(l.age_s, 0) + " s old" : "none: free flow assumed"}</dd>
        ${camData(l) && m ? `<dt>Density</dt><dd>${fmt(m.density, 1)} veh/km/lane</dd><dt>Speed</dt><dd>${fmt(m.speed_kph, 1)} km/h</dd>
          <dt>Queue</dt><dd>${fmt(m.queue_m, 0)} m (${fmt(m.queue_veh, 1)} veh)</dd><dt>Occupancy</dt><dd>${fmt(m.occupancy, 2)}</dd><dt>Waiting</dt><dd>${fmt(m.wait_s, 0)} s</dd>` : ""}
        ${l.camera_zone ? `<dt>Camera zone</dt><dd>${esc(l.camera_zone)} (camera ${l.camera_id ?? "any"})</dd>` : ""}</dl>`;
    } else if (s.type === "camera") {
      const c = L.cameras.find((x) => x.id === s.id); if (!c) return el.classList.add("hidden");
      h = `<div class="card-head"><h2>${esc(c.name)}</h2><span class="spacer"></span><button class="btn small ghost" id="info-x" aria-label="Close">x</button></div>
        <p class="small muted">Camera ${c.id}, ${esc(c.state)}. Column height = measured density.</p>
        <table class="t"><tr><th>Zone</th><th>Dens.</th><th>Speed</th><th>Queue</th><th>Age</th></tr>
        ${c.zones.map((z) => z.usable ? `<tr><td>${esc(z.zone)}</td><td>${fmt(z.density, 1)}</td><td>${fmt(z.speed_kph, 0)}</td><td>${fmt(z.queue_m, 0)} m</td><td>${z.age_s == null ? "-" : fmt(z.age_s, 0) + " s"}${z.fresh ? "" : " (stale)"}</td></tr>`
          : `<tr><td>${esc(z.zone)}</td><td colspan="4" class="muted">no usable sample</td></tr>`).join("") || '<tr><td colspan="5" class="muted">no live zone data (is the camera running with impact on?)</td></tr>'}</table>
        <p class="small">Attached to: ${c.links.length ? c.links.map(esc).join(", ") : "no link yet"}</p>`;
    } else if (s.type === "node") {
      const n = L.nodes.find((x) => x.id === s.id), sig = S.rec && S.rec.signals.find((x) => x.node === s.id);
      h = `<div class="card-head"><h2>${esc(n.name || n.id)}</h2><span class="spacer"></span><button class="btn small ghost" id="info-x" aria-label="Close">x</button></div>
        <p class="small muted">${esc(n.kind)} (${esc(n.id)})</p>${sig ? recHtml(sig) : ""}`;
    }
    el.innerHTML = h;
    $("#info-x").onclick = () => { S.sel = null; renderInfo(); };
  }

  // ---------------------------------------------------------------- panel: recommendation
  function recHtml(s) {
    const mx = Math.max(...s.phases.map((p) => p.green_s), 1);
    return `<p class="small">${esc(s.message)}</p><p class="small">Cycle ${fmt(s.cycle_s, 0)} s (${fmt(s.lost_s, 0)} s clearance), demand: <b>${esc(s.demand)}</b></p>
      ${s.phases.map((p, i) => `<div class="phase-row"><span style="width:56px">${i === s.favoured_phase ? "<b>" : ""}${esc(p.name)}${i === s.favoured_phase ? "</b>" : ""}</span>
        <span class="pbar"><i style="width:${(100 * p.green_s / mx).toFixed(0)}%"></i></span><b>${fmt(p.green_s, 0)} s</b></div>
        <div class="muted small" style="margin:0 0 4px 62px">serves ${p.links.map(esc).join(", ") || "-"} · queue ${fmt(p.queue_veh, 1)} veh</div>`).join("")}`;
  }
  function renderRec() {
    const R_ = S.rec, el = $("#p-rec");
    if (!R_) { el.innerHTML = '<div class="card-head"><h2>Signal timing recommendation</h2></div><p class="muted">Loading...</p>'; return; }
    el.innerHTML = `<div class="card-head"><h2>Signal timing recommendation</h2></div>
      <p class="small"><b>${esc(R_.disclaimer)}</b> ${esc(R_.method)}</p>
      <p class="small muted">The green dots on the map show the phase with the most green in this suggestion ("favoured"), not the real signal's state, which this system does not read.</p>
      ${R_.signals.map((s) => `<h3 data-node="${esc(s.node)}" style="cursor:pointer">${esc(s.name || s.node)} <span class="muted small">(${s.approaches_measured}/${s.approaches} approaches measured)</span></h3>${recHtml(s)}`).join("") || '<p class="muted">The network has no signalised junction.</p>'}`;
    el.querySelectorAll("[data-node]").forEach((h) => h.onclick = () => { const n = nodeMap()[h.dataset.node]; if (n) S.map.easeTo({ center: [n.lon, n.lat], zoom: 17, pitch: 60, duration: 700 }); });
  }

  // ---------------------------------------------------------------- panel: simulation
  function simFrame() {
    if (S.tab !== "sim" || !S.sim) return null;
    const r = S.sim.runs[S.simRun]; return r && r.frames ? r.frames[Math.min(S.simIdx, r.frames.length - 1)] : null;
  }
  function nodeOptions(kinds) {
    const ns = ((S.live && S.live.nodes) || []).filter((n) => !kinds || kinds.includes(n.kind));
    return ns.map((n) => `<option value="${esc(n.id)}">${esc(n.name || n.id)} (${esc(n.id)})</option>`).join("");
  }
  const noNet = (what) => `<p class="small warn" id="no-net">${what} needs a network: there is none yet. Place cameras on the <a href="/map">map</a> then draw a network in the Edit tab (administrators) or use "Generate demo around my cameras".</p>`;
  function renderSim() {
    const el = $("#p-sim"), hasNet = S.live && S.live.nodes.length;
    const sim = S.sim;
    el.innerHTML = `<div class="card-head"><h2>Simulate: fixed-time vs max pressure</h2></div>
      <p class="small"><b>Synthetic.</b> A queue simulator run on this network's layout with invented demand. Both controllers see exactly the same arrivals.</p>
      <div class="form-grid"><label>Horizon (s)<input id="s-h" type="number" min="60" max="900" step="30" value="${sim ? sim.params.horizon_s : 300}"></label>
        <label>Load (x)<input id="s-l" type="number" min="0.1" max="2" step="0.1" value="${sim ? sim.params.load : 0.8}"></label>
        <label>Seed<input id="s-seed" type="number" min="0" step="1" value="${sim ? sim.params.seed : 1}"></label>
        <label>Emergency vehicle<select id="s-ev"><option value="">none</option><option value="1" ${sim && sim.params.ev ? "selected" : ""}>one ambulance</option></select></label>
        <label class="ev-f">From<select id="s-o">${nodeOptions(["boundary"])}</select></label>
        <label class="ev-f">To<select id="s-d">${nodeOptions(["boundary"])}</select></label></div>
      <div class="row"><button class="btn primary small" id="s-run" ${hasNet ? "" : "disabled"}>Run simulation</button> <span class="muted small" id="s-msg"></span></div>
      ${hasNet ? "" : noNet("Simulation")}
      ${sim ? simResultHtml() : ""}`;
    const evOn = () => $("#s-ev").value === "1";
    const sync = () => el.querySelectorAll(".ev-f").forEach((x) => x.classList.toggle("hidden", !evOn()));
    $("#s-ev").onchange = sync; sync();
    const d = $("#s-d"); if (d && d.options.length > 1) d.selectedIndex = d.options.length - 1;
    $("#s-run").onclick = runSim;
    if (sim) wireSim();
  }
  async function runSim() {
    const btn = $("#s-run"), msg = $("#s-msg");
    const body = { horizon_s: +$("#s-h").value, load: +$("#s-l").value, seed: +$("#s-seed").value, frames: true };
    if ($("#s-ev").value === "1") body.ev = { origin: $("#s-o").value, dest: $("#s-d").value, depart_s: Math.round(body.horizon_s * 0.3) };
    btn.disabled = true; msg.textContent = "Running (can take up to a minute)...";
    try { S.sim = await api("/api/optimize/simulate", { method: "POST", body }); S.simRun = 0; S.simIdx = 0; S.playing = false; S.sim.params.horizon_s = body.horizon_s; renderSim(); renderPlay(); drawLive(); }
    catch (e) { msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; btn.disabled = false; }
  }
  function simResultHtml() {
    const sim = S.sim, rows = [["Vehicles generated", "generated", 0], ["Finished", "finished", 0], ["Unfinished at the end", "unfinished", 0], ["Mean delay (s)", "mean_delay_s", 1],
      ["95th pct delay (s)", "p95_delay_s", 1], ["Throughput (veh/h)", "throughput_vph", 0], ["Max queue (veh)", "max_queue_veh", 0], ["Forced releases", "forced_releases", 0]];
    const c = sim.comparison, pct = (v) => (v > 0 ? "+" : "") + v + "%";
    const evRows = sim.runs.some((r) => r.ev.length) ? `<tr><td>Ambulance trip (s)</td>${sim.runs.map((r) => `<td>${r.ev.length ? fmt(r.ev[0].travel_time_s, 0) : "-"}</td>`).join("")}</tr>` : "";
    return `<h3>Summary (synthetic)</h3>
      <table class="t"><tr><th></th>${sim.runs.map((r, i) => `<th class="${i === S.simRun ? "sel" : ""}">${esc(r.label.split(" (")[0])}</th>`).join("")}</tr>
      ${rows.map(([t, k, d]) => `<tr><td>${t}</td>${sim.runs.map((r) => `<td>${fmt(r.summary[k], d)}</td>`).join("")}</tr>`).join("")}${evRows}</table>
      ${"max_pressure_vs_fixed_mean_delay_pct" in c ? `<p class="small">Max pressure vs fixed-time, mean delay: <b>${pct(c.max_pressure_vs_fixed_mean_delay_pct)}</b> (one seed; the project's multi-seed study is in tools/signal_sim.py).</p>` : ""}
      ${"corridor_ev_time_pct" in c ? `<p class="small">Emergency corridor: ambulance time <b>${pct(c.corridor_ev_time_pct)}</b>, other vehicles' delay <b>${pct(c.corridor_others_delay_pct ?? 0)}</b>.</p>` : ""}
      <h3>Total queue over time</h3>${spark()}
      <h3>Show on the map</h3><div class="stack" id="s-pick">${sim.runs.map((r, i) => `<label style="flex-direction:row;gap:6px"><input type="radio" name="run" value="${i}" ${i === S.simRun ? "checked" : ""}> ${esc(r.label)}</label>`).join("")}</div>
      <p class="muted small">Red dots = queued vehicles, green = moving, signal dots = phase being shown (amber = clearance), pink = ambulance. Computed in ${sim.elapsed_s} s. ${esc(sim.notes[sim.notes.length - 1])}</p>`;
  }
  function spark() {
    const runs = S.sim.runs, W = 300, H = 60, cols = [css("--text-muted"), css("--accent"), "#e07b00"];
    const T = Math.max(1, ...runs.map((r) => r.queue_series.length ? r.queue_series[r.queue_series.length - 1][0] : 1)), Q = Math.max(1, ...runs.map((r) => Math.max(0, ...r.queue_series.map((p) => p[1]))));
    const paths = runs.map((r, i) => `<polyline fill="none" stroke="${cols[i % 3]}" stroke-width="1.6" points="${r.queue_series.map((p) => `${(p[0] / T * W).toFixed(1)},${(H - p[1] / Q * H).toFixed(1)}`).join(" ")}"/>`).join("");
    return `<svg class="spark" viewBox="0 0 ${W} ${H + 14}" preserveAspectRatio="none" role="img" aria-label="Total queue over time per controller">${paths}
      <text x="0" y="${H + 11}" font-size="9" fill="${css("--text-muted")}">0 s</text><text x="${W - 40}" y="${H + 11}" font-size="9" fill="${css("--text-muted")}">${T.toFixed(0)} s · max ${Q.toFixed(0)} veh</text></svg>
      <div class="muted small">${runs.map((r, i) => `<span style="color:${cols[i % 3]}">&#9632;</span> ${esc(r.label.split(" (")[0])}`).join(" &nbsp; ")}</div>`;
  }
  function wireSim() {
    document.querySelectorAll("#s-pick input").forEach((r) => r.onchange = () => { S.simRun = +r.value; S.simIdx = 0; renderSim(); renderPlay(); drawLive(); });
  }
  let playTimer = null;
  function renderPlay() {
    const el = $("#play"), r = S.sim && S.sim.runs[S.simRun];
    if (S.tab !== "sim" || !r || !r.frames || !r.frames.length) { el.classList.add("hidden"); clearInterval(playTimer); S.playing = false; renderBanner(); return; }
    S.simFrame = true; renderBanner();
    el.classList.remove("hidden");
    const n = r.frames.length;
    el.innerHTML = `<button class="btn small" id="p-btn">${S.playing ? "Pause" : "Play"}</button>
      <input type="range" id="p-rng" min="0" max="${n - 1}" value="${S.simIdx}" aria-label="Simulation time">
      <span class="small mono" id="p-t">${fmt(r.frames[S.simIdx].t, 0)} s</span><span class="small muted">${esc(r.label.split(" (")[0])}</span>`;
    $("#p-btn").onclick = () => { S.playing = !S.playing; clearInterval(playTimer); if (S.playing) playTimer = setInterval(() => { S.simIdx = (S.simIdx + 1) % n; $("#p-rng").value = S.simIdx; $("#p-t").textContent = fmt(r.frames[S.simIdx].t, 0) + " s"; drawLive(); }, 150); renderPlay(); };
    $("#p-rng").oninput = (e) => { S.simIdx = +e.target.value; $("#p-t").textContent = fmt(r.frames[S.simIdx].t, 0) + " s"; drawLive(); };
  }

  // ---------------------------------------------------------------- panel: routing
  function renderRoute() {
    const el = $("#p-route"), R_ = S.route;
    el.innerHTML = `<div class="card-head"><h2>Route on live travel times</h2></div>
      <p class="small">Fastest routes use camera-measured travel times where a fresh measurement exists and free flow elsewhere. Recent recorded violations with a scored impact add an incident delay.</p>
      <div class="form-grid"><label>From<select id="r-o">${nodeOptions()}</select></label><label>To<select id="r-d">${nodeOptions()}</select></label>
        <label>Alternatives<select id="r-k"><option>1</option><option>2</option><option selected>3</option><option>4</option></select></label>
        <label>Incidents from the last (min)<input id="r-w" type="number" min="1" max="720" value="30"></label></div>
      <div class="row"><button class="btn small" id="r-po">Pick start on map</button> <button class="btn small" id="r-pd">Pick end on map</button>
        <button class="btn primary small" id="r-go" ${S.live && S.live.nodes.length ? "" : "disabled"}>Find routes</button></div>${S.live && S.live.nodes.length ? "" : noNet("Routing")}<div id="r-msg"></div>
      ${R_ ? routeHtml() : ""}`;
    const d = $("#r-d"); if (d && d.options.length > 1) d.selectedIndex = d.options.length - 1;
    $("#r-po").onclick = () => { S.pick = "o"; toast("Click the start intersection on the map"); };
    $("#r-pd").onclick = () => { S.pick = "d"; toast("Click the end intersection on the map"); };
    $("#r-go").onclick = findRoute;
    el.querySelectorAll(".route-card").forEach((c) => c.onclick = () => { S.routeSel = +c.dataset.i; renderRoute(); drawRoutes(); });
  }
  async function findRoute() {
    const msg = $("#r-msg");
    try {
      S.route = await api("/api/optimize/route", { method: "POST", body: { origin: $("#r-o").value, dest: $("#r-d").value, k: +$("#r-k").value, incident_window_min: +$("#r-w").value } });
      S.routeSel = 0; renderRoute(); drawRoutes();
    } catch (e) { msg.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
  function routeHtml() {
    const R_ = S.route, mm = (s) => (s / 60).toFixed(1) + " min";
    const inc = R_.incidents.length ? `<h3>Incidents counted</h3>${R_.incidents.map((i) => `<div class="small">${esc(i.link)}: +${fmt(i.extra_delay_s, 0)} s from case #${i.violation_id} (${esc(i.code)}, ${fmt(i.age_s / 60, 0)} min ago)</div>`).join("")}` : '<p class="muted small">No recent incident on the roads cameras watch.</p>';
    const cmp = R_.incident_comparison ? `<p class="small">${R_.incident_comparison.usual_route_closed ? "The usual route is closed." : `Going round the incident saves <b>${fmt(R_.incident_comparison.saving_s, 0)} s</b> against the usual route.`}</p>` : "";
    const cards = R_.routes.map((r, i) => {
      const mix = r.source_mix, total = r.breakdown.length;
      const seg = (k, c) => `<i style="width:${(100 * (mix[k] || 0) / total).toFixed(0)}%;background:${c}"></i>`;
      return `<div class="route-card ${i === S.routeSel ? "sel" : ""}" data-i="${i}"><b>${esc(r.label)}</b>: ${mm(r.time_s)} (${fmt(r.length_m / 1000, 2)} km, ${fmt(r.delay_s, 0)} s over free flow${r.extra_delay_s ? `, ${fmt(r.extra_delay_s, 0)} s incident` : ""})
        <div class="bar" title="camera / no data">${seg("camera", css("--good"))}${seg("free_flow", GREY)}</div>
        <div class="muted small">${mix.camera || 0} of ${total} links measured by cameras${r.stale_legs ? `, ${r.stale_legs} stale` : ""}; the rest free flow assumed.</div>
        ${i === S.routeSel ? `<table class="t"><tr><th>Link</th><th>Time</th><th>Data</th><th>Age</th></tr>${r.breakdown.map((b) => `<tr><td>${esc(b.link)}</td><td>${fmt(b.travel_time_s, 0)}${b.extra_delay_s ? " +" + fmt(b.extra_delay_s, 0) : ""} s</td>
          <td>${esc(b.data_source)}</td><td>${b.age_s == null ? "-" : fmt(b.age_s, 0) + " s"}${b.status === "stale" ? " stale" : ""}</td></tr>`).join("")}</table>` : ""}</div>`;
    }).join("");
    return `<h3>Routes</h3>${cards}${cmp}${inc}`;
  }

  // ---------------------------------------------------------------- panel: editor (admin)
  const clone = (x) => JSON.parse(JSON.stringify(x));
  function startEdit() {
    if (!S.draft) {
      const d = S.net && S.net.doc ? clone(S.net.doc) : { name: "My network", synthetic: false, nodes: [], roads: [] };
      S.draft = d; S.dirty = false; S.tool = "select"; S.selNode = S.selRoad = S.roadFrom = null;
    }
  }
  const touch = () => { S.dirty = true; renderEdit(); drawLive(); };
  function nextId(prefix) { const used = new Set(S.draft.nodes.map((n) => n.id)); let i = 1; while (used.has(prefix + i)) i++; return prefix + i; }
  function addNode(lat, lon) {
    const kind = $("#e-kind") ? $("#e-kind").value : "signal";
    const id = nextId(kind === "boundary" ? "B" : kind === "signal" ? "J" : "U");
    S.draft.nodes.push({ id, kind, lat: +lat.toFixed(7), lon: +lon.toFixed(7), name: "" }); S.selNode = id; touch();
  }
  function addRoad(a, b) {
    if (S.draft.roads.some((r) => (r.from === a && r.to === b) || (r.from === b && r.to === a))) { toast("Those two are already joined by a road"); return; }
    S.draft.roads.push({ from: a, to: b, name: "", lanes: 2, speed_kph: 40, length_m: null, two_way: true, cameras: [] }); S.selRoad = S.draft.roads.length - 1; S.selNode = null; touch();
  }
  function delNode(id) { S.draft.nodes = S.draft.nodes.filter((n) => n.id !== id); S.draft.roads = S.draft.roads.filter((r) => r.from !== id && r.to !== id); S.selNode = null; S.selRoad = null; touch(); }
  function drawDraft() {
    const D = S.draft, nm = {}; D.nodes.forEach((n) => { nm[n.id] = n; });
    setData("draft-roads", D.roads.map((r, i) => nm[r.from] && nm[r.to] ? ln([[nm[r.from].lon, nm[r.from].lat], [nm[r.to].lon, nm[r.to].lat]],
      { idx: i, color: i === S.selRoad ? css("--accent") : r.cameras.length ? css("--good") : "#555", w: i === S.selRoad ? 7 : 5 }) : null).filter(Boolean));
    setData("draft-nodes", D.nodes.map((n) => pt([n.lon, n.lat], { id: n.id, r: 8, fill: n.kind === "signal" ? css("--accent") : n.kind === "boundary" ? "#fff" : GREY,
      stroke: n.id === S.selNode || n.id === S.roadFrom ? "#ff3b8b" : "#111" })));
    setData("labels", D.nodes.map((n) => pt([n.lon, n.lat], { t: n.name || n.id })));
  }
  function editorDrag() {
    const m = S.map; let drag = null;
    m.on("mousedown", "draft-nodes", (e) => { if (S.tool !== "select" || S.tab !== "edit") return; e.preventDefault(); drag = e.features[0].properties.id; m.dragPan.disable(); S.selNode = drag; S.selRoad = null; });
    m.on("mousemove", (e) => { if (!drag) return; const n = S.draft.nodes.find((x) => x.id === drag); if (n) { n.lat = +e.lngLat.lat.toFixed(7); n.lon = +e.lngLat.lng.toFixed(7); S.dirty = true; drawDraft(); } });
    const up = () => { if (drag) { drag = null; m.dragPan.enable(); renderEdit(); } };
    m.on("mouseup", up); m.on("touchend", up);
  }
  function cameraOptions() {
    return ((S.net && S.net.cameras) || []).filter((c) => c.placed).map((c) => `<option value="${c.id}">${esc(c.name)} (#${c.id})</option>`).join("");
  }
  function renderEdit() {
    const el = $("#p-edit"); if (S.tab !== "edit") return;
    startEdit();
    const D = S.draft, tool = (t, label) => `<button class="btn small ${S.tool === t ? "primary" : ""}" data-tool="${t}">${label}</button>`;
    const n = D.nodes.find((x) => x.id === S.selNode), r = D.roads[S.selRoad];
    let sel = '<p class="muted small">Select Intersection to add one by clicking the map; Road, then click two intersections to join them. Drag intersections in Select mode.</p>';
    if (n) sel = `<h3>Intersection ${esc(n.id)}</h3><div class="form-grid"><label>Name<input id="n-name" value="${esc(n.name)}"></label>
        <label>Kind<select id="n-kind">${["signal", "uncontrolled", "boundary"].map((k) => `<option ${k === n.kind ? "selected" : ""}>${k}</option>`).join("")}</select></label></div>
        <p class="muted small">signal: has phases and is controlled. uncontrolled: all movements always flow. boundary: where traffic enters or leaves the modelled area.</p>
        <button class="btn small danger" id="n-del">Delete intersection</button>`;
    if (r) sel = `<h3>Road ${esc(r.from)} - ${esc(r.to)}</h3><div class="form-grid"><label class="wide">Name<input id="r-name" value="${esc(r.name)}"></label>
        <label>Lanes (each way)<input id="r-lanes" type="number" min="1" max="8" value="${r.lanes}"></label><label>Speed limit (km/h)<input id="r-speed" type="number" min="5" max="120" value="${r.speed_kph}"></label>
        <label>Length (m, blank = straight line)<input id="r-len" type="number" min="5" value="${r.length_m ?? ""}"></label>
        <label>Two-way<select id="r-two"><option value="1" ${r.two_way ? "selected" : ""}>yes</option><option value="0" ${r.two_way ? "" : "selected"}>no (${esc(r.from)} to ${esc(r.to)} only)</option></select></label></div>
        <h3>Cameras on this road</h3>${r.cameras.map((c, i) => `<div class="small">camera #${c.camera_id}, zone "${esc(c.zone)}", watches ${c.dir === "fwd" ? esc(r.from) + " to " + esc(r.to) : esc(r.to) + " to " + esc(r.from)} <button class="btn small ghost" data-rmcam="${i}">remove</button></div>`).join("") || '<p class="muted small">none</p>'}
        <div class="form-grid"><label>Camera<select id="c-id">${cameraOptions()}</select></label><label>Road zone name<input id="c-zone" list="c-zones" placeholder="as named in the camera's zones"><datalist id="c-zones"></datalist></label>
          <label>Direction watched<select id="c-dir"><option value="fwd">${esc(r.from)} to ${esc(r.to)}</option>${r.two_way ? `<option value="back">${esc(r.to)} to ${esc(r.from)}</option>` : ""}</select></label>
          <label>&nbsp;<button class="btn small" id="c-add">Attach camera</button></label></div>
        <button class="btn small danger" id="r-del">Delete road</button>`;
    el.innerHTML = `<div class="card-head"><h2>Edit the network</h2></div>
      <p class="small">Draw intersections and roads, then attach each placed camera and its road zone to the road it watches. Saved networks are what the live view, recommendation, simulator and router all use.</p>
      ${D.synthetic ? '<p class="warn">This network is marked SYNTHETIC (a generated demo). Uncheck below once you have corrected it to the real street plan.</p>' : ""}
      <div class="row">${tool("select", "Select / move")}${tool("node", "Intersection")}${tool("road", "Road")}${tool("delete", "Delete")}
        <select id="e-kind" class="small" aria-label="Kind of new intersection"><option>signal</option><option>uncontrolled</option><option>boundary</option></select></div>
      ${S.roadFrom ? `<p class="small">Road from ${esc(S.roadFrom)}: now click the other end.</p>` : ""}
      <div class="stack"><label>Network name<input id="d-name" value="${esc(D.name)}"></label>
        <label class="check"><input type="checkbox" id="d-syn" ${D.synthetic ? "checked" : ""}> Mark as synthetic demo layout</label></div>
      ${sel}
      <div id="e-msg"></div>
      <div class="row" style="margin-top:10px"><button class="btn primary small" id="e-save" ${S.dirty ? "" : ""}>Save network</button>
        <button class="btn small" id="e-revert">Revert</button><button class="btn small" id="e-demo">Generate demo around my cameras</button>
        ${S.net && S.net.source === "saved" ? '<button class="btn small danger" id="e-delete">Delete saved network</button>' : ""}</div>
      <p class="muted small">${D.nodes.length} intersections, ${D.roads.length} roads${S.dirty ? " - unsaved changes" : ""}.
      The demo is invented geometry joining your cameras; it is labelled synthetic and is not saved until you press Save.</p>`;
    el.querySelectorAll("[data-tool]").forEach((b) => b.onclick = () => { S.tool = b.dataset.tool; S.roadFrom = null; renderEdit(); });
    $("#d-name").onchange = (e) => { D.name = e.target.value; S.dirty = true; };
    $("#d-syn").onchange = (e) => { D.synthetic = e.target.checked; S.dirty = true; renderEdit(); };
    if (n) { $("#n-name").onchange = (e) => { n.name = e.target.value; touch(); }; $("#n-kind").onchange = (e) => { n.kind = e.target.value; touch(); }; $("#n-del").onclick = () => delNode(n.id); }
    if (r) {
      const num = (id, k, f) => { $(id).onchange = (e) => { r[k] = f(e.target.value); touch(); }; };
      $("#r-name").onchange = (e) => { r.name = e.target.value; S.dirty = true; };
      num("#r-lanes", "lanes", Number); num("#r-speed", "speed_kph", Number); num("#r-len", "length_m", (v) => v === "" ? null : Number(v));
      $("#r-two").onchange = (e) => { r.two_way = e.target.value === "1"; if (!r.two_way) r.cameras = r.cameras.filter((c) => c.dir === "fwd"); touch(); };
      $("#r-del").onclick = () => { D.roads.splice(S.selRoad, 1); S.selRoad = null; touch(); };
      el.querySelectorAll("[data-rmcam]").forEach((b) => b.onclick = () => { r.cameras.splice(+b.dataset.rmcam, 1); touch(); });
      const fillZones = () => { const c = S.net.cameras.find((x) => x.id === +$("#c-id").value); $("#c-zones").innerHTML = ((c && c.zones) || []).map((z) => `<option value="${esc(z)}">`).join(""); if (c && c.zones[0] && !$("#c-zone").value) $("#c-zone").value = c.zones[0]; };
      if ($("#c-id")) { $("#c-id").onchange = () => { $("#c-zone").value = ""; fillZones(); }; fillZones(); }
      $("#c-add").onclick = () => {
        const cid = +$("#c-id").value, zone = $("#c-zone").value.trim();
        if (!cid || !zone) { toast("Choose a placed camera and give its road zone name"); return; }
        r.cameras.push({ camera_id: cid, zone, dir: $("#c-dir").value }); touch();
      };
    }
    $("#e-save").onclick = saveNet;
    $("#e-revert").onclick = () => { S.draft = null; S.dirty = false; renderEdit(); drawLive(); };
    $("#e-demo").onclick = async () => {
      try { const out = await api("/api/optimize/network/demo", { method: "POST" }); S.draft = out.doc; S.dirty = true; S.selNode = S.selRoad = null; toast(out.notes[0]); renderEdit(); drawLive(); fitDraft(); }
      catch (e) { $("#e-msg").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
    };
    const del = $("#e-delete");
    if (del) del.onclick = async () => { if (!confirm("Delete the saved network? Reads fall back to the synthetic demo.")) return; try { await api("/api/optimize/network", { method: "DELETE" }); S.draft = null; await refresh(true); renderEdit(); } catch (e) { toast(e.message); } };
  }
  async function saveNet() {
    const msg = $("#e-msg");
    try {
      const out = await api("/api/optimize/network", { method: "PUT", body: S.draft });
      S.draft = out.doc; S.dirty = false; await refresh(true); renderEdit();
      $("#e-msg").innerHTML = out.warnings.map((w) => `<p class="warn">${esc(w)}</p>`).join(""); toast("Saved");
    } catch (e) { msg.innerHTML = `<p class="err">${esc(e.message)}</p>` + (e.problems || []).map((p) => `<p class="err">- ${esc(p)}</p>`).join(""); }
  }
  function fitDraft() {
    const ns = S.draft ? S.draft.nodes : []; if (!ns.length) return;
    const b = new maplibregl.LngLatBounds(); ns.forEach((n) => b.extend([n.lon, n.lat])); S.map.fitBounds(b, { padding: 80, maxZoom: 17, pitch: 55 });
  }

  // ---------------------------------------------------------------- data
  async function refresh(full) {
    if (S.busy && !full) return;
    S.busy = true;
    try {
      S.net = await api("/api/optimize/network");
      try { S.live = await api("/api/optimize/live"); S.liveErr = ""; if (S.live.network === false) { S.liveErr = S.live.reason || "No network yet"; S.live = null; } } catch (e) { S.live = null; S.liveErr = e.message; }
      try { S.rec = S.live ? await api("/api/optimize/recommendation") : null; } catch (e) { S.rec = null; }
      renderBanner(); renderLegend(); renderLive(); renderRec(); renderInfo();
      if (S.tab !== "sim" || !S.sim) { if (S.tab === "route" && document.activeElement && ["r-o", "r-d"].includes(document.activeElement.id)) { /* do not reset a menu in use */ } else if (S.tab === "route") { const o = $("#r-o") && $("#r-o").value, d = $("#r-d") && $("#r-d").value; renderRoute(); if (o) $("#r-o").value = o; if (d) $("#r-d").value = d; } }
      drawLive();
      if (full && S.live && !S.fitted) { S.fitted = true; fit(); }
    } catch (e) { toast(e.message); } finally { S.busy = false; }
  }
  function fit() {
    const pts = [...((S.live && S.live.nodes) || []).map((n) => [n.lon, n.lat]), ...((S.net && S.net.cameras) || []).filter((c) => c.geo).map((c) => [c.geo.lon, c.geo.lat])];
    if (!pts.length) return;
    const b = new maplibregl.LngLatBounds(); pts.forEach((p) => b.extend(p));
    S.map.fitBounds(b, { padding: 90, maxZoom: 17, pitch: 60, bearing: -20, duration: 0 });
  }

  function setTab(t) {
    S.tab = t; S.pick = null;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("sel", b.dataset.tab === t));
    ["live", "rec", "sim", "route", "edit"].forEach((k) => $("#p-" + k).classList.toggle("hidden", k !== t));
    if (t !== "sim") { S.playing = false; clearInterval(playTimer); S.simFrame = false; }
    if (t === "sim") renderSim();
    if (t === "route") renderRoute();
    if (t === "edit") { renderEdit(); if (S.draft && S.draft.nodes.length) fitDraft(); }
    else if (t !== "sim") S.draft = null;
    renderPlay(); renderBanner(); drawLive();
  }

  async function start() {
    try { S.me = await api("/api/me"); } catch (e) { return; }
    if (!S.me.signed_in) { location.href = "/login"; return; }
    if (S.me.role === "driver") { location.href = "/portal"; return; }
    $("#who").textContent = (S.me.name || S.me.username || "") + (isAdmin() ? " (administrator)" : " (view only)");
    if (isAdmin()) $("#tab-edit").classList.remove("hidden");
    renderHonest();
    if (typeof maplibregl === "undefined") { $("#city-map").innerHTML = '<p class="err" style="padding:16px">The map library did not load (it comes from unpkg.com, and the 3D tiles from tiles.openfreemap.org). Check the internet connection.</p>'; return; }
    S.map = new maplibregl.Map({ container: "city-map", style: await basemapStyle(), center: [90.4075, 23.7806], zoom: 13, pitch: 60, bearing: -20, maxPitch: 80, attributionControl: { compact: true } });
    S.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    S.map.on("error", (e) => { if (e && e.error && /tiles\.openfreemap|style/i.test(String(e.error.message || e.error))) toast("Map tiles could not be loaded: " + String(e.error.message || "").slice(0, 80)); });
    S.map.on("load", async () => { initLayers(); S.ready = true; await refresh(true); });
    document.querySelectorAll(".tab").forEach((b) => b.onclick = () => setTab(b.dataset.tab));
    $("#metric").onchange = (e) => { S.metric = e.target.value; renderLegend(); drawLive(); };
    setInterval(() => { if (S.tab === "live" || S.tab === "rec") refresh(false); }, 6000);
  }
  document.addEventListener("DOMContentLoaded", () => { try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ } start(); });
})();
