/* Zone editor: draw road zones, special areas and solid lane lines on a frame of the camera. */
(function () {
  "use strict";
  const COLORS = { road: "#ffc800", no_parking: "#ff5a4f", bus_stop: "#2ecc71", intersection: "#e8e8e8", signal: "#ffdc00", parking: "#3d8bff", uturn: "#c86bfa", line: "#ffffff" };
  const LABEL = { road: "Road zone", no_parking: "No-parking kerb", bus_stop: "Bus stop", intersection: "Intersection", signal: "Signal queue area", parking: "Legal parking", uturn: "U-turn opening", line: "Solid lane line" };
  const AREA_ORDER = ["no_parking", "bus_stop", "intersection", "signal", "parking", "uturn"];
  const HELP = {
    select: "Click a shape to select it, drag its corners or body to adjust. Delete removes it. Ctrl/⌘+Z undoes.",
    road: "Click the 4 corners of a real rectangle on the road surface that you can measure – e.g. two lane lines, between two dash ends. Any order.",
    no_parking: "Click around the kerb strip where standing is illegal parking (V3) in peak hours or heavy traffic – or select a road zone and use “Add kerb strips”. Enter to finish.",
    bus_stop: "Click around the bus-stop bay. It only counts if it is just AFTER an intersection (mark that too); elsewhere buses standing here are reported (V1). Enter to finish.",
    intersection: "Mark the junction mouth where side-road traffic joins or turns. Vehicles waiting here are never stop violations; bus stops must come right after one. Enter to finish.",
    signal: "Mark where vehicles queue for the signal (the lanes before the stop line). Stops inside are never violations. Enter to finish.",
    parking: "Mark a legal parking bay. Vehicles standing inside are not illegal parking. Enter to finish.",
    uturn: "Mark a legal U-turn opening in the median. Turning there is not wrong-side driving. Enter to finish.",
    line: "Click both ends of a SOLID (unbroken) lane line. Only solid lines prohibit changing lanes; skip broken lines.",
  };
  const CORNER_NAMES = ["far-left", "far-right", "near-right", "near-left"];
  const DEFAULT_PEAK = [["07:00", "10:30"], ["16:00", "20:30"]];

  async function render(main, cid, App) {
    const { api, esc, toast, fail, modal } = App;
    const $ = (s, r = main) => r.querySelector(s);
    const $$ = (s, r = main) => Array.from(r.querySelectorAll(s));
    const cam = await api(`/api/cameras/${cid}`);
    const saved = await api(`/api/cameras/${cid}/zones`);
    const meta = App.state.meta;
    const ruleList = Object.entries(meta.rules).map(([key, r]) => ({ key, ...r })).sort((a, b) => a.code.localeCompare(b.code));
    let cfg = saved.config ? JSON.parse(JSON.stringify(saved.config)) :
      { frame_size: null, peak_hours: DEFAULT_PEAK, rules: {}, zones: [], lane_lines: [] };
    cfg.zones = cfg.zones || [];
    cfg.lane_lines = cfg.lane_lines || [];
    cfg.rules = cfg.rules || {};
    if (!("peak_hours" in cfg)) cfg.peak_hours = DEFAULT_PEAK;
    const learning = (saved.learning && saved.learning.zones) || {};

    let tool = "select", sel = null, drawing = [], mouse = null, drag = null;
    let undo = [], redo = [], dirty = false, check = { roads: [], roi: null, error: "" };
    let img = new Image(), W = 0, H = 0, frameT = 60;

    main.innerHTML = `
      <div class="page-head"><div><h1>Zones · <span id="cam-title">${esc(cam.name)}</span></h1>
        <p><span id="cam-loc">${esc(cam.location || "No location set")}</span> · ${saved.version ? `saved version ${saved.version}` : "no zones yet"} · draw on a clear frame of this camera</p></div><span class="spacer"></span>
        <button class="btn" id="edit-cam">${App.ICON.edit}Edit name &amp; location</button>
        <a class="btn" href="#/camera/${cid}">Camera page</a></div>
      <div class="toolbar" id="tools">
        <button class="tool" data-tool="select">Select</button>
        ${["road", ...AREA_ORDER, "line"].map((t) => `<button class="tool" data-tool="${t}"><span class="sw" style="background:${COLORS[t]}"></span>${LABEL[t]}</button>`).join("")}
        <span class="spacer"></span>
        <button class="btn small primary" id="auto-detect" title="Learn the zones from this camera's traffic">${App.ICON.auto}Auto-detect zones</button>
        <button class="btn small" id="undo" title="Undo (Ctrl/⌘+Z)">Undo</button><button class="btn small" id="redo" title="Redo">Redo</button>
      </div>
      <div class="editor">
        <div class="stack">
          <div class="canvas-wrap"><canvas id="cv"></canvas><div class="canvas-hint" id="hint"></div></div>
          <div class="row">
            ${cam.kind === "file" ? `<label class="row small">Frame at <input type="number" id="t" value="${frameT}" min="0" step="5" style="width:80px"> s</label>
              <button class="btn small" id="load-frame">Load frame</button><span class="small muted">pick a moment with little traffic so the road is visible</span>`
              : `<button class="btn small" id="load-frame">Grab new frame</button><span class="small muted">the camera must be stopped to grab a clean frame</span>`}
            <span class="spacer"></span>
            <label class="btn small">Import JSON<input type="file" id="import" accept=".json,application/json" hidden></label>
            <button class="btn small" id="export">Export JSON</button>
          </div>
        </div>
        <div class="panel">
          <div class="card"><div class="row"><input id="note" placeholder="What changed? (optional)" style="flex:1">
            <button class="btn primary" id="save">Save zones</button></div>
            <div id="check" class="small" style="margin-top:8px"></div></div>
          <div class="card hidden" id="auto-card"></div>
          <div class="card"><div class="card-head"><h3>Shapes</h3></div><div class="shape-list" id="shapes"></div></div>
          <div class="card" id="props"></div>
          <div class="card" id="rules"></div>
        </div>
      </div>`;

    $("#edit-cam").onclick = () => App.editCameraModal(cam, (upd) => {
      Object.assign(cam, upd);
      $("#cam-title").textContent = upd.name;
      $("#cam-loc").textContent = upd.location || "No location set";
    });
    const cv = $("#cv"), ctx = cv.getContext("2d");

    // -------------------------------------------------- frame
    async function loadFrame() {
      const t = cam.kind === "file" ? +($("#t").value || 0) : 0;
      frameT = t;
      $("#hint").textContent = "Loading frame...";
      try {
        const blob = await api(`/api/cameras/${cid}/frame?t=${t}`);
        const url = URL.createObjectURL(blob);
        await new Promise((res, rej) => { img = new Image(); img.onload = res; img.onerror = rej; img.src = url; });
        const nw = img.naturalWidth, nh = img.naturalHeight;
        if (cfg.frame_size && (cfg.frame_size[0] !== nw || cfg.frame_size[1] !== nh)) scaleAll(nw / cfg.frame_size[0], nh / cfg.frame_size[1]);
        cfg.frame_size = [nw, nh];
        W = nw; H = nh;
        cv.width = W; cv.height = H;
        setTool(tool);
        draw();
        validate();
      } catch (e) {
        $("#hint").textContent = "Could not load a frame: " + (e.message || "error") + (cam.kind === "live" ? " – stop the camera and try again." : "");
      }
    }
    function scaleAll(sx, sy) {
      const sc = (p) => [Math.round(p[0] * sx * 10) / 10, Math.round(p[1] * sy * 10) / 10];
      cfg.zones.forEach((z) => { z.points = z.points.map(sc); });
      cfg.lane_lines.forEach((l) => { l.points = l.points.map(sc); });
    }

    // -------------------------------------------------- shapes
    const shapes = () => [...cfg.zones.map((z, i) => ({ list: "zones", i, kind: z.type || "road", obj: z })),
      ...cfg.lane_lines.map((l, i) => ({ list: "lane_lines", i, kind: "line", obj: l }))];
    const selected = () => (sel ? cfg[sel.list][sel.i] : null);
    const selKind = () => (sel ? (sel.list === "lane_lines" ? "line" : cfg.zones[sel.i].type || "road") : null);

    function snapshot() { undo.push(JSON.stringify(cfg)); if (undo.length > 100) undo.shift(); redo = []; dirty = true; }
    function restore(json) { cfg = JSON.parse(json); sel = null; refresh(); }

    function orderRoad(pts) {
      // far = the two upper points (left, right); near = the two lower points (right, left)
      const s = [...pts].sort((a, b) => a[1] - b[1]);
      const far = s.slice(0, 2).sort((a, b) => a[0] - b[0]);
      const near = s.slice(2).sort((a, b) => b[0] - a[0]);
      return [far[0], far[1], near[0], near[1]];
    }

    function finishDrawing() {
      const pts = drawing.map((p) => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]);
      if (tool === "road") {
        if (pts.length !== 4) return;
        snapshot();
        const n = cfg.zones.filter((z) => (z.type || "road") === "road").length + 1;
        cfg.zones.push({ type: "road", name: `Road ${n}`, points: orderRoad(pts), width_m: null, length_m: null, road_category: "city",
          allowed_direction: "both", speed_limit_kph: null, banned_classes: [], stopping_allowed: false });
        sel = { list: "zones", i: cfg.zones.length - 1 };
      } else if (tool === "line") {
        if (pts.length !== 2) return;
        snapshot();
        cfg.lane_lines.push({ name: `Solid line ${cfg.lane_lines.length + 1}`, points: pts });
        sel = { list: "lane_lines", i: cfg.lane_lines.length - 1 };
      } else {
        if (pts.length < 3) { toast("An area needs at least 3 points", "error"); return; }
        snapshot();
        const n = cfg.zones.filter((z) => z.type === tool).length + 1;
        cfg.zones.push({ type: tool, name: `${LABEL[tool]} ${n}`, points: pts });
        sel = { list: "zones", i: cfg.zones.length - 1 };
      }
      drawing = [];
      setTool("select");
      refresh();
      if (selKind() === "road") setTimeout(() => $("#p-width") && $("#p-width").focus(), 50);
    }

    function deleteSelected() {
      if (!sel) return;
      snapshot();
      cfg[sel.list].splice(sel.i, 1);
      sel = null;
      refresh();
    }

    // -------------------------------------------------- geometry helpers
    const px = () => W / Math.max(1, cv.clientWidth); // image pixels per screen pixel
    function toImg(e) {
      const r = cv.getBoundingClientRect();
      return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
    }
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    function inPoly(p, poly) {
      let inside = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const [xi, yi] = poly[i], [xj, yj] = poly[j];
        if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
      }
      return inside;
    }
    function distToSeg(p, a, b) {
      const l2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2;
      let t = l2 ? ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / l2 : 0;
      t = Math.max(0, Math.min(1, t));
      return dist(p, [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    function hitShape(p) {
      const list = shapes();
      for (let k = list.length - 1; k >= 0; k--) {           // lines first (drawn on top), then areas, then roads
        const s = list[k];
        if (s.kind === "line" && distToSeg(p, s.obj.points[0], s.obj.points[1]) < 10 * px()) return s;
      }
      for (const kind of [...AREA_ORDER, "road"]) {
        for (let k = list.length - 1; k >= 0; k--) {
          const s = list[k];
          if (s.kind === kind && inPoly(p, s.obj.points)) return s;
        }
      }
      return null;
    }

    // -------------------------------------------------- drawing
    function draw() {
      if (!W) return;
      const k = px();
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(img, 0, 0);
      if (check.roi) {
        const [x0, y0, x1, y1] = check.roi;
        if (x1 - x0 < W || y1 - y0 < H) {
          ctx.save(); ctx.fillStyle = "rgba(0,0,0,0.35)";
          ctx.fillRect(0, 0, W, y0); ctx.fillRect(0, y1, W, H - y1); ctx.fillRect(0, y0, x0, y1 - y0); ctx.fillRect(x1, y0, W - x1, y1 - y0);
          ctx.restore();
          label("detection area", x0 + 6 * k, y0 + 16 * k, "#bbbbbb", k);
        }
      }
      const list = shapes();
      const order = ["road", ...AREA_ORDER, "line"];
      list.sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind)).forEach((s) => drawShape(s, k));
      // reliable measurement lines
      check.roads.forEach((r) => {
        if (r.reliable_from_m <= 0.5) return;
        const [a, b] = r.line;
        ctx.save(); ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 2.5 * k; ctx.setLineDash([8 * k, 6 * k]);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.restore();
        label(`measured from here: ${r.reliable_m} of ${r.length_m} m`, Math.min(a[0], b[0]), Math.min(a[1], b[1]) - 6 * k, "#00e5ff", k);
      });
      // in-progress shape
      if (drawing.length) {
        const col = COLORS[tool] || "#fff";
        ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2 * k;
        ctx.beginPath(); drawing.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
        if (mouse) ctx.lineTo(mouse[0], mouse[1]);
        ctx.stroke();
        drawing.forEach((p, i) => { ctx.beginPath(); ctx.arc(p[0], p[1], (i === 0 && tool !== "road" && tool !== "line" ? 7 : 5) * k, 0, 7); ctx.fill(); });
        ctx.restore();
      }
    }
    function label(text, x, y, color, k) {
      ctx.save();
      ctx.font = `600 ${Math.round(12.5 * k)}px system-ui, sans-serif`;
      const w = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(10,10,10,0.78)";
      ctx.fillRect(x - 3 * k, y - 13 * k, w + 6 * k, 17 * k);
      ctx.fillStyle = color;
      ctx.fillText(text, x, y);
      ctx.restore();
    }
    function drawShape(s, k) {
      const isSel = sel && sel.list === s.list && sel.i === s.i;
      const col = COLORS[s.kind];
      const pts = s.obj.points;
      ctx.save();
      if (s.kind === "line") {
        ctx.lineCap = "round";
        ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = (isSel ? 8 : 6) * k;
        ctx.beginPath(); ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); ctx.stroke();
        ctx.strokeStyle = col; ctx.lineWidth = (isSel ? 4.5 : 3) * k; ctx.stroke();
        label(s.obj.name, pts[0][0] + 6 * k, pts[0][1] - 6 * k, col, k);
      } else {
        ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
        ctx.fillStyle = col + (s.kind === "road" ? (isSel ? "33" : "1a") : (isSel ? "55" : "33"));
        ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = (isSel ? 3.5 : 2) * k; ctx.stroke();
        if (s.kind === "road") drawRoadExtras(s.obj, k, col);
        const top = pts.reduce((a, p) => (p[1] < a[1] ? p : a), pts[0]);
        const sizeTxt = s.kind === "road" ? (s.obj.width_m && s.obj.length_m ? ` · ${s.obj.width_m}×${s.obj.length_m} m` : " · size?") : "";
        const stop = s.kind === "bus_stop" ? (check.bus_stops || []).find((b) => b.name === s.obj.name) : null;
        const bad = stop && !stop.valid;
        if (bad) { ctx.setLineDash([10 * k, 7 * k]); ctx.strokeStyle = "#ff3b30"; ctx.lineWidth = 3 * k; ctx.stroke(); ctx.setLineDash([]); }
        label(s.obj.name + sizeTxt + (bad ? " · NOT a valid stop" : ""), top[0] + 4 * k, top[1] - 8 * k, bad ? "#ff6b61" : col, k);
      }
      if (isSel) {
        pts.forEach((p, i) => {
          ctx.beginPath(); ctx.arc(p[0], p[1], 7 * k, 0, 7);
          ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2.5 * k; ctx.strokeStyle = col; ctx.stroke();
          if (s.kind === "road") label(CORNER_NAMES[i], p[0] + 10 * k, p[1] + 4 * k, "#fff", k);
        });
      }
      ctx.restore();
    }
    function drawRoadExtras(z, k, col) {
      const p = z.points;
      if (z.allowed_direction === "both" || p.length !== 4) return;
      const far = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2];
      const near = [(p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2];
      const a = [far[0] + (near[0] - far[0]) * 0.3, far[1] + (near[1] - far[1]) * 0.3];
      const b = [far[0] + (near[0] - far[0]) * 0.75, far[1] + (near[1] - far[1]) * 0.75];
      const [from, to] = z.allowed_direction === "toward_camera" ? [a, b] : [b, a];
      const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
      ctx.save();
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 6 * k; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(...to);
      ctx.lineTo(to[0] - 26 * k * Math.cos(ang - 0.45), to[1] - 26 * k * Math.sin(ang - 0.45));
      ctx.lineTo(to[0] - 26 * k * Math.cos(ang + 0.45), to[1] - 26 * k * Math.sin(ang + 0.45));
      ctx.closePath(); ctx.fill();
      ctx.restore();
    }

    // -------------------------------------------------- pointer
    cv.addEventListener("pointerdown", (e) => {
      if (!W) return;
      const p = toImg(e);
      if (tool !== "select") {
        if (drawing.length >= 3 && !["road", "line"].includes(tool) && dist(p, drawing[0]) < 12 * px()) { finishDrawing(); return; }
        drawing.push(p);
        if ((tool === "road" && drawing.length === 4) || (tool === "line" && drawing.length === 2)) finishDrawing();
        else draw();
        return;
      }
      const cur = selected();
      if (cur) {
        const vi = cur.points.findIndex((q) => dist(p, q) < 12 * px());
        if (vi >= 0) { drag = { type: "vertex", vi, start: p, before: JSON.stringify(cfg) }; cv.setPointerCapture(e.pointerId); return; }
      }
      const hit = hitShape(p);
      sel = hit ? { list: hit.list, i: hit.i } : null;
      if (hit) { drag = { type: "move", last: p, before: JSON.stringify(cfg), moved: false }; cv.setPointerCapture(e.pointerId); }
      refresh();
    });
    cv.addEventListener("pointermove", (e) => {
      if (!W) return;
      const p = toImg(e);
      mouse = p;
      if (drag && sel) {
        const pts = selected().points;
        if (drag.type === "vertex") pts[drag.vi] = [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10];
        else {
          const dx = p[0] - drag.last[0], dy = p[1] - drag.last[1];
          pts.forEach((q) => { q[0] = Math.round((q[0] + dx) * 10) / 10; q[1] = Math.round((q[1] + dy) * 10) / 10; });
          drag.last = p;
          drag.moved = true;
        }
        draw();
        return;
      }
      if (tool === "select") {
        const cur = selected();
        const onVertex = cur && cur.points.some((q) => dist(p, q) < 12 * px());
        cv.style.cursor = onVertex ? "grab" : hitShape(p) ? "move" : "default";
      }
      if (drawing.length) draw();
    });
    cv.addEventListener("pointerup", () => {
      if (drag) {
        const changed = drag.before !== JSON.stringify(cfg);
        if (changed) {
          undo.push(drag.before); redo = []; dirty = true;
          if (selKind() === "road" && drag.type === "vertex") selected().points = orderRoad(selected().points);
          refresh();
        }
        drag = null;
      }
    });
    cv.addEventListener("dblclick", () => { if (drawing.length >= 3 && !["road", "line"].includes(tool)) { drawing.pop(); finishDrawing(); } });
    cv.addEventListener("pointerleave", () => { mouse = null; if (drawing.length) draw(); });

    const onKey = (e) => {
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        if (typing) return;
        e.preventDefault();
        if (e.shiftKey) doRedo(); else doUndo();
        return;
      }
      if (typing) return;
      if (e.key === "Enter" && drawing.length >= 3 && !["road", "line"].includes(tool)) finishDrawing();
      else if (e.key === "Escape") { if (drawing.length) { drawing = []; draw(); } else { sel = null; setTool("select"); refresh(); } }
      else if ((e.key === "Delete" || e.key === "Backspace") && sel) { e.preventDefault(); deleteSelected(); }
      else if (e.key === "Backspace" && drawing.length) { drawing.pop(); draw(); }
    };
    document.addEventListener("keydown", onKey);
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    const beforeUnload = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } };
    window.addEventListener("beforeunload", beforeUnload);
    App.onLeave(() => {
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("beforeunload", beforeUnload);
    });
    App.setLeaveGuard(() => !dirty || confirm("You have unsaved zone changes. Leave without saving?"));

    function doUndo() { if (!undo.length) return; redo.push(JSON.stringify(cfg)); restore(undo.pop()); dirty = true; }
    function doRedo() { if (!redo.length) return; undo.push(JSON.stringify(cfg)); restore(redo.pop()); dirty = true; }
    $("#undo").onclick = doUndo;
    $("#redo").onclick = doRedo;

    function setTool(t) {
      tool = t;
      drawing = [];
      $$("#tools [data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
      cv.classList.toggle("select", t === "select");
      $("#hint").textContent = W ? HELP[t] : $("#hint").textContent;
      draw();
    }
    $$("#tools [data-tool]").forEach((b) => b.onclick = () => { setTool(b.dataset.tool); if (b.dataset.tool !== "select") { sel = null; refresh(); } });

    // -------------------------------------------------- panel
    function refresh() {
      renderShapes();
      renderProps();
      renderRules();
      draw();
      validate();
    }

    function renderShapes() {
      const list = shapes();
      $("#shapes").innerHTML = list.length ? list.map((s) => `<div class="shape-item${sel && sel.list === s.list && sel.i === s.i ? " sel" : ""}" data-l="${s.list}" data-i="${s.i}">
        <span class="sw" style="background:${COLORS[s.kind]}"></span>${esc(s.obj.name)}<span class="spacer"></span><span class="small muted">${LABEL[s.kind]}</span></div>`).join("")
        : `<div class="small muted">Nothing drawn yet. Start with <b>Road zone</b>: one per carriageway, as long as you can measure accurately (30–90 m).</div>`;
      $$("#shapes .shape-item").forEach((el) => el.onclick = () => { sel = { list: el.dataset.l, i: +el.dataset.i }; setTool("select"); refresh(); });
    }

    function renderProps() {
      const box = $("#props");
      const z = selected();
      if (!z) {
        box.innerHTML = `<div class="card-head"><h3>How to set up a camera</h3></div><ol class="small secondary" style="margin:0;padding-left:18px">
          <li>Pick <b>Road zone</b> and click 4 corners of a rectangle on each carriageway; type its real width and length (measure on site, or with Google Maps' <i>Measure distance</i>).</li>
          <li>Check <b>Top-down view</b>: lane lines must come out straight and parallel.</li>
          <li>Set the allowed direction (use the suggestion once traffic has been analysed), banned vehicle classes and road type.</li>
          <li>Add <b>No-parking kerb</b> strips along the kerbs (select a road zone → <i>Add kerb strips</i>).</li>
          <li>Mark <b>Intersections</b> (junction mouths), then <b>Bus stops</b> just after them – a bus stop anywhere else does not count. Add signal queue areas, legal parking and U-turn openings; draw solid lane lines if painted.</li>
          <li>Save, then start the camera.</li></ol>`;
        return;
      }
      const kind = selKind();
      let html = `<div class="card-head"><span class="sw" style="width:12px;height:12px;border-radius:3px;background:${COLORS[kind]}"></span><h3>${LABEL[kind]}</h3>
        <span class="spacer"></span><button class="btn small ghost danger" id="p-del">Delete</button></div>
        <div class="stack"><label class="field">Name<input id="p-name" value="${esc(z.name)}" maxlength="60"></label>`;
      if (kind === "road") {
        const learn = learning[z.name];
        const rel = check.roads.find((r) => r.name === z.name);
        const dirBtn = (v, txt) => `<button data-dir="${v}" class="${z.allowed_direction === v ? "on" : ""}">${txt}</button>`;
        html += `
          <div class="grid2">
            <label class="field">Width (m)<input id="p-width" type="number" min="1" max="60" step="0.1" value="${z.width_m ?? ""}" class="${z.width_m ? "" : "invalid"}"></label>
            <label class="field">Length (m)<input id="p-length" type="number" min="5" max="300" step="0.5" value="${z.length_m ?? ""}" class="${z.length_m ? "" : "invalid"}"></label>
          </div>
          <span class="hint">Width: left edge to right edge (Dhaka lanes are about 3.0–3.5 m). Length: far edge to near edge. Speeds are only as accurate as these two numbers.</span>
          ${z.scale_source === "auto" ? `<div class="suggest">Size from <b>automatic calibration</b> (speeds ±${z.speed_uncertainty_pct ?? 15}%, so speeding is only reported well above the limit). Type a measured width or length to switch this zone to exact speed enforcement.</div>` : ""}
          ${rel ? `<div class="small ${rel.reliable_m < 20 ? "errors" : "okline"}">Reliable measurement over ${rel.reliable_m} of ${rel.length_m} m${rel.reliable_m < 20 ? " – too short for speed: extend the zone toward the camera or use the 960 px detector" : ""}</div>` : ""}
          <button class="btn small" id="p-top" ${z.width_m && z.length_m ? "" : "disabled"}>Top-down view check</button>
          <label class="field">Road type<select id="p-cat">${meta.road_categories.map((c) => `<option value="${c}" ${z.road_category === c ? "selected" : ""}>${{ city: "City road – 60 km/h", highway: "National highway – 80 km/h", expressway: "Expressway / flyover – 80 km/h" }[c]}</option>`).join("")}</select>
            <span class="hint">${speedHint(z.road_category)}</span></label>
          <label class="field">Lower local speed limit (km/h)<input id="p-cap" type="number" min="5" max="120" value="${z.speed_limit_kph ?? ""}" placeholder="none – e.g. 20 near a school"></label>
          <div class="field">Allowed direction<div class="dirs">${dirBtn("both", "Both ways")}${dirBtn("toward_camera", "↓ Toward camera")}${dirBtn("away_from_camera", "↑ Away from camera")}</div>
            <span class="hint">Bangladesh drives on the left. One-way zones enable the wrong-side rule (V4).</span></div>
          ${learn ? `<div class="suggest">Seen in traffic: <b>${learn.toward}</b> toward, <b>${learn.away}</b> away.
            ${learn.suggestion && learn.suggestion !== z.allowed_direction ? `<br>Suggested: <b>${{ both: "both ways", toward_camera: "toward camera", away_from_camera: "away from camera" }[learn.suggestion]}</b>
            <button class="btn small" id="p-apply" style="margin-left:6px">Apply</button>` : learn.suggestion ? "<br>Matches the current setting." : "<br>Not enough one-way evidence yet."}</div>` : ""}
          <div class="field">Vehicles banned on this road (V8)<div class="classes">${meta.classes.map((c) => `<label class="${z.banned_classes.includes(c) ? "on" : ""}"><input type="checkbox" value="${esc(c)}" ${z.banned_classes.includes(c) ? "checked" : ""}>${esc(c)}</label>`).join("")}</div>
            <span class="hint">e.g. Rickshaw / Easybike on main roads, CNG / Leguna / Rickshaw on flyovers.</span></div>
          <div class="field">No-parking kerb strips (V3)<div class="row"><label class="small">Width <input id="p-kw" type="number" min="1.5" max="6" step="0.5" value="3" style="width:64px"> m</label>
            <button class="btn small" data-kerb="left">+ Left edge</button><button class="btn small" data-kerb="right">+ Right edge</button></div>
            <span class="hint">Bangladesh drives on the left: the kerb is usually the left edge for traffic away from the camera and the right edge for traffic toward it. Standing there is illegal parking in peak hours or heavy traffic; standing in the other lanes is V7.</span></div>
          <label class="row small"><input type="checkbox" id="p-stop" ${z.stopping_allowed ? "checked" : ""}> Stopping is allowed in this whole zone (service lane)</label>`;
      } else if (kind !== "line") {
        const stop = kind === "bus_stop" ? (check.bus_stops || []).find((b) => b.name === z.name) : null;
        html += `<label class="field">Type<select id="p-type">${meta.area_types.map((t) => `<option value="${t}" ${z.type === t ? "selected" : ""}>${LABEL[t]}</option>`).join("")}</select></label>
          <div class="hint">${HELP[z.type]}</div>
          ${kind === "bus_stop" ? `${stop ? `<div class="small ${stop.valid ? "okline" : "errors"}">${stop.valid ? "Valid bus stop: " : "Not a bus stop for enforcement: "}${esc(stop.reason)}</div>` : ""}
            <label class="row small"><input type="checkbox" id="p-approved" ${z.approved ? "checked" : ""}> Officially approved mid-block bus bay (skip the after-intersection check)</label>` : ""}`;
      } else {
        html += `<div class="hint">Crossing this line from one side to the other is reported as V5. Vehicles riding on the line are not.</div>`;
      }
      box.innerHTML = html + "</div>";
      const commit = (fn) => { snapshot(); fn(); renderShapes(); draw(); validate(); };
      $("#p-del", box).onclick = deleteSelected;
      $("#p-name", box).onchange = (e) => commit(() => { z.name = e.target.value.trim() || z.name; });
      if (kind === "road") {
        const num = (v) => (v === "" ? null : +v);
        const measured = () => { z.scale_source = "measured"; delete z.speed_uncertainty_pct; };
        $("#p-width", box).onchange = (e) => { commit(() => { z.width_m = num(e.target.value); measured(); }); renderProps(); };
        $("#p-length", box).onchange = (e) => { commit(() => { z.length_m = num(e.target.value); measured(); }); renderProps(); };
        $("#p-cat", box).onchange = (e) => { commit(() => { z.road_category = e.target.value; }); renderProps(); };
        $("#p-cap", box).onchange = (e) => commit(() => { z.speed_limit_kph = num(e.target.value); });
        $$("[data-dir]", box).forEach((b) => b.onclick = () => { commit(() => { z.allowed_direction = b.dataset.dir; }); renderProps(); });
        if ($("#p-apply", box)) $("#p-apply", box).onclick = () => { commit(() => { z.allowed_direction = learning[z.name].suggestion; }); renderProps(); };
        $$(".classes input", box).forEach((cb) => cb.onchange = () => {
          commit(() => { z.banned_classes = $$(".classes input:checked", box).map((x) => x.value); });
          cb.parentElement.classList.toggle("on", cb.checked);
        });
        $("#p-stop", box).onchange = (e) => commit(() => { z.stopping_allowed = e.target.checked; });
        $$("[data-kerb]", box).forEach((b) => b.onclick = async () => {
          if (!(z.width_m && z.length_m)) { toast("Enter the zone's width and length first", "error"); return; }
          try {
            const r = await api(`/api/kerb-strip`, { method: "POST", body: { points: z.points, width_m: z.width_m, length_m: z.length_m, side: b.dataset.kerb, strip_m: +$("#p-kw", box).value || 3 } });
            snapshot();
            const name = `${z.name} – ${b.dataset.kerb} kerb`;
            cfg.zones.push({ type: "no_parking", name: cfg.zones.some((x) => x.name === name) ? name + " 2" : name, points: r.points });
            sel = { list: "zones", i: cfg.zones.length - 1 };
            refresh();
          } catch (e) { fail(e); }
        });
        $("#p-top", box).onclick = () => topDown(z);
      } else if (kind !== "line") {
        $("#p-type", box).onchange = (e) => { commit(() => { z.type = e.target.value; }); renderProps(); };
        if ($("#p-approved", box)) $("#p-approved", box).onchange = (e) => commit(() => { z.approved = e.target.checked; });
      }
    }

    function speedHint(cat) {
      const t = meta.speed_limits[cat] || {};
      const groups = {};
      Object.entries(t).forEach(([c, v]) => { (groups[v] = groups[v] || []).push(c); });
      return "Limits: " + Object.entries(groups).sort((a, b) => b[0] - a[0]).map(([v, cs]) => `${v} km/h ${cs.join(", ")}`).join("; ");
    }

    async function topDown(z) {
      try {
        const blob = await api(`/api/cameras/${cid}/topdown`, { method: "POST", body: { points: z.points, width_m: z.width_m, length_m: z.length_m, t: frameT } });
        const m = modal(`<div class="modal-head"><h2>Top-down view · ${esc(z.name)}</h2><span class="spacer"></span><button class="btn ghost small" data-close>Close</button></div>
          <div class="row" style="align-items:flex-start;flex-wrap:nowrap;gap:16px"><img id="td" style="max-height:70vh;border-radius:8px" alt="Road seen from above">
          <div class="small secondary"><p><b>If the size and corners are right:</b></p><ul style="padding-left:16px">
            <li>lane lines are straight and parallel to the sides,</li><li>vehicles keep the same size from top to bottom,</li>
            <li>lane-marking dashes are evenly spaced; the orange marks are every 10 m.</li></ul>
            <p>Lines that spread apart or narrow toward the top mean a corner is off – drag it and check again. Vehicles stretched along the road mean the length is too long.</p></div></div>`, "wide");
        $("#td", m.el).src = URL.createObjectURL(blob);
      } catch (e) { fail(e); }
    }

    function renderRules() {
      const r = cfg.rules;
      const val = (key, field) => (r[key] && r[key][field] != null ? r[key][field] : meta.rules[key].defaults[field]);
      const enabled = (key) => !(r[key] && r[key].enabled === false);
      const allDay = !cfg.peak_hours;
      $("#rules").innerHTML = `<div class="card-head"><h3>Rules for this camera</h3></div><div class="stack">
        <div class="grid2">
          <label class="field">Bus stopped (V1) after, s<input type="number" min="5" max="300" id="r-bus" value="${val("bus_blocking", "dwell_sec")}"></label>
          <label class="field">Parked at kerb (V3) after, s<input type="number" min="15" max="1800" id="r-park" value="${val("illegal_parking", "dwell_sec")}"></label>
          <label class="field">Blocking a lane (V7) after, s<input type="number" min="5" max="600" id="r-block" value="${val("road_blocking", "dwell_sec")}"></label>
          <label class="field">Heavy traffic (V3), vehicles<input type="number" min="3" max="200" id="r-busy" value="${val("illegal_parking", "busy_vehicles")}"></label>
        </div>
        <span class="hint">A stand only counts when at least ${val("bus_blocking", "min_passed")} vehicles drove past it and it was not just waiting behind another standing vehicle – red lights and jams are never violations.</span>
        <div class="field">Kerb no-parking hours (V3) <span class="hint">– also enforced whenever traffic is heavy</span>
          <label class="row small"><input type="checkbox" id="r-allday" ${allDay ? "checked" : ""}> Enforce all day</label>
          <div id="r-peaks" class="stack ${allDay ? "hidden" : ""}" style="gap:6px">${(cfg.peak_hours || []).map((p, i) => `<div class="row">
            <input type="time" value="${esc(p[0])}" data-pk="${i}" data-e="0"> – <input type="time" value="${esc(p[1])}" data-pk="${i}" data-e="1">
            <button class="btn small ghost" data-pkdel="${i}">Remove</button></div>`).join("")}
            <button class="btn small" id="r-pkadd" style="align-self:flex-start">Add time range</button></div></div>
        <div class="field">Active rules<div class="stack" style="gap:4px">${ruleList.map((x) => `<label class="row small"><input type="checkbox" data-rule="${x.key}" ${enabled(x.key) ? "checked" : ""}>
          <span class="code">${x.code}</span>${esc(x.title)}</label>`).join("")}</div>
          <span class="hint">V3 needs a no-parking kerb strip, V5 solid lane lines, V8 banned classes, V6 a helmet model; V4 needs a one-way zone.</span></div></div>`;
      const setRule = (key, field, value) => { snapshot(); r[key] = { ...(r[key] || {}), [field]: value }; };
      $("#r-bus").onchange = (e) => setRule("bus_blocking", "dwell_sec", +e.target.value);
      $("#r-park").onchange = (e) => setRule("illegal_parking", "dwell_sec", +e.target.value);
      $("#r-block").onchange = (e) => setRule("road_blocking", "dwell_sec", +e.target.value);
      $("#r-busy").onchange = (e) => setRule("illegal_parking", "busy_vehicles", +e.target.value);
      $("#r-allday").onchange = (e) => { snapshot(); cfg.peak_hours = e.target.checked ? null : DEFAULT_PEAK.map((p) => [...p]); renderRules(); };
      $$("[data-pk]").forEach((inp) => inp.onchange = () => { snapshot(); cfg.peak_hours[+inp.dataset.pk][+inp.dataset.e] = inp.value; });
      $$("[data-pkdel]").forEach((b) => b.onclick = () => { snapshot(); cfg.peak_hours.splice(+b.dataset.pkdel, 1); if (!cfg.peak_hours.length) cfg.peak_hours = null; renderRules(); });
      if ($("#r-pkadd")) $("#r-pkadd").onclick = () => { snapshot(); cfg.peak_hours.push(["12:00", "14:00"]); renderRules(); };
      $$("[data-rule]").forEach((cb) => cb.onchange = () => { setRule(cb.dataset.rule, "enabled", cb.checked); if (cb.checked) delete r[cb.dataset.rule].enabled; });
    }

    // -------------------------------------------------- validation & save
    let vTimer = null;
    function clientErrors() {
      const errs = [];
      const names = new Set();
      cfg.zones.forEach((z) => {
        if (names.has(z.name)) errs.push(`Two shapes are called "${z.name}"`);
        names.add(z.name);
        if ((z.type || "road") === "road" && !(z.width_m > 0 && z.length_m > 0)) errs.push(`Enter the width and length of "${z.name}"`);
      });
      if (!cfg.zones.some((z) => (z.type || "road") === "road")) errs.push("Draw at least one road zone");
      return errs;
    }
    function validate() {
      clearTimeout(vTimer);
      vTimer = setTimeout(async () => {
        const errs = clientErrors();
        if (errs.length) { check = { roads: [], roi: null, error: errs.join(" · ") }; showCheck(); draw(); return; }
        try {
          const r = await api(`/api/cameras/${cid}/zones/validate`, { method: "POST", body: { config: cfg } });
          check = { roads: r.roads, roi: r.roi, error: "", imgsz: r.imgsz, bus_stops: r.bus_stops || [] };
        } catch (e) { check = { roads: [], roi: null, error: e.message }; }
        showCheck();
        draw();
        if (selKind() === "road" && document.activeElement.tagName !== "INPUT") renderProps();
      }, 350);
    }
    function showCheck() {
      const el = $("#check");
      const badStops = (check.bus_stops || []).filter((b) => !b.valid);
      el.innerHTML = check.error ? `<span class="errors">${esc(check.error)}</span>` :
        `<span class="okline">Ready to save.</span> <span class="muted">${dirty ? "Unsaved changes." : "No changes."} Detector ${check.imgsz || ""} px.</span>` +
        badStops.map((b) => `<div class="warnline">Bus stop “${esc(b.name)}”: ${esc(b.reason)}</div>`).join("");
      $("#save").disabled = !!check.error;
    }
    $("#save").onclick = async () => {
      try {
        const r = await api(`/api/cameras/${cid}/zones`, { method: "PUT", body: { config: cfg, note: $("#note").value } });
        dirty = false;
        $("#note").value = "";
        toast(`Saved zones version ${r.version}${r.restarted ? " – camera restarted with the new zones" : ""}`, "ok");
        $(".page-head p").innerHTML = `<span id="cam-loc">${esc(cam.location || "No location set")}</span> · saved version ${r.version} · draw on a clear frame of this camera`;
        showCheck();
      } catch (e) { fail(e); }
    };
    // -------------------------------------------------- automatic zones
    function showAutoReport(rep) {
      const card = $("#auto-card");
      card.classList.toggle("hidden", !rep);
      if (rep) card.innerHTML = `<div class="card-head"><h3>${App.ICON.auto.replace("<svg", '<svg width="16" height="16"')} Learned automatically</h3></div>${App.autoReport(rep)}
        <div class="small muted" style="margin-top:6px">Check the shapes on the picture, adjust or delete anything wrong, then <b>Save zones</b>.</div>`;
    }
    if (cfg.auto && cfg.auto.report) showAutoReport(cfg.auto.report);
    $("#auto-detect").onclick = async () => {
      if (cfg.zones.length && !confirm("Replace the current drawing with zones learned automatically from traffic? Nothing is saved until you press Save zones.")) return;
      let stopped = false;
      const m = modal(`<div class="modal-head"><h2>Learning the road layout</h2></div>
        <p class="secondary small" id="al-msg">Opening the video…</p>
        <div class="progress learn"><div id="al-bar" style="width:0"></div></div>
        <p class="small muted">The AI watches the traffic (about 2–5 minutes of video), finds the road direction, calibrates the camera
          from vehicle sizes and vertical lines, then marks carriageways, kerbs, junctions, bus stops and signal queues.</p>
        <div class="row" style="justify-content:flex-end"><button class="btn" id="al-cancel">Cancel</button></div>`, "", () => { stopped = true; });
      $("#al-cancel", m.el).onclick = async () => { stopped = true; try { await api(`/api/cameras/${cid}/auto-zones`, { method: "DELETE" }); } catch (e) { /* ignore */ } m.close(); };
      try {
        await api(`/api/cameras/${cid}/auto-zones`, { method: "POST", body: { apply: false, start: false } });
      } catch (e) { m.close(); fail(e); return; }
      while (!stopped) {
        await new Promise((r) => setTimeout(r, 1500));
        let st;
        try { st = await api(`/api/cameras/${cid}/auto-zones`); } catch (e) { continue; }
        if ($("#al-msg", m.el)) { $("#al-msg", m.el).textContent = st.message || ""; $("#al-bar", m.el).style.width = ((st.progress || 0) * 100).toFixed(1) + "%"; }
        if (st.state === "done" && st.result) {
          m.close();
          snapshot();
          const next = JSON.parse(JSON.stringify(st.result.config));
          const fs = next.frame_size;
          cfg = next;
          cfg.peak_hours = cfg.peak_hours || DEFAULT_PEAK;
          const refT = cfg.auto && cfg.auto.reference_t;
          if (cam.kind === "file" && refT != null && Math.abs(refT - frameT) > 0.5) {
            $("#t").value = refT;   // the zones were learned in the view of this frame (the camera may have moved since)
            await loadFrame();
          } else if (W && fs && (fs[0] !== W || fs[1] !== H)) { scaleAll(W / fs[0], H / fs[1]); cfg.frame_size = [W, H]; }
          sel = null;
          refresh();
          showAutoReport(cfg.auto && cfg.auto.report);
          toast("Zones learned. Check them and press Save zones.", "ok");
          return;
        }
        if (["error", "cancelled", "idle"].includes(st.state)) {
          m.close();
          if (st.state === "error") fail(new Error("Automatic setup failed: " + st.message));
          return;
        }
      }
    };

    $("#load-frame").onclick = loadFrame;
    $("#export").onclick = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" }));
      a.download = `zones_${cam.name.replace(/\W+/g, "_")}.json`;
      a.click();
    };
    $("#import").onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const data = JSON.parse(await file.text());
        if (!Array.isArray(data.zones)) throw new Error("No zones in this file");
        snapshot();
        cfg = { frame_size: data.frame_size || [W, H], peak_hours: "peak_hours" in data ? data.peak_hours : DEFAULT_PEAK,
          rules: data.rules || {}, zones: data.zones.map((z) => ({ type: "road", ...z })), lane_lines: data.lane_lines || [] };
        cfg.zones.forEach((z) => { if (z.type === "road") z.banned_classes = z.banned_classes || []; });
        if (W && (cfg.frame_size[0] !== W || cfg.frame_size[1] !== H)) { scaleAll(W / cfg.frame_size[0], H / cfg.frame_size[1]); cfg.frame_size = [W, H]; }
        sel = null;
        refresh();
        toast(`Imported ${cfg.zones.length} zones from ${esc(file.name)}`, "ok");
      } catch (err) { fail(err); }
      e.target.value = "";
    };

    refresh();
    await loadFrame();
  }

  window.Editor = { render };
})();
