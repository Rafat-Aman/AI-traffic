/* Small SVG charts: single-series bars (horizontal) and columns (vertical), with hover tooltips and a table view.
   Labels are inserted with textContent (never innerHTML). */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  let tip = null;

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function showTip(evt, value, label) {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tip";
      document.body.appendChild(tip);
    }
    tip.replaceChildren();
    const b = document.createElement("b");
    b.textContent = value;
    const s = document.createElement("span");
    s.textContent = label;
    tip.append(b, s);
    tip.style.display = "block";
    const x = Math.min(window.innerWidth - 160, evt.clientX + 14);
    tip.style.left = x + "px";
    tip.style.top = evt.clientY - 10 + "px";
  }

  function hideTip() {
    if (tip) tip.style.display = "none";
  }

  function niceMax(v) {
    if (v <= 5) return Math.max(1, Math.ceil(v));
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  // Rounded data-end only (4px), square at the baseline.
  function barPath(x, y, w, h, horizontal) {
    const r = Math.min(4, horizontal ? h / 2 : w / 2, horizontal ? w : h);
    if (w <= 0 || h <= 0) return "";
    if (horizontal) return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z`;
    return `M${x},${y + h}v-${h - r}a${r},${r} 0 0 1 ${r},-${r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
  }

  /** Horizontal bars: categories with labels on the left, values at the bar ends. */
  function bars(container, items, opts = {}) {
    container.replaceChildren();
    if (!items.length) return empty(container);
    const fmt = opts.format || ((v) => String(v));
    const unit = opts.unit || "";
    const W = Math.max(280, container.clientWidth || 520), valueW = 44;
    const labelW = Math.min(opts.labelWidth || 150, Math.round(W * 0.42)), rowH = 26, gap = 6;
    const H = items.length * (rowH + gap);
    const max = niceMax(Math.max(...items.map((d) => d.value)));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title || "bar chart" }, container);
    const plotW = W - labelW - valueW;
    items.forEach((d, i) => {
      const y = i * (rowH + gap);
      const w = (d.value / max) * plotW;
      const t = el("text", { x: labelW - 8, y: y + rowH / 2 + 4, "text-anchor": "end", class: "cat" }, svg);
      t.textContent = d.label;
      const hit = el("rect", { x: labelW, y: y - gap / 2, width: plotW + valueW, height: rowH + gap, class: "hit" }, svg);
      el("path", { d: barPath(labelW, y + 3, Math.max(w, 2), rowH - 6, true), class: "bar" }, svg);
      const v = el("text", { x: labelW + w + 6, y: y + rowH / 2 + 4, class: "val" }, svg);
      v.textContent = fmt(d.value);
      hit.addEventListener("pointermove", (e) => showTip(e, fmt(d.value) + unit, d.label));
      hit.addEventListener("pointerleave", hideTip);
    });
  }

  /** Vertical columns over an ordered x axis (hours, days). */
  function columns(container, items, opts = {}) {
    container.replaceChildren();
    if (!items.some((d) => d.value > 0)) return empty(container);
    const fmt = opts.format || ((v) => String(v));
    const W = Math.max(280, container.clientWidth || 560), H = 220, L = 34, B = 24, T = 8, R = 6;
    const max = niceMax(Math.max(...items.map((d) => d.value)));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title || "column chart" }, container);
    const plotW = W - L - R, plotH = H - T - B;
    const grid = el("g", { class: "grid" }, svg);
    for (let i = 0; i <= 4; i++) {
      const y = T + plotH - (i / 4) * plotH;
      el("line", { x1: L, x2: W - R, y1: y, y2: y }, grid);
      const t = el("text", { x: L - 6, y: y + 4, "text-anchor": "end" }, svg);
      t.textContent = fmt(Math.round((max * i) / 4 * 10) / 10);
    }
    const step = plotW / items.length;
    const bw = Math.max(2, Math.min(28, step - 2));
    const every = opts.labelEvery || Math.ceil(items.length / 12);
    items.forEach((d, i) => {
      const x = L + i * step + (step - bw) / 2;
      const h = (d.value / max) * plotH;
      const hit = el("rect", { x: L + i * step, y: T, width: step, height: plotH, class: "hit" }, svg);
      if (d.value > 0) el("path", { d: barPath(x, T + plotH - h, bw, h, false), class: "bar" }, svg);
      if (i % every === 0) {
        const t = el("text", { x: x + bw / 2, y: H - 6, "text-anchor": "middle" }, svg);
        t.textContent = d.label;
      }
      hit.addEventListener("pointermove", (e) => showTip(e, fmt(d.value), d.tip || d.label));
      hit.addEventListener("pointerleave", hideTip);
    });
    el("line", { x1: L, x2: W - R, y1: T + plotH, y2: T + plotH, stroke: "var(--axis)" }, svg);
  }

  function table(container, items, headers) {
    container.replaceChildren();
    const t = document.createElement("table");
    t.className = "chart-table";
    const hr = t.insertRow();
    headers.forEach((h) => { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); });
    items.forEach((d) => {
      const r = t.insertRow();
      r.insertCell().textContent = d.tip || d.label;
      r.insertCell().textContent = d.value;
    });
    container.appendChild(t);
  }

  function empty(container) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = "No data yet";
    container.appendChild(d);
  }

  window.Charts = { bars, columns, table };
})();
