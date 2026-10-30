(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // Pictures and the clip arrive as signed links that expire (an hour), already complete URLs: nothing here
  // builds a path to a file, and there is no path in the page to build one from.
  const api = async (u, o) => {
    const r = await fetch(u, { headers: { "Content-Type": "application/json" }, ...o });
    if (r.status === 401) { location.href = "/login"; throw new Error("signed out"); }
    let d = {}; try { d = await r.json(); } catch (e) { /* empty */ }
    if (!r.ok) throw new Error(d.detail || r.statusText);
    return d;
  };
  const when = (v) => v.wall_time || (v.video_time_s != null ? `${(+v.video_time_s).toFixed(0)} s into the recording` : "");

  $("#out").onclick = async () => { await api("/api/logout", { method: "POST" }); location.href = "/login"; };

  async function load() {
    const me = await api("/api/me");
    $("#who").textContent = me.name || me.licence_no || "";
    const r = await api("/api/portal/violations");
    const vehicles = r.vehicles || [];
    const plates = vehicles.map((v) => `<span class="plate${v.verified ? "" : " pending"}">${esc(v.plate_en || v.plate_key)}${
      v.verified ? "" : " · awaiting verification"}</span>`).join("");

    $("#body").innerHTML = `
      <div class="card">
        <div class="card-head"><h2>My vehicles</h2></div>
        <div class="plates">${plates || '<span class="muted small">No vehicles registered yet.</span>'}</div>
        <div class="row">
          <input id="plate" placeholder="Add a number plate, e.g. DHAKA METRO-GA 12-3456" style="flex:1">
          <button class="btn" id="add">Add</button>
        </div>
        <div class="small muted" style="margin-top:7px">A vehicle you add shows nothing until the traffic
          authority has verified that it is yours.</div>
      </div>
      <div class="card">
        <div class="card-head"><h2>Citations</h2><span class="spacer" style="flex:1"></span>
          <span class="small muted">${r.total} citation${r.total === 1 ? "" : "s"}</span></div>
        <div class="p-note">${esc(r.note)}</div>
        <div id="cases">${r.items.length ? r.items.map(card).join("")
          : '<div class="empty">No citations have been issued against your registered vehicles.</div>'}</div>
      </div>`;

    $("#add").onclick = async () => {
      const plate = $("#plate").value.trim();
      if (!plate) return;
      try { const res = await api("/api/portal/vehicles", { method: "POST", body: JSON.stringify({ plate }) });
            alert(res.message || "Added."); load(); } catch (e) { alert(e.message); }
    };
    document.querySelectorAll(".case").forEach((el) => el.onclick = () => detail(+el.dataset.id));
    returnBanner();
    annotateStates();
  }

  // ---- paying a fine (WP18). Nothing here handles card details: the SSLCommerz button sends the driver to the
  // gateway's own page, and the test gateway has two buttons and no fields.
  const STATE = { unpaid: ["Unpaid", ""], paid: ["Paid", "ok"], paused: ["Paused by dispute", "review"], cancelled: ["Cancelled", "void"] };
  const stateBadge = (st) => { const [t, k] = STATE[st.state] || [st.state, ""]; return `<span class="badge ${k}">${esc(t)}</span>`; };
  const money = (st) => st.amount_bdt == null ? "Amount not set by the authority" : "BDT " + st.amount_bdt.toLocaleString("en-US");

  // The gateway sends the browser back to /portal?payment=...; say what happened, then clean the address.
  function returnBanner() {
    const q = new URLSearchParams(location.search), r = q.get("payment");
    if (!r) return;
    const msg = { paid: ["ok", "Payment received. Your invoice is on the citation page."],
                  checking: ["", "Your payment is being confirmed. Refresh in a minute."],
                  cancelled: ["bad", "Payment cancelled. Nothing was charged."],
                  failed: ["bad", "The payment was not completed, or could not be confirmed. If money left your account, quote the citation number to the authority."] }[r];
    history.replaceState(null, "", "/portal");
    if (msg) $("#body").insertAdjacentHTML("afterbegin", `<div class="pay-banner ${msg[0]}">${esc(msg[1])}</div>`);
  }

  async function annotateStates() {
    try {
      const r = await api("/api/portal/citations/states");
      document.querySelectorAll(".case[data-no]").forEach((el) => {
        const st = r.items[el.dataset.no];
        if (st) el.querySelector(".meta").insertAdjacentHTML("beforeend", `<br>${esc(money(st))} ${stateBadge(st)}`);
      });
    } catch (e) { /* the list still works without amounts */ }
  }

  async function payBox(no, back) {
    const box = $("#paybox");
    if (!box) return;
    const st = await api(`/api/portal/citations/${encodeURIComponent(no)}/payment`);
    box.innerHTML = `<div class="small"><b>Payment</b> ${stateBadge(st)}</div>
      <div class="pay-amount" style="margin:6px 0">${esc(money(st))}</div>
      <div class="small muted">Due by ${esc(st.due_at)}.${st.note ? " " + esc(st.note) : ""}</div>
      ${st.reason ? `<div class="small" style="margin-top:6px">${esc(st.reason)}</div>` : ""}
      ${st.state === "paid" ? `<div class="small" style="margin-top:6px">Paid ${esc((st.paid_at || "").replace("T", " "))}${st.provider === "test" ? ' <span class="pay-test">Test - no money moved</span>' : ""}
        <br><a class="btn small" style="margin-top:8px" href="/api/portal/citations/${encodeURIComponent(no)}/invoice.pdf">Download invoice (PDF)</a></div>` : ""}
      ${st.can_pay ? '<button class="btn primary" id="paybtn" style="margin-top:10px">Pay the fine</button>' : ""}`;
    const b = $("#paybtn");
    if (b) b.onclick = () => payDialog(no, st, back);
  }

  function payDialog(no, st, back) {
    const pr = st.providers;
    const ov = document.createElement("div");
    ov.className = "modal-back";
    ov.innerHTML = `<div class="modal" role="dialog" aria-modal="true" aria-label="Pay the fine">
      <div class="modal-head"><h2 style="margin:0">Pay the fine</h2><span class="spacer" style="flex:1"></span><button class="btn small" id="pclose">Close</button></div>
      <div class="pay-amount">${esc(money(st))}</div><div class="small muted">Citation ${esc(no)}</div>
      <div class="pay-actions">
        ${pr.sslcommerz ? `<button class="btn primary" id="p-ssl">Pay with SSLCommerz (card, bKash, Nagad, bank)</button>
          <div class="small muted">You will leave this site to pay on SSLCommerz${pr.sslcommerz_live ? "" : " (sandbox: not real money)"}. Your card details are never sent to us.</div>` : ""}
        ${pr.test ? `<div><span class="pay-test">Test — no money moves</span></div>
          <button class="btn primary" id="p-ok">Simulate successful payment</button>
          <button class="btn" id="p-no">Simulate declined payment</button>` : ""}
        ${!pr.sslcommerz && !pr.test ? '<div class="small">No way of paying is available right now.</div>' : ""}
        <div class="small errors" id="perr"></div></div></div>`;
    document.body.appendChild(ov);
    const close = () => ov.remove();
    $("#pclose", ov).onclick = close;
    const start = async (provider) => api(`/api/portal/citations/${encodeURIComponent(no)}/pay`, { method: "POST", body: JSON.stringify({ provider }) });
    const err = (e) => { $("#perr", ov).textContent = e.message; ov.querySelectorAll("button").forEach((x) => x.disabled = false); };
    const lock = () => ov.querySelectorAll(".pay-actions button").forEach((x) => x.disabled = true);
    if ($("#p-ssl", ov)) $("#p-ssl", ov).onclick = async () => { lock(); try { const r = await start("sslcommerz"); location.href = r.redirect; } catch (e) { err(e); } };
    const sim = (outcome) => async () => {
      lock();
      try {
        const r = await start("test");
        await api(`/api/portal/citations/${encodeURIComponent(no)}/pay/simulate`, { method: "POST", body: JSON.stringify({ tran_id: r.tran_id, outcome }) });
        close(); back();
      } catch (e) { err(e); }
    };
    if ($("#p-ok", ov)) { $("#p-ok", ov).onclick = sim("success"); $("#p-no", ov).onclick = sim("declined"); }
  }

  // One line for a list row: the citation, and where a dispute stands. Withdrawn citations stay listed as void.
  const badge = (c) => c.status === "withdrawn" ? '<span class="badge void">withdrawn</span>'
    : c.dispute && c.dispute.status === "open" ? '<span class="badge review">dispute under review</span>'
    : c.dispute && c.dispute.status === "upheld" ? '<span class="badge ok">dispute decided</span>' : "";
  const citedLine = (v) => !v.citation ? "" : `<br><b>Citation ${esc(v.citation.number)}</b>${badge(v.citation)}`;

  // The citation as the driver reads it. "Set by the authority" is the honest fine text until the authority's
  // schedule is loaded: the system does not guess an amount.
  const citationBox = (v) => {
    const c = v.citation;
    if (!c) return "";
    return `<div class="card" style="background:var(--surface-2)">
        <div class="small"><b>Citation ${esc(c.number)}</b>${badge(c)}</div>
        <div class="small secondary" style="margin-top:5px">Issued ${esc((c.issued_at || "").replace("T", " "))}.
          Fine: ${esc(c.fine_text)}.${c.status === "withdrawn"
            ? " The authority has withdrawn this citation; it is void and no longer applies." : ""}</div>
        <div class="small muted" style="margin-top:7px">Quote this number when you contact the traffic authority.</div></div>
      <div class="card pay-box" id="paybox"><div class="small muted">Loading payment...</div></div>
      ${disputeBox(c)}`;
  };

  // Dispute: status and the final reason are all a driver is shown of the review. What the authority's tools
  // made of the pictures is working material for the reviewer and is not displayed here.
  const disputeBox = (c) => {
    const d = c.dispute;
    if (d) {
      const done = d.status !== "open";
      return `<div class="card dispute"><div class="small"><b>Your dispute</b><span class="badge ${
        d.status === "accepted" ? "ok" : d.status === "open" ? "review" : ""}">${esc(d.label)}</span></div>
        <div class="small muted" style="margin-top:4px">Filed ${esc((d.filed_at || "").replace("T", " "))}</div>
        <div class="small quote">${esc(d.statement)}</div>
        ${done && d.reason ? `<div class="small" style="margin-top:8px"><b>Decision${d.decided_at ? " (" + esc(d.decided_at.replace("T", " ")) + ")" : ""}</b></div>
          <div class="small secondary quote">${esc(d.reason)}</div>` : ""}
        ${!done ? '<div class="small muted" style="margin-top:6px">A member of staff will review the evidence and your statement. You will see the decision here.</div>' : ""}
        ${d.status === "upheld" ? '<div class="small muted" style="margin-top:6px">The citation stands. There is no further dispute in this system; contact the traffic authority if you wish to take it further.</div>' : ""}
      </div>`;
    }
    if (!c.can_dispute) return c.cannot_reason ? `<div class="small muted" style="margin:10px 0">${esc(c.cannot_reason)}</div>` : "";
    return `<div class="card dispute"><div class="small"><b>Dispute this citation</b></div>
      <div class="small muted" style="margin:4px 0 8px">You can dispute it until <b>${esc(c.until)}</b> (${c.window_days} days from the day it was issued),
        once. Write what you believe is wrong, in plain text, 20 to 2000 characters. Look at the pictures and the video above first and say what
        they do or do not show. A member of staff decides; you cannot attach files in this version.</div>
      <textarea id="stmt" maxlength="2000" placeholder="Why do you believe this citation is wrong?"></textarea>
      <div class="row" style="margin-top:6px"><span class="small muted" id="cnt">0 / 2000</span><span class="spacer" style="flex:1"></span>
        <button class="btn" id="send">Submit dispute</button></div><div class="small errors" id="derr"></div></div>`;
  };

  const card = (v) => `<div class="case" data-id="${v.id}"${v.citation ? ` data-no="${esc(v.citation.number)}"` : ""}>
      <img src="${esc(v.crop || v.snapshot || "")}" alt="" loading="lazy">
      <div><h3>${esc(v.title)}</h3>
        <div class="meta">${esc(when(v))}${v.camera ? " · " + esc(v.camera) : ""}<br>
          ${esc(v.cls || "")}${v.plate_text_en ? " · " + esc(v.plate_text_en) : ""}
          ${v.kph != null ? " · " + Math.round(v.kph) + " km/h" : ""}${citedLine(v)}</div></div></div>`;

  async function detail(id) {
    const v = await api(`/api/portal/violation/${id}`);
    const items = [];
    if (v.snapshot) items.push([v.snapshot, "The scene, with the vehicle outlined", "img"]);
    if (v.crop) items.push([v.crop, "Close-up", "img"]);
    (v.evidence || []).forEach((e) => items.push([e.url, e.label, "img"]));
    if (v.plate_img) items.push([v.plate_img, "Number plate as read", "img"]);
    if (v.clip) items.push([v.clip, "Video clip around the event", "video"]);
    const x = v.vlm || {};
    $("#body").innerHTML = `<div class="card">
      <div class="card-head"><button class="btn small" id="back">&larr; Back</button>
        <span class="spacer" style="flex:1"></span><span class="small muted">Case TV-${v.id}</span></div>
      <h2 style="margin:10px 0 2px">${esc(v.title)}</h2>
      <div class="meta small muted">${esc(when(v))}${v.camera ? " · " + esc(v.camera) : ""}
        ${v.plate_text_en ? " · " + esc(v.plate_text_en) : ""}</div>
      ${citationBox(v)}
      <div class="ev">${items.map(([src, cap, kind]) => `<figure style="margin:0">${
        kind === "video" ? `<video src="${esc(src)}" controls muted playsinline preload="metadata"></video>`
                         : `<img src="${esc(src)}" alt="${esc(cap)}" loading="lazy">`}
        <figcaption>${esc(cap)}</figcaption></figure>`).join("")}</div>
      <div class="small muted">The links to these pictures and the video stop working after an hour. If one
        stops opening, go back and open the case again.</div>
      ${x.summary ? `<div class="card" style="background:var(--surface-2)">
        <div class="small"><b>What the system recorded</b></div>
        <div class="small secondary" style="margin-top:5px">${esc(x.summary)}</div>
        <div class="small muted" style="margin-top:7px">Written automatically from the images above. The
          images and the video are the evidence; this description is a summary of them.</div></div>` : ""}
      <div class="small muted" style="margin-top:14px">If you believe this is wrong, use the dispute form above. If the vehicle is not yours, contact the traffic authority quoting case TV-${v.id}.</div>
    </div>`;
    $("#back").onclick = load;
    if (v.citation) payBox(v.citation.number, () => detail(id)).catch((e) => { $("#paybox").textContent = e.message; });
    const send = $("#send");
    if (send) {
      const box = $("#stmt");
      box.oninput = () => { $("#cnt").textContent = `${box.value.trim().length} / 2000`; };
      send.onclick = async () => {
        const statement = box.value.trim();
        if (statement.length < 20) { $("#derr").textContent = "Please write at least 20 characters."; return; }
        if (!confirm("Submit this dispute? You can dispute a citation only once.")) return;
        send.disabled = true;
        try { await api(`/api/portal/citations/${encodeURIComponent(v.citation.number)}/dispute`,
                        { method: "POST", body: JSON.stringify({ statement }) });
              detail(id); }
        catch (e) { $("#derr").textContent = e.message; send.disabled = false; }
      };
    }
  }

  load().catch((e) => { $("#body").innerHTML = `<div class="card errors">${esc(e.message)}</div>`; });
})();
