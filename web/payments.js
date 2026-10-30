/* Admin page: Fines, Registry (with CSV import), Payments, Outbox. Every value from the server is escaped before
   it reaches the page, and nothing here ever sees a card number or a gateway credential. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pane = $("#pane");
  let tab = "fines";

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
  const fail = (e) => toast(e.message || String(e));
  const when = (s) => esc((s || "").replace("T", " "));

  // ---------------------------------------------------------------- Fines
  async function fines() {
    const d = await api("/api/fines");
    pane.innerHTML = `<div class="card-head"><h2>Fines</h2></div>
      <p class="note">Amounts are whole taka and are the authority's to set; this system holds none by default. An amount without a legal
        reference is a placeholder: check it against the Road Transport Act 2018 before it is used. Advisory codes are never fined.
        A citation with no amount cannot be paid.</p>
      <div class="scroll"><table><thead><tr><th>Code</th><th>Violation</th><th class="num">Amount (BDT)</th><th>Legal reference</th><th></th></tr></thead><tbody>
      ${d.items.map((r) => `<tr data-code="${esc(r.code)}"><td><b>${esc(r.code)}</b></td><td>${esc(r.title)}</td>
        ${r.advisory ? `<td colspan="3"><span class="chip">advisory - no fine</span></td>` : `
        <td class="num"><input type="number" min="1" step="1" class="amt" value="${r.amount_bdt ?? ""}" placeholder="not set" style="width:100px"></td>
        <td><input class="ref" value="${esc(r.legal_ref)}" placeholder="e.g. section and rule" maxlength="200" style="width:100%">
          ${r.note ? `<div class="small ${r.placeholder ? "chip warn" : "muted"}">${esc(r.note)}</div>` : ""}
          ${r.source === "environment" ? '<div class="small muted">from FINE_SCHEDULE_BDT</div>' : ""}</td>
        <td><button class="btn small save">Save</button> ${r.source === "database" ? '<button class="btn small clear">Clear</button>' : ""}</td>`}
      </tr>`).join("")}</tbody></table></div>`;
    pane.querySelectorAll("tr[data-code]").forEach((tr) => {
      const code = tr.dataset.code;
      const save = $(".save", tr), clear = $(".clear", tr);
      if (save) save.onclick = async () => {
        const amount = parseInt($(".amt", tr).value, 10);
        if (!Number.isInteger(amount)) { toast("Enter a whole number of taka"); return; }
        try { await api(`/api/fines/${code}`, { method: "PUT", body: { amount_bdt: amount, legal_ref: $(".ref", tr).value } }); toast(`${code} saved`); fines(); } catch (e) { fail(e); }
      };
      if (clear) clear.onclick = async () => {
        if (!confirm(`Remove the stored amount for ${code}?`)) return;
        try { await api(`/api/fines/${code}`, { method: "DELETE" }); fines(); } catch (e) { fail(e); }
      };
    });
  }

  // ---------------------------------------------------------------- Registry
  async function registry(q = "") {
    const d = await api("/api/owners?q=" + encodeURIComponent(q));
    pane.innerHTML = `<div class="card-head"><h2>Vehicle owner registry</h2><span class="spacer"></span><span class="small muted">${d.total} entries</span></div>
      <p class="note">Plate to owner. The email here is where notices and receipts go; a licence number also lets that driver pay for the plate in the portal.</p>
      <div class="row"><input id="q" placeholder="Search plate, name, email, licence" value="${esc(q)}"><button class="btn small" id="go">Search</button></div>
      <div class="row"><input id="f-plate" placeholder="Plate *"><input id="f-name" placeholder="Owner name"><input id="f-mobile" placeholder="Mobile">
        <input id="f-email" placeholder="Email"><input id="f-lic" placeholder="Licence no."><button class="btn primary small" id="add">Add</button></div>
      <div class="scroll"><table><thead><tr><th>Plate</th><th>Owner</th><th>Mobile</th><th>Email</th><th>Licence</th><th></th></tr></thead><tbody>
      ${d.items.map((r) => `<tr data-id="${r.id}"><td><b>${esc(r.plate_text || r.plate_key)}</b></td><td>${esc(r.owner_name)}</td><td>${esc(r.mobile)}</td>
        <td>${esc(r.email)}</td><td>${esc(r.licence_no)}</td><td><button class="btn small edit">Edit</button> <button class="btn small del">Delete</button></td></tr>`).join("")
        || '<tr><td colspan="6" class="muted">No entries.</td></tr>'}</tbody></table></div>
      <h3 style="margin-top:18px">Import from CSV</h3>
      <p class="note">Columns: plate (required), owner_name, mobile, email, licence_no. Preview first; nothing changes until you apply.</p>
      <input type="file" id="file" accept=".csv,text/csv"><textarea id="csv" placeholder="plate,owner_name,mobile,email,licence_no"></textarea>
      <div class="row"><button class="btn small" id="preview">Preview (dry run)</button><button class="btn primary small" id="apply" disabled>Apply</button></div>
      <div id="import-out"></div>`;
    $("#go").onclick = () => registry($("#q").value);
    $("#q").onkeydown = (e) => { if (e.key === "Enter") registry($("#q").value); };
    const form = () => ({ plate: $("#f-plate").value, owner_name: $("#f-name").value, mobile: $("#f-mobile").value, email: $("#f-email").value, licence_no: $("#f-lic").value });
    $("#add").onclick = async () => { try { await api("/api/owners", { method: "POST", body: form() }); toast("Added"); registry(q); } catch (e) { fail(e); } };
    pane.querySelectorAll("tr[data-id]").forEach((tr) => {
      const row = d.items.find((r) => r.id === +tr.dataset.id);
      $(".del", tr).onclick = async () => { if (confirm(`Delete ${row.plate_text || row.plate_key}?`)) { try { await api(`/api/owners/${row.id}`, { method: "DELETE" }); registry(q); } catch (e) { fail(e); } } };
      $(".edit", tr).onclick = () => {
        tr.innerHTML = `<td><input class="e-plate" value="${esc(row.plate_text || row.plate_key)}"></td><td><input class="e-name" value="${esc(row.owner_name)}"></td>
          <td><input class="e-mobile" value="${esc(row.mobile)}"></td><td><input class="e-email" value="${esc(row.email)}"></td><td><input class="e-lic" value="${esc(row.licence_no)}"></td>
          <td><button class="btn primary small ok">Save</button></td>`;
        $(".ok", tr).onclick = async () => {
          try { await api(`/api/owners/${row.id}`, { method: "PUT", body: { plate: $(".e-plate", tr).value, owner_name: $(".e-name", tr).value, mobile: $(".e-mobile", tr).value, email: $(".e-email", tr).value, licence_no: $(".e-lic", tr).value } }); registry(q); } catch (e) { fail(e); }
        };
      };
    });
    $("#file").onchange = async (e) => { const f = e.target.files[0]; if (f) { $("#csv").value = await f.text(); $("#apply").disabled = true; } };
    $("#csv").oninput = () => { $("#apply").disabled = true; };
    const run = async (apply) => {
      try {
        const r = await api("/api/owners/import", { method: "POST", body: { csv: $("#csv").value, apply } });
        const c = r.counts;
        $("#import-out").innerHTML = `<p class="note"><b>${r.applied ? "Applied" : "Preview"}:</b> ${c.add} to add, ${c.update} to update, ${c.same} unchanged, ${c.error} refused.</p>
          <div class="scroll"><table><thead><tr><th>Line</th><th>Plate</th><th>Result</th></tr></thead><tbody>${r.rows.map((x) =>
            `<tr><td>${x.line}</td><td>${esc(x.plate)}</td><td><span class="chip ${x.action === "error" ? "bad" : x.action === "same" ? "" : "ok"}">${esc(x.action)}</span> ${esc(x.reason)}</td></tr>`).join("")}</tbody></table></div>`;
        $("#apply").disabled = r.applied || (c.add + c.update) === 0;
        if (r.applied) { toast("Import applied"); registry(q); }
      } catch (e) { fail(e); }
    };
    $("#preview").onclick = () => run(false);
    $("#apply").onclick = () => { if (confirm("Apply this import? Lines marked refused are skipped.")) run(true); };
  }

  // ---------------------------------------------------------------- Payments
  async function payments() {
    const d = await api("/api/payments");
    const p = d.providers;
    pane.innerHTML = `<div class="card-head"><h2>Payments</h2></div>
      ${p.test ? '<div class="banner-test">The built-in test gateway is on: payments made with it are labelled "Test - no money moves".</div>' : ""}
      <p class="note">SSLCommerz: ${p.sslcommerz ? (p.sslcommerz_live ? "<b>LIVE - real money</b>" : "configured (sandbox)") : "not configured"}.
        Email: ${esc(d.smtp)}. A payment counts only after the gateway's validation API confirms it.</p>
      <div class="scroll"><table><thead><tr><th>Transaction</th><th>Citation</th><th>Provider</th><th class="num">BDT</th><th>Status</th><th>Time</th><th></th></tr></thead><tbody>
      ${d.items.map((r) => `<tr><td>${esc(r.tran_id)}</td><td>${esc(r.citation_no)}</td><td>${esc(r.provider)}</td><td class="num">${r.amount_bdt}</td>
        <td><span class="chip ${r.status === "paid" ? "ok" : r.status === "initiated" ? "" : "bad"}">${esc(r.status)}</span>
          ${r.detail ? `<div class="small muted">${esc(r.detail)}</div>` : ""}</td><td>${when(r.paid_at || r.created_at)}</td>
        <td>${r.status === "paid" ? `<a class="btn small" href="/api/payments/${encodeURIComponent(r.tran_id)}/invoice.pdf">Invoice</a>` : ""}</td></tr>`).join("")
        || '<tr><td colspan="7" class="muted">No payments yet.</td></tr>'}</tbody></table></div>`;
  }

  // ---------------------------------------------------------------- Outbox
  async function outbox() {
    const d = await api("/api/outbox");
    pane.innerHTML = `<div class="card-head"><h2>Outbox</h2></div>
      <p class="note">${d.mode === "development" ? "Development mode: no SMTP server is set, so messages are saved here and as .eml files in data/outbox/. Nothing is sent." : "Messages are sent through the configured SMTP server; failures are retried with a growing delay."}</p>
      <div class="scroll"><table><thead><tr><th>#</th><th>To</th><th>Subject</th><th>Status</th><th>Created</th><th></th></tr></thead><tbody>
      ${d.items.map((m) => `<tr><td>${m.id}</td><td>${esc(m.recipient) || '<span class="muted">no address</span>'}</td><td>${esc(m.subject)}</td>
        <td><span class="chip ${m.status === "failed" ? "bad" : m.status === "sent" || m.status === "saved" ? "ok" : "warn"}">${esc(m.status)}</span>
          ${m.last_error ? `<div class="small muted">${esc(m.last_error)}</div>` : ""}</td><td>${when(m.created_at)}</td>
        <td><button class="btn small view" data-id="${m.id}">Open</button>${m.status === "failed" ? ` <button class="btn small retry" data-id="${m.id}">Retry</button>` : ""}</td></tr>`).join("")
        || '<tr><td colspan="6" class="muted">No messages yet.</td></tr>'}</tbody></table></div><div id="msg"></div>`;
    pane.querySelectorAll(".view").forEach((b) => b.onclick = async () => {
      try {
        const m = await api("/api/outbox/" + b.dataset.id);
        $("#msg").innerHTML = `<h3>${esc(m.subject)}</h3><div class="small muted">From ${esc(m.from)} to ${esc(m.to) || "(nobody)"}</div>
          <div class="mail-view">${esc(m.text)}</div>${m.attachments.length ? `<div class="small">Attachments: ${m.attachments.map((a) => `${esc(a.name)} (${a.bytes} bytes)`).join(", ")}</div>` : ""}`;
        $("#msg").scrollIntoView({ behavior: "smooth" });
      } catch (e) { fail(e); }
    });
    pane.querySelectorAll(".retry").forEach((b) => b.onclick = async () => { try { await api(`/api/outbox/${b.dataset.id}/retry`, { method: "POST" }); outbox(); } catch (e) { fail(e); } });
  }

  const views = { fines, registry, payments, outbox };
  const show = (t) => {
    tab = t;
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    pane.innerHTML = '<div class="empty">Loading...</div>';
    views[t]().catch((e) => { pane.innerHTML = `<div class="errors">${esc(e.message)}</div>`; });
  };
  document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => show(b.dataset.tab));
  fetch("/api/me").then((r) => r.json()).then((m) => { $("#who").textContent = m.name || m.username || ""; }).catch(() => {});
  show(tab);
})();
