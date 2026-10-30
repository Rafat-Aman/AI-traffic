"""HTTP surface of fines, the owner registry, online payment, invoices and the outbox (WP18): a feature module.

Wiring (the only edits server.py needs; the exact PUBLIC_API / DRIVER_API lines are in docs/PAYMENTS.md):
    from . import api_payments
    pay = api_payments.build(store, DATA, current_user, require, CONTROLLER, ADMIN, DRIVER)
    app.include_router(pay.router)
    cit.on_issued.append(pay.on_issued)             # the citation-issued email
    disputes_mod.on_decided.append(pay.on_decided)  # the dispute-decided email
    pay.mailer.start()   # in lifespan, before `yield`;  pay.mailer.stop() after it

Endpoints
  admin    GET /payments (page) | GET /api/fines | PUT,DELETE /api/fines/{code}
           GET,POST /api/owners | PUT,DELETE /api/owners/{id} | POST /api/owners/import
           GET /api/payments | GET /api/outbox | GET /api/outbox/{id} | POST /api/outbox/{id}/retry
  staff    GET /api/payments/{tran_id}/invoice.pdf
  driver   GET /api/portal/citations/states | GET /api/portal/citations/{no}/payment
           POST /api/portal/citations/{no}/pay | POST /api/portal/citations/{no}/pay/simulate
           GET /api/portal/citations/{no}/invoice.pdf          (own citations only; anyone else's is a 404)
  public   POST,GET /api/payments/sslcommerz/{success,fail,cancel,ipn}   (the gateway calls these; see payments.py)
"""

import json
import os
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import citation as cit
from . import disputes as disputes_mod
from . import invoice
from . import mailer as mailer_mod
from . import payments as pay
from . import registry as reg

WEB = Path(__file__).resolve().parents[1] / "web"
MAX_CALLBACK_BYTES = 64_000


class FineIn(BaseModel):
    amount_bdt: int
    legal_ref: str = Field("", max_length=200)


class EntryIn(BaseModel):
    plate: str = ""
    owner_name: str = ""
    mobile: str = ""
    email: str = ""
    licence_no: str = ""


class ImportIn(BaseModel):
    csv: str = Field("", max_length=1_500_000)
    apply: bool = False


class PayIn(BaseModel):
    provider: str = ""


class SimulateIn(BaseModel):
    tran_id: str = ""
    outcome: str = ""


def _plate_of(c):
    try:
        return json.loads(c.get("manifest") or "{}").get("plate") or ""
    except ValueError:
        return ""


def build(store, data_dir, current_user, require, controller="controller", admin="admin", driver="driver", *,
          env=None, http=None, mailer=None, web_dir=None):
    """Make the router, the mailer and the two hook functions."""
    env = os.environ if env is None else env
    pay.ensure_schema(store)
    reg.ensure_schema(store)
    mail = mailer or mailer_mod.Mailer(store, Path(data_dir) / "outbox", env=env)
    ssl = pay.SslCommerz(env, http)

    def base_url():
        return (env.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")

    def portal_link():
        return (base_url() or "") + "/portal"

    def owner_email(plate):
        row = reg.by_plate(store, plate)
        return (row or {}).get("email", "")

    # ------------------------------------------------------------ emails
    def on_issued(c, v):
        """Citation-issued email to the registered owner. No evidence is attached: the evidence is for the portal."""
        try:
            amount = pay.citation_amount(store, c)
            plate = _plate_of(c) or (v or {}).get("plate_text_en") or ""
            window = disputes_mod.WINDOW_DAYS
            lines = [f"A traffic citation has been issued against vehicle {plate or '(plate unreadable)'}.", "",
                     f"Case number: {c['citation_no']}", f"Violation: {(v or {}).get('title') or pay.citation_code(c)}",
                     f"Issued: {c['issued_at'].replace('T', ' ')}", f"Fine: {cit.fine_text(amount)}", "",
                     "What you can do", f"  - Dispute it within {window} days of issue: sign in to the portal, open the "
                     "case, look at the evidence and press Dispute. A member of staff decides.",
                     "  - Pay it online from the same page once the amount is set and no dispute is open.",
                     f"  Portal: {portal_link()}", "",
                     "Quote the case number if you contact the traffic authority.",
                     "This message contains no photographs or video; they are shown on the portal after you sign in."]
            mail.enqueue("citation_issued", owner_email(plate), f"Traffic citation {c['citation_no']}",
                         "\n".join(lines), related=c["citation_no"])
        except Exception as e:      # an email problem must never undo or block issuing the citation
            print(f"[payments] citation email not queued: {e}")

    def on_decided(d):
        try:
            c = store.citation(d["citation_no"])
            if not c or d["status"] not in ("upheld", "accepted"):
                return
            plate = _plate_of(c)
            verdict = ("accepted: the citation has been withdrawn and nothing is owed" if d["status"] == "accepted"
                       else "not accepted: the citation stands and payment is due again")
            body = (f"Your dispute of citation {c['citation_no']} has been decided.\n\nDecision: {verdict}.\n"
                    f"Reason given: {d['decision_reason']}\n\nPortal: {portal_link()}\n")
            mail.enqueue("dispute_decided", owner_email(plate), f"Dispute decided: {c['citation_no']}", body,
                         related=c["citation_no"])
        except Exception as e:
            print(f"[payments] dispute email not queued: {e}")

    def invoice_bytes(p, c):
        v = store.violation(c["violation_id"]) or {}
        return invoice.invoice_pdf(p, c, _plate_of(c), v.get("title") or pay.citation_code(c))

    def after_paid(p):
        try:
            c = store.citation(p["citation_no"])
            body = (f"Thank you. Your payment for citation {c['citation_no']} was received.\n\n"
                    f"Amount: BDT {p['amount_bdt']:,}\nTransaction id: {p['tran_id']}\n"
                    + ("Test - no money moved.\n" if p["provider"] == "test" else "")
                    + "The invoice is attached.\n")
            mail.enqueue("receipt", owner_email(_plate_of(c)), f"Receipt for citation {c['citation_no']}", body,
                         [(f"invoice-{c['citation_no']}.pdf", invoice_bytes(p, c), "application/pdf")],
                         related=c["citation_no"])
        except Exception as e:
            print(f"[payments] receipt not queued: {e}")

    # ------------------------------------------------------------ router
    router = APIRouter()
    admin_only, staff, drv = require(admin), require(controller), require(driver)
    page = (web_dir or WEB) / "payments.html"
    who = lambda u: u.get("address", "")      # noqa: E731

    def guarded(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except (pay.PayError, reg.RegistryError) as e:
            raise HTTPException(e.status, str(e)) from None

    @router.get("/payments")
    def payments_page(user=Depends(current_user)):
        if user is None:
            return RedirectResponse("/login", status_code=302)
        if user["role"] != admin:
            return RedirectResponse("/portal" if user["role"] == driver else "/", status_code=302)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    # -- the gateway's callbacks: public, believed for nothing but "go and ask the gateway"
    async def callback_fields(request):
        raw = (await request.body())[:MAX_CALLBACK_BYTES]
        fields = {k: v[0] for k, v in urllib.parse.parse_qs(raw.decode("utf-8", "replace")).items()}
        for k, v in request.query_params.items():
            fields.setdefault(k, v)
        return fields

    def back_to_portal(outcome, p=None):
        q = {"payment": outcome}
        if p:
            q["case"] = p["citation_no"]
        return RedirectResponse("/portal?" + urllib.parse.urlencode(q), status_code=303)

    @router.api_route("/api/payments/sslcommerz/success", methods=["GET", "POST"])
    async def ssl_success(request: Request):
        f = await callback_fields(request)
        outcome, p, newly = pay.settle(store, ssl, f.get("val_id"), f.get("tran_id", ""))
        if newly:
            after_paid(p)
        return back_to_portal({"paid": "paid", "already": "paid", "pending": "checking"}.get(outcome, "failed"), p)

    @router.api_route("/api/payments/sslcommerz/fail", methods=["GET", "POST"])
    async def ssl_fail(request: Request):
        f = await callback_fields(request)
        pay.return_marker(store, f.get("tran_id", ""), "failed")
        return back_to_portal("failed")

    @router.api_route("/api/payments/sslcommerz/cancel", methods=["GET", "POST"])
    async def ssl_cancel(request: Request):
        f = await callback_fields(request)
        pay.return_marker(store, f.get("tran_id", ""), "cancelled")
        return back_to_portal("cancelled")

    @router.api_route("/api/payments/sslcommerz/ipn", methods=["GET", "POST"])
    async def ssl_ipn(request: Request):
        f = await callback_fields(request)
        outcome, p, newly = pay.settle(store, ssl, f.get("val_id"), f.get("tran_id", ""))
        if newly:
            after_paid(p)
        # 503 asks the gateway to call again when we could not reach its validation API; anything else is final.
        return JSONResponse({"ok": outcome in ("paid", "already")}, status_code=503 if outcome == "pending" else 200)

    # -- fines (admin)
    @router.get("/api/fines")
    def get_fines(user=Depends(admin_only)):
        return {"items": pay.fines_table(store), "placeholder_text": pay.REF_PLACEHOLDER}

    @router.put("/api/fines/{code}")
    def put_fine(code: str, body: FineIn, user=Depends(admin_only)):
        guarded(pay.set_fine, store, user, code, body.amount_bdt, body.legal_ref)
        return {"items": pay.fines_table(store)}

    @router.delete("/api/fines/{code}")
    def del_fine(code: str, user=Depends(admin_only)):
        guarded(pay.clear_fine, store, user, code)
        return {"items": pay.fines_table(store)}

    # -- registry (admin)
    @router.get("/api/owners")
    def get_registry(q: str = "", limit: int = 500, offset: int = 0, user=Depends(admin_only)):
        rows, total = reg.listing(store, q, min(max(limit, 1), 1000), max(offset, 0))
        return {"items": rows, "total": total}

    @router.post("/api/owners")
    def add_entry(body: EntryIn, user=Depends(admin_only)):
        return guarded(reg.save, store, user, body.model_dump())

    @router.put("/api/owners/{rid}")
    def edit_entry(rid: int, body: EntryIn, user=Depends(admin_only)):
        return guarded(reg.save, store, user, body.model_dump(), rid)

    @router.delete("/api/owners/{rid}")
    def del_entry(rid: int, user=Depends(admin_only)):
        guarded(reg.delete, store, user, rid)
        return {"ok": True}

    @router.post("/api/owners/import")
    def import_entries(body: ImportIn, user=Depends(admin_only)):
        return guarded(reg.import_csv, store, user, body.csv, body.apply)

    # -- payments and the outbox (admin), invoices (staff)
    @router.get("/api/payments")
    def get_payments(user=Depends(admin_only)):
        return {"items": pay.listing(store), "providers": pay.provider_options(ssl, env),
                "sslcommerz_host": ssl.host if ssl.configured else None,
                "smtp": "configured" if not mail.dev_mode else "development outbox (nothing is sent)"}

    @router.get("/api/payments/{tran_id}/invoice.pdf")
    def staff_invoice(tran_id: str, user=Depends(staff)):
        p = pay.payment_by_tran(store, tran_id)
        c = store.citation(p["citation_no"]) if p else None
        if not p or p["status"] != "paid" or not c:
            raise HTTPException(404, "No such paid payment")
        return Response(invoice_bytes(p, c), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="invoice-{c["citation_no"]}.pdf"'})

    @router.get("/api/outbox")
    def get_outbox(user=Depends(admin_only)):
        return {"items": mail.listing(), "mode": "development" if mail.dev_mode else "smtp"}

    @router.get("/api/outbox/{mid}")
    def get_message(mid: int, user=Depends(admin_only)):
        m = mail.message(mid)
        if not m:
            raise HTTPException(404, "No such message")
        return m

    @router.post("/api/outbox/{mid}/retry")
    def retry_message(mid: int, user=Depends(admin_only)):
        if not mail.retry(mid, user):
            raise HTTPException(409, "Only a failed or waiting message can be retried")
        return {"ok": True}

    # -- the driver's own citations
    def my_citation(no, u):
        """The citation if it is on one of this driver's plates, else the same 404 as for a number that does not exist."""
        c = store.citation(no) if cit.NUMBER_RE.match(no or "") else None
        mine = reg.driver_plate_keys(store, u)
        if not c or disputes_mod.owner_plate_key(c, store.violation(c["violation_id"])) not in mine:
            if c:
                store.audit(u, "portal.denied", "citation", no, "payment or invoice for a citation that is not the driver's", who(u))
            raise HTTPException(404, "No such citation")
        return c

    @router.get("/api/portal/citations/states")
    def my_states(user=Depends(drv)):
        rows, _ = store.violations_for_plates(sorted(reg.driver_plate_keys(store, user)), limit=200, offset=0)
        return {"items": {r["citation_no"]: pay.payment_state(store, store.citation(r["citation_no"]))
                          for r in rows if r.get("citation_no")}}

    @router.get("/api/portal/citations/{no}/payment")
    def my_payment(no: str, user=Depends(drv)):
        c = my_citation(no, user)
        return {**pay.payment_state(store, c), "providers": pay.provider_options(ssl, env),
                "test_label": "Test - no money moves"}

    @router.post("/api/portal/citations/{no}/pay")
    def my_pay(no: str, body: PayIn, user=Depends(drv)):
        c = my_citation(no, user)
        return guarded(pay.start_payment, store, user, c, body.provider, ssl, base_url(), env)

    @router.post("/api/portal/citations/{no}/pay/simulate")
    def my_simulate(no: str, body: SimulateIn, user=Depends(drv)):
        c = my_citation(no, user)
        p, newly = guarded(pay.finish_test, store, user, c, body.tran_id, body.outcome, ssl, env)
        if newly:
            after_paid(p)
        return {"status": p["status"], **pay.payment_state(store, store.citation(no))}

    @router.get("/api/portal/citations/{no}/invoice.pdf")
    def my_invoice(no: str, user=Depends(drv)):
        c = my_citation(no, user)
        p = pay.paid_payment(store, no)
        if not p:
            raise HTTPException(404, "No invoice: this citation has not been paid")
        return Response(invoice_bytes(p, c), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="invoice-{no}.pdf"'})

    return SimpleNamespace(router=router, mailer=mail, on_issued=on_issued, on_decided=on_decided, ssl=ssl,
                           after_paid=after_paid)
