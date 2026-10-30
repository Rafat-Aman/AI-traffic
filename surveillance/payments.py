"""Fines, and paying a citation online.

Fines
    The amount of a fine is the authority's to set (Road Transport Act 2018 and the rules under it); this project
    holds no copy, so by default no code has an amount and nothing can be paid. An administrator can enter an amount
    per violation code, with the legal reference it comes from. An amount with no reference is shown as a
    placeholder ("check against the Road Transport Act 2018") so a number typed in for a demo is never mistaken for
    the law. Advisory codes (citation.ADVISORY_CODES) are never fined. FINE_SCHEDULE_BDT in the environment is the
    fallback when the database has no amount for a code.
    A citation issued before its amount was set keeps "no amount" until someone tries to pay it; the amount then in
    force is copied onto the citation (audited), after which later changes to the table never touch it.

What decides whether a citation can be paid (payment_state)
    withdrawn -> cancelled, nothing to pay; paid -> paid; an open dispute -> paused; no amount -> cannot pay
    ("amount not set by the authority"); otherwise unpaid. A dispute that was upheld makes the citation due again,
    with a due date at least 7 days after the decision. An accepted dispute withdraws the citation (disputes.decide).

Paying
    Two providers. "sslcommerz": we open a session on the gateway (server to server) and send the driver to its page,
    so card or wallet details never reach this server. A payment is counted ONLY after the gateway's validation API,
    asked with the val_id, says VALID/VALIDATED for the transaction id, amount and currency we recorded. The browser's
    return and the gateway's IPN call only trigger that check; nothing they carry is believed. Settling is idempotent.
    "test": a built-in gateway with no card fields, two buttons, for demos and for development; never offered when
    SSLCOMMERZ_LIVE is on, so no pretend payment can ever be mistaken for a real one.
    Credentials are read from the environment (SSLCOMMERZ_STORE_ID, SSLCOMMERZ_STORE_PASSWORD) and never leave it.
"""

import json
import os
import re
import secrets
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

from . import citation as cit
from . import disputes as disputes_mod

SCHEMA = """
CREATE TABLE IF NOT EXISTS fines (
    code TEXT PRIMARY KEY,
    amount_bdt INTEGER NOT NULL CHECK (amount_bdt > 0),
    legal_ref TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY,
    tran_id TEXT NOT NULL UNIQUE,            -- ours, one per attempt; what the gateway echoes back
    citation_no TEXT NOT NULL,
    provider TEXT NOT NULL CHECK (provider IN ('test', 'sslcommerz')),
    amount_bdt INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'BDT',
    status TEXT NOT NULL DEFAULT 'initiated'
        CHECK (status IN ('initiated', 'paid', 'failed', 'cancelled', 'rejected')),
    driver_id INTEGER,
    val_id TEXT NOT NULL DEFAULT '',
    bank_tran_id TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    paid_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_payment_citation ON payments(citation_no);
"""

CODES = tuple(f"V{i}" for i in range(1, 12))
CURRENCY = "BDT"
MAX_FINE_BDT = 1_000_000
REF_PLACEHOLDER = "placeholder - check against the Road Transport Act 2018"
SANDBOX_HOST, LIVE_HOST = "https://sandbox.sslcommerz.com", "https://securepay.sslcommerz.com"
REDISPUTE_GRACE_DAYS = 7


class PayError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def ensure_schema(store):
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _flag(env, name):
    return (env.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def due_days(env=None):
    try:
        n = int((os.environ if env is None else env).get("PAYMENT_DUE_DAYS", 30))
    except ValueError:
        n = 30
    return max(1, min(n, 365))


# ------------------------------------------------------------------ fines

def _fine_row(store, code):
    r = store.q("SELECT * FROM fines WHERE code=?", (code,))
    return r[0] if r else None


def effective_amount(store, code, schedule=None, advisory=None):
    """(amount or None, source). Advisory codes never have one; the database wins over the environment."""
    code = (code or "").strip().upper()
    if cit.is_advisory(code, advisory):
        return None, "advisory"
    row = _fine_row(store, code)
    if row:
        return row["amount_bdt"], "database"
    env_amount = cit.fine_for(code, schedule)
    return (env_amount, "environment") if env_amount else (None, "")


def fines_table(store, schedule=None, advisory=None):
    codes = list(CODES) + sorted(({r["code"] for r in store.q("SELECT code FROM fines")}
                                  | set(cit.FINE_SCHEDULE if schedule is None else schedule)) - set(CODES))
    titles = {r["code"]: r["title"] for r in store.q("SELECT code, MAX(title) AS title FROM violations GROUP BY code")}
    out = []
    for code in codes:
        amount, source = effective_amount(store, code, schedule, advisory)
        row = _fine_row(store, code)
        ref = (row or {}).get("legal_ref", "")
        out.append({"code": code, "title": titles.get(code, ""), "advisory": source == "advisory",
                    "amount_bdt": amount, "source": source, "legal_ref": ref,
                    "placeholder": amount is not None and not ref,
                    "note": ("advisory - never fined" if source == "advisory" else
                             "set by the authority" if amount is None else
                             REF_PLACEHOLDER if not ref else ""),
                    "updated_at": (row or {}).get("updated_at"), "updated_by": (row or {}).get("updated_by", "")})
    return out


def set_fine(store, user, code, amount, legal_ref="", advisory=None):
    code = (code or "").strip().upper()
    if not re.fullmatch(r"V\d{1,3}", code):
        raise PayError("Unknown violation code", 400)
    if cit.is_advisory(code, advisory):
        raise PayError(f"{code} is advisory: it never supports a citation, so it has no fine.")
    if isinstance(amount, bool) or not isinstance(amount, int) or not (0 < amount <= MAX_FINE_BDT):
        raise PayError(f"The amount must be a whole number of taka between 1 and {MAX_FINE_BDT:,}.", 400)
    ref = " ".join(str(legal_ref or "").split())[:200]
    old = _fine_row(store, code)
    store.x("INSERT INTO fines (code, amount_bdt, legal_ref, updated_at, updated_by) VALUES (?,?,?,?,?)"
            " ON CONFLICT(code) DO UPDATE SET amount_bdt=excluded.amount_bdt, legal_ref=excluded.legal_ref,"
            " updated_at=excluded.updated_at, updated_by=excluded.updated_by",
            (code, amount, ref, _now(), user.get("name") or user.get("username") or ""))
    store.audit(user, "fine.set", "fine", code,
                f"{('BDT %d' % old['amount_bdt']) if old else 'unset'} -> BDT {amount}; ref: {ref or '(none)'}",
                user.get("address", ""))


def clear_fine(store, user, code):
    code = (code or "").strip().upper()
    old = _fine_row(store, code)
    if not old:
        raise PayError("No amount is stored for that code", 404)
    store.x("DELETE FROM fines WHERE code=?", (code,))
    store.audit(user, "fine.clear", "fine", code, f"was BDT {old['amount_bdt']}", user.get("address", ""))


# ------------------------------------------------------------------ what a citation owes

def citation_code(c):
    try:
        return (json.loads(c.get("manifest") or "{}").get("code") or "").upper()
    except ValueError:
        return ""


def citation_amount(store, c, schedule=None, advisory=None):
    if cit.is_advisory(citation_code(c), advisory):
        return None
    if c.get("fine_bdt"):
        return c["fine_bdt"]
    return effective_amount(store, citation_code(c), schedule, advisory)[0]


def due_at(c, dispute, env=None):
    """When payment is due: the issue date plus PAYMENT_DUE_DAYS, and after an upheld dispute never sooner than
    7 days after the decision (the driver was not allowed to pay while it was open, and needs time afterwards)."""
    due = datetime.fromisoformat(c["issued_at"]) + timedelta(days=due_days(env))
    if dispute and dispute["status"] == "upheld" and dispute.get("decided_at"):
        due = max(due, datetime.fromisoformat(dispute["decided_at"]) + timedelta(days=REDISPUTE_GRACE_DAYS))
    return due


def payment_state(store, c, schedule=None, advisory=None, env=None):
    """{state, can_pay, reason, amount_bdt, due_at, ...} for a citation row."""
    d = disputes_mod.dispute_for_citation(store, c["citation_no"])
    amount = citation_amount(store, c, schedule, advisory)
    out = {"citation_no": c["citation_no"], "amount_bdt": amount, "amount_text": cit.fine_text(amount),
           "due_at": due_at(c, d, env).strftime("%Y-%m-%d"), "can_pay": False, "reason": "", "invoice": False}
    if c["status"] == "withdrawn":
        out.update(state="cancelled", reason="This citation was withdrawn; there is nothing to pay.")
    elif c["status"] == "paid":
        out.update(state="paid", invoice=True)
        p = paid_payment(store, c["citation_no"])
        if p:
            out["paid_at"], out["provider"], out["tran_id"] = p["paid_at"], p["provider"], p["tran_id"]
    elif d and d["status"] == "open":
        out.update(state="paused", reason="Payment is paused while your dispute is under review.")
    elif amount is None:
        out.update(state="unpaid", reason="Amount not set by the authority, so this cannot be paid yet.")
    else:
        out.update(state="unpaid", can_pay=True)
        if d and d["status"] == "upheld":
            out["note"] = "Your dispute was not accepted; the fine is due again."
    return out


def paid_payment(store, no):
    r = store.q("SELECT * FROM payments WHERE citation_no=? AND status='paid' ORDER BY id LIMIT 1", (no,))
    return r[0] if r else None


def payment_by_tran(store, tran_id):
    r = store.q("SELECT * FROM payments WHERE tran_id=?", (tran_id,))
    return r[0] if r else None


# ------------------------------------------------------------------ gateways

class Http:
    """The two calls SSLCommerz needs, over HTTPS with a timeout. Tests replace this object."""

    def post_form(self, url, data, timeout=30):
        req = urllib.request.Request(url, urllib.parse.urlencode(data).encode(), method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 (https URLs only, built below)
            return json.loads(r.read(1_000_000).decode("utf-8", "replace"))

    def get_json(self, url, params, timeout=30):
        with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params), timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read(1_000_000).decode("utf-8", "replace"))


class SslCommerz:
    def __init__(self, env=None, http=None):
        self.env = os.environ if env is None else env
        self.http = http or Http()

    @property
    def live(self):
        return _flag(self.env, "SSLCOMMERZ_LIVE")

    @property
    def configured(self):
        return bool(self.env.get("SSLCOMMERZ_STORE_ID") and self.env.get("SSLCOMMERZ_STORE_PASSWORD"))

    @property
    def host(self):
        return LIVE_HOST if self.live else SANDBOX_HOST

    def _auth(self):
        return {"store_id": self.env["SSLCOMMERZ_STORE_ID"], "store_passwd": self.env["SSLCOMMERZ_STORE_PASSWORD"]}

    def init_session(self, tran_id, amount, base_url, citation_no, who):
        """Open a checkout session; returns the GatewayPageURL to send the driver to."""
        if not self.configured:
            raise PayError("Online payment is not set up on this server.", 503)
        cb = base_url.rstrip("/") + "/api/payments/sslcommerz/"
        data = {**self._auth(), "total_amount": f"{amount}.00", "currency": CURRENCY, "tran_id": tran_id,
                "success_url": cb + "success", "fail_url": cb + "fail", "cancel_url": cb + "cancel",
                "ipn_url": cb + "ipn", "shipping_method": "NO", "product_name": f"Traffic fine {citation_no}",
                "product_category": "Government fee", "product_profile": "non-physical-goods",
                "cus_name": (who or "Driver")[:50], "cus_email": "driver@example.invalid", "cus_add1": "Dhaka",
                "cus_city": "Dhaka", "cus_country": "Bangladesh", "cus_phone": "01700000000",
                "value_a": citation_no}
        try:
            r = self.http.post_form(self.host + "/gwprocess/v4/api.php", data)
        except Exception:
            raise PayError("The payment gateway could not be reached. Try again in a minute.", 502) from None
        if not isinstance(r, dict) or r.get("status") != "SUCCESS" or not r.get("GatewayPageURL"):
            raise PayError("The payment gateway did not open the payment.", 502)
        url = str(r["GatewayPageURL"])
        if not url.startswith("https://"):
            raise PayError("The payment gateway returned an unusable address.", 502)
        return url

    def validate(self, val_id):
        """The gateway's own answer about a val_id (a dict), or raises on a network failure."""
        r = self.http.get_json(self.host + "/validator/api/validationserverAPI.php",
                               {"val_id": val_id, **self._auth(), "format": "json"})
        return r if isinstance(r, dict) else {}


def test_gateway_enabled(ssl, env=None):
    env = os.environ if env is None else env
    if ssl.live:
        return False
    return _flag(env, "PAYMENTS_TEST_GATEWAY") or not ssl.configured


def provider_options(ssl, env=None):
    return {"sslcommerz": ssl.configured, "sslcommerz_live": ssl.configured and ssl.live,
            "test": test_gateway_enabled(ssl, env)}


# ------------------------------------------------------------------ the payment steps

_lock = threading.Lock()      # payments settle one at a time: "check, then mark paid" must not interleave


def start_payment(store, user, c, provider, ssl, base_url="", env=None, schedule=None, advisory=None):
    """Begin paying citation row `c`. Returns {"tran_id", "provider", "redirect"?}. The caller has already checked the
    citation belongs to this driver."""
    options = provider_options(ssl, env)
    if provider not in options or provider not in ("test", "sslcommerz") or not options[provider]:
        raise PayError("That way of paying is not available.", 400)
    with _lock:
        st = payment_state(store, c, schedule, advisory)
        if not st["can_pay"]:
            raise PayError(st["reason"] or "This citation cannot be paid.")
        amount = st["amount_bdt"]
        if c.get("fine_bdt") is None:       # freeze the amount in force onto the citation
            store.x("UPDATE citations SET fine_bdt=? WHERE citation_no=? AND fine_bdt IS NULL", (amount, c["citation_no"]))
            store.audit(user, "citation.fine_fixed", "citation", c["citation_no"], f"BDT {amount}", user.get("address", ""))
        if provider == "sslcommerz" and not base_url:
            raise PayError("Online payment is not set up on this server (PUBLIC_BASE_URL).", 503)
        for _ in range(5):
            tran_id = "TV" + secrets.token_hex(8).upper()
            try:
                store.x("INSERT INTO payments (tran_id, citation_no, provider, amount_bdt, currency, driver_id, created_at)"
                        " VALUES (?,?,?,?,?,?,?)", (tran_id, c["citation_no"], provider, amount, CURRENCY,
                                                    user.get("user_id"), _now()))
                break
            except Exception:
                continue
        else:
            raise PayError("Could not start the payment; try again.", 500)
        store.audit(user, "payment.start", "payment", tran_id, f"{c['citation_no']} via {provider}, BDT {amount}",
                    user.get("address", ""))
    if provider == "test":
        return {"tran_id": tran_id, "provider": "test"}
    try:
        url = ssl.init_session(tran_id, amount, base_url, c["citation_no"], user.get("name") or "")
    except PayError as e:
        _set_status(store, tran_id, "failed", f"session not opened: {e}")
        raise
    return {"tran_id": tran_id, "provider": "sslcommerz", "redirect": url}


def _set_status(store, tran_id, status, detail, user=None, only_from=("initiated",)):
    marks = ",".join("?" * len(only_from))
    with store.lock:
        cur = store.db.execute(f"UPDATE payments SET status=?, detail=? WHERE tran_id=? AND status IN ({marks})",
                               (status, detail[:300], tran_id, *only_from))
        store.db.commit()
    if cur.rowcount:
        store.audit(user, "payment." + status, "payment", tran_id, detail[:300])
    return bool(cur.rowcount)


def _mark_paid(store, p, val_id="", bank_tran_id="", detail="", user=None):
    """Mark paid exactly once. Returns True only for the call that made the change."""
    with store.lock:
        cur = store.db.execute("UPDATE payments SET status='paid', paid_at=?, val_id=?, bank_tran_id=?, detail=?"
                               " WHERE tran_id=? AND status<>'paid'", (_now(), val_id, bank_tran_id, detail[:300], p["tran_id"]))
        if cur.rowcount:
            store.db.execute("UPDATE citations SET status='paid' WHERE citation_no=? AND status='issued'", (p["citation_no"],))
        store.db.commit()
    if cur.rowcount:
        store.audit(user, "payment.paid", "payment", p["tran_id"],
                    f"{p['citation_no']} BDT {p['amount_bdt']} via {p['provider']}")
    return bool(cur.rowcount)


def finish_test(store, user, c, tran_id, outcome, ssl, env=None):
    """The test gateway's two buttons. Returns (payment row, newly_paid)."""
    if not test_gateway_enabled(ssl, env):
        raise PayError("The test gateway is not enabled.", 400)
    with _lock:
        p = payment_by_tran(store, tran_id)
        if not p or p["citation_no"] != c["citation_no"] or p["provider"] != "test":
            raise PayError("No such payment", 404)
        if p["status"] != "initiated":
            return p, False
        if outcome == "success":
            if c["status"] != "issued" or payment_state(store, c)["state"] == "paused":
                _set_status(store, tran_id, "failed", "the citation can no longer be paid", user)
                raise PayError("This citation can no longer be paid.")
            newly = _mark_paid(store, p, detail="Test - no money moved", user=user)
        elif outcome == "declined":
            _set_status(store, tran_id, "failed", "Test - simulated decline", user)
            newly = False
        else:
            raise PayError("Choose success or declined.", 400)
    return payment_by_tran(store, tran_id), newly


def return_marker(store, tran_id, status):
    """The browser came back from the gateway's fail or cancel page. Informational only (a forged call could do no
    more than this): the attempt is closed unless it is still open, and a genuine validated payment can still
    settle it afterwards."""
    if tran_id and payment_by_tran(store, tran_id):
        _set_status(store, tran_id, status, f"gateway sent the browser to its {status} page")


def settle(store, ssl, val_id, claimed_tran_id=""):
    """Count an SSLCommerz payment if, and only if, the validation API confirms it.

    `val_id` and `claimed_tran_id` come from the caller (the browser or the IPN) and are untrusted: val_id is only
    the question we ask the gateway, and the answer decides. Returns (outcome, payment row or None, newly_paid) with
    outcome one of: paid, already, pending (gateway unreachable), rejected, unknown."""
    val_id = (val_id or "").strip()
    if not val_id or len(val_id) > 100 or not ssl.configured:
        return "rejected", None, False
    try:
        r = ssl.validate(val_id)
    except Exception:
        return "pending", None, False
    tran_id = str(r.get("tran_id") or "")
    p = payment_by_tran(store, tran_id) if tran_id else None
    if p is None or p["provider"] != "sslcommerz":
        store.audit(None, "payment.unknown", "payment", tran_id or "-", "validation answered for a transaction we did not start")
        return "unknown", None, False
    if claimed_tran_id and claimed_tran_id != tran_id:
        _reject(store, p, "the callback named a different transaction than the gateway confirmed")
        return "rejected", p, False
    with _lock:
        p = payment_by_tran(store, tran_id)
        if p["status"] == "paid":
            return "already", p, False
        status = str(r.get("status") or "").upper()
        if status not in ("VALID", "VALIDATED"):
            _reject(store, p, f"validation status {status or 'missing'}")
            return "rejected", p, False
        currency = str(r.get("currency_type") or r.get("currency") or "").upper()
        try:
            same = (Decimal(str(r.get("amount"))).quantize(Decimal("0.01")) == Decimal(p["amount_bdt"]).quantize(Decimal("0.01")))
        except (InvalidOperation, ValueError, TypeError):
            same = False
        if currency != CURRENCY or not same:
            _reject(store, p, f"amount or currency does not match (got {r.get('amount')} {currency})")
            return "rejected", p, False
        if str(r.get("risk_level") or "0") == "1":
            _reject(store, p, "the gateway flagged this payment as risky; held for review")
            return "rejected", p, False
        c = store.citation(p["citation_no"])
        if not c or c["status"] == "withdrawn":
            _reject(store, p, "the citation was withdrawn; the payment needs a manual refund")
            return "rejected", p, False
        newly = _mark_paid(store, p, val_id=val_id, bank_tran_id=str(r.get("bank_tran_id") or "")[:60],
                           detail="validated by the gateway")
        return ("paid" if newly else "already"), payment_by_tran(store, tran_id), newly


def _reject(store, p, why):
    if p["status"] != "paid":
        _set_status(store, p["tran_id"], "rejected", why, only_from=("initiated", "failed", "cancelled"))


def listing(store, limit=300):
    return store.q("SELECT id, tran_id, citation_no, provider, amount_bdt, currency, status, detail, created_at, paid_at"
                   " FROM payments ORDER BY id DESC LIMIT ?", (limit,))
