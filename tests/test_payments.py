"""Fines, paying a citation (test gateway and SSLCommerz with every HTTP call mocked), disputes pausing payment,
ownership of a driver's citations, and the invoice download.

Nothing here touches the network or the real database: the gateway's `http` is a fake that returns what the
tests tell it to, and the app is a small FastAPI with a header-based stand-in for sign-in.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_citation as tc  # noqa: E402  (sets the scratch data folder first)

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402

from surveillance import api_payments  # noqa: E402
from surveillance import auth  # noqa: E402
from surveillance import citation as cit  # noqa: E402
from surveillance import disputes  # noqa: E402
from surveillance import payments as pay  # noqa: E402

STATEMENT = "The signal was red and the whole road was stopped, so I could not move my car at all."
SSL_ENV = {"SSLCOMMERZ_STORE_ID": "teststore", "SSLCOMMERZ_STORE_PASSWORD": "not-a-real-password",
           "PUBLIC_BASE_URL": "https://traffic.example.test", "PAYMENTS_TEST_GATEWAY": "0"}


class FakeHttp:
    """Stands in for the gateway. `validation` is what the validation API answers; every call is recorded."""

    def __init__(self):
        self.posts, self.gets = [], []
        self.session = {"status": "SUCCESS", "GatewayPageURL": "https://sandbox.sslcommerz.com/gw/abc"}
        self.validation = {}
        self.fail_validation = False

    def post_form(self, url, data, timeout=30):
        self.posts.append((url, data))
        return self.session

    def get_json(self, url, params, timeout=30):
        self.gets.append((url, params))
        if self.fail_validation:
            raise OSError("unreachable")
        return self.validation


def make_app(store, root, env, http):
    def current_user(request: Request):
        role = request.headers.get("x-role")
        if not role:
            return None
        u = store.user(int(request.headers.get("x-uid", "0"))) or {}
        return {"user_id": u.get("id", 0), "role": role, "name": u.get("name", role), "username": u.get("username"),
                "licence_no": u.get("licence_no"), "address": "10.0.0.7"}

    def require(*roles):
        def guard(user=Depends(current_user)):
            if user is None:
                raise HTTPException(401, "Sign in")
            if not any(auth.can(user["role"], r) for r in roles):
                raise HTTPException(403, "no")
            return user
        return guard

    built = api_payments.build(store, root, current_user, require, env=env, http=http)
    app = FastAPI()
    app.include_router(built.router)
    return app, built


class PayBase(tc.StoreScratch):
    ENV = {}

    def setUp(self):
        super().setUp()
        disputes.ensure_schema(self.store)
        self.env = {**self.ENV}
        self.http = FakeHttp()
        self.app, self.built = make_app(self.store, self.root, self.env, self.http)
        self.uid = self.store.add_user("driver", "Karim", licence_no="DK1234567", phone="+8801700000000")
        self.store.add_vehicle("DHAKAMETROGA123456", self.uid, plate_en="DHAKA METRO-GA 12-3456", verified=1)
        self.other = self.store.add_user("driver", "Other", licence_no="DK7654321", phone="+8801700000001")
        self.store.add_vehicle("DHAKAMETROKHA991111", self.other, plate_en="DHAKA METRO-KHA 99-1111", verified=1)
        pay.set_fine(self.store, self.admin, "V3", 1500, "Road Transport Act 2018 s.0 (test)")
        self.row = self.issue(self.case())[0]
        self.no = self.row["citation_no"]

    def call(self, method, path, role="driver", uid=None, body=None, raw=None):
        headers = {"x-role": role, "x-uid": str(uid if uid is not None else self.uid)} if role else {}
        data = b""
        if body is not None:
            data, headers["content-type"] = json.dumps(body).encode(), "application/json"
        if raw is not None:
            data, headers["content-type"] = raw.encode(), "application/x-www-form-urlencoded"
        status, hdrs, out = tc.asgi_call(self.app, method, path, headers, data)
        return status, hdrs, out

    def j(self, *a, **kw):
        status, _, out = self.call(*a, **kw)
        try:
            return status, json.loads(out)
        except ValueError:
            return status, out

    def citation(self):
        return self.store.citation(self.no)

    def state(self):
        return pay.payment_state(self.store, self.citation())

    def start(self, provider="test", **kw):
        status, out = self.j("POST", f"/api/portal/citations/{self.no}/pay", body={"provider": provider}, **kw)
        self.assertEqual(status, 200, out)
        return out

    def open_dispute(self):
        return disputes.file(self.store, self.no, {"user_id": self.uid, "role": "driver", "name": "Karim"}, STATEMENT,
                             ["DHAKAMETROGA123456"])


class FineTests(PayBase):
    def test_schedule_defaults_empty_and_an_admin_amount_is_a_placeholder_until_referenced(self):
        pay.clear_fine(self.store, self.admin, "V3")
        rows = {r["code"]: r for r in pay.fines_table(self.store, schedule={})}
        self.assertTrue(all(r["amount_bdt"] is None for r in rows.values()))
        pay.set_fine(self.store, self.admin, "V5", 500)
        r = {r["code"]: r for r in pay.fines_table(self.store, schedule={})}["V5"]
        self.assertTrue(r["placeholder"])
        self.assertIn("Road Transport Act 2018", r["note"])
        pay.set_fine(self.store, self.admin, "V5", 500, "Act s.1")
        self.assertFalse({r["code"]: r for r in pay.fines_table(self.store, schedule={})}["V5"]["placeholder"])

    def test_environment_schedule_is_the_fallback(self):
        self.assertEqual(pay.effective_amount(self.store, "V4", {"V4": 700}), (700, "environment"))
        pay.set_fine(self.store, self.admin, "V4", 800)
        self.assertEqual(pay.effective_amount(self.store, "V4", {"V4": 700}), (800, "database"))

    def test_advisory_codes_have_no_fine(self):
        for code in ("V2", "V10", "V11"):
            with self.assertRaises(pay.PayError):
                pay.set_fine(self.store, self.admin, code, 100)
        self.assertTrue(all(r["advisory"] and r["amount_bdt"] is None for r in pay.fines_table(self.store)
                            if r["code"] in ("V2", "V10", "V11")))

    def test_an_advisory_citation_is_unpayable_even_with_a_stored_amount(self):
        v2 = self.case(code="V2")
        c = cit.issue(self.store, v2, self.officer, self.root, advisory=frozenset(), schedule={})[0]            # as if the deployment had made V2 citable
        self.store.x("UPDATE citations SET fine_bdt=900 WHERE citation_no=?", (c["citation_no"],))
        st = pay.payment_state(self.store, self.store.citation(c["citation_no"]))
        self.assertFalse(st["can_pay"])

    def test_bad_amounts_are_refused_and_changes_are_audited(self):
        for bad in (0, -5, True, "100", 2_000_000):
            with self.assertRaises(pay.PayError):
                pay.set_fine(self.store, self.admin, "V6", bad)
        pay.set_fine(self.store, self.admin, "V6", 300)
        pay.clear_fine(self.store, self.admin, "V6")
        acts = [a["action"] for a in self.store.q("SELECT action FROM audit_log WHERE object_type='fine'")]
        self.assertEqual(acts[-2:], ["fine.set", "fine.clear"])

    def test_unset_amount_cannot_be_paid(self):
        pay.clear_fine(self.store, self.admin, "V3")
        st = self.state()
        self.assertFalse(st["can_pay"])
        self.assertIn("not set by the authority", st["reason"])
        status, out = self.j("POST", f"/api/portal/citations/{self.no}/pay", body={"provider": "test"})
        self.assertEqual(status, 409)

    def test_only_admin_edits_fines_over_http(self):
        self.assertEqual(self.call("PUT", "/api/fines/V4", role="controller", body={"amount_bdt": 5})[0], 403)
        self.assertEqual(self.call("PUT", "/api/fines/V4", role="admin", uid=self.admin["user_id"],
                                   body={"amount_bdt": 5})[0], 200)


class TestGatewayTests(PayBase):
    def test_pay_with_the_test_gateway_marks_the_citation_paid_and_emails_a_receipt(self):
        tran = self.start()["tran_id"]
        status, out = self.j("POST", f"/api/portal/citations/{self.no}/pay/simulate",
                             body={"tran_id": tran, "outcome": "success"})
        self.assertEqual((status, out["state"]), (200, "paid"))
        self.assertEqual(self.citation()["status"], "paid")
        self.assertEqual(self.citation()["fine_bdt"], 1500)
        mails = self.built.mailer.listing()
        self.assertEqual([m["kind"] for m in mails], ["receipt"])
        self.assertEqual(self.built.mailer.message(mails[0]["id"])["attachments"][0]["name"], f"invoice-{self.no}.pdf")
        # a second press does nothing twice
        again = self.j("POST", f"/api/portal/citations/{self.no}/pay/simulate", body={"tran_id": tran, "outcome": "success"})
        self.assertEqual(again[0], 200)
        self.assertEqual(len(self.built.mailer.listing()), 1)
        self.assertEqual(len(self.store.q("SELECT * FROM payments WHERE status='paid'")), 1)

    def test_declined_leaves_it_unpaid_and_payable_again(self):
        tran = self.start()["tran_id"]
        status, out = self.j("POST", f"/api/portal/citations/{self.no}/pay/simulate",
                             body={"tran_id": tran, "outcome": "declined"})
        self.assertEqual(out["status"], "failed")
        self.assertTrue(self.state()["can_pay"])
        self.assertNotEqual(self.start()["tran_id"], tran)           # a fresh transaction id per attempt

    def test_payment_state_changes_are_audited(self):
        tran = self.start()["tran_id"]
        self.j("POST", f"/api/portal/citations/{self.no}/pay/simulate", body={"tran_id": tran, "outcome": "success"})
        acts = [a["action"] for a in self.store.q("SELECT action FROM audit_log WHERE object_type='payment'")]
        self.assertEqual(acts, ["payment.start", "payment.paid"])

    def test_test_gateway_is_off_when_live_even_if_forced_and_when_ssl_is_configured(self):
        ssl = pay.SslCommerz({**SSL_ENV, "SSLCOMMERZ_LIVE": "1", "PAYMENTS_TEST_GATEWAY": "1"})
        self.assertFalse(pay.test_gateway_enabled(ssl, {"PAYMENTS_TEST_GATEWAY": "1"}))
        sandbox = pay.SslCommerz(SSL_ENV)
        self.assertFalse(pay.test_gateway_enabled(sandbox, SSL_ENV))
        self.assertTrue(pay.test_gateway_enabled(sandbox, {"PAYMENTS_TEST_GATEWAY": "1"}))
        self.assertTrue(pay.test_gateway_enabled(pay.SslCommerz({}), {}))

    def test_starting_a_test_payment_is_refused_when_disabled(self):
        self.env.update(SSL_ENV)                        # configured, test gateway not forced -> off
        status, _ = self.j("POST", f"/api/portal/citations/{self.no}/pay", body={"provider": "test"})
        self.assertEqual(status, 400)


class SslCommerzTests(PayBase):
    ENV = SSL_ENV

    def init(self):
        out = self.start("sslcommerz")
        self.assertEqual(out["redirect"], "https://sandbox.sslcommerz.com/gw/abc")
        return out["tran_id"]

    def valid(self, tran, **over):
        self.http.validation = {"status": "VALID", "tran_id": tran, "amount": "1500.00", "currency_type": "BDT",
                                "bank_tran_id": "BANK123", "risk_level": "0", **over}

    def ipn(self, val_id="VAL1", tran=""):
        return self.call("POST", "/api/payments/sslcommerz/ipn", role=None,
                         raw=urlencode({"val_id": val_id, "tran_id": tran}))

    def test_init_goes_to_the_sandbox_server_side_with_our_urls_and_whole_taka(self):
        tran = self.init()
        url, data = self.http.posts[0]
        self.assertEqual(url, "https://sandbox.sslcommerz.com/gwprocess/v4/api.php")
        self.assertEqual((data["total_amount"], data["currency"], data["tran_id"]), ("1500.00", "BDT", tran))
        self.assertEqual(data["success_url"], "https://traffic.example.test/api/payments/sslcommerz/success")
        self.assertEqual(data["ipn_url"], "https://traffic.example.test/api/payments/sslcommerz/ipn")
        self.assertNotIn("TV", self.no)
        self.assertLessEqual(len(tran), 30)

    def test_live_uses_the_live_host(self):
        self.assertEqual(pay.SslCommerz({**SSL_ENV, "SSLCOMMERZ_LIVE": "1"}).host, "https://securepay.sslcommerz.com")

    def test_the_provider_page_never_shows_credentials(self):
        status, out = self.j("GET", f"/api/portal/citations/{self.no}/payment")
        self.assertNotIn("not-a-real-password", json.dumps(out))
        self.assertNotIn("teststore", json.dumps(out))
        self.assertEqual((out["providers"]["sslcommerz"], out["providers"]["test"]), (True, False))

    def test_valid_payment_is_counted_once_and_a_repeated_ipn_does_nothing(self):
        tran = self.init()
        self.valid(tran)
        self.assertEqual(self.ipn("VAL1", tran)[0], 200)
        self.assertEqual(self.citation()["status"], "paid")
        self.assertEqual(self.http.gets[0][1]["val_id"], "VAL1")
        self.assertEqual(self.ipn("VAL1", tran)[0], 200)
        self.assertEqual(len(self.store.q("SELECT * FROM payments WHERE status='paid'")), 1)
        self.assertEqual([m["kind"] for m in self.built.mailer.listing()], ["receipt"])
        paid_events = [a for a in self.store.q("SELECT * FROM audit_log WHERE action='payment.paid'")]
        self.assertEqual(len(paid_events), 1)

    def test_validated_status_and_the_browser_return_also_count(self):
        tran = self.init()
        self.valid(tran, status="VALIDATED")
        status, hdrs, _ = self.call("POST", "/api/payments/sslcommerz/success", role=None,
                                    raw=urlencode({"val_id": "V", "tran_id": tran, "status": "VALID"}))
        self.assertEqual(status, 303)
        self.assertIn("payment=paid", hdrs["location"])
        self.assertEqual(self.citation()["status"], "paid")

    def test_mismatches_and_invalid_are_rejected(self):
        cases = {"amount": {"amount": "1.00"}, "currency": {"currency_type": "USD"}, "invalid": {"status": "INVALID_TRANSACTION"},
                 "failed": {"status": "FAILED"}, "risky": {"risk_level": "1"}, "no amount": {"amount": None}}
        for label, over in cases.items():
            tran = self.init()
            self.valid(tran, **over)
            self.ipn("V-" + label, tran)
            self.assertEqual(self.citation()["status"], "issued", label)
            self.assertEqual(self.store.q("SELECT status FROM payments WHERE tran_id=?", (tran,))[0]["status"],
                             "rejected", label)

    def test_a_validation_for_a_different_transaction_is_rejected(self):
        t1, t2 = self.init(), self.init()
        self.valid(t2)                                          # the gateway says t2 is paid ...
        self.ipn("V", t1)                                        # ... but the callback claimed t1
        self.assertEqual(self.citation()["status"], "issued")
        self.http.validation = {"status": "VALID", "tran_id": "TVUNKNOWN", "amount": "1500.00", "currency_type": "BDT"}
        self.ipn("V")
        self.assertEqual(self.citation()["status"], "issued")

    def test_forged_success_query_without_validation_is_not_counted(self):
        tran = self.init()
        self.valid(tran, status="INVALID_TRANSACTION")
        for path in ("/api/payments/sslcommerz/success?status=VALID&tran_id=%s&val_id=FORGED&amount=1500.00" % tran,):
            for method in ("GET", "POST"):
                status, hdrs, _ = self.call(method, path, role=None)
                self.assertIn("payment=failed", hdrs["location"])
        self.assertEqual(self.citation()["status"], "issued")
        # and with no val_id at all the gateway is not even asked
        before = len(self.http.gets)
        self.call("GET", f"/api/payments/sslcommerz/success?status=VALID&tran_id={tran}", role=None)
        self.assertEqual(len(self.http.gets), before)
        self.assertEqual(self.citation()["status"], "issued")

    def test_unreachable_validation_leaves_it_pending_and_asks_for_a_retry(self):
        tran = self.init()
        self.http.fail_validation = True
        self.assertEqual(self.ipn("V", tran)[0], 503)
        self.assertEqual(self.citation()["status"], "issued")
        self.http.fail_validation = False
        self.valid(tran)
        self.assertEqual(self.ipn("V", tran)[0], 200)
        self.assertEqual(self.citation()["status"], "paid")

    def test_fail_and_cancel_returns_close_only_open_attempts(self):
        tran = self.init()
        self.call("POST", "/api/payments/sslcommerz/cancel", role=None, raw=urlencode({"tran_id": tran}))
        self.assertEqual(self.store.q("SELECT status FROM payments")[0]["status"], "cancelled")
        self.valid(tran)                                       # a genuine validated payment still settles it
        self.ipn("V", tran)
        self.assertEqual(self.citation()["status"], "paid")
        self.call("POST", "/api/payments/sslcommerz/fail", role=None, raw=urlencode({"tran_id": tran}))
        self.assertEqual(self.store.q("SELECT status FROM payments")[0]["status"], "paid")

    def test_gateway_refusing_to_open_a_session_fails_the_attempt(self):
        self.http.session = {"status": "FAILED", "failedreason": "Invalid store"}
        status, out = self.j("POST", f"/api/portal/citations/{self.no}/pay", body={"provider": "sslcommerz"})
        self.assertEqual(status, 502)
        self.assertEqual(self.store.q("SELECT status FROM payments")[0]["status"], "failed")

    def test_sslcommerz_needs_a_public_base_url(self):
        self.env["PUBLIC_BASE_URL"] = ""
        status, _ = self.j("POST", f"/api/portal/citations/{self.no}/pay", body={"provider": "sslcommerz"})
        self.assertEqual(status, 503)


class DisputeInteractionTests(PayBase):
    def decide(self, how):
        return disputes.decide(self.store, self.open_dispute()["id"], self.admin, how, "the photographs show it clearly")

    def test_an_open_dispute_pauses_payment(self):
        self.open_dispute()
        st = self.state()
        self.assertEqual((st["state"], st["can_pay"]), ("paused", False))
        status, _ = self.j("POST", f"/api/portal/citations/{self.no}/pay", body={"provider": "test"})
        self.assertEqual(status, 409)

    def test_an_accepted_dispute_cancels_the_citation(self):
        self.decide("accept")
        st = self.state()
        self.assertEqual((st["state"], st["can_pay"]), ("cancelled", False))

    def test_a_rejected_dispute_makes_it_due_again_at_least_seven_days_later(self):
        self.store.x("UPDATE citations SET issued_at=? WHERE citation_no=?",
                     ((datetime.now() - timedelta(days=29)).isoformat(timespec="seconds"), self.no))
        before = self.state()["due_at"]
        self.decide("uphold")
        st = self.state()
        self.assertTrue(st["can_pay"])
        self.assertGreaterEqual(datetime.fromisoformat(st["due_at"]), (datetime.now() + timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0))
        self.assertGreater(st["due_at"], before)

    def test_decisions_queue_a_dispute_email_through_the_hook(self):
        reg_row = {"plate": "DHAKA METRO-GA 12-3456", "email": "owner@example.test"}
        from surveillance import registry
        registry.save(self.store, self.admin, reg_row)
        disputes.on_decided.append(self.built.on_decided)
        self.addCleanup(lambda: disputes.on_decided.remove(self.built.on_decided))
        self.decide("uphold")
        kinds = [m["kind"] for m in self.built.mailer.listing()]
        self.assertEqual(kinds, ["dispute_decided"])


class OwnershipAndInvoiceTests(PayBase):
    def paid(self):
        tran = self.start()["tran_id"]
        self.j("POST", f"/api/portal/citations/{self.no}/pay/simulate", body={"tran_id": tran, "outcome": "success"})
        return tran

    def test_another_driver_cannot_pay_see_or_download(self):
        self.paid()
        for method, path, body in (("GET", f"/api/portal/citations/{self.no}/payment", None),
                                   ("POST", f"/api/portal/citations/{self.no}/pay", {"provider": "test"}),
                                   ("GET", f"/api/portal/citations/{self.no}/invoice.pdf", None),
                                   ("POST", f"/api/portal/citations/{self.no}/pay/simulate",
                                    {"tran_id": "X", "outcome": "success"})):
            self.assertEqual(self.call(method, path, uid=self.other, body=body)[0], 404, path)
        self.assertEqual(self.j("GET", "/api/portal/citations/states", uid=self.other)[1]["items"], {})
        denied = self.store.q("SELECT * FROM audit_log WHERE action='portal.denied'")
        self.assertGreaterEqual(len(denied), 4)

    def test_the_owner_downloads_a_valid_pdf_and_staff_can_too(self):
        tran = self.paid()
        status, hdrs, body = self.call("GET", f"/api/portal/citations/{self.no}/invoice.pdf")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"%PDF-1.4"))
        self.assertIn(b"Test - no money moved", body)
        self.assertEqual(hdrs["content-type"], "application/pdf")
        self.assertEqual(self.call("GET", f"/api/payments/{tran}/invoice.pdf", role="controller",
                                   uid=self.officer["user_id"])[0], 200)
        self.assertEqual(self.call("GET", f"/api/payments/{tran}/invoice.pdf")[0], 403)    # a driver is not staff

    def test_no_invoice_before_payment(self):
        self.assertEqual(self.call("GET", f"/api/portal/citations/{self.no}/invoice.pdf")[0], 404)

    def test_the_registry_licence_lets_a_driver_pay_a_plate_they_did_not_claim(self):
        from surveillance import registry
        registry.save(self.store, self.admin, {"plate": "DHAKA METRO-GA 12-3456", "licence_no": "DK7654321"})
        self.assertEqual(self.call("GET", f"/api/portal/citations/{self.no}/payment", uid=self.other)[0], 200)

    def test_admin_endpoints_are_admin_only(self):
        for path in ("/api/fines", "/api/owners", "/api/payments", "/api/outbox"):
            self.assertEqual(self.call("GET", path)[0], 403, path)
            self.assertEqual(self.call("GET", path, role="controller")[0], 403, path)
            self.assertEqual(self.call("GET", path, role="admin")[0], 200, path)


if __name__ == "__main__":
    unittest.main()
