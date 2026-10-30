"""The mail queue: development outbox, SMTP retry with backoff, skipped messages, and the citation hook."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_citation as tc  # noqa: E402  (sets the scratch data folder first)

from surveillance import api_payments  # noqa: E402
from surveillance import citation as cit  # noqa: E402
from surveillance import mailer as mm  # noqa: E402
from surveillance import registry as reg  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class MailerTests(tc.StoreScratch):
    def make(self, env, send_fn=None, clock=None):
        return mm.Mailer(self.store, self.root / "outbox", env=env, send_fn=send_fn, clock=clock or FakeClock(),
                         log=lambda *_: None)

    def test_development_mode_writes_an_eml_file_and_sends_nothing(self):
        def never(*_):
            raise AssertionError("no network in development mode")
        m = self.make({}, never)
        mid = m.enqueue("receipt", "owner@example.test", "Hello", "Body text",
                        [("invoice.pdf", b"%PDF-1.4 fake", "application/pdf")], related="DMP-2026-000001")
        files = list((self.root / "outbox").glob("*.eml"))
        self.assertEqual(len(files), 1)
        self.assertIn(b"Subject: Hello", files[0].read_bytes())
        self.assertEqual(m.deliver_due(), 0)
        msg = m.message(mid)
        self.assertEqual((msg["status"], msg["to"], msg["attachments"][0]["name"]),
                         ("saved", "owner@example.test", "invoice.pdf"))
        self.assertEqual(m.listing()[0]["id"], mid)

    def test_no_recipient_is_recorded_as_skipped(self):
        m = self.make({})
        mid = m.enqueue("citation_issued", "", "S", "B")
        self.assertEqual(m.message(mid)["status"], "skipped")
        self.assertEqual(list((self.root / "outbox").glob("*.eml")) if (self.root / "outbox").exists() else [], [])

    def test_header_injection_is_flattened(self):
        m = self.make({})
        mid = m.enqueue("x", "a@example.test", "Hi\r\nBcc: evil@example.test", "B")
        raw = bytes(self.store.q("SELECT raw FROM mail_queue WHERE id=?", (mid,))[0]["raw"])
        self.assertNotIn(b"\nBcc:", raw)

    def test_failed_sends_are_kept_and_retried_with_backoff(self):
        calls, clock = [], FakeClock()

        def flaky(raw, to):
            calls.append(to)
            if len(calls) < 3:
                raise OSError("connection refused")
        m = self.make({"SMTP_HOST": "smtp.example.test"}, flaky, clock)
        mid = m.enqueue("receipt", "owner@example.test", "S", "B")
        self.assertEqual(m.message(mid)["status"], "pending")
        self.assertEqual(m.deliver_due(), 0)
        row = self.store.q("SELECT * FROM mail_queue WHERE id=?", (mid,))[0]
        self.assertEqual((row["status"], row["attempts"]), ("pending", 1))
        self.assertIn("refused", row["last_error"])
        self.assertEqual(m.deliver_due(), 0)           # not due yet: it waits
        self.assertEqual(len(calls), 1)
        clock.t += mm.backoff(1) + 1
        m.deliver_due()
        clock.t += mm.backoff(2) + 1
        self.assertEqual(m.deliver_due(), 1)
        self.assertEqual(m.message(mid)["status"], "sent")
        self.assertEqual(len(calls), 3)

    def test_backoff_grows_and_is_capped(self):
        self.assertEqual([mm.backoff(n) for n in (1, 2, 3)], [60, 120, 240])
        self.assertEqual(mm.backoff(40), mm.BACKOFF_CAP_SEC)

    def test_gives_up_after_the_limit_and_can_be_retried(self):
        clock = FakeClock()
        m = self.make({"SMTP_HOST": "smtp.example.test"}, lambda *_: (_ for _ in ()).throw(OSError("down")), clock)
        mid = m.enqueue("receipt", "owner@example.test", "S", "B")
        for _ in range(mm.MAX_ATTEMPTS):
            clock.t += mm.BACKOFF_CAP_SEC + 1
            m.deliver_due()
        self.assertEqual(m.message(mid)["status"], "failed")
        self.assertTrue(m.retry(mid))
        self.assertEqual(m.message(mid)["status"], "pending")


class CitationHookTests(tc.StoreScratch):
    def test_issuing_a_citation_queues_a_notice_to_the_registry_email(self):
        reg.ensure_schema(self.store)
        reg.save(self.store, self.admin, {"plate": "DHAKA METRO-GA 12-3456", "email": "owner@example.test"})
        pay = api_payments.build(self.store, self.root, lambda: None, lambda *r: (lambda: None),
                                 env={"PUBLIC_BASE_URL": "https://traffic.example.test"})
        self.addCleanup(lambda: cit.on_issued.remove(pay.on_issued))
        cit.on_issued.append(pay.on_issued)
        row, created, _ = self.issue(self.case())
        self.assertTrue(created)
        msgs = pay.mailer.listing()
        self.assertEqual(len(msgs), 1)
        m = pay.mailer.message(msgs[0]["id"])
        self.assertEqual((m["to"], m["status"]), ("owner@example.test", "saved"))
        self.assertIn(row["citation_no"], m["subject"])
        self.assertIn("https://traffic.example.test/portal", m["text"])
        self.assertIn("set by the authority", m["text"])
        self.assertEqual(m["attachments"], [])
        self.issue(row["violation_id"])                       # asking again does not send a second notice
        self.assertEqual(len(pay.mailer.listing()), 1)

    def test_a_failing_hook_never_blocks_issuing(self):
        def boom(*_):
            raise RuntimeError("mail is down")
        cit.on_issued.append(boom)
        self.addCleanup(lambda: cit.on_issued.remove(boom))
        row, created, _ = self.issue(self.case())
        self.assertTrue(created)


if __name__ == "__main__":
    unittest.main()
