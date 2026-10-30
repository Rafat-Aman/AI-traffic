"""Outgoing email: a queue that never blocks a request, with a local outbox for development.

Why a queue: an SMTP server can be slow, down, or refusing; a driver paying a fine, or an officer issuing a
citation, must not wait for it or fail because of it. Every message is first written to the `mail_queue` table
(fast, local) and a background thread does the sending.

Two modes, chosen by the environment
    SMTP_HOST set     messages are sent through that server (SMTP_PORT default 587, SMTP_USER / SMTP_PASSWORD
                      optional, SMTP_FROM the sender, SMTP_STARTTLS=0 to turn STARTTLS off; port 465 uses SSL).
                      A failed send is kept with its error and tried again after 1, 2, 4 ... minutes (capped at an
                      hour) for up to MAX_ATTEMPTS tries, then marked failed; an administrator can retry it.
    SMTP_HOST unset   nothing leaves the machine. Each message is written as an .eml file in data/outbox/ (open it
                      in any mail client) and shown on the admin Outbox tab, the way Mailpit works.
The credentials come from the environment only and are never stored or logged.

A message with no recipient (the plate is not in the registry, or has no email) is recorded as "skipped" with the
reason, so the Outbox tab answers "why did nobody get a notice" instead of the message vanishing.
"""

import os
import smtplib
import threading
import time
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS mail_queue (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT '',          -- citation_issued | receipt | dispute_decided
    related TEXT NOT NULL DEFAULT '',       -- the citation number
    recipient TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    raw BLOB NOT NULL,                      -- the whole message, as bytes (an .eml)
    status TEXT NOT NULL CHECK (status IN ('pending','sent','saved','failed','skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_try REAL NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    done_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_mail_status ON mail_queue(status, next_try);
"""

MAX_ATTEMPTS = 8
BACKOFF_BASE_SEC = 60
BACKOFF_CAP_SEC = 3600


def ensure_schema(store):
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


def backoff(attempts):
    """Seconds to wait after the `attempts`-th failure: 60, 120, 240 ... up to an hour."""
    return min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * (2 ** max(0, attempts - 1)))


def _now():
    return datetime.now().isoformat(timespec="seconds")


def one_line(s, n=200):
    """Header values cannot contain line breaks (that is how header injection works)."""
    return " ".join(str(s or "").split())[:n]


class Mailer:
    def __init__(self, store, outbox_dir, env=None, send_fn=None, clock=time.time, log=print):
        self.store, self.outbox = store, Path(outbox_dir)
        self.env = os.environ if env is None else env
        self.send_fn = send_fn or self._smtp_send
        self.clock, self.log = clock, log
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None
        ensure_schema(store)

    # ------------------------------------------------------------ configuration
    @property
    def smtp_host(self):
        return (self.env.get("SMTP_HOST") or "").strip()

    @property
    def dev_mode(self):
        return not self.smtp_host

    @property
    def sender(self):
        return one_line(self.env.get("SMTP_FROM") or "traffic-authority@localhost", 200)

    # ------------------------------------------------------------ queueing
    def enqueue(self, kind, to, subject, body, attachments=(), related=""):
        """Queue one message and return its row id. Never raises for a mail problem and never touches the network."""
        to = one_line(to, 254)
        msg = EmailMessage(policy=policy.SMTP)
        msg["From"], msg["Subject"] = self.sender, one_line(subject)
        if to:
            msg["To"] = to
        msg.set_content(body)
        for name, data, mime in attachments:
            main, _, sub = mime.partition("/")
            msg.add_attachment(data, maintype=main, subtype=sub or "octet-stream", filename=one_line(name, 100))
        raw = bytes(msg)
        if not to:
            status, err = "skipped", "no email address on record for the owner"
        elif self.dev_mode:
            status, err = "saved", ""
        else:
            status, err = "pending", ""
        mid = self.store.x("INSERT INTO mail_queue (kind, related, recipient, subject, raw, status, last_error,"
                           " created_at, done_at) VALUES (?,?,?,?,?,?,?,?,?)",
                           (kind, related, to, one_line(subject), raw, status, err, _now(),
                            _now() if status != "pending" else None))
        if status == "saved":
            self._write_eml(mid, kind, raw)
        elif status == "pending":
            self._wake.set()
        self.store.audit(None, "mail." + status, "mail", mid, f"{kind} {related}".strip())
        return mid

    def _write_eml(self, mid, kind, raw):
        try:
            self.outbox.mkdir(parents=True, exist_ok=True)
            (self.outbox / f"{mid:06d}-{kind or 'mail'}.eml").write_bytes(raw)
        except OSError as e:
            self.log(f"[mailer] could not write the outbox file: {e}")

    # ------------------------------------------------------------ sending
    def _smtp_send(self, raw, to):
        port = int(self.env.get("SMTP_PORT") or 587)
        starttls = (self.env.get("SMTP_STARTTLS", "1").strip().lower() not in ("0", "false", "no", "off"))
        cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
        with cls(self.smtp_host, port, timeout=20) as s:
            if port != 465 and starttls:
                s.starttls()
            if self.env.get("SMTP_USER"):
                s.login(self.env["SMTP_USER"], self.env.get("SMTP_PASSWORD", ""))
            s.sendmail(self.sender, [to], raw)

    def deliver_due(self):
        """Try every pending message that is due. Returns how many were sent. Called by the thread; also by tests."""
        now, sent = self.clock(), 0
        for row in self.store.q("SELECT id, recipient, raw, attempts FROM mail_queue WHERE status='pending'"
                                " AND next_try<=? ORDER BY id LIMIT 50", (now,)):
            try:
                self.send_fn(bytes(row["raw"]), row["recipient"])
            except Exception as e:      # any failure: keep the message, remember why, try again later
                attempts = row["attempts"] + 1
                give_up = attempts >= MAX_ATTEMPTS
                self.store.x("UPDATE mail_queue SET attempts=?, last_error=?, next_try=?, status=?, done_at=?"
                             " WHERE id=?", (attempts, str(e)[:300], now + backoff(attempts),
                                             "failed" if give_up else "pending", _now() if give_up else None,
                                             row["id"]))
                self.store.audit(None, "mail.failed" if give_up else "mail.retry", "mail", row["id"],
                                 f"attempt {attempts}: {str(e)[:150]}")
            else:
                self.store.x("UPDATE mail_queue SET status='sent', attempts=attempts+1, last_error='', done_at=?"
                             " WHERE id=?", (_now(), row["id"]))
                self.store.audit(None, "mail.sent", "mail", row["id"], "")
                sent += 1
        return sent

    def retry(self, mid, user=None):
        """Put a failed message back in the queue (administrator)."""
        r = self.store.q("SELECT status FROM mail_queue WHERE id=?", (mid,))
        if not r or r[0]["status"] not in ("failed", "pending"):
            return False
        self.store.x("UPDATE mail_queue SET status='pending', next_try=0, attempts=0 WHERE id=?", (mid,))
        self.store.audit(user, "mail.retry_requested", "mail", mid, "", (user or {}).get("address", ""))
        self._wake.set()
        return True

    def start(self, interval=15.0):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    if not self.dev_mode:
                        self.deliver_due()
                except Exception as e:      # the sender thread must outlive any one bad message
                    self.log(f"[mailer] {e}")
                self._wake.wait(interval)
                self._wake.clear()
        self._thread = threading.Thread(target=loop, name="mailer", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------ the Outbox tab
    def listing(self, limit=200):
        return self.store.q("SELECT id, kind, related, recipient, subject, status, attempts, last_error, created_at,"
                            " done_at FROM mail_queue ORDER BY id DESC LIMIT ?", (limit,))

    def message(self, mid):
        """One message parsed for display (headers, text, attachment names), or None."""
        r = self.store.q("SELECT * FROM mail_queue WHERE id=?", (mid,))
        if not r:
            return None
        row = r[0]
        msg = BytesParser(policy=policy.default).parsebytes(bytes(row["raw"]))
        body = msg.get_body(preferencelist=("plain",))
        out = {k: row[k] for k in ("id", "kind", "related", "status", "attempts", "last_error", "created_at",
                                    "done_at")}
        out.update({"from": str(msg["From"] or ""), "to": str(msg["To"] or ""), "subject": str(msg["Subject"] or ""),
                    "text": body.get_content() if body else "",
                    "attachments": [{"name": a.get_filename(), "bytes": len(a.get_content())}
                                    for a in msg.iter_attachments()]})
        return out
