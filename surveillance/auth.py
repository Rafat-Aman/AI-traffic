"""Who is using the system, and what they are allowed to see.

Three kinds of account, because they have genuinely different relationships to the data:

    driver      a member of the public. Sees violations recorded against vehicles registered to them, and
                nothing else - not other people's cases, not the cameras, not the review queue.
    controller  traffic police. Works the review queue, confirms or rejects cases, issues citations.
    admin       runs the system: cameras, zones, accounts, retention.

How each proves who they are
    Staff sign in with a username and password. Drivers sign in with their **driving licence number and a
    one-time code sent to the mobile registered against it**, and the distinction matters: a licence number
    is printed on a card, handed to police and copied onto rental forms, so it identifies a person but
    keeps no secret. On its own it would let anyone who has seen a licence read that person's violation
    history, and licence numbers are regular enough to walk through one by one. The code is what actually
    authenticates; the licence number only says who is being asked about.

    Both paths are rate limited for the same reason - without that, the licence number alone is enough to
    enumerate the database a request at a time.

Hashing uses `hashlib.scrypt` from the standard library, so no password or code is ever stored in a form
that is useful to whoever reads the database, and nothing new had to be installed to get there.
"""

import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque

# --- roles ------------------------------------------------------------------
DRIVER, CONTROLLER, ADMIN = "driver", "controller", "admin"
ROLES = (DRIVER, CONTROLLER, ADMIN)
STAFF = (CONTROLLER, ADMIN)

# --- password / code hashing ------------------------------------------------
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1    # ~100 ms per hash on the development CPU
SALT_BYTES = 16

# --- one-time codes ---------------------------------------------------------
OTP_DIGITS = 6
OTP_TTL_SEC = 300
OTP_MAX_ATTEMPTS = 5

# --- sessions ---------------------------------------------------------------
SESSION_COOKIE = "tsid"
SESSION_TTL_SEC = 12 * 3600
STAFF_SESSION_TTL_SEC = 8 * 3600


def hash_secret(secret, salt=None):
    """scrypt hash of a password or one-time code, as one self-describing string."""
    salt = salt or secrets.token_bytes(SALT_BYTES)
    h = hashlib.scrypt(secret.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${h.hex()}"


def verify_secret(secret, stored):
    """Constant-time check of a password or code against its stored hash."""
    if not stored or not secret:
        return False
    try:
        kind, n, r, p, salt_hex, want = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(secret.encode(), salt=bytes.fromhex(salt_hex),
                             n=int(n), r=int(r), p=int(p), dklen=len(want) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), want)


def new_token():
    return secrets.token_urlsafe(32)


def token_fingerprint(token):
    """What is stored for a session. The token itself never touches the database, so reading the database
    does not hand anyone a working session."""
    return hashlib.sha256(token.encode()).hexdigest()


def new_code():
    """A numeric one-time code, uniform over its range."""
    return f"{secrets.randbelow(10 ** OTP_DIGITS):0{OTP_DIGITS}d}"


def normalise_licence(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def normalise_plate(s):
    """Plates are compared without spaces, dashes or case, matching how the store already searches them."""
    return "".join(ch for ch in (s or "").upper() if ch not in " -\t\n")


def mask_phone(phone):
    """What a login page may safely echo back: enough to recognise your own number, not enough to learn
    someone else's."""
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    return f"{'*' * max(0, len(digits) - 3)}{digits[-3:]}" if len(digits) >= 3 else "***"


class RateLimiter:
    """Fixed allowance per key over a sliding window, in memory.

    Enough for one process, which is what this runs as today. A hosted deployment behind more than one
    worker wants this in Redis or the database instead, or each worker will allow the full quota.
    """

    def __init__(self, limit, window_sec):
        self.limit, self.window = limit, window_sec
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key):
        """(allowed, seconds to wait). Records the attempt when allowed."""
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(self.window - (now - q[0])) + 1
            q.append(now)
            return True, 0

    def clear(self, key):
        with self.lock:
            self.hits.pop(key, None)


# Deliberately tight on anything that takes a licence number: that is the enumeration route.
code_requests = RateLimiter(limit=3, window_sec=300)      # codes sent, per licence number
code_attempts = RateLimiter(limit=10, window_sec=900)     # codes tried, per licence number
login_attempts = RateLimiter(limit=10, window_sec=900)    # staff password attempts, per username
by_address = RateLimiter(limit=40, window_sec=300)        # any login traffic, per client address


def deliver_code(code, phone, licence, log=print):
    """Send a one-time code to a driver.

    There is no SMS gateway wired up, so by default the code is written to the server log and the operator
    reads it out - which is fine for a demonstration and useless in production. Setting OTP_DELIVERY=echo
    additionally returns the code in the HTTP response so the login page can show it; that hands the code
    to whoever asked for it, which defeats the point of having one, so it warns every time and must never
    be set on a hosted instance. A real deployment adds an "sms" branch here and nothing else changes.
    """
    mode = os.environ.get("OTP_DELIVERY", "log").lower()
    log(f"[login code] licence {licence} -> {mask_phone(phone)}: {code} (valid {OTP_TTL_SEC // 60} min)")
    if mode == "echo":
        log("[login code] OTP_DELIVERY=echo - the code is being returned in the HTTP response. "
            "Demonstration only; never set this on a hosted instance.")
        return code
    return None


def can(role, need):
    """Is `role` allowed to act as `need`? Admin covers controller; nothing covers driver but driver."""
    if role == ADMIN:
        return need in (ADMIN, CONTROLLER)
    return role == need
