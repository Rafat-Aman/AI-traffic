"""Signing in, staying signed in, and being refused - the FastAPI side of surveillance/auth.py.

Everything the dashboard exposes was open to anyone who could reach the port before this: the review queue,
the camera list, and a CSV export of every case with its number plate. These dependencies are what close
that, and they fail shut - a route guarded by `require(CONTROLLER)` cannot be reached without a session,
and there is no configuration that turns the check off.

Bootstrapping
    A fresh database has no accounts, and an installation nobody can sign in to is useless. The first time
    the server starts without any admin it creates one and prints its password to the console once. That is
    the only time a password is ever displayed, and it has to be changed on first sign-in.
"""

import os
import secrets
import time

from fastapi import Cookie, Depends, HTTPException, Request

from .auth import (ADMIN, CONTROLLER, DRIVER, SESSION_COOKIE, SESSION_TTL_SEC, STAFF_SESSION_TTL_SEC,
                   can, hash_secret, new_token, token_fingerprint)


def client_address(request):
    """The caller's address, trusting a proxy header only when one is configured - otherwise anyone could
    forge it and sidestep the per-address rate limit."""
    if os.environ.get("TRUST_PROXY_HEADER"):
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def secure_cookies():
    """Mark cookies Secure unless explicitly told the deployment is plain HTTP (local development)."""
    return os.environ.get("INSECURE_COOKIES", "").lower() not in ("1", "true", "yes")


def set_session_cookie(response, token, ttl):
    response.set_cookie(SESSION_COOKIE, token, max_age=ttl, httponly=True, samesite="lax",
                        secure=secure_cookies(), path="/")


def clear_session_cookie(response):
    response.delete_cookie(SESSION_COOKIE, path="/")


class Sessions:
    """Issues and resolves sessions against the store."""

    def __init__(self, store):
        self.store = store
        self._last_purge = 0.0

    def start(self, response, user, address=""):
        token = new_token()
        ttl = SESSION_TTL_SEC if user["role"] == DRIVER else STAFF_SESSION_TTL_SEC
        self.store.add_session(token_fingerprint(token), user["id"], user["role"], time.time() + ttl,
                               address)
        self.store.touch_login(user["id"])
        set_session_cookie(response, token, ttl)
        return token

    def end(self, response, token):
        if token:
            self.store.drop_session(token_fingerprint(token))
        clear_session_cookie(response)

    def resolve(self, token):
        """The signed-in user for a cookie, or None. Expired and disabled accounts resolve to nothing."""
        if not token:
            return None
        now = time.time()
        if now - self._last_purge > 600:
            self._last_purge = now
            self.store.purge_sessions(now)
        row = self.store.session(token_fingerprint(token))
        if not row or row["expires_at"] < now or not row["active"]:
            return None
        return {"user_id": row["user_id"], "role": row["role"], "name": row["name"],
                "username": row["username"], "licence_no": row["licence_no"]}


def make_dependencies(sessions):
    """Returns (current_user, require) bound to this Sessions instance."""

    def current_user(request: Request, tsid: str = Cookie(default=None)):
        user = sessions.resolve(tsid)
        if user:
            user = {**user, "address": client_address(request)}
        return user

    def require(*roles):
        """Dependency that admits only these roles. Admin implicitly covers controller."""
        def guard(user=Depends(current_user)):
            if user is None:
                raise HTTPException(401, "Sign in to continue")
            if not any(can(user["role"], r) for r in roles):
                raise HTTPException(403, "Your account does not have access to this")
            return user
        return guard

    return current_user, require


def _emit(line):
    """Print and flush. Redirected to a file, Python block-buffers stdout, and a password that is shown
    once but sits in a buffer for an hour has not been shown at all."""
    print(line, flush=True)


def ensure_admin(store, log=_emit):
    """Create the first administrator if the database has none, and say so loudly, once."""
    if store.count_users(ADMIN):
        return None
    username = os.environ.get("ADMIN_USER", "admin")
    password = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    store.add_user(ADMIN, name="Administrator", username=username, password_hash=hash_secret(password))
    store.audit(None, "user.create", "user", username, "first administrator created at startup")
    if not os.environ.get("ADMIN_PASSWORD"):
        log("\n" + "=" * 68)
        log("  No accounts existed, so an administrator has been created:")
        log(f"      username  {username}")
        log(f"      password  {password}")
        log("  This is shown once. Sign in and change it.")
        log("=" * 68 + "\n")
    return username
