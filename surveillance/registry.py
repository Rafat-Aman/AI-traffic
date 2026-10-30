"""Vehicle owner registry: plate -> owner's name, mobile, email and licence number.

Why it exists: the camera reads a plate, not a person. To send an owner a notice, or a receipt, the authority needs
somewhere to look the plate up. This table is that, kept by an administrator (one row at a time or from a CSV).
It is separate from the `vehicles` table, which records which signed-in driver has claimed which plate: a person
can own a car long before they have ever opened the portal.

Two uses
    * the owner's email for the notice (citation issued) and the receipt;
    * a second way for a driver to be matched to a plate: the registry row that carries the driver's licence
      number counts as that driver's vehicle, in addition to the plates they claimed and an administrator verified.
      Only an administrator can write the registry, so this adds no way for a driver to claim a plate on their own.

CSV import is always previewed first (dry run: what would be added, changed, left alone, or refused, with a
reason per line) and only then applied, and only for the lines that were valid.
"""

import csv
import io
import re
import sqlite3

from . import auth

SCHEMA = """
CREATE TABLE IF NOT EXISTS owner_registry (
    id INTEGER PRIMARY KEY,
    plate_key TEXT NOT NULL UNIQUE,        -- the plate with spaces and dashes removed, upper case
    plate_text TEXT NOT NULL DEFAULT '',   -- as the administrator typed it
    owner_name TEXT NOT NULL DEFAULT '',
    mobile TEXT NOT NULL DEFAULT '',
    email TEXT NOT NULL DEFAULT '',
    licence_no TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_registry_licence ON owner_registry(licence_no);
"""

FIELDS = ("plate", "owner_name", "mobile", "email", "licence_no")
MAX_CSV_BYTES = 1_000_000
MAX_CSV_ROWS = 5000
EMAIL_RE = re.compile(r"^[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[^@\s,;<>\"']+$")


class RegistryError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def ensure_schema(store):
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


def _now():
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def licence_key(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def clean_entry(raw):
    """(entry, error): one registry row validated and normalised. Control characters and over-long values are
    refused: this text ends up in email headers and bodies."""
    e = {k: " ".join(str(raw.get(k) or "").split()) for k in FIELDS}
    plate_key = auth.normalise_plate(e["plate"])
    if len(plate_key) < 4 or len(plate_key) > 40:
        return None, "the number plate is missing or too short"
    if any(len(e[k]) > 120 for k in FIELDS) or any(ord(ch) < 32 for v in raw.values() for ch in str(v or "")
                                                   if ch not in "\t\n\r"):
        return None, "a value is too long or contains control characters"
    if e["email"] and not EMAIL_RE.match(e["email"]):
        return None, "the email address does not look right"
    if e["mobile"] and not re.fullmatch(r"\+?[0-9][0-9 \-]{8,16}[0-9]", e["mobile"]):
        return None, "the mobile number does not look right"
    return {"plate_key": plate_key, "plate_text": e["plate"], "owner_name": e["owner_name"],
            "mobile": e["mobile"], "email": e["email"].lower(), "licence_no": e["licence_no"]}, None


def get(store, rid):
    r = store.q("SELECT * FROM owner_registry WHERE id=?", (rid,))
    return r[0] if r else None


def by_plate(store, plate):
    r = store.q("SELECT * FROM owner_registry WHERE plate_key=?", (auth.normalise_plate(plate),))
    return r[0] if r else None


def listing(store, query="", limit=500, offset=0):
    where, args = "", ()
    if query.strip():
        like = "%" + query.strip().upper().replace("%", "").replace("_", "") + "%"
        where = (" WHERE UPPER(plate_key) LIKE ? OR UPPER(owner_name) LIKE ? OR UPPER(email) LIKE ?"
                 " OR UPPER(licence_no) LIKE ? OR mobile LIKE ?")
        args = (like.replace(" ", ""), like, like, like, like)
    total = store.q("SELECT COUNT(*) AS n FROM owner_registry" + where, args)[0]["n"]
    rows = store.q("SELECT * FROM owner_registry" + where + " ORDER BY plate_key LIMIT ? OFFSET ?",
                   args + (limit, offset))
    return rows, total


def save(store, user, raw, rid=None):
    """Add (rid None) or edit one row. Audited. Returns the row."""
    e, err = clean_entry(raw)
    if err:
        raise RegistryError(err)
    who = user.get("name") or user.get("username") or ""
    try:
        if rid is None:
            rid = store.x("INSERT INTO owner_registry (plate_key, plate_text, owner_name, mobile, email, licence_no,"
                          " updated_at, updated_by) VALUES (?,?,?,?,?,?,?,?)",
                          (e["plate_key"], e["plate_text"], e["owner_name"], e["mobile"], e["email"],
                           e["licence_no"], _now(), who))
            action = "registry.add"
        else:
            if not get(store, rid):
                raise RegistryError("No such registry entry", 404)
            store.x("UPDATE owner_registry SET plate_key=?, plate_text=?, owner_name=?, mobile=?, email=?,"
                    " licence_no=?, updated_at=?, updated_by=? WHERE id=?",
                    (e["plate_key"], e["plate_text"], e["owner_name"], e["mobile"], e["email"], e["licence_no"],
                     _now(), who, rid))
            action = "registry.edit"
    except sqlite3.IntegrityError:
        raise RegistryError("That plate is already in the registry", 409) from None
    # The audit detail names the plate and which fields are filled, not the personal data itself.
    store.audit(user, action, "registry", rid, f"plate {e['plate_key']}; filled: "
                + ", ".join(k for k in ("owner_name", "mobile", "email", "licence_no") if e[k]),
                user.get("address", ""))
    return get(store, rid)


def delete(store, user, rid):
    row = get(store, rid)
    if not row:
        raise RegistryError("No such registry entry", 404)
    store.x("DELETE FROM owner_registry WHERE id=?", (rid,))
    store.audit(user, "registry.delete", "registry", rid, f"plate {row['plate_key']}", user.get("address", ""))


def parse_csv(text):
    """(rows, error). The first line must name the columns; plate is required, the rest optional. Column names
    are matched without case or spaces, and a few common spellings are accepted."""
    if len(text.encode("utf-8", "replace")) > MAX_CSV_BYTES:
        return None, f"The file is larger than {MAX_CSV_BYTES // 1000} KB; split it."
    reader = csv.reader(io.StringIO(text.lstrip("﻿"), newline=""))
    try:
        head = next(reader)
    except StopIteration:
        return None, "The file is empty."
    alias = {"plate": "plate", "numberplate": "plate", "platenumber": "plate", "ownername": "owner_name",
             "name": "owner_name", "owner": "owner_name", "mobile": "mobile", "phone": "mobile",
             "email": "email", "licence": "licence_no", "licenceno": "licence_no", "licenceno.": "licence_no",
             "license": "licence_no", "licenseno": "licence_no", "licencenumber": "licence_no"}
    cols = [alias.get(re.sub(r"[\s_\-]", "", h).lower()) for h in head]
    if "plate" not in cols:
        return None, "The first line must name the columns, including plate (plate, owner_name, mobile, email, licence_no)."
    rows = []
    for line_no, rec in enumerate(reader, 2):
        if not any(c.strip() for c in rec):
            continue
        if len(rows) >= MAX_CSV_ROWS:
            return None, f"More than {MAX_CSV_ROWS} rows; split the file."
        rows.append((line_no, {c: rec[i] for i, c in enumerate(cols) if c and i < len(rec)}))
    return rows, None


def import_csv(store, user, text, apply=False):
    """Dry run (apply=False) or apply. Returns {"rows": [...], "counts": {...}, "applied": bool}. Each row says
    what happens: add, update, same (no change) or error with the reason. Lines naming the same plate twice:
    the later one is an error, so a typo cannot silently overwrite the line above it."""
    rows, err = parse_csv(text)
    if err:
        raise RegistryError(err)
    out, seen = [], set()
    for line_no, raw in rows:
        e, why = clean_entry(raw)
        if why:
            out.append({"line": line_no, "plate": raw.get("plate", ""), "action": "error", "reason": why})
            continue
        if e["plate_key"] in seen:
            out.append({"line": line_no, "plate": e["plate_text"], "action": "error",
                        "reason": "this plate appears twice in the file"})
            continue
        seen.add(e["plate_key"])
        have = by_plate(store, e["plate_key"])
        if not have:
            action = "add"
        elif all(have[k] == e[k] for k in ("owner_name", "mobile", "email", "licence_no")):
            action = "same"
        else:
            action = "update"
        out.append({"line": line_no, "plate": e["plate_text"], "action": action, "reason": "", "entry": e})
    counts = {a: sum(1 for r in out if r["action"] == a) for a in ("add", "update", "same", "error")}
    if apply:
        for r in out:
            if r["action"] in ("add", "update"):
                e = r["entry"]
                have = by_plate(store, e["plate_key"])
                save(store, user, {"plate": e["plate_text"], **{k: e[k] for k in FIELDS[1:]}}, have["id"] if have else None)
        store.audit(user, "registry.import", "registry", "", f"added {counts['add']}, updated {counts['update']}, "
                    f"unchanged {counts['same']}, refused {counts['error']}", user.get("address", ""))
    for r in out:
        r.pop("entry", None)
    return {"rows": out, "counts": counts, "applied": bool(apply)}


def plates_for_licence(store, licence_no):
    """Plate keys the registry attaches to a driver's licence number (empty if the licence is blank)."""
    key = licence_key(licence_no)
    if not key:
        return []
    sql = "SELECT plate_key, licence_no FROM owner_registry WHERE licence_no<>''"
    try:
        rows = store.q(sql)
    except sqlite3.OperationalError:  # the portal asks this of whatever store the server holds, which need not be
        ensure_schema(store)          # the one the payments module was built on (tests swap in scratch stores)
        rows = store.q(sql)
    return [r["plate_key"] for r in rows if licence_key(r["licence_no"]) == key]


def driver_plate_keys(store, user):
    """Every plate a signed-in driver may act on: their verified claimed vehicles plus the registry's plates for
    their licence number."""
    keys = {v["plate_key"] for v in store.vehicles_of(user["user_id"], verified_only=True)}
    keys.update(plates_for_licence(store, user.get("licence_no")))
    return keys
