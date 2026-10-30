"""E-prosecution: what may be cited, how a citation seals its evidence, and how that evidence reaches a driver.

Three problems live here because they share one failure mode - evidence that cannot be trusted, or that reaches
the wrong person - and one set of readers: the officer issuing a citation, whoever later disputes it, and the
driver it is addressed to.

What may be cited (advisory codes)
    The project report (SDP 3.1.2) says speed "enters the system only as a descriptor feeding the impact score
    and the behavioural rules. It never supports a speeding citation, because speed enforcement demands a
    calibration and legal-metrology standard beyond this project." The code used to disagree with the report:
    V2 produced notices like any other rule. A code in ADVISORY_CODES is still detected, reviewed and shown to
    staff - the reviewers' verdicts are what measure the rule's precision - but no citation can be issued for
    it, it never appears in a driver's portal, and its notice is refused. The default is the report's position.
    ADVISORY_CODES in the environment changes it (an empty value means nothing is advisory), so reversing the
    decision is a deployment setting somebody has to make on purpose, not a code change nobody notices.

Fines
    Amounts come from the Road Transport Act 2018 and the rules under it. This project holds no authoritative
    copy of them, so none are written down here: every code's fine is None, shown as "set by the authority".
    FINE_SCHEDULE_BDT='{"V3": 1000}' fills in amounts once the authority supplies them, and the amount in force
    is copied onto each citation when it is issued, so a later change to the schedule never rewrites a
    citation somebody already holds.

Sealing the evidence
    A citation says: these images and this video show this vehicle doing this. The files sit on an ordinary
    disk that a backup restore, a clean-up script or a person can change without anybody noticing. At issuance
    every evidence file is hashed (SHA-256) and the hashes are written, with the facts of the case, into a
    manifest in canonical form; the SHA-256 of that text is the bundle hash printed on the notice. Verifying
    recomputes all of it and names the file that differs, which is more useful than "something changed".
    The canonical text is pure ASCII (non-ASCII is escaped), so the hash does not depend on how a tool
    normalises Bangla text.
    What this is and is not: tamper EVIDENCE, not prevention. The manifest sits in the same database as the
    rest, so somebody who can write to the database and the disk can alter a file and re-seal it. This catches
    accidents, bit rot, a replaced or deleted file and a careless edit. Against a deliberate insider the bundle
    hash has to be somewhere the operator cannot write, which is the reason it is printed on the notice; a
    stronger deployment would also anchor it with an external timestamp or an append-only log.

Signed, expiring links for drivers
    A driver's evidence used to be served by the same unauthenticated static mount as everything else, so a
    guessed or shared path was enough. Now the portal hands out /m/<expiry>.<signature>/<path> links:
      * the signature is HMAC-SHA256 over the exact path and the expiry, keyed with a server secret, so a link
        for one file cannot be pointed at another, and the expiry cannot be extended by editing the URL;
      * the signature is checked before the expiry, so a caller without a valid link learns nothing from the
        difference between "invalid" and "expired", and cannot tell whether a file exists;
      * the lifetime is an hour by default - long enough to watch a clip and save the photos, short enough that
        a link left in a forwarded message or a browser history is dead by tomorrow;
      * a link is a bearer token: anyone holding it can open that one file until it expires. That is the
        trade-off for being able to put it in an <img> or <video> tag, and the reason for the short life.
    The path is then treated as hostile anyway: dot segments, absolute paths, backslashes and NUL bytes are
    refused before the disk is touched, and the resolved file (symlinks followed) must still be inside
    data/evidence, which catches a symlink planted inside the folder. Staff use the same checks on the
    cookie-authenticated /media routes.
    The secret is MEDIA_SECRET from the environment or, failing that, 32 random bytes generated once and kept
    in data/.media_secret, created owner-only and never written anywhere else. Deleting the file invalidates
    every outstanding link, which is the whole rotation procedure. Limit worth knowing: on a filesystem that
    ignores Unix permissions (an NTFS or exFAT mount) the file cannot be made private; this module says so on
    start-up and the environment variable is the right choice there.
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import quote

# ------------------------------------------------------------------ what may be cited

# V2: the report says speed "never supports a speeding citation". V10/V11 are kinematic candidates that the
# vision-language model decides (report Table 3.1 #7, #8); their false-positive rate on Dhaka footage is unmeasured.
DEFAULT_ADVISORY = ("V2", "V10", "V11")
CITABLE_PLATE = ("read", "partial")       # a citation needs a vehicle to name; "partial" is the six-digit number
NUMBER_RE = re.compile(r"^[A-Z]{2,8}-\d{4}-\d{6,}$")


def advisory_codes(env=None):
    """Codes that are detected and reviewed but never cited. ADVISORY_CODES="V2,V6" replaces the default."""
    raw = (os.environ if env is None else env).get("ADVISORY_CODES")
    if raw is None:
        return frozenset(DEFAULT_ADVISORY)
    return frozenset(c.strip().upper() for c in raw.split(",") if c.strip())


ADVISORY_CODES = advisory_codes()


def is_advisory(code, codes=None):
    return (code or "").strip().upper() in (ADVISORY_CODES if codes is None else codes)


def fine_schedule(env=None, log=print):
    """{code: whole taka} from FINE_SCHEDULE_BDT. A code that is absent has no fine set, which is the default
    for every code. An entry that is not a positive whole number is ignored loudly rather than guessed at:
    the safe reading of a typo is "the authority has not set this", never a different amount."""
    raw = (os.environ if env is None else env).get("FINE_SCHEDULE_BDT", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        log('[citation] FINE_SCHEDULE_BDT must be a JSON object such as {"V3": 1000}; no fines are set')
        return {}
    out = {}
    for code, amount in data.items():
        # bool is a subclass of int: without the second test `true` would become a fine of one taka
        if isinstance(amount, int) and not isinstance(amount, bool) and amount > 0:
            out[str(code).strip().upper()] = amount
        else:
            log(f"[citation] ignoring the fine for {code!r}: {amount!r} is not a positive whole number of taka")
    return out


FINE_SCHEDULE = fine_schedule()


def fine_for(code, schedule=None):
    return (FINE_SCHEDULE if schedule is None else schedule).get((code or "").strip().upper())


def fine_text(amount):
    return "set by the authority" if amount is None else f"BDT {amount:,}"


def citation_prefix(env=None):
    p = (os.environ if env is None else env).get("CITATION_PREFIX", "DMP").strip().upper()
    return p if re.fullmatch(r"[A-Z]{2,8}", p) else "DMP"


class Refused(Exception):
    """A request the rules do not allow. `status` is the HTTP code the server answers with."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ paths: nothing from a URL reaches the disk raw

def clean_rel(rel):
    """`rel` if it is a plain relative path with no way out of its folder, else None."""
    if not isinstance(rel, str) or not rel or len(rel) > 1024 or "\0" in rel or "\\" in rel:
        return None
    # A leading "/" gives an empty first part, so absolute paths, "a//b" and "a/./b" are all refused here.
    if any(part in ("", ".", "..") for part in rel.split("/")):
        return None
    return rel


def resolve_under(root, rel):
    """The existing file `rel` names inside `root`, or None.

    resolve() follows symlinks and collapses anything clean_rel let through, so what is tested is where the
    file really is: a symlink inside the folder that points out of it is refused."""
    if clean_rel(rel) is None:
        return None
    root = Path(root).resolve()
    try:
        path = (root / rel).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    return path if path.is_relative_to(root) and path.is_file() else None


def evidence_file(data_dir, rel):
    """A file under data/evidence, from a path relative to data/ - the form the database stores."""
    rel = clean_rel(rel)
    if rel is None or not rel.startswith("evidence/"):
        return None
    return resolve_under(Path(data_dir) / "evidence", rel[len("evidence/"):])


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def delivery_path(clip_rel):
    """Where the driver's shorter copy of a clip lives: next to the original, which it never replaces."""
    p = PurePosixPath(clip_rel)
    return str(p.with_name(p.stem + "_delivery.mp4"))


# ------------------------------------------------------------------ the seal

MANIFEST_VERSION = 1


def case_files(v):
    """(role, path relative to data/) for every file that belongs to a case, each once."""
    out, seen = [], set()
    named = [("snapshot", v.get("snapshot")), ("crop", v.get("crop")), ("clip", v.get("clip")),
             ("plate", v.get("plate_img"))]
    named += [(f"evidence:{e.get('label', '')}", e.get("path")) for e in v.get("evidence") or []]
    for role, rel in named:
        if rel and rel not in seen:
            seen.add(rel)
            out.append((role, rel))
    return out


def sealed_facts(v):
    """The facts of a case that the seal covers: who, what, when, where."""
    return {"violation_id": v["id"], "code": v["code"], "plate": v.get("plate_text_en") or "",
            "plate_bn": v.get("plate_text") or "", "wall_time": v.get("wall_time") or "",
            "video_time_s": v.get("video_time_s"), "camera_id": v["camera_id"]}


def canonical(obj):
    """One byte sequence per value: sorted keys, no whitespace, ASCII only."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_manifest(v, data_dir):
    """(canonical manifest text, bundle hash) for a case, hashing every file as it is on the disk right now."""
    files = []
    for role, rel in case_files(v):
        path = evidence_file(data_dir, rel)
        if path is None:
            raise Refused(f"An evidence file of this case is missing from the disk ({rel}), "
                          "so the bundle cannot be sealed.")
        files.append({"path": rel, "role": role, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    files.sort(key=lambda f: f["path"])
    text = canonical({"version": MANIFEST_VERSION, **sealed_facts(v), "files": files})
    return text, sha256_text(text)


def verify_seal(c, violation, data_dir):
    """Recompute everything a citation sealed and report, per file, match / mismatch / missing.

    `c` is the citation row, `violation` the case as it is now (None if it has been deleted). The stored
    manifest text is itself hashed and compared with the stored bundle hash, so an edit to the manifest is
    caught as well as an edit to a file."""
    text = c["manifest"]
    bundle = {"expected": c["bundle_sha256"], "actual": sha256_text(text)}
    bundle["status"] = "match" if bundle["expected"] == bundle["actual"] else "mismatch"
    try:
        manifest = json.loads(text)
        listed = manifest.get("files") or []
    except (ValueError, AttributeError):
        manifest, listed = {}, []
    files = []
    for f in listed:
        want = f.get("sha256")
        path = evidence_file(data_dir, f.get("path"))
        got = sha256_file(path) if path else None
        files.append({"path": f.get("path"), "role": f.get("role"), "expected": want, "actual": got,
                      "status": "missing" if path is None else ("match" if got == want else "mismatch")})
    if violation is None:
        case = {k: "missing" for k in sealed_facts({"id": 0, "code": "", "camera_id": 0})}
    else:
        now_facts = sealed_facts(violation)
        case = {k: "match" if manifest.get(k) == val else "mismatch" for k, val in now_facts.items()}
    counts = Counter(f["status"] for f in files)
    ok = (bundle["status"] == "match" and bool(files) and counts["match"] == len(files)
          and all(s == "match" for s in case.values()))
    return {"citation_no": c["citation_no"], "ok": ok, "bundle": bundle, "files": files, "case": case,
            "counts": {k: counts.get(k, 0) for k in ("match", "mismatch", "missing")}}


def verify_summary(result):
    n = result["counts"]
    if result["ok"]:
        return f"intact: all {n['match']} files match the seal"
    bad = [f"{n[k]} {k}" for k in ("mismatch", "missing") if n[k]]
    if result["bundle"]["status"] != "match":
        bad.append("manifest altered")
    bad += [f"case {k} {s}" for k, s in result["case"].items() if s != "match"]
    return "NOT intact: " + (", ".join(bad) or "no files listed")


# ------------------------------------------------------------------ eligibility and the operations on the store

on_issued = []      # callbacks f(citation row, case) run after a NEW citation is issued (the notice email hooks in here)

_issue_lock = threading.Lock()     # one citation is issued at a time: hashing and transcoding are slow, and two
                                   # requests for the same case must not both write the same delivery file


def blockers(v, data_dir, advisory=None):
    """Everything standing between this case and a citation, as sentences for the officer. [] means it may be cited."""
    why = []
    if is_advisory(v["code"], advisory):
        why.append(f"{v['code']} is advisory: it is detected and reviewed, but it never supports a citation.")
    if v["status"] != "confirmed":
        why.append(f"Only a case a controller has confirmed can be cited (this one is {v['status']}).")
    plate = v.get("plate_status") or ""
    if plate not in CITABLE_PLATE:
        why.append("The number plate is still being read." if plate == "pending" else
                   "The number plate could not be read, so the vehicle cannot be named in a citation.")
    rel = clean_rel(v.get("clip")) if v.get("clip") else None
    if rel is None:
        why.append("This case has no video clip, so its evidence is incomplete.")
    else:
        final = Path(data_dir) / rel
        if final.with_suffix(".part.mp4").exists():       # the name clips.write_mp4 gives a clip being written
            why.append("The video clip is still being encoded. Try again in a few seconds.")
        elif not final.is_file():
            why.append("The video clip is not on the disk yet - it may still be encoding, or it could not be "
                       "written. A citation is not issued without its video.")
    return why


def _make_delivery(clip_rel, data_dir, delivery, warnings):
    if delivery is None or not clip_rel:
        return None
    src, dst_rel = evidence_file(data_dir, clip_rel), delivery_path(clip_rel)
    if src is None:
        return None
    try:
        delivery(src, Path(data_dir) / dst_rel)
    except Exception as e:      # the shorter copy is a convenience; failing to make it must not stop a citation
        warnings.append(f"The shorter video for the driver could not be made ({e}); "
                        "the driver is shown the original until it is.")
        return None
    return dst_rel


def issue(store, vid, user, data_dir, delivery=None, schedule=None, advisory=None, issued_at=None, prefix=None):
    """Issue the citation for a case. Returns (citation row, created, warnings).

    Idempotent: asking again for a case that already has a live citation returns it unchanged (and, if the
    driver's shorter video is still missing, tries to make it again). A withdrawn citation is not revived -
    that would put a number back into circulation that the driver was told is void.

    A refusal is audited as well as raised: somebody asking for a speed case to be cited is worth a row in
    the trail, and it is how a clip that never finishes encoding gets noticed."""
    with _issue_lock:
        try:
            return _issue(store, vid, user, data_dir, delivery, schedule, advisory, issued_at, prefix)
        except Refused as e:
            if e.status != 404:      # asking about a case that does not exist is not worth a row
                store.audit(user, "citation.refused", "violation", vid, str(e), user.get("address", ""))
            raise


def _issue(store, vid, user, data_dir, delivery, schedule, advisory, issued_at, prefix):
    v = store.violation(vid)
    if not v:
        raise Refused("Violation not found", 404)
    warnings = []
    have = store.citation_by_violation(vid)
    if have:
        if have["status"] == "withdrawn":
            raise Refused(f"Citation {have['citation_no']} was issued for this case and has been withdrawn; "
                          "it cannot be issued again.")
        if not have["delivery_clip"]:
            rel = _make_delivery(v.get("clip"), data_dir, delivery, warnings)
            if rel:
                store.set_delivery_clip(have["citation_no"], rel)
                store.audit(user, "citation.delivery", "citation", have["citation_no"], rel,
                            user.get("address", ""))
                have = store.citation(have["citation_no"])
        return have, False, warnings
    why = blockers(v, data_dir, advisory)
    if why:
        raise Refused(" ".join(why))
    text, bundle = build_manifest(v, data_dir)
    fine = fine_for(v["code"], schedule)
    rel = _make_delivery(v["clip"], data_dir, delivery, warnings)
    row, created = store.create_citation(vid, user, fine, bundle, text, rel, issued_at,
                                         prefix or citation_prefix())
    if created:
        store.audit(user, "citation.issue", "citation", row["citation_no"],
                    f"case TV-{vid} {v['code']} plate {v.get('plate_text_en') or '-'}; bundle {bundle}; "
                    f"fine {fine_text(fine)}", user.get("address", ""))
        for callback in on_issued:
            try:
                callback(row, v)
            except Exception as e:      # a notice that cannot be sent never undoes a citation that is issued
                print(f"[citation] on_issued callback failed: {e}")
    return row, created, warnings


def withdraw(store, no, user, reason):
    """Withdraw a citation. Returns (row, changed). Withdrawing twice is not an error and is not audited twice."""
    reason = (reason or "").strip()[:500]
    if len(reason) < 5:
        raise Refused("Say why the citation is being withdrawn; it is kept on record with the reason.", 400)
    c = store.citation(no)
    if not c:
        raise Refused("No such citation", 404)
    if c["status"] == "withdrawn":
        return c, False
    if c["status"] == "paid":
        raise Refused("A paid citation cannot be withdrawn here; refunds are outside this system.")
    store.withdraw_citation(no, user, reason)
    store.audit(user, "citation.withdraw", "citation", no, reason, user.get("address", ""))
    return store.citation(no), True


def verify_citation(store, no, user, data_dir):
    """Re-hash a citation's evidence and report. Audited: who checked the chain of custody, and what they found."""
    c = store.citation(no)
    if not c:
        raise Refused("No such citation", 404)
    result = verify_seal(c, store.violation(c["violation_id"]), data_dir)
    store.audit(user, "citation.verify", "citation", no, verify_summary(result), user.get("address", ""))
    return result


def view(c, manifest=True):
    """A citation row as the API returns it: the manifest parsed, and the fine in words."""
    out = {k: v for k, v in c.items() if k != "manifest"}
    out["fine_text"] = fine_text(c.get("fine_bdt"))
    if manifest:
        try:
            out["manifest"] = json.loads(c["manifest"])
        except ValueError:
            out["manifest"] = None
    return out


# ------------------------------------------------------------------ signed links

MEDIA_AREAS = ("evidence", "autozone")     # what staff may browse; a driver's link can only ever name evidence/
SECRET_FILE = ".media_secret"
MIN_SECRET_CHARS = 16
_PURPOSE = b"media-link-v1"                # keeps this MAC distinct from any other use of the same secret


def _link_ttl(env=None):
    try:
        ttl = int((os.environ if env is None else env).get("MEDIA_LINK_TTL_SEC", 3600))
    except ValueError:
        ttl = 3600
    return max(60, min(ttl, 7 * 24 * 3600))


LINK_TTL_SEC = _link_ttl()


def _read_existing(path):
    for _ in range(40):
        key = Path(path).read_text().strip()
        if len(key) >= 32:
            return key
        time.sleep(0.05)       # another process has just created the file and has not finished writing it
    raise RuntimeError(f"{path} exists but holds no usable key. Delete it to have a new one generated "
                       "(every outstanding media link stops working when you do).")


def _private(path, log):
    """Make a key file owner-only, and say so if the filesystem will not let us."""
    try:
        if os.stat(path).st_mode & 0o077:
            os.chmod(path, 0o600)
        if os.stat(path).st_mode & 0o077:
            log(f"[citation] {path} is readable by other users and this filesystem ignores chmod; "
                "set MEDIA_SECRET in the environment instead of keeping the key in a file.")
    except OSError as e:
        log(f"[citation] could not check the permissions of {path}: {e}")


def media_secret(data_dir, env=None, log=print):
    """The key that signs media links: MEDIA_SECRET, else one generated on first use and kept in data/."""
    env = os.environ if env is None else env
    given = env.get("MEDIA_SECRET")
    if given:
        if len(given) < MIN_SECRET_CHARS:
            raise RuntimeError(f"MEDIA_SECRET must be at least {MIN_SECRET_CHARS} characters: a short one can "
                               "be guessed offline from a single link.")
        return given.encode("utf-8")
    path = Path(data_dir) / SECRET_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL with mode 0600: the file is created private, never created open and tightened afterwards, and
        # of two processes starting together exactly one writes the key and the other reads it back.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = _read_existing(path)
    else:
        key = secrets.token_hex(32)
        with os.fdopen(fd, "w") as f:
            f.write(key + "\n")
    _private(path, log)
    return key.encode("utf-8")


def _mac(secret, path, expires):
    msg = _PURPOSE + b"\n" + str(int(expires)).encode("ascii") + b"\n" + path.encode("utf-8")
    return base64.urlsafe_b64encode(hmac.new(secret, msg, hashlib.sha256).digest()).rstrip(b"=")


def sign(path, secret, ttl=None, now=None):
    """The token for `path` (relative to data/, under evidence/): "<expiry>.<signature>". None if it cannot be signed."""
    rel = clean_rel(path)
    if rel is None or not rel.startswith("evidence/"):
        return None
    expires = int((time.time() if now is None else now) + (LINK_TTL_SEC if ttl is None else ttl))
    return f"{expires}.{_mac(secret, rel, expires).decode('ascii')}"


def check_token(token, path, secret, now=None):
    """"ok", "expired" or "invalid". The signature is judged first and the expiry second, so only a link that
    was genuinely issued can ever be told it has expired."""
    exp, dot, sig = (token or "").partition(".")
    if not dot or not exp.isascii() or not exp.isdigit() or len(exp) > 12 or not isinstance(path, str):
        return "invalid"
    if not hmac.compare_digest(sig.encode("utf-8"), _mac(secret, path, int(exp))):
        return "invalid"
    return "ok" if (time.time() if now is None else now) <= int(exp) else "expired"


def signed_url(path, secret, ttl=None, now=None):
    """/m/<token>/<path> for a file under data/evidence, or None if the path cannot be shared."""
    token = sign(path, secret, ttl, now)
    return None if token is None else f"/m/{token}/{quote(path, safe='/')}"
