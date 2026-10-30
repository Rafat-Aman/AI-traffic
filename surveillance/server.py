"""Web API + dashboard for the traffic surveillance system."""

import asyncio
import atexit
import csv
from html import escape
import io
import json
import os
import re
import shutil
import socket
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, RedirectResponse,
                               Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from violations.base import MOTORISED, NON_MOTORISED
from violations.engine import RULES, ViolationEngine
from violations.zones import (AREA_TYPES, DEFAULT_SPEED_LIMITS, DIRECTIONS, ROAD_CATEGORIES, RoadZone,
                              camera_config_from_dict)

from . import api_map
from . import api_optimize
from . import api_payments
from . import api_query
from . import api_storage
from . import api_vehicles
from . import auth as auth_mod
from . import citation as cit
from . import disputes as disputes_mod
from . import impact as impact_mod
from . import registry as registry_mod
from . import topology
from . import vlm as vlm_mod
from . import vlm_cascade
from . import webauth
from .link import LinkJob, OnlineLinker, SiteRun, run_linking
from .anpr import find_models
from .camera_motion import save_reference
from .clips import make_delivery_clip
from .detector import MODELS_DIR, find_helmet_model
from .pipeline import reliable_limits, zones_roi
from .sources import detect_start_time, grab_frame, is_live_source, probe_mac_cameras
from .store import DEFAULT_SETTINGS, Store
from .worker import WorkerManager

ROOT = Path(__file__).resolve().parents[1]
# Importing this module opens the database, so the folder it lives in can be redirected: the tests point it at a
# scratch directory instead of the one a running installation is using.
DATA = Path(os.environ.get("TRAFFIC_DATA_DIR") or ROOT / "data")
WEB = ROOT / "web"
EDITOR_WIDTH = 1920  # zones are drawn on a 1920-px frame and scaled to the analysis resolution
EDITOR_FRAMES = {}   # camera id -> last frame shown in the zone editor
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".mts", ".webm", ".ogv", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp"}

store = Store(DATA / "surveillance.db")
subscribers: set = set()
loop_ref = {}


def notify(msg):
    """Called from worker threads: push to every open dashboard (server-sent events)."""
    loop = loop_ref.get("loop")
    if loop is None:
        return
    data = json.dumps(msg, default=str)
    for q in list(subscribers):
        loop.call_soon_threadsafe(q.put_nowait, data)


linker = OnlineLinker(store)
sites = {}            # site name -> SiteRun
manager = WorkerManager(store, DATA, notify, linker)
link_jobs = {}        # "current" -> the running / last LinkJob


# What an administrator may change from the dashboard (saved in data/vlm_settings.json), with the type of each.
VLM_ADMIN_KEYS = {
    "vlm_mode": ("auto", "cascade", "single"),
    "vlm_screen_url": str, "vlm_screen_model": str,
    "vlm_reason_backend": ("auto", "local", "cloud"),
    "vlm_reason_url": str, "vlm_reason_model": str,
    "vlm_reason_idle_sec": (30, 7200), "vlm_reason_start_sec": (30, 900), "vlm_reason_timeout_sec": (60, 3600),
    "vlm_escalation": vlm_cascade.POLICIES,
    "vlm_stop_score": (0.0, 1.0), "vlm_no_confidence": (0.0, 1.0),
    "vlm_local_url": str, "vlm_local_model": str,
}


def _vlm_saved():
    try:
        saved = json.loads((DATA / "vlm_settings.json").read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in saved.items() if k in VLM_ADMIN_KEYS} if isinstance(saved, dict) else {}


def vlm_settings(backend=None):
    """VLM options: defaults, then the administrator's saved choices, then the environment, then one request."""
    s = dict(vlm_mod.DEFAULTS)
    s.update(_vlm_saved())
    s["vlm_backend"] = backend or os.environ.get("VLM_BACKEND") or s["vlm_backend"]
    if os.environ.get("VLM_URL"):
        s["vlm_local_url"] = os.environ["VLM_URL"]
    return s


def vlm_done(vid, explanation):
    """Called from the reasoning thread when a case has been explained."""
    store.set_vlm(vid, "done" if explanation.ok else "failed", explanation.as_dict())
    v = store.violation(vid)
    notify({"type": "explanation", "camera_id": v["camera_id"] if v else None, "violation": v})


cascade = vlm_cascade.Cascade(DATA, vlm_settings())


def vlm_stage(info):
    """Progress of one explanation, pushed to every open dashboard (stage events precede the final "explanation")."""
    v = store.violation(info["violation_id"])
    notify({"type": "explain_stage", "camera_id": v["camera_id"] if v else None, **info})


cascade.on_stage = vlm_stage
explainer = vlm_mod.Explainer(DATA, vlm_settings(), vlm_done, cascade=cascade)
atexit.register(lambda: cascade._server and cascade._server.stop())     # never leave an 8B behind
# Re-checks of drivers' disputes by the reasoning tier: advisory, one at a time, on their own thread.
dispute_checker = disputes_mod.Checker(lambda: cascade, vlm_settings,
                                       on_done=lambda did: notify({"type": "dispute", "id": did}))


sessions = webauth.Sessions(store)
current_user, require = webauth.make_dependencies(sessions)
CONTROLLER, ADMIN, DRIVER = auth_mod.CONTROLLER, auth_mod.ADMIN, auth_mod.DRIVER

_media_key = {}


def media_key():
    """The secret that signs a driver's media links. Loaded on first use rather than at import, so merely
    importing this module never creates a key file; and loaded at start-up (below), so a bad MEDIA_SECRET
    stops the server from booting instead of failing on the first driver who opens a case."""
    if "key" not in _media_key:
        _media_key["key"] = cit.media_secret(DATA)
    return _media_key["key"]


@asynccontextmanager
async def lifespan(_app):
    loop_ref["loop"] = asyncio.get_running_loop()
    (DATA / "videos").mkdir(parents=True, exist_ok=True)
    webauth.ensure_admin(store)
    media_key()
    dispute_checker.resume(store, DATA)     # any open dispute whose model check was lost to a restart
    storage.queue.start()           # ingest queue: runs uploaded videos max_parallel at a time
    pay.mailer.start()              # sends (or files in the outbox) notices and receipts, retrying failures
    vehicles.purge.start()          # daily: passed-vehicle records older than PASSED_RETENTION_DAYS go
    yield
    vehicles.purge.stop()
    pay.mailer.stop()
    storage.queue.stop()
    await asyncio.to_thread(manager.stop_all)


app = FastAPI(title="Dhaka Traffic Surveillance", lifespan=lifespan)


def cam_or_404(cid):
    cam = store.camera(cid)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


def keep_cited_evidence(cid):
    """Refuse to delete a camera, or its cases, while a citation stands for any of them.

    A citation seals files that live under this camera's evidence folder, and the cases it was made from are
    rows that cascade away with the camera. Without this, one click on "delete camera" would silently turn
    every standing citation's evidence into "missing". Withdrawing the citations first makes it a decision
    somebody takes, and leaves the audit trail saying who."""
    held = store.live_citations_for_camera(cid)
    if held:
        shown = ", ".join(held[:3]) + (f" and {len(held) - 3} more" if len(held) > 3 else "")
        raise HTTPException(409, f"Citations are standing for cases from this camera ({shown}). Their "
                                 "evidence has to be kept: withdraw them first.")


def jpeg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def rule_coverage(cam, z):
    """{code: "" if the rule runs on this camera, else why not} for the camera's saved zones and settings."""
    if not z:
        return {cls.code: "no zones yet" for cls in RULES.values()}
    try:
        w, h = z["config"].get("frame_size") or [1920, 1080]
        cfg = camera_config_from_dict(z["config"], w, h)
    except (ValueError, KeyError, TypeError) as e:
        return {cls.code: f"zones invalid: {e}" for cls in RULES.values()}
    helmet = cam["settings"].get("helmet", True) and find_helmet_model() is not None
    engine = ViolationEngine(cfg, helmet_detector=(lambda crop: 0.0) if helmet else None)
    out = {cls.code: engine.inactive.get(cls.code, "") for cls in RULES.values()}
    if not helmet and out.get("V6") == "needs a helmet model":
        out["V6"] = "helmet check switched off" if find_helmet_model() else "needs a helmet model"
    return out


def with_impact(v):
    """A case as staff see it: its stored impact (None until scored) and the priority worked out now from that and
    the plate's confirmed history. Drivers never get either: the portal builds its answers from a whitelist."""
    v["impact"] = impact_mod.load_impact(store, v["id"])
    v["priority"] = impact_mod.priority_for(store, v)
    return v


def camera_view(cam):
    z = store.zones(cam["id"])
    auto = (z or {}).get("config", {}).get("auto") if z else None
    learn = manager.autozone_status(cam["id"])
    learn.pop("result", None)
    return {**cam, "status": manager.status(cam["id"]), "zone_version": z["version"] if z else None,
            "zones_ready": bool(z), "pending": store.count_violations(camera_id=cam["id"], status="pending"),
            "total": store.count_violations(camera_id=cam["id"]), "rules": rule_coverage(cam, z),
            "zone_mode": "auto" if auto else ("manual" if z else cam["settings"].get("zone_mode", "manual")),
            "auto_report": auto.get("report") if auto else None, "auto_preview": auto.get("preview") if auto else None,
            "learning": learn}


# ---------------------------------------------------------------- meta

# Paths reachable without a session. Everything else under /api/ needs one: the check below fails shut, so
# a route added later is protected until someone deliberately lists it here.
PUBLIC_API = {"/api/login", "/api/login/code", "/api/login/verify", "/api/logout", "/api/me",
              # SSLCommerz's browser returns and its server-to-server IPN carry no session. They never trust
              # their parameters: a payment counts only after SSLCommerz's validation API confirms it.
              "/api/payments/sslcommerz/success", "/api/payments/sslcommerz/fail",
              "/api/payments/sslcommerz/cancel", "/api/payments/sslcommerz/ipn"}
# What a driver may reach. Their portal is a different surface from the operator dashboard, not a filtered
# view of it, so the two never share an endpoint.
DRIVER_API = {"/api/me", "/api/logout", "/api/portal/violations", "/api/portal/vehicles",
              "/api/portal/violation", "/api/portal/citations"}


@app.middleware("http")
async def guard(request: Request, call_next):
    # Only /api/ is guarded here. Everything else decides for itself: /media/* and /notice/* check the session
    # in their own dependency, and /m/* is open on purpose - its credential is the signature in the URL, and a
    # driver's <img> and <video> tags could not send anything else.
    path = request.url.path
    if not path.startswith("/api/") or path in PUBLIC_API:
        return await call_next(request)
    user = sessions.resolve(request.cookies.get(auth_mod.SESSION_COOKIE))
    if user is None:
        return JSONResponse({"detail": "Sign in to continue"}, status_code=401)
    if user["role"] == DRIVER and not any(path == d or path.startswith(d + "/") for d in DRIVER_API):
        return JSONResponse({"detail": "Your account does not have access to this"}, status_code=403)
    request.state.user = user
    return await call_next(request)


# ---------------------------------------------------------------- signing in

class StaffLogin(BaseModel):
    username: str
    password: str


class CodeRequest(BaseModel):
    licence_no: str


class CodeVerify(BaseModel):
    licence_no: str
    code: str


def _limited(ok_wait, what):
    ok, wait = ok_wait
    if not ok:
        raise HTTPException(429, f"Too many {what}. Try again in {wait} seconds.")


@app.post("/api/login")
def staff_login(body: StaffLogin, request: Request, response: Response):
    """Traffic controllers and administrators."""
    address = webauth.client_address(request)
    _limited(auth_mod.by_address.check(address), "attempts from this address")
    _limited(auth_mod.login_attempts.check(body.username.strip().lower()), "sign-in attempts")
    user = store.user_by_username(body.username.strip())
    # The same answer whether the account is unknown or the password is wrong: anything else tells an
    # attacker which usernames exist.
    if not user or user["role"] == DRIVER or not auth_mod.verify_secret(body.password,
                                                                        user["password_hash"]):
        store.audit(None, "login.fail", "user", body.username.strip(), "bad username or password", address)
        raise HTTPException(401, "Wrong username or password")
    auth_mod.login_attempts.clear(body.username.strip().lower())
    sessions.start(response, user, address)
    store.audit({"user_id": user["id"], "name": user["name"], "username": user["username"],
                 "role": user["role"]}, "login.ok", "user", user["id"], "", address)
    return {"id": user["id"], "role": user["role"], "name": user["name"], "username": user["username"]}


@app.post("/api/login/code")
def request_code(body: CodeRequest, request: Request):
    """Step one for a driver: send a one-time code to the mobile registered against this licence.

    The reply is identical whether or not the licence is known. A licence number is printed on a card and
    handed over routinely, so telling a caller "no such licence" turns this endpoint into a way of testing
    whether a person holds one."""
    address = webauth.client_address(request)
    _limited(auth_mod.by_address.check(address), "attempts from this address")
    licence = auth_mod.normalise_licence(body.licence_no)
    if not licence:
        raise HTTPException(400, "Enter your driving licence number")
    _limited(auth_mod.code_requests.check(licence), "codes requested for this licence")
    user = store.user_by_licence(licence)
    out = {"sent": True, "message": "If that licence is registered, a code has been sent to the mobile "
                                    "number held for it."}
    if not user:
        store.audit(None, "login.code.unknown", "licence", licence, "", address)
        return out
    code = auth_mod.new_code()
    store.add_login_code(user["id"], auth_mod.hash_secret(code),
                         time.time() + auth_mod.OTP_TTL_SEC)
    shown = auth_mod.deliver_code(code, user["phone"], licence)
    store.audit(None, "login.code.sent", "user", user["id"], "", address)
    # No hint about the number here, however helpful it would be. Returning one for a licence that exists
    # and nothing for one that does not turns this endpoint into a test for whether a person holds a
    # licence, which is exactly the question the identical wording above is meant to refuse to answer.
    if shown:
        out["demo_code"] = shown      # only when OTP_DELIVERY=echo, which warns about itself
    return out


@app.post("/api/login/verify")
def verify_code(body: CodeVerify, request: Request, response: Response):
    """Step two for a driver: exchange the code for a session."""
    address = webauth.client_address(request)
    _limited(auth_mod.by_address.check(address), "attempts from this address")
    licence = auth_mod.normalise_licence(body.licence_no)
    _limited(auth_mod.code_attempts.check(licence), "codes tried for this licence")
    user = store.user_by_licence(licence)
    bad = HTTPException(401, "That code is wrong or has expired")
    if not user:
        raise bad
    row = store.live_login_code(user["id"], time.time())
    if not row or row["attempts"] >= auth_mod.OTP_MAX_ATTEMPTS:
        raise bad
    store.bump_code_attempts(row["id"])
    if not auth_mod.verify_secret(body.code.strip(), row["code_hash"]):
        store.audit(None, "login.code.bad", "user", user["id"], "", address)
        raise bad
    store.spend_login_code(row["id"])
    auth_mod.code_attempts.clear(licence)
    sessions.start(response, user, address)
    store.audit({"user_id": user["id"], "name": user["name"], "licence_no": user["licence_no"],
                 "role": user["role"]}, "login.ok", "user", user["id"], "", address)
    return {"id": user["id"], "role": user["role"], "name": user["name"]}


@app.post("/api/logout")
def logout(response: Response, tsid: str = Cookie(default=None), user=Depends(current_user)):
    if user:
        store.audit(user, "logout", "user", user["user_id"], "", user.get("address", ""))
    sessions.end(response, tsid)
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(current_user)):
    """Who is signed in, for the page to decide what to show. Null when nobody is."""
    if not user:
        return {"signed_in": False}
    out = {"signed_in": True, **user}
    if user["role"] == DRIVER:
        out["vehicles"] = store.vehicles_of(user["user_id"])
    return out


@app.get("/api/meta")
def meta(request: Request):
    return {
        "classes": sorted(MOTORISED | NON_MOTORISED), "motorised": sorted(MOTORISED),
        "road_categories": ROAD_CATEGORIES, "directions": DIRECTIONS, "area_types": AREA_TYPES,
        "speed_limits": DEFAULT_SPEED_LIMITS, "default_settings": DEFAULT_SETTINGS,
        "rules": {key: {"code": cls.code, "title": cls.title, "defaults": cls.DEFAULTS} for key, cls in RULES.items()},
        "lan_url": lan_url(request.url.port or 80),
        "advisory_codes": sorted(cit.ADVISORY_CODES),    # detected and reviewable, but never cited
        "features": {"anpr": all(find_models(MODELS_DIR)), "helmet": find_helmet_model() is not None,
                     "vlm": vlm_mod.backend_status(vlm_settings())["enabled"],
                     "reid": True},
    }


def lan_url(port=8000):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return f"http://{ip}:{port}"
    except OSError:
        return None


# ---------------------------------------------------------------- cameras

class CameraIn(BaseModel):
    name: str
    source: str
    location: str = ""
    start_time: Optional[str] = None
    zone_mode: str = "manual"     # "auto": learn the zones from traffic, then start; "manual": draw them


class CameraPatch(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    start_time: Optional[str] = None
    settings: Optional[dict] = None


def default_settings(kind):
    """Recorded video: complete, most accurate result (960 px detector input sees more far / small vehicles and
    measures ~40% more of each zone reliably, ~30 FPS on an M5). Live: 640 px, ~60 FPS, always real time."""
    return {"imgsz": 960 if kind == "file" else 640}


@app.get("/api/cameras")
def list_cameras():
    return [camera_view(c) for c in store.cameras()]


@app.get("/api/cameras/{cid}")
def get_camera(cid: int):
    cam = cam_or_404(cid)
    return {**camera_view(cam), "runs": store.runs(cid), "zone_history": store.zone_history(cid)}


@app.post("/api/cameras")
def add_camera(body: CameraIn, user=Depends(require(ADMIN))):
    """A live camera (phone / IP / device index) or a video file already on this Mac."""
    source = body.source.strip()
    if is_live_source(source):
        kind, start, how = "live", None, ""
    else:
        p = Path(source).expanduser()
        if not p.is_file():
            raise HTTPException(400, f"No video file at {source}")
        source, kind = str(p), "file"
        start, how = detect_start_time(p)
        start = start.isoformat(timespec="seconds") if start else None
    if body.start_time:
        start, how = body.start_time, "entered by you"
    cid = store.add_camera(body.name.strip() or "Camera", kind, source, body.location.strip(), start, how,
                           settings={**default_settings(kind), "zone_mode": body.zone_mode})
    if body.zone_mode == "auto":
        manager.start_autozone(cid, apply=True, start_after=True)
    return camera_view(store.camera(cid))


@app.post("/api/cameras/upload")
def upload_camera(name: str = Form(...), location: str = Form(""), start_time: str = Form(""),
                  zone_mode: str = Form("manual"), file: UploadFile = File(...),
                  user=Depends(require(ADMIN))):
    if storage.storage.ingest_paused():
        raise HTTPException(507, "Disk nearly full: uploads are paused until space is freed (see Storage)")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Not a video file: {file.filename}")
    dest = DATA / "videos" / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(file.filename).name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f, 16 * 1024 * 1024)
    start, how = detect_start_time(file.filename)  # the original name usually holds the recording time
    if not start:
        start, how = detect_start_time(dest)
    if start_time:
        start, how = datetime.fromisoformat(start_time), "entered by you"
    cid = store.add_camera(name.strip() or Path(file.filename).stem, "file", str(dest), location,
                           start.isoformat(timespec="seconds") if start else None, how,
                           {**default_settings("file"), "zone_mode": zone_mode})
    if zone_mode == "auto":
        storage.queue.enqueue(cid, label=name)   # learns zones, then analyses, when a slot is free
    return camera_view(store.camera(cid))


@app.post("/api/sites/upload")
def upload_site(name: str = Form(""), location: str = Form(""), zone_mode: str = Form("auto"),
                files: list[UploadFile] = File(...), user=Depends(require(ADMIN))):
    """Several camera views of one place at once: one camera per video, learned and analysed in turn, then
    linked so a vehicle keeps one identity across all of them.

    They run one after another rather than together on purpose - the detector already uses every core, and
    two analyses at once would simply halve each other's frame rate - and the batch link runs when the last
    one finishes."""
    if storage.storage.ingest_paused():
        raise HTTPException(507, "Disk nearly full: uploads are paused until space is freed (see Storage)")
    if len(files) < 1:
        raise HTTPException(400, "Choose at least one video")
    site = name.strip() or f"Site {datetime.now():%H:%M}"
    made = []
    for n, file in enumerate(files, 1):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            raise HTTPException(400, f"Not a video file: {file.filename}")
        dest = DATA / "videos" / f"{datetime.now():%Y%m%d_%H%M%S}_{n}_{Path(file.filename).name}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f, 16 * 1024 * 1024)
        start, how = detect_start_time(file.filename)
        if not start:
            start, how = detect_start_time(dest)
        cid = store.add_camera(f"{site} - {Path(file.filename).stem}", "file", str(dest), location,
                               start.isoformat(timespec="seconds") if start else None, how,
                               {**default_settings("file"), "zone_mode": zone_mode, "site": site})
        made.append(cid)
    if not any(store.camera(c)["start_time"] for c in made):
        pass     # no recording times: linking still works, it just uses each video's own clock
    sites[site] = SiteRun(manager, store, made, zone_mode, notify=notify, linker=linker)
    sites[site].start()
    return {"site": site, "cameras": [camera_view(store.camera(c)) for c in made],
            "status": sites[site].status}


@app.get("/api/sites")
def list_sites():
    return {s: dict(j.status, cameras=j.cameras) for s, j in sites.items()}


@app.patch("/api/cameras/{cid}")
def patch_camera(cid: int, body: CameraPatch, user=Depends(require(ADMIN))):
    """Rename / move a camera position, or change its settings. The source of a running camera cannot change."""
    cam = cam_or_404(cid)
    fields = body.model_dump(exclude_none=True)
    if "name" in fields:
        fields["name"] = fields["name"].strip()[:80]
        if not fields["name"]:
            raise HTTPException(400, "The camera needs a name")
    if "location" in fields:
        fields["location"] = fields["location"].strip()[:120]
    if "source" in fields:
        fields["source"] = fields["source"].strip()
        if fields["source"] == cam["source"]:
            del fields["source"]
        elif manager.running(cid):
            raise HTTPException(400, "Stop the camera before changing its source")
        elif is_live_source(fields["source"]) != (cam["kind"] == "live"):
            raise HTTPException(400, "Add a new camera to switch between a video file and a live camera")
        elif cam["kind"] == "file" and not Path(fields["source"]).expanduser().is_file():
            raise HTTPException(400, f"No video file at {fields['source']}")
    if "settings" in fields:
        fields["settings"] = {**cam["settings"], **fields["settings"]}
    if "start_time" in fields:
        if fields["start_time"]:
            datetime.fromisoformat(fields["start_time"])
        fields["start_time_source"] = "entered by you" if fields["start_time"] else ""
        fields["start_time"] = fields["start_time"] or None
    store.update_camera(cid, **fields)
    return camera_view(store.camera(cid))


@app.delete("/api/cameras/{cid}")
def delete_camera(cid: int, user=Depends(require(ADMIN))):
    cam_or_404(cid)
    keep_cited_evidence(cid)
    manager.stop_autozone(cid)
    manager.stop(cid, wait=True)
    store.delete_camera(cid)
    shutil.rmtree(DATA / "evidence" / f"cam{cid}", ignore_errors=True)
    return {"ok": True}


@app.post("/api/cameras/{cid}/start")
def start_camera(cid: int, start_sec: float = 0.0, user=Depends(require(CONTROLLER))):
    cam_or_404(cid)
    if not store.zones(cid):
        raise HTTPException(400, "Set up the zones first: learn them automatically or draw them in the zone editor")
    try:
        manager.start(cid, start_sec)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return manager.status(cid)


class AutoZoneIn(BaseModel):
    apply: bool = True       # save the learned zones as the camera's zones (False: only propose, for the editor)
    start: bool = True       # then start the camera


@app.post("/api/cameras/{cid}/auto-zones")
def start_auto_zones(cid: int, body: AutoZoneIn, user=Depends(require(ADMIN))):
    """Learn the zones from the camera's own traffic (surveillance/autozone.py)."""
    cam_or_404(cid)
    try:
        manager.start_autozone(cid, apply=body.apply, start_after=body.start)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    if body.apply:
        store.update_camera(cid, settings={**store.camera(cid)["settings"], "zone_mode": "auto"})
    return manager.autozone_status(cid)


@app.get("/api/cameras/{cid}/auto-zones")
def auto_zones_status(cid: int):
    cam_or_404(cid)
    return manager.autozone_status(cid)


@app.delete("/api/cameras/{cid}/auto-zones")
def cancel_auto_zones(cid: int, user=Depends(require(ADMIN))):
    manager.stop_autozone(cid)
    return {"ok": True}


@app.post("/api/cameras/{cid}/stop")
def stop_camera(cid: int, user=Depends(require(CONTROLLER))):
    manager.stop(cid)
    return {"ok": True}


@app.delete("/api/cameras/{cid}/violations")
def clear_camera_violations(cid: int, user=Depends(require(ADMIN))):
    if manager.running(cid):
        raise HTTPException(400, "Stop the camera first")
    keep_cited_evidence(cid)
    store.clear_violations(cid)
    shutil.rmtree(DATA / "evidence" / f"cam{cid}", ignore_errors=True)
    store.audit(user, "camera.clear_violations", "camera", cid, "all cases and tracklets deleted", user.get("address", ""))
    return {"ok": True}

@app.get("/api/status")
def all_status():
    return {c["id"]: manager.status(c["id"]) for c in store.cameras()}


@app.get("/api/devices")
def devices():
    """Cameras this Mac can open (built-in, iPhone Continuity Camera, USB)."""
    return probe_mac_cameras()


@app.post("/api/test-source")
def test_source(body: CameraIn, user=Depends(require(ADMIN))):
    try:
        frame = grab_frame(body.source.strip())
    except Exception as e:
        raise HTTPException(400, f"Could not get a picture: {e}")
    return jpeg(cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1]))))


# ---------------------------------------------------------------- frames & live view

@app.get("/api/cameras/{cid}/frame")
def camera_frame(cid: int, t: float = 0.0):
    """A clean frame for the zone editor (from the running worker for live cameras)."""
    cam = cam_or_404(cid)
    if cam["kind"] == "live" and manager.running(cid):
        raise HTTPException(409, "Stop the camera to grab a clean frame")
    try:
        frame = grab_frame(cam["source"], t, EDITOR_WIDTH)
    except Exception as e:
        raise HTTPException(400, str(e))
    EDITOR_FRAMES[cid] = frame   # zones saved next are drawn on this view: kept as their reference
    return jpeg(frame, 90)


@app.get("/api/cameras/{cid}/traffic")
def camera_traffic(cid: int):
    """Live traffic state of each road zone of a running camera: queue, mean speed, occupancy, density and waiting
    time over the last window of video, and whether a time-of-day baseline exists for the zone yet."""
    cam_or_404(cid)
    w = manager.get(cid)
    col = w.traffic if w is not None and w.is_alive() else None
    if col is None:
        why = ("The camera is not running." if w is None or not w.is_alive() else
               "Traffic measuring is switched off for this camera." if not w.cam["settings"].get("impact", True)
               else "The camera is still starting.")
        return {"camera_id": cid, "running": False, "message": why, "zones": {}}
    zones = col.snapshot()
    for name, z in zones.items():
        wall = col.wall_at(z["t"]) if z["t"] is not None else None
        ref = w.baseline.stats(cid, name, wall) if wall is not None and w.baseline is not None else None
        z["baseline"] = {"source": "history", "days": ref.detail["days"]} if ref else {"source": "none"}
    return {"camera_id": cid, "running": True, "window_s": col.window, "video_time": w.status.get("video_time"),
            "zones": zones}


@app.get("/api/cameras/{cid}/stream")
async def stream(cid: int):
    """Annotated live view as MJPEG (works in an <img> tag on any browser, including phones)."""
    cam_or_404(cid)

    async def gen():
        last = -1
        idle = 0
        while True:
            w = manager.get(cid)
            if w and w.jpeg is not None and w.seq != last:
                last, idle = w.seq, 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(w.jpeg)).encode() +
                       b"\r\n\r\n" + w.jpeg + b"\r\n")
            else:
                idle += 1
                if idle > 600 and not (w and w.is_alive()):  # stop streaming 30 s after the worker ended
                    return
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/events")
async def events(request: Request):
    """Server-sent events: new violations and camera status changes."""
    q = asyncio.Queue()
    subscribers.add(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- zones

@app.get("/api/cameras/{cid}/zones")
def get_zones(cid: int):
    cam = cam_or_404(cid)
    z = store.zones(cid)
    return {"config": z["config"] if z else None, "version": z["version"] if z else None,
            "history": store.zone_history(cid), "learning": cam["learning"]}


class ZonesIn(BaseModel):
    config: dict
    note: str = ""


def validate_config(config):
    w, h = config.get("frame_size") or [None, None]
    if not w or not h:
        raise HTTPException(400, "frame_size missing")
    try:
        camera_config_from_dict(config, w, h)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.put("/api/cameras/{cid}/zones")
def save_zones(cid: int, body: ZonesIn, user=Depends(require(ADMIN))):
    cam = cam_or_404(cid)
    validate_config(body.config)
    if not body.config.get("auto"):  # drawn / edited by hand
        store.update_camera(cid, settings={**cam["settings"], "zone_mode": "manual"})
    _, version = store.save_zones(cid, body.config, body.note)
    frame = EDITOR_FRAMES.get(cid)
    if frame is not None and list(frame.shape[1::-1]) == list(body.config.get("frame_size") or []):
        save_reference(DATA, cid, version, frame)   # lets the camera follow its zones if the view moves later
    restarted = False
    if manager.running(cid):  # apply the new zones right away
        manager.stop(cid, wait=True)
        manager.start(cid)
        restarted = True
    return {"version": version, "restarted": restarted}


@app.post("/api/cameras/{cid}/zones/validate")
def check_zones(cid: int, body: ZonesIn, user=Depends(require(ADMIN))):
    """Validates a draft and returns what the detector will do with it: the detection area and, per road zone,
    the line from which positions are precise enough to measure (drawn in the editor)."""
    cam = cam_or_404(cid)
    validate_config(body.config)
    w, h = body.config["frame_size"]
    cfg = camera_config_from_dict(body.config, w, h, require_road=False)
    if not cfg.roads:
        return {"ok": True, "roads": [], "roi": None, "bus_stops": []}
    roi = zones_roi(cfg, w, h) if cam["settings"]["auto_roi"] else (0, 0, w, h)
    limits = reliable_limits(cfg, roi, cam["settings"]["imgsz"])
    roads = []
    for r in cfg.roads:
        y = limits[id(r)]
        line = r.to_pixels([[0, y], [r.width_m, y]]).round(1).tolist()
        roads.append({"name": r.name, "reliable_from_m": round(y, 1), "reliable_m": round(r.length_m - y, 1),
                      "length_m": r.length_m, "line": line})
    return {"ok": True, "roads": roads, "roi": roi, "imgsz": cam["settings"]["imgsz"], "bus_stops": cfg.check_bus_stops()}


class KerbIn(BaseModel):
    points: list
    width_m: float
    length_m: float
    side: str = "left"
    strip_m: float = 3.0


@app.post("/api/kerb-strip")
def kerb_strip(body: KerbIn, user=Depends(require(ADMIN))):
    """A no-parking strip `strip_m` wide along the left or right edge of a road zone, as image points (follows the
    road's perspective, which is hard to click by hand)."""
    try:
        zone = RoadZone("kerb", body.points, body.width_m, body.length_m)
    except ValueError as e:
        raise HTTPException(400, str(e))
    w = min(max(0.5, body.strip_m), zone.width_m)
    x0, x1 = (0.0, w) if body.side == "left" else (zone.width_m - w, zone.width_m)
    pts = zone.to_pixels([[x0, 0], [x1, 0], [x1, zone.length_m], [x0, zone.length_m]])
    return {"points": [[round(float(x), 1), round(float(y), 1)] for x, y in pts]}


class TopDownIn(BaseModel):
    points: list
    width_m: float
    length_m: float
    t: float = 0.0


@app.post("/api/cameras/{cid}/topdown")
def topdown(cid: int, body: TopDownIn, user=Depends(require(ADMIN))):
    """Bird's-eye view of a road zone: lane lines must come out straight and parallel if the size is right."""
    cam = cam_or_404(cid)
    if cam["kind"] == "live" and manager.running(cid):
        raise HTTPException(409, "Stop the camera to check the top-down view")
    try:
        zone = RoadZone("check", body.points, body.width_m, body.length_m)
        frame = grab_frame(cam["source"], body.t if cam["kind"] == "file" else 0, EDITOR_WIDTH)
    except Exception as e:
        raise HTTPException(400, str(e))
    height = 700
    ppm = height / zone.length_m
    out_w = max(60, int(zone.width_m * ppm))
    top = cv2.warpPerspective(frame, np.diag([ppm, ppm, 1.0]) @ zone.H, (out_w, height))
    for d in np.arange(10, zone.length_m, 10):
        y = int(d * ppm)
        cv2.line(top, (0, y), (out_w, y), (0, 200, 255), 1)
        cv2.putText(top, f"{d:g} m", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    return jpeg(top)


# ---------------------------------------------------------------- violations

class ReviewIn(BaseModel):
    status: str
    note: str = ""


@app.get("/api/violations")
def list_violations(camera_id: Optional[int] = None, code: Optional[str] = None, status: Optional[str] = None,
                    run_id: Optional[int] = None, plate: Optional[str] = None, limit: int = 60, offset: int = 0,
                    order: str = "DESC"):
    rows = store.violations(camera_id, code, status, run_id, limit=min(limit, 500), offset=offset, order=order,
                            plate=plate)
    rows = [with_impact(v) for v in rows]
    total = store.count(camera_id=camera_id, code=code, status=status, run_id=run_id, plate=plate)
    return {"items": rows, "total": total}


@app.get("/api/violations/{vid}")
def get_violation(vid: int):
    v = store.violation(vid)
    if not v:
        raise HTTPException(404, "Violation not found")
    return with_impact(v)


@app.patch("/api/violations/{vid}")
def review_violation(vid: int, body: ReviewIn, user=Depends(require(CONTROLLER))):
    if body.status not in ("pending", "confirmed", "rejected"):
        raise HTTPException(400, "status must be pending, confirmed or rejected")
    if not store.violation(vid):
        raise HTTPException(404, "Violation not found")
    held = store.citation_by_violation(vid)
    if held and held["status"] != "withdrawn":
        # The seal records that this case was confirmed when it was cited. Letting a decision change underneath
        # a standing citation would leave a citation for a case nobody now stands behind.
        raise HTTPException(409, f"Citation {held['citation_no']} was issued for this case. An administrator "
                                 "has to withdraw it before the decision can change.")
    store.review(vid, body.status, body.note, by=user)
    store.audit(user, f"violation.{body.status}", "violation", vid, body.note, user.get("address", ""))
    return with_impact(store.violation(vid))


class SeverityIn(BaseModel):
    grade: Optional[int] = None     # 1 (minor) .. 5 (severe); null takes the rating back


@app.post("/api/violations/{vid}/severity")
def rate_severity(vid: int, body: SeverityIn, user=Depends(require(CONTROLLER))):
    """A reviewer's own grade of how badly the event hurt traffic: the independent ordering the impact score is
    validated against (impact.validate). Only a case that has an impact row can be graded - a grade is only
    useful next to a score."""
    if not store.violation(vid):
        raise HTTPException(404, "Violation not found")
    if body.grade is not None and not 1 <= body.grade <= 5:
        raise HTTPException(400, "grade must be 1 (minor) to 5 (severe), or null to take the rating back")
    if impact_mod.load_impact(store, vid) is None:
        raise HTTPException(409, "This case has no impact measurement yet (it is scored 30 s of video after the "
                                 "event), so there is nothing for a rating to be compared with.")
    impact_mod.set_human_severity(store, vid, body.grade)
    store.audit(user, "violation.severity", "violation", vid, "cleared" if body.grade is None else f"{body.grade}/5",
                user.get("address", ""))
    return with_impact(store.violation(vid))


def weights_view(w):
    d = w.to_dict()
    d["calibrated"] = w.any_calibrated()
    d["label"] = "calibrated" if d["calibrated"] else "equal weights (uncalibrated)"
    return d


@app.post("/api/impact/calibrate")
def calibrate_impact(user=Depends(require(ADMIN))):
    """Fit the five weights to the excess delay of the events scored so far, and adopt them only if they agree with
    the reviewers' severity grades better than equal weights do (impact.compare_weights). Otherwise equal weights
    stay in force - or return, if an earlier fit had been adopted - and the answer says why. New weights apply to
    cases scored from the next camera start."""
    fitted = impact_mod.calibrate(impact_mod.calibration_rows(store))
    verdict = impact_mod.compare_weights(impact_mod.graded_rows(store), fitted)
    before = impact_mod.load_weights(store)
    if verdict["adopt"]:
        impact_mod.save_weights(store, fitted)
        in_use = fitted
    elif before.any_calibrated():
        in_use = impact_mod.Weights()
        impact_mod.save_weights(store, in_use)
    else:
        in_use = before
    store.audit(user, "impact.calibrate", "impact", "weights",
                f"{'adopted' if verdict['adopt'] else 'kept equal weights'}: {verdict['reason']}; "
                f"{fitted.n_rows} events", user.get("address", ""))
    return {"adopted": verdict["adopt"], "reason": verdict["reason"], "n_events": fitted.n_rows,
            "n_skipped": fitted.skipped, "min_events": fitted.min_events, "comparison": verdict,
            "candidate": weights_view(fitted), "weights_in_use": weights_view(in_use)}


@app.get("/api/impact/validation")
def impact_validation(user=Depends(require(ADMIN))):
    """Rank agreement (Spearman rho, bootstrap interval, permutation p) between the impact score and the reviewers'
    grades, for equal weights and for the weights in force, each applied to the same graded cases."""
    w = impact_mod.load_weights(store)
    graded = impact_mod.graded_rows(store)
    n = store.q("SELECT COUNT(*) AS n, COALESCE(SUM(score IS NOT NULL), 0) AS scored FROM event_impact")[0]
    return {"n_impacts": n["n"], "n_scored": n["scored"], "n_graded": len(graded),
            "min_graded": impact_mod.MIN_GRADED,
            "equal": impact_mod.validate(*impact_mod.score_graded(graded, impact_mod.Weights())),
            "in_use": impact_mod.validate(*impact_mod.score_graded(graded, w)), "weights_in_use": weights_view(w)}


# ---------------------------------------------------------------- citations (e-prosecution)

def citation_or_404(no):
    if not cit.NUMBER_RE.match(no or "") or not store.citation(no):
        raise HTTPException(404, "No such citation")
    return store.citation(no)


@app.post("/api/violations/{vid}/citation")
def issue_citation(vid: int, user=Depends(require(CONTROLLER))):
    """Issue the citation for a case: number it, seal its evidence, make the driver's smaller copy of the clip.

    Idempotent - asking again returns the citation already issued. Refused with 409, and the reason, for an
    advisory code (speed), a case no controller has confirmed, a plate that was not read, or a clip that is
    still being encoded. Takes a few seconds: it hashes every evidence file and transcodes the clip."""
    try:
        row, created, warnings = cit.issue(store, vid, user, DATA, delivery=make_delivery_clip)
    except cit.Refused as e:
        raise HTTPException(e.status, str(e))
    return {**cit.view(row), "created": created, "warnings": warnings}


@app.get("/api/citations")
def list_citations(status: Optional[str] = None, plate: Optional[str] = None, limit: int = 100, offset: int = 0,
                   user=Depends(require(CONTROLLER))):
    if status and status not in ("issued", "withdrawn", "paid"):
        raise HTTPException(400, "status must be issued, withdrawn or paid")
    rows, total = store.citations(status, plate, min(max(limit, 1), 500), max(offset, 0))
    return {"items": [{**r, "fine_text": cit.fine_text(r["fine_bdt"])} for r in rows], "total": total}


@app.get("/api/citations/{no}")
def get_citation(no: str, user=Depends(require(CONTROLLER))):
    c = citation_or_404(no)
    v = store.violation(c["violation_id"])
    case = ({k: v[k] for k in ("id", "code", "title", "status", "plate_text_en", "wall_time", "camera_name")}
            if v else None)       # None once the case itself has been deleted: the citation outlives it
    return {**cit.view(c), "case": case}


@app.get("/api/citations/{no}/verify")
def verify_citation(no: str, user=Depends(require(CONTROLLER))):
    """Re-hash every file the citation sealed and report match / mismatch / missing per file. Audited."""
    citation_or_404(no)
    try:
        return cit.verify_citation(store, no, user, DATA)
    except cit.Refused as e:
        raise HTTPException(e.status, str(e))


class WithdrawIn(BaseModel):
    reason: str = ""


@app.post("/api/citations/{no}/withdraw")
def withdraw_citation(no: str, body: WithdrawIn, user=Depends(require(ADMIN))):
    """Withdraw a citation (administrators only). The reason is required and kept; the number is never reused."""
    citation_or_404(no)
    try:
        row, changed = cit.withdraw(store, no, user, body.reason)
    except cit.Refused as e:
        raise HTTPException(e.status, str(e))
    if changed:
        disputes_mod.close_as_moot(store, no, user)     # an open dispute of it has nothing left to decide
    return {**cit.view(row), "changed": changed}


class ExplainIn(BaseModel):
    backend: Optional[str] = None   # auto (default) / local / cloud; one case only
    force: bool = False             # explain again even though an explanation is stored


@app.post("/api/violations/{vid}/explain")
def explain_violation(vid: int, body: ExplainIn = ExplainIn()):
    """Ask the vision-language model to explain this case. Returns at once: the answer takes seconds to a
    minute on the local backend and arrives over /api/events as an "explanation" message."""
    v = store.violation(vid)
    if not v:
        raise HTTPException(404, "Violation not found")
    settings = vlm_settings(body.backend)
    if not cascade.usable(settings) and vlm_mod.resolve_backend(settings) is None:
        raise HTTPException(503, "No reasoning model available. Start it with ./tools/vlm_server.sh --tier screen, "
                                 "or set ANTHROPIC_API_KEY for the cloud backend.")
    if v.get("vlm_status") == "pending":
        return {"queued": False, "already": "pending", "violation": v}
    if v.get("vlm") and not body.force:
        return {"queued": False, "already": "done", "violation": v}
    # Mark it pending before queueing, never after: a fast backend can finish and write its answer while
    # this request is still running, and a "pending" written afterwards would erase it.
    store.set_vlm(vid, "pending")
    # Hand the reasoner where else this vehicle was seen: one camera cannot distinguish a vehicle caught in
    # a jam from one that stopped dead, and the camera before it usually can.
    v = {**v, "journey": store.journey_for_violation(v)}
    if not explainer.request(v, settings):
        store.set_vlm(vid, "")
        raise HTTPException(429, "The reasoning queue is full; try again shortly")
    return {"queued": True, "violation": store.violation(vid)}


@app.get("/api/vlm")
def vlm_status():
    """Which reasoning backend could answer right now (shown in the dashboard)."""
    s = vlm_settings()
    return {**vlm_mod.backend_status(s), "busy": explainer.busy(), "cascade": cascade.status(s)}


class VlmSettingsIn(BaseModel):
    settings: dict


@app.get("/api/vlm/settings")
def vlm_settings_get(user=Depends(require(ADMIN))):
    s = vlm_settings()
    return {"settings": {k: s[k] for k in VLM_ADMIN_KEYS}, "options": {
        k: (list(v) if isinstance(v, tuple) and isinstance(v[0], str) else
            {"min": v[0], "max": v[1]} if isinstance(v, tuple) else "text") for k, v in VLM_ADMIN_KEYS.items()},
        "cascade": cascade.status(s)}


@app.put("/api/vlm/settings")
def vlm_settings_put(body: VlmSettingsIn, user=Depends(require(ADMIN))):
    """Validate every value against VLM_ADMIN_KEYS, then save; takes effect on the next Explain request."""
    clean = _vlm_saved()
    for k, v in body.settings.items():
        rule = VLM_ADMIN_KEYS.get(k)
        if rule is None:
            raise HTTPException(400, f"Unknown VLM setting {k}")
        if rule is str:
            if not isinstance(v, str) or not v.strip() or len(v) > 200:
                raise HTTPException(400, f"{k} must be a short text")
            if k.endswith("_url") and not v.startswith(("http://127.0.0.1", "http://localhost")):
                raise HTTPException(400, f"{k} must point at this machine (http://127.0.0.1:PORT)")
            clean[k] = v.strip()
        elif isinstance(rule[0], str):
            if v not in rule:
                raise HTTPException(400, f"{k} must be one of {', '.join(rule)}")
            clean[k] = v
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not rule[0] <= v <= rule[1]:
                raise HTTPException(400, f"{k} must be between {rule[0]} and {rule[1]}")
            clean[k] = v
    (DATA / "vlm_settings.json").write_text(json.dumps(clean, indent=1))
    store.audit(user, "vlm.settings", "vlm", "settings", ", ".join(sorted(body.settings)), user.get("address", ""))
    return vlm_settings_get(user)


class LinkIn(BaseModel):
    since: Optional[float] = None    # epoch seconds; default: everything stored
    until: Optional[float] = None


@app.post("/api/link")
def start_linking(body: LinkIn = LinkIn()):
    """Match vehicles across every camera and assign global vehicle ids."""
    job = link_jobs.get("current")
    if job and job.is_alive():
        return {"started": False, "status": job.status}

    def done(msg):
        notify(msg)
        linker.learn_from()          # the online path picks up the routes this run just learned

    job = LinkJob(store, body.since, body.until, notify=done)
    link_jobs["current"] = job
    job.start()
    return {"started": True, "status": job.status}


@app.get("/api/link")
def linking_status():
    job = link_jobs.get("current")
    if not job:
        return {"state": "idle", "routes_known": len(linker.params)}
    st = dict(job.status)
    st["routes_known"] = len(linker.params)
    if job.result:
        st["result"] = job.result
    return st


@app.get("/api/vehicles")
def list_vehicles(min_cameras: int = 2, limit: int = 100):
    """Vehicles recognised on more than one camera."""
    items = store.vehicles(limit=min(limit, 500), min_cameras=min_cameras)
    for v in items:
        v["display_id"] = topology.display_id(v["global_id"])
    return {"items": items, "tracklets": store.q("SELECT COUNT(*) AS n FROM tracklets")[0]["n"]}


@app.get("/api/vehicles/{vehicle}")
def vehicle_journey(vehicle: str):
    """Every camera one vehicle passed, in order, with how long it took between them. Takes "DV-000123", "dv123"
    or "123"; an id that was later merged into another vehicle (records that arrived out of order) opens that one."""
    gid = topology.parse_display_id(vehicle)
    if gid is None:
        raise HTTPException(404, "No such vehicle")
    try:
        gid = topology.resolve(store, gid) or gid
        merged = topology.merged_from(store, gid)
    except sqlite3.OperationalError:     # no merge table yet: nothing was ever merged
        merged = []
    rows = store.journey(gid)
    if not rows:
        raise HTTPException(404, "No such vehicle")
    for a, b in zip(rows, rows[1:]):
        gap = (b["t_cross"] or 0) - (a["t_cross"] or 0)
        b["seconds_from_previous"] = round(gap, 1)
        b["kmh_between"] = round(3.6 * (a["speed"] or 0), 1) if gap > 0 else None
    return {"global_id": gid, "display_id": topology.display_id(gid), "merged_ids": merged, "passes": rows}


# ---------------------------------------------------------------- disputes (staff)

def dispute_or_404(did):
    d = disputes_mod.dispute(store, did)
    if not d:
        raise HTTPException(404, "No such dispute")
    return d


@app.get("/api/disputes")
def list_disputes(status: str = "open", limit: int = 100, offset: int = 0, user=Depends(require(CONTROLLER))):
    """The dispute queue, open ones first in the order to read them (those the model thinks the evidence does not
    support go first, then the rest, oldest first). `open` in the answer is the badge count."""
    if status not in ("open", "decided", "all"):
        raise HTTPException(400, "status must be open, decided or all")
    rows, total = disputes_mod.queue_rows(store, status, min(max(limit, 1), 500), max(offset, 0))
    return {"items": rows, "total": total, "open": disputes_mod.open_count(store)}


@app.get("/api/disputes/{did}")
def get_dispute(did: int, user=Depends(require(CONTROLLER))):
    d = dispute_or_404(did)
    c = store.citation(d["citation_no"])
    v = store.violation(d["violation_id"])
    case = None
    if v:
        case = {k: v.get(k) for k in ("id", "code", "title", "cls", "status", "plate_text_en", "plate_text", "wall_time",
                                     "video_time_s", "zone", "kph", "details", "camera_name", "snapshot", "crop",
                                     "clip", "plate_img", "evidence", "reviewed_by_name", "note")}
        case["code_claim"] = vlm_mod.RULE_CLAIM.get(case["code"] or "", ("", ""))[0]
    return {**disputes_mod.staff_view(d), "case": case, "citation": cit.view(c, manifest=False) if c else None,
            "window_days": disputes_mod.WINDOW_DAYS, "trail": store.audit_trail("dispute", d["id"])}


class DecideIn(BaseModel):
    decision: str = ""
    reason: str = Field("", max_length=5000)


@app.post("/api/disputes/{did}/decide")
def decide_dispute(did: int, body: DecideIn, user=Depends(require(CONTROLLER))):
    """UPHOLD the citation or ACCEPT the dispute (which withdraws the citation with reason "dispute accepted: ...").
    A controller or an administrator may decide; the reason is required and is shown to the driver."""
    dispute_or_404(did)
    try:
        d = disputes_mod.decide(store, did, user, body.decision, body.reason)
    except cit.Refused as e:
        raise HTTPException(e.status, str(e))
    notify({"type": "dispute", "id": did})
    return disputes_mod.staff_view(d)


@app.post("/api/disputes/{did}/recheck")
def recheck_dispute(did: int, user=Depends(require(CONTROLLER))):
    """Ask the model to look again (for instance after "model check unavailable"). Advisory, like the first."""
    d = dispute_or_404(did)
    if d["status"] != "open":
        raise HTTPException(409, "This dispute is already closed")
    store.x("UPDATE disputes SET model_state='pending' WHERE id=?", (did,))
    store.audit(user, "dispute.model_check", "dispute", did, "re-check requested", user.get("address", ""))
    if not dispute_checker.submit(store, DATA, did):
        raise HTTPException(429, "A check of this dispute is already queued, or the queue is full")
    return {"queued": True}


# ---------------------------------------------------------------- the driver's own portal

def _driver(request: Request):
    u = getattr(request.state, "user", None)
    if not u or u["role"] != DRIVER:
        raise HTTPException(403, "Drivers only")
    return u


def _my_plate_keys(user):
    """Plates a driver may see, dispute and pay: their verified vehicles plus the owner contacts that carry their
    licence number. Both are set by an administrator (driver accounts are admin-created), and payments use the
    same function, so what a driver can see and what they can pay never disagree."""
    return sorted(registry_mod.driver_plate_keys(store, user))


def shared_link(path):
    """A signed, expiring URL for a file under data/evidence, or None when there is no such file (a dangling
    link would only show the driver a broken picture)."""
    return cit.signed_url(path, media_key()) if path and cit.evidence_file(DATA, path) else None


def _dispute_facts(c, d):
    """What a driver may know about disputing one citation: the window, whether they can, and their dispute."""
    ok, why, until = disputes_mod.eligibility(c, d)
    return {"window_days": disputes_mod.WINDOW_DAYS, "until": until, "can_dispute": ok,
            "cannot_reason": "" if ok else why, "dispute": disputes_mod.driver_view(d)}


def _for_driver(v, dispute=None):
    """A case as its registered keeper may see it: the evidence, and nothing about anyone else.

    Internal review material is stripped - the reviewer's name, the operator's note, the camera's
    identifiers - because none of it is theirs to see and some of it names a police officer.

    Files are never given as paths. Every picture and the clip is a signed link that expires (one hour by
    default), so a path that leaks - in a screenshot, a log, a forwarded message - opens nothing. The clip is
    the smaller copy made when the citation was issued, if there is one; the original stays on the server."""
    keep = ("id", "code", "title", "cls", "wall_time", "video_time_s", "zone", "kph", "details",
            "plate_text", "plate_text_en", "camera_location", "status", "vlm")
    out = {k: v.get(k) for k in keep}
    out["camera"] = v.get("camera_location") or ""
    out.pop("camera_location", None)
    out["snapshot"] = shared_link(v.get("snapshot"))
    out["crop"] = shared_link(v.get("crop"))
    out["plate_img"] = shared_link(v.get("plate_img"))
    out["clip"] = shared_link(v.get("delivery_clip")) or shared_link(v.get("clip"))
    photos = [(e, shared_link(e.get("path"))) for e in v.get("evidence") or []]
    out["evidence"] = [{"label": e.get("label"), "time_s": e.get("time_s"), "wall_time": e.get("wall_time"),
                        "url": url} for e, url in photos if url]
    out["citation"] = None if not v.get("citation_no") else {
        "number": v["citation_no"], "status": v.get("citation_status"), "issued_at": v.get("citation_issued_at"),
        "fine_bdt": v.get("citation_fine_bdt"), "fine_text": cit.fine_text(v.get("citation_fine_bdt")),
        **_dispute_facts({"issued_at": v.get("citation_issued_at") or "", "status": v.get("citation_status")},
                         dispute)}
    return out


@app.get("/api/portal/violations")
def portal_violations(request: Request, limit: int = 50, offset: int = 0):
    """Cases against the signed-in driver's verified vehicles for which a citation has been issued to them.

    Only cited cases (team decision 2026-10-30): a case that is confirmed but not cited has not become a
    liability for anyone yet, and one still in the review queue has not been decided at all. A withdrawn citation
    stays in the list, marked withdrawn, so the driver sees the outcome."""
    u = _driver(request)
    plates = _my_plate_keys(u)
    rows, total = store.violations_for_plates(plates, limit=min(limit, 200), offset=offset)
    found = disputes_mod.disputes_for_citations(store, [r["citation_no"] for r in rows])
    return {"items": [_for_driver(r, found.get(r["citation_no"])) for r in rows], "total": total,
            "vehicles": store.vehicles_of(u["user_id"]),
            "note": ("These are citations issued against vehicles registered to you. The system "
                     "identifies vehicles by their number plate, not drivers, so a case shows the vehicle "
                     "that was involved rather than who was at the wheel.")}


@app.get("/api/portal/violation/{vid}")
def portal_violation(vid: int, request: Request):
    u = _driver(request)
    plates = set(_my_plate_keys(u))
    v = store.violation(vid)
    # An uncited case, and an advisory one (speed), are answered exactly like one that does not exist: the list
    # already leaves them out, and a driver who guesses a number must not be able to tell it is there.
    if not v or not v.get("citation_no") or cit.is_advisory(v["code"]):
        raise HTTPException(404, "No such case")
    if auth_mod.normalise_plate(v.get("plate_text_en") or "") not in plates:
        store.audit(u, "portal.denied", "violation", vid, "case is not against a registered vehicle",
                    u.get("address", ""))
        raise HTTPException(404, "No such case")
    return _for_driver(v, disputes_mod.dispute_for_citation(store, v["citation_no"]))


def _my_citation(no, u):
    """The citation if it is on one of this driver's verified plates, else 404 - the same answer as for a number
    that does not exist, so another driver's citation cannot be told apart from none."""
    c = store.citation(no) if cit.NUMBER_RE.match(no or "") else None
    mine = set(_my_plate_keys(u))
    if not c or disputes_mod.owner_plate_key(c, store.violation(c["violation_id"])) not in mine:
        if c:
            store.audit(u, "portal.denied", "citation", no, "citation is not the driver's", u.get("address", ""))
        raise HTTPException(404, "No such citation")
    return c


class DisputeIn(BaseModel):
    statement: str = Field("", max_length=20000)


@app.get("/api/portal/citations/{no}/dispute")
def portal_dispute(no: str, request: Request):
    """The window, whether this citation can still be disputed, and the driver's dispute if they filed one."""
    u = _driver(request)
    c = _my_citation(no, u)
    return _dispute_facts(c, disputes_mod.dispute_for_citation(store, no))


@app.post("/api/portal/citations/{no}/dispute")
def portal_file_dispute(no: str, body: DisputeIn, request: Request):
    """Dispute an issued citation: statement of 20-2000 characters, within the window, one per citation. The
    model re-check starts in the background; a human decides."""
    u = _driver(request)
    try:
        d = disputes_mod.file(store, no, u, body.statement, _my_plate_keys(u))
    except cit.Refused as e:
        raise HTTPException(e.status, str(e))
    dispute_checker.submit(store, DATA, d["id"])
    notify({"type": "dispute", "id": d["id"]})
    return disputes_mod.driver_view(d)


class PlateClaim(BaseModel):
    plate: str


@app.post("/api/portal/vehicles")
def claim_vehicle(body: PlateClaim, request: Request):
    """A driver says a plate is theirs. It shows nothing until an administrator verifies it - otherwise
    anyone could read any vehicle's history by claiming its plate."""
    u = _driver(request)
    key = auth_mod.normalise_plate(body.plate)
    if len(key) < 4:
        raise HTTPException(400, "Enter the full number plate")
    existing = store.vehicle_by_plate(key)
    if existing:
        if existing["owner_id"] == u["user_id"]:
            return {"ok": True, "already": True, "verified": bool(existing["verified"])}
        raise HTTPException(409, "That plate is already registered to someone else. "
                                 "Contact the traffic authority if this is wrong.")
    vid = store.add_vehicle(key, u["user_id"], plate_en=body.plate.strip(), source="claimed", verified=0)
    store.audit(u, "vehicle.claim", "vehicle", vid, key, u.get("address", ""))
    return {"ok": True, "verified": False,
            "message": "Added. It will show your violations once the traffic authority has verified it."}


@app.delete("/api/portal/vehicles/{vid}")
def unclaim_vehicle(vid: int, request: Request):
    u = _driver(request)
    rows = [v for v in store.vehicles_of(u["user_id"]) if v["id"] == vid]
    if not rows:
        raise HTTPException(404, "Not one of your vehicles")
    store.delete_vehicle(vid)
    store.audit(u, "vehicle.unclaim", "vehicle", vid, rows[0]["plate_key"], u.get("address", ""))
    return {"ok": True}


# ---------------------------------------------------------------- accounts and registry (admin)

class UserIn(BaseModel):
    role: str
    name: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    licence_no: Optional[str] = None
    phone: str = ""


@app.get("/api/users")
def list_users(role: Optional[str] = None, user=Depends(require(ADMIN))):
    return {"items": store.users(role)}


@app.post("/api/users")
def create_user(body: UserIn, user=Depends(require(ADMIN))):
    if body.role not in auth_mod.ROLES:
        raise HTTPException(400, f"role must be one of {', '.join(auth_mod.ROLES)}")
    if body.role == DRIVER:
        licence = auth_mod.normalise_licence(body.licence_no or "")
        if not licence:
            raise HTTPException(400, "A driver needs a licence number")
        if not (body.phone or "").strip():
            raise HTTPException(400, "A driver needs a mobile number to receive sign-in codes")
        if store.user_by_licence(licence):
            raise HTTPException(409, "That licence number is already registered")
        uid = store.add_user(DRIVER, body.name, licence_no=licence, phone=body.phone.strip())
    else:
        if not body.username or not body.password:
            raise HTTPException(400, "Staff need a username and a password")
        if store.user_by_username(body.username.strip()):
            raise HTTPException(409, "That username is taken")
        uid = store.add_user(body.role, body.name, username=body.username.strip(),
                             password_hash=auth_mod.hash_secret(body.password))
    store.audit(user, "user.create", "user", uid, body.role, user.get("address", ""))
    return store.user(uid)


class UserPatch(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    active: Optional[bool] = None
    password: Optional[str] = None


@app.patch("/api/users/{uid}")
def update_user(uid: int, body: UserPatch, user=Depends(require(ADMIN))):
    target = store.user(uid)
    if not target:
        raise HTTPException(404, "No such account")
    fields = {k: v for k, v in body.dict().items() if v is not None and k != "password"}
    if body.active is not None:
        fields["active"] = int(body.active)
    if body.password:
        fields["password_hash"] = auth_mod.hash_secret(body.password)
    store.update_user(uid, **fields)
    if body.password or body.active is False:
        store.drop_sessions_of(uid)       # a changed password or a disabled account ends every session
    store.audit(user, "user.update", "user", uid, ",".join(fields), user.get("address", ""))
    return store.user(uid)


class PasswordChange(BaseModel):
    current: str
    new: str


@app.post("/api/me/password")
def change_own_password(body: PasswordChange, request: Request):
    """Staff change their own password. The bootstrap administrator's printed password must not survive."""
    u = getattr(request.state, "user", None)
    if not u or u["role"] == DRIVER:
        raise HTTPException(403, "Staff accounts only")
    row = store.user(u["user_id"])
    if not auth_mod.verify_secret(body.current, row["password_hash"]):
        raise HTTPException(401, "Current password is wrong")
    if len(body.new) < 10:
        raise HTTPException(400, "Use at least 10 characters")
    store.update_user(u["user_id"], password_hash=auth_mod.hash_secret(body.new))
    store.audit(u, "user.password", "user", u["user_id"], "changed own password", u.get("address", ""))
    return {"ok": True}


@app.get("/api/registry")
def list_registry(pending: bool = False, user=Depends(require(ADMIN))):
    return {"items": store.all_vehicles(pending_only=pending)}


class RegistryIn(BaseModel):
    plate: str
    licence_no: str
    plate_bn: str = ""


@app.post("/api/registry")
def register_vehicle(body: RegistryIn, user=Depends(require(ADMIN))):
    """Link a plate to a licence holder, standing in for the registry data a real deployment would import."""
    owner = store.user_by_licence(auth_mod.normalise_licence(body.licence_no))
    if not owner:
        raise HTTPException(404, "No driver with that licence number")
    key = auth_mod.normalise_plate(body.plate)
    if store.vehicle_by_plate(key):
        raise HTTPException(409, "That plate is already registered")
    vid = store.add_vehicle(key, owner["id"], plate_en=body.plate.strip(), plate_bn=body.plate_bn,
                            source="admin", verified=1)
    store.audit(user, "vehicle.register", "vehicle", vid, f"{key} -> {owner['licence_no']}",
                user.get("address", ""))
    return store.vehicle_by_plate(key)


@app.post("/api/registry/{vid}/verify")
def verify_vehicle(vid: int, user=Depends(require(ADMIN))):
    store.set_vehicle(vid, verified=1)
    store.audit(user, "vehicle.verify", "vehicle", vid, "", user.get("address", ""))
    return {"ok": True}


@app.delete("/api/registry/{vid}")
def remove_vehicle(vid: int, user=Depends(require(ADMIN))):
    store.delete_vehicle(vid)
    store.audit(user, "vehicle.remove", "vehicle", vid, "", user.get("address", ""))
    return {"ok": True}


@app.get("/api/audit")
def audit_trail(object_type: Optional[str] = None, object_id: Optional[str] = None,
                user_id: Optional[int] = None, limit: int = 200, user=Depends(require(ADMIN))):
    """Who did what. Append-only by convention: nothing in the API deletes from it."""
    return {"items": store.audit_trail(object_type, object_id, user_id, min(limit, 1000))}


@app.get("/api/export.csv")
def export_csv(camera_id: Optional[int] = None, status: Optional[str] = None):
    rows = store.violations(camera_id, status=status, limit=1_000_000, order="ASC")
    buf = io.StringIO()
    cols = ["id", "camera_name", "camera_location", "code", "title", "cls", "class_conf", "wall_time",
            "video_time_s", "zone", "kph", "plate_status", "plate_text", "plate_text_en", "plate_conf", "details",
            "vlm_assessment", "vlm_confidence", "vlm_summary", "vlm_cause", "vlm_model",
            "status", "note", "snapshot", "crop", "clip", "plate_img", "evidence", "created_at"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        x = r.get("vlm") or {}
        w.writerow({**r, "details": json.dumps(r["details"]),
                    "vlm_assessment": x.get("assessment", ""), "vlm_confidence": x.get("confidence", ""),
                    "vlm_summary": x.get("summary", ""), "vlm_cause": x.get("cause", ""),
                    "vlm_model": x.get("model", ""),
                    "evidence": " | ".join(f"{e['label']}: {e['path']}" for e in r.get("evidence") or [])})
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",  # BOM: Excel shows Bangla
                    headers={"Content-Disposition": "attachment; filename=violations.csv"})


@app.get("/api/stats")
def stats(camera_id: Optional[int] = None, days: int = 30, include_rejected: bool = False, basis: str = "event"):
    """basis=event: when the violation happened (real time, if known); basis=detected: when it was detected."""
    return store.stats(camera_id, include_rejected, days, basis)


def plate_line(v):
    st = v.get("plate_status") or ""
    if st in ("read", "partial"):
        return f"{v['plate_text']} ({v['plate_text_en']})" + (" - number only" if st == "partial" else "")
    return {"unreadable": "Not readable in the video - identify from the images", "pending": "Being read"}.get(st, "-")


def evidence_figure(e):
    when = e.get("wall_time") or f"video {e['time_s']:.1f} s"
    return (f'<figure><img src="/media/{escape(e["path"])}" alt=""><figcaption>{escape(e["label"].capitalize())}: '
            f'{escape(when)}</figcaption></figure>')


VERIFY_TEXT = ("The issuing authority can re-check this evidence at any time (a traffic controller opens "
               "/api/citations/{no}/verify). The check recomputes the SHA-256 of every file listed below and of "
               "the record that lists them, and names any file that differs. The bundle hash above is the "
               "SHA-256 of that record: keep this printout, because a copy of the hash held outside the "
               "system is what makes later alteration detectable.")


def citation_fields(c):
    """The citation block of the notice, as template fields: number, state, issuer, fine, and the evidence seal."""
    if not c:
        return {"cit_no": "Not issued", "cit_state": "none", "cit_status": "", "cit_issued": "-", "cit_fine": "-",
                "cit_bundle": "-", "sealed": "",
                "cit_note": "No citation has been issued for this case, so this page is a record of the case "
                            "only and carries no penalty."}
    try:
        files = json.loads(c["manifest"]).get("files") or []
    except (ValueError, AttributeError):
        files = []
    rows = "".join(f'<tr><td>{escape(str(f.get("role", "")))}</td>'
                   f'<td>{escape(str(f.get("path", "")).rsplit("/", 1)[-1])}</td>'
                   f'<td class="hash">{escape(str(f.get("sha256", "")))}</td></tr>' for f in files)
    sealed = (f'<table class="seal"><caption>Sealed evidence files (SHA-256)</caption>{rows}</table>'
              if rows else "")
    note = VERIFY_TEXT.replace("{no}", c["citation_no"])
    if c["status"] == "withdrawn":
        note = (f"WITHDRAWN on {c['withdrawn_at'].replace('T', ' ')} by "
                f"{c['withdrawn_by_name'] or 'an administrator'}: {c['withdraw_reason']}. "
                "This citation is void. " + note)
    return {"cit_no": c["citation_no"], "cit_state": c["status"], "cit_status": c["status"],
            "cit_issued": f"{c['issued_at'].replace('T', ' ')} by {c['issued_by_name'] or 'unknown'}",
            "cit_fine": cit.fine_text(c["fine_bdt"]), "cit_bundle": c["bundle_sha256"], "cit_note": note,
            "sealed": sealed}


# Template fields whose value is HTML this module has already escaped; every other field is escaped on the way in.
NOTICE_HTML_FIELDS = ("details", "extra", "sealed")


@app.get("/notice/{vid}", response_class=HTMLResponse)
def notice(vid: int, user=Depends(current_user)):
    """Printable violation notice for a case (browser: Print -> Save as PDF), with its citation if one was issued.

    Refused for an advisory code: speed is detected and shown to staff but never printed as a notice.
    A page, not an API: anonymous visitors go to /login and drivers to their own /portal."""
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["role"] == DRIVER:
        return RedirectResponse("/portal", status_code=302)
    v = store.violation(vid)
    if not v:
        raise HTTPException(404, "Violation not found")
    if cit.is_advisory(v["code"]):
        raise HTTPException(409, f"{v['code']} is advisory: it is detected and reviewed, but it is never printed "
                                 "as a notice or cited.")
    tpl = (WEB / "notice.html").read_text()
    details = "".join(f"<tr><th>{escape(k.replace('_', ' '))}</th><td>{escape(str(val))}</td></tr>"
                      for k, val in v["details"].items())
    fields = {"id": v["id"], "code": v["code"], "title": v["title"], "cls": v["cls"], "track": v["track_id"],
              "camera": v["camera_name"], "location": v["camera_location"] or "-", "zone": v["zone"],
              "when": v["wall_time"] or f"video time {v['video_time_s']:.1f} s", "status": v["status"],
              "kph": f"{v['kph']:.0f} km/h" if v["kph"] is not None else "-", "details": details,
              "snapshot": f"/media/{v['snapshot']}", "crop": f"/media/{v['crop']}" if v["crop"] else "",
              "note": v["note"] or "", "printed": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "plate": plate_line(v), "plate_img": f"/media/{v['plate_img']}" if v.get("plate_img") else "",
              "extra": "".join(evidence_figure(e) for e in v.get("evidence") or []),
              **citation_fields(store.citation_by_violation(vid))}

    def fill(m):
        k = m.group(1)
        if k not in fields:
            return m.group(0)
        return str(fields[k]) if k in NOTICE_HTML_FIELDS else escape(str(fields[k]))

    # One pass over the template. Replacing field by field would re-read text that an earlier field had
    # inserted, so an officer's note containing "{{plate}}" would be expanded as though it were part of the page.
    return re.sub(r"\{\{(\w+)\}\}", fill, tpl)


# ---------------------------------------------------------------- static

# ---------------------------------------------------------------- media

# Never `data/` as a whole: that once served the database itself at /media/surveillance.db - every plate, every
# offender record - along with the source footage, to anyone who could reach the port. Only the two folders the
# dashboard reads from are reachable, and neither is a public mount any more: even with the database out of
# reach, evidence is pictures of vehicles and their number plates.
#   /media/<folder>/...   staff (controller or admin) with a session. The paths stored in the database are
#                         relative to `data/`, so every link the dashboard already builds keeps working.
#   /m/<token>/<path>     whoever holds a link the portal signed: one file, until it expires.
for _name in cit.MEDIA_AREAS:
    (DATA / _name).mkdir(parents=True, exist_ok=True)


def media_response(path, max_age=300):
    # private: a shared cache must not keep pictures of number plates; nosniff and no referrer: a link carries
    # its own credential, and must not be handed on to whatever site the page happens to be embedded from.
    return FileResponse(path, headers={"Cache-Control": f"private, max-age={max_age}",
                                       "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})


@app.get("/media/{area}/{path:path}")
def staff_media(area: str, path: str, user=Depends(require(CONTROLLER))):
    """Evidence and learned-zone pictures for the dashboard. Needs a controller or admin session: a driver is
    refused here too (they get their evidence through signed links), and an unknown folder is a plain 404."""
    if area not in cit.MEDIA_AREAS:
        raise HTTPException(404, "Not found")
    f = cit.resolve_under(DATA / area, path)
    if f is None:
        raise HTTPException(404, "Not found")
    return media_response(f)


@app.get("/m/{token}/{path:path}")
def shared_media(token: str, path: str):
    """One evidence file for the holder of a signed link. No session: the signature is the credential.

    The answers are deliberately few. A link that does not verify is 403 whether or not the file exists, so
    this cannot be used to probe the evidence folder; only a link that was really issued can be told it has
    expired (410), and an expired link is not served."""
    state = cit.check_token(token, path, media_key())
    if state == "invalid":
        raise HTTPException(403, "This link is not valid")
    if state == "expired":
        raise HTTPException(410, "This link has expired. Open the case again to get a new one.")
    f = cit.evidence_file(DATA, path)
    if f is None:
        raise HTTPException(404, "Not found")
    remaining = max(0, int(token.split(".", 1)[0]) - int(time.time()))
    return media_response(f, max_age=min(300, remaining))


@app.get("/")
def index(request: Request):
    """Each actor lands where they belong: staff on the dashboard, drivers on their own portal, and
    anyone else at the sign-in page."""
    user = sessions.resolve(request.cookies.get(auth_mod.SESSION_COOKIE))
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["role"] == DRIVER:
        return RedirectResponse("/portal", status_code=302)
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/login")
def login_page(request: Request):
    user = sessions.resolve(request.cookies.get(auth_mod.SESSION_COOKIE))
    if user:
        return RedirectResponse("/portal" if user["role"] == DRIVER else "/", status_code=302)
    return FileResponse(WEB / "login.html", headers={"Cache-Control": "no-store"})


@app.get("/portal")
def portal_page(request: Request):
    """A driver's own violations. Deliberately a separate page from the operator dashboard rather than a
    filtered view of it: there is then no state in which the wrong thing can be shown to the wrong person."""
    user = sessions.resolve(request.cookies.get(auth_mod.SESSION_COOKIE))
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["role"] != DRIVER:
        return RedirectResponse("/", status_code=302)
    return FileResponse(WEB / "portal.html", headers={"Cache-Control": "no-store"})


# Camera map: placement + heading, section speeds, Google keys (feature module; owns its own tables and /map page)
app.include_router(api_map.make_router(store, current_user, require, CONTROLLER, ADMIN, status_fn=manager.status))
# 3D city + optimiser: network editor, live camera-measured state, signal recommendation, simulation, routing.
# Google travel times only on request from the Google-provider map, never on the MapLibre view (Google terms).
app.include_router(api_optimize.make_router(
    store, current_user, require, CONTROLLER, ADMIN,
    snapshots_fn=lambda: api_optimize.snapshots_from_workers(manager.workers),
    status_fn=manager.status))
# Query panel: persistent segment index (built from stored analytics, never re-detects), Stage A grounding,
# Stage B descriptive answers from the index or VLM reasoning over a few retrieved segments' frames.
app.include_router(api_query.make_router(store, current_user, require, CONTROLLER, ADMIN, data_dir=DATA,
                                         backend_fn=lambda: vlm_mod.resolve_backend(vlm_settings())))
# Storage: retention (dry run, admin applies), disk monitor, and the ingest queue that uploads go through.
storage = api_storage.build(store, DATA, current_user, require, CONTROLLER, ADMIN, manager, max_parallel=1)
app.include_router(storage.router)
pay = api_payments.build(store, DATA, current_user, require, CONTROLLER, ADMIN, DRIVER)
app.include_router(pay.router)
cit.on_issued.append(pay.on_issued)              # citation notice to the registered owner's email
disputes_mod.on_decided.append(pay.on_decided)   # payment paused / cancelled / due again, and the decision email
vehicles = api_vehicles.build(store, DATA, current_user, require, CONTROLLER, ADMIN)
app.include_router(vehicles.router)

@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    return RedirectResponse("/static/favicon.svg", status_code=301)


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
