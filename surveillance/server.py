"""Web API + dashboard for the traffic surveillance system."""

import asyncio
import csv
from html import escape
import io
import json
import shutil
import socket
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from violations.base import MOTORISED, NON_MOTORISED
from violations.engine import RULES, ViolationEngine
from violations.zones import (AREA_TYPES, DEFAULT_SPEED_LIMITS, DIRECTIONS, ROAD_CATEGORIES, RoadZone,
                              camera_config_from_dict)

from .anpr import find_models
from .camera_motion import save_reference
from .detector import MODELS_DIR, find_helmet_model
from .pipeline import reliable_limits, zones_roi
from .sources import detect_start_time, grab_frame, is_live_source, probe_mac_cameras
from .store import DEFAULT_SETTINGS, Store
from .worker import WorkerManager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
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


manager = WorkerManager(store, DATA, notify)


@asynccontextmanager
async def lifespan(_app):
    loop_ref["loop"] = asyncio.get_running_loop()
    (DATA / "videos").mkdir(parents=True, exist_ok=True)
    yield
    await asyncio.to_thread(manager.stop_all)


app = FastAPI(title="Dhaka Traffic Surveillance", lifespan=lifespan)


def cam_or_404(cid):
    cam = store.camera(cid)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


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

@app.get("/api/meta")
def meta(request: Request):
    return {
        "classes": sorted(MOTORISED | NON_MOTORISED), "motorised": sorted(MOTORISED),
        "road_categories": ROAD_CATEGORIES, "directions": DIRECTIONS, "area_types": AREA_TYPES,
        "speed_limits": DEFAULT_SPEED_LIMITS, "default_settings": DEFAULT_SETTINGS,
        "rules": {key: {"code": cls.code, "title": cls.title, "defaults": cls.DEFAULTS} for key, cls in RULES.items()},
        "lan_url": lan_url(request.url.port or 80),
        "features": {"anpr": all(find_models(MODELS_DIR)), "helmet": find_helmet_model() is not None},
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
def add_camera(body: CameraIn):
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
                  zone_mode: str = Form("manual"), file: UploadFile = File(...)):
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
        manager.start_autozone(cid, apply=True, start_after=True)
    return camera_view(store.camera(cid))


@app.patch("/api/cameras/{cid}")
def patch_camera(cid: int, body: CameraPatch):
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
def delete_camera(cid: int):
    cam_or_404(cid)
    manager.stop_autozone(cid)
    manager.stop(cid, wait=True)
    store.delete_camera(cid)
    shutil.rmtree(DATA / "evidence" / f"cam{cid}", ignore_errors=True)
    return {"ok": True}


@app.post("/api/cameras/{cid}/start")
def start_camera(cid: int, start_sec: float = 0.0):
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
def start_auto_zones(cid: int, body: AutoZoneIn):
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
def cancel_auto_zones(cid: int):
    manager.stop_autozone(cid)
    return {"ok": True}


@app.post("/api/cameras/{cid}/stop")
def stop_camera(cid: int):
    manager.stop(cid)
    return {"ok": True}


@app.delete("/api/cameras/{cid}/violations")
def clear_camera_violations(cid: int):
    if manager.running(cid):
        raise HTTPException(400, "Stop the camera first")
    store.clear_violations(cid)
    shutil.rmtree(DATA / "evidence" / f"cam{cid}", ignore_errors=True)
    return {"ok": True}


@app.get("/api/status")
def all_status():
    return {c["id"]: manager.status(c["id"]) for c in store.cameras()}


@app.get("/api/devices")
def devices():
    """Cameras this Mac can open (built-in, iPhone Continuity Camera, USB)."""
    return probe_mac_cameras()


@app.post("/api/test-source")
def test_source(body: CameraIn):
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
def save_zones(cid: int, body: ZonesIn):
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
def check_zones(cid: int, body: ZonesIn):
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
def kerb_strip(body: KerbIn):
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
def topdown(cid: int, body: TopDownIn):
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
    total = store.count(camera_id=camera_id, code=code, status=status, run_id=run_id, plate=plate)
    return {"items": rows, "total": total}


@app.get("/api/violations/{vid}")
def get_violation(vid: int):
    v = store.violation(vid)
    if not v:
        raise HTTPException(404, "Violation not found")
    return v


@app.patch("/api/violations/{vid}")
def review_violation(vid: int, body: ReviewIn):
    if body.status not in ("pending", "confirmed", "rejected"):
        raise HTTPException(400, "status must be pending, confirmed or rejected")
    if not store.violation(vid):
        raise HTTPException(404, "Violation not found")
    store.review(vid, body.status, body.note)
    return store.violation(vid)


@app.get("/api/export.csv")
def export_csv(camera_id: Optional[int] = None, status: Optional[str] = None):
    rows = store.violations(camera_id, status=status, limit=1_000_000, order="ASC")
    buf = io.StringIO()
    cols = ["id", "camera_name", "camera_location", "code", "title", "cls", "class_conf", "wall_time",
            "video_time_s", "zone", "kph", "plate_status", "plate_text", "plate_text_en", "plate_conf", "details",
            "status", "note", "snapshot", "crop", "clip", "plate_img", "evidence", "created_at"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({**r, "details": json.dumps(r["details"]),
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


@app.get("/notice/{vid}", response_class=HTMLResponse)
def notice(vid: int):
    """Printable violation notice for a confirmed case (browser: Print -> Save as PDF)."""
    v = store.violation(vid)
    if not v:
        raise HTTPException(404, "Violation not found")
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
              "extra": "".join(evidence_figure(e) for e in v.get("evidence") or [])}
    for k, val in fields.items():
        tpl = tpl.replace("{{" + k + "}}", str(val) if k in ("details", "extra") else escape(str(val)))
    return tpl


# ---------------------------------------------------------------- static

app.mount("/media", StaticFiles(directory=str(DATA), check_dir=False), name="media")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
