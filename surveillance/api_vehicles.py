"""HTTP surface of the passed-vehicle list, "find a vehicle" and look-alike suggestions (WP17, P5): a feature module
with its own router.

Wiring (the only edits server.py needs; see docs/VEHICLES.md):
    from . import api_vehicles
    vehicles = api_vehicles.build(store, DATA, current_user, require, CONTROLLER, ADMIN)
    app.include_router(vehicles.router)
    vehicles.purge.start()                      # on startup (expires old records daily); vehicles.purge.stop() on shutdown

Endpoints (staff = controller or admin; drivers get 403 from the server's fail-shut /api guard AND from `require`
here - do NOT list any of these in DRIVER_API). `/api/vehicles` already belongs to the cross-camera journeys, so
everything here is under /api/passed.
    GET  /vehicles                       page
    GET  /api/passed/cameras             cameras with how many passed vehicles each has
    GET  /api/passed/runs?camera_id=     the runs of one camera that have records, newest first
    GET  /api/passed                     the list. camera_id, run_id (default: the camera's newest run; "all" for every
                                         run), q (free text), plate, cls, colour, gid, since, until, status, limit, offset
    GET  /api/passed/export.csv          the same filter, streamed as CSV (audited)
    GET  /api/passed/find                across the network by gid / plate / colour / cls / q, each with where it was last seen
    GET  /api/passed/{id}                one vehicle: the record, its sighting and its journey if linked
    GET  /api/passed/{id}/thumb          its thumbnail
    GET  /api/passed/{id}/similar?k=     look-alike suggestions (read-only; never links anything)
    GET  /api/passed/lookalike/status    what the look-alike index holds
"""

from __future__ import annotations

import csv
import io
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from . import lookalike as la
from . import passed as pv

WEB = Path(__file__).resolve().parents[1] / "web"
MAX_LIMIT = 500
FIND_LIMIT = 50
EXPORT_BATCH = 1000
MAX_EXPORT_ROWS = 500_000

# A record plus what lives elsewhere: the camera, and the global id / tracklet of the same vehicle (read, never
# copied: the linker revises global ids after the fact). Correlated sub-selects, not a join, so a tracker id that
# somehow has two tracklets cannot double a row.
_TRACK = "t.camera_id=p.camera_id AND t.run_id=p.run_id AND t.track_id=p.track_id"
BASE = ("SELECT p.*, c.name AS camera_name, c.location AS camera_location,"
        f" (SELECT t.global_id FROM tracklets t WHERE {_TRACK} ORDER BY t.id DESC LIMIT 1) AS global_id,"
        f" (SELECT t.id FROM tracklets t WHERE {_TRACK} ORDER BY t.id DESC LIMIT 1) AS tracklet_id"
        " FROM passed_vehicles p LEFT JOIN cameras c ON c.id=p.camera_id")
FROM_WHERE = " FROM passed_vehicles p"

_GID = re.compile(r"^g(?:id)?[:#]?(\d+)$", re.I)
_TID = re.compile(r"^(?:#|id:|t:|track:)(\d+)$", re.I)


def to_epoch(value, name) -> Optional[float]:
    """A time as given in a query: epoch seconds, or an ISO date/time in the server's local time."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        raise HTTPException(400, f"{name} must be epoch seconds or an ISO date like 2026-10-30T09:00") from None


def like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def build_filter(store, *, camera_id=None, run_id=None, q=None, plate=None, cls=None, colour=None, gid=None,
                 since=None, until=None, status=None, newest_run=True) -> tuple:
    """(where sql, args, resolved run id). Every part is a bound parameter; nothing typed by a user reaches the SQL."""
    where, args = [], []
    run = None
    if camera_id is not None:
        where.append("p.camera_id=?")
        args.append(camera_id)
        if run_id in (None, "") and newest_run:
            r = store.q("SELECT MAX(run_id) AS r FROM passed_vehicles WHERE camera_id=?", (camera_id,))
            run = r[0]["r"] if r else None
        elif run_id not in (None, "", "all"):
            run = _int(run_id, "run_id")
    elif run_id not in (None, "", "all"):
        run = _int(run_id, "run_id")
    if run is not None:
        where.append("p.run_id=?")
        args.append(run)
    if plate:
        n = pv.normalise_plate(plate)
        if n:
            where.append("p.plate_norm LIKE ? ESCAPE '\\'")
            args.append(like(n))
    for col, val in (("cls", cls),):
        if val:
            names = [v.strip().lower() for v in str(val).split(",") if v.strip()]
            if names:
                where.append(f"LOWER(p.{col}) IN ({','.join('?' * len(names))})")
                args += names
    if colour:
        names = [v.strip().lower() for v in str(colour).split(",") if v.strip()]
        if names:
            where.append(f"p.colour IN ({','.join('?' * len(names))})")
            args += names
    if status:
        where.append("p.plate_status=?")
        args.append(status)
    if gid not in (None, ""):
        where.append(f"(p.camera_id, p.run_id, p.track_id) IN (SELECT t.camera_id, t.run_id, t.track_id FROM tracklets t"
                     " WHERE t.global_id=?)")
        args.append(_int(gid, "gid"))
    lo, hi = to_epoch(since, "since"), to_epoch(until, "until")
    if lo is not None:                       # records without a wall clock cannot be placed in time: left out
        where.append("p.last_epoch >= ?")
        args.append(lo)
    if hi is not None:
        where.append("p.first_epoch <= ?")
        args.append(hi)
    if q:
        known = {r["cls"].lower() for r in store.q("SELECT DISTINCT cls FROM passed_vehicles WHERE cls != ''")}
        for tok in str(q).split():
            low = tok.lower()
            if low in pv.COLOURS or low == pv.MIXED:
                where.append("p.colour=?")
                args.append(low)
            elif low in known:
                where.append("LOWER(p.cls)=?")
                args.append(low)
            elif _TID.match(tok):
                where.append("p.track_id=?")
                args.append(int(_TID.match(tok).group(1)))
            elif _GID.match(tok):
                where.append("(p.camera_id, p.run_id, p.track_id) IN (SELECT t.camera_id, t.run_id, t.track_id"
                             " FROM tracklets t WHERE t.global_id=?)")
                args.append(int(_GID.match(tok).group(1)))
            else:
                n = pv.normalise_plate(tok)
                where.append("(p.plate_norm LIKE ? ESCAPE '\\' OR LOWER(p.look_vlm) LIKE ? ESCAPE '\\')")
                args += [like(n) if n else like(low), like(low)]
    return (" WHERE " + " AND ".join(where)) if where else "", args, run


def _int(v, name):
    try:
        return int(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{name} must be a number") from None


ORDER = " ORDER BY COALESCE(p.first_epoch, p.created_at) DESC, p.id DESC"


def public(row: dict) -> dict:
    """What leaves the server: no file paths, no search key."""
    out = {k: v for k, v in row.items() if k not in ("thumb", "plate_norm")}
    out["thumb_url"] = f"/api/passed/{row['id']}/thumb" if row.get("thumb") else ""
    out["plate"] = row.get("plate_bn") or row.get("plate_en") or ""
    return out


CSV_COLUMNS = ("id", "camera_id", "camera", "run_id", "track_id", "global_id", "class", "colour", "first_seen",
               "last_seen", "first_video_s", "last_video_s", "plate_status", "plate_bn", "plate_en",
               "plate_confidence", "plate_width_px", "vehicle_width_px")


def _cell(v):
    """A CSV cell that a spreadsheet will not run as a formula."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return "" if v is None else v


def csv_row(r: dict) -> list:
    return [_cell(v) for v in (r["id"], r["camera_id"], r.get("camera_name") or "", r["run_id"], r["track_id"],
                               r.get("global_id"), r["cls"], r["colour"], r["first_wall"], r["last_wall"],
                               r["first_video_t"], r["last_video_t"], r["plate_status"], r["plate_bn"], r["plate_en"],
                               r["plate_conf"], r["plate_px"], r["width_px"])]


def make_router(store, current_user, require, controller: str = "controller", admin: str = "admin", *,
                data_dir=None, web_dir: Optional[Path] = None, index: Optional[la.LookalikeIndex] = None,
                now_fn: Callable = time.time) -> APIRouter:
    pv.ensure_schema(store)
    router = APIRouter()
    staff = require(controller)
    page = (web_dir or WEB) / "vehicles.html"
    root = pv.passed_dir(data_dir) if data_dir is not None else None
    index = index or la.LookalikeIndex(store)

    def row_or_404(pid: int) -> dict:
        r = store.q(BASE + " WHERE p.id=?", (pid,))
        if not r:
            raise HTTPException(404, "No such vehicle")
        return r[0]

    @router.get("/passed")
    def vehicles_page(user=Depends(current_user)):
        if user is None:
            return RedirectResponse("/login", status_code=302)
        if user["role"] not in (controller, admin):
            return RedirectResponse("/portal", status_code=302)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    @router.get("/api/passed/cameras")
    def cameras(user=Depends(staff)):
        return {"items": store.q(
            "SELECT c.id, c.name, c.location, COUNT(p.id) AS vehicles, MAX(p.run_id) AS latest_run"
            " FROM cameras c LEFT JOIN passed_vehicles p ON p.camera_id=c.id GROUP BY c.id ORDER BY c.id"),
            "retention_days": pv.retention_days()}

    @router.get("/api/passed/runs")
    def runs(camera_id: int, user=Depends(staff)):
        return {"items": store.q(
            "SELECT p.run_id, COUNT(*) AS vehicles, MIN(p.first_wall) AS first_seen, MAX(p.last_wall) AS last_seen,"
            " r.started_at, r.mode, r.status FROM passed_vehicles p LEFT JOIN runs r ON r.id=p.run_id"
            " WHERE p.camera_id=? GROUP BY p.run_id ORDER BY p.run_id DESC LIMIT 100", (camera_id,))}

    def filters(camera_id, run_id, q, plate, cls, colour, gid, since, until, status, newest_run=True):
        return build_filter(store, camera_id=camera_id, run_id=run_id, q=q, plate=plate, cls=cls, colour=colour,
                            gid=gid, since=since, until=until, status=status, newest_run=newest_run)

    @router.get("/api/passed")
    def listing(camera_id: Optional[int] = None, run_id: Optional[str] = None, q: Optional[str] = None,
                plate: Optional[str] = None, cls: Optional[str] = None, colour: Optional[str] = None,
                gid: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None,
                status: Optional[str] = None, limit: int = Query(100, ge=1, le=MAX_LIMIT),
                offset: int = Query(0, ge=0), user=Depends(staff)):
        where, args, run = filters(camera_id, run_id, q, plate, cls, colour, gid, since, until, status)
        total = store.q("SELECT COUNT(*) AS n" + FROM_WHERE + where, args)[0]["n"]
        rows = store.q(BASE + where + ORDER + " LIMIT ? OFFSET ?", (*args, limit, offset))
        # how the plates of this selection came out, ignoring a status filter so the chips stay clickable
        w2, a2, _ = filters(camera_id, run_id, q, plate, cls, colour, gid, since, until, None)
        counts = {r["plate_status"]: r["n"] for r in store.q(
            "SELECT plate_status, COUNT(*) AS n" + FROM_WHERE + w2 + " GROUP BY plate_status", a2)}
        return {"items": [public(r) for r in rows], "total": total, "limit": limit, "offset": offset,
                "run_id": run, "plate_status_counts": counts}

    @router.get("/api/passed/export.csv")
    def export(camera_id: Optional[int] = None, run_id: Optional[str] = None, q: Optional[str] = None,
               plate: Optional[str] = None, cls: Optional[str] = None, colour: Optional[str] = None,
               gid: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None,
               status: Optional[str] = None, user=Depends(staff)):
        where, args, run = filters(camera_id, run_id, q, plate, cls, colour, gid, since, until, status)
        store.audit(user, "passed.export", "passed_vehicles", "", f"camera {camera_id} run {run} q={q!r} plate={plate!r}",
                    user.get("address", ""))

        def chunks():
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(CSV_COLUMNS)
            yield "﻿" + buf.getvalue()                  # BOM: Excel opens the Bangla plates as UTF-8
            last, sent = 0, 0
            while sent < MAX_EXPORT_ROWS:
                rows = store.q(BASE + (where + " AND " if where else " WHERE ") + "p.id > ? ORDER BY p.id LIMIT ?",
                               (*args, last, EXPORT_BATCH))
                if not rows:
                    return
                buf = io.StringIO()
                w = csv.writer(buf)
                for r in rows:
                    w.writerow(csv_row(r))
                yield buf.getvalue()
                last, sent = rows[-1]["id"], sent + len(rows)

        name = f"passed-vehicles-cam{camera_id or 'all'}-run{run or 'all'}.csv"
        return StreamingResponse(chunks(), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"',
                                          "Cache-Control": "no-store"})

    @router.get("/api/passed/lookalike/status")
    def lookalike_status(user=Depends(staff)):
        return index.stats()

    @router.get("/api/passed/find")
    def find(q: Optional[str] = None, plate: Optional[str] = None, cls: Optional[str] = None,
             colour: Optional[str] = None, gid: Optional[str] = None, limit: int = Query(FIND_LIMIT, ge=1, le=200),
             user=Depends(staff)):
        """Find one vehicle anywhere in the network. Records are grouped by global vehicle id (a vehicle seen on three
        cameras is one result); each result says where it was last seen, counting the cross-camera sightings too."""
        if not any((q, plate, cls, colour, gid not in (None, ""))):
            raise HTTPException(400, "give a global id, a plate, a class, a colour or some text to look for")
        where, args, _ = filters(None, "all", q, plate, cls, colour, gid, None, None, None)
        rows = store.q(BASE + where + ORDER + " LIMIT 1000", args)
        groups: dict = {}
        for r in rows:
            groups.setdefault(("g", r["global_id"]) if r["global_id"] else ("p", r["id"]), []).append(r)
        if gid not in (None, "") and ("g", _int(gid, "gid")) not in groups:
            groups[("g", _int(gid, "gid"))] = []             # known to the linker even if no record of it survives
        items = [item for item in (_find_item(key, rs) for key, rs in groups.items()) if item]
        items.sort(key=lambda i: -(i["last_seen"]["epoch"] or 0) if i["last_seen"] else 0)
        return {"items": items[:limit], "total": len(items)}

    def _sighting(r, source):
        epoch = r.get("last_epoch") if source == "passed" else r.get("t_cross")
        return {"source": source, "camera_id": r["camera_id"], "camera_name": r.get("camera_name") or "",
                "camera_location": r.get("camera_location") or "", "run_id": r.get("run_id"), "epoch": epoch,
                "wall": r.get("last_wall") if source == "passed" else (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch)) if epoch else ""),
                "video_t": r.get("last_video_t") if source == "passed" else r.get("video_t"),
                "passed_id": r.get("id") if source == "passed" else None,
                "tracklet_id": r.get("tracklet_id") if source == "passed" else r.get("id")}

    def _journey(gid):
        return store.q(
            "SELECT t.id, t.camera_id, t.run_id, t.track_id, t.cls, t.t_cross, t.video_t, t.speed, t.direction,"
            " t.link_source, c.name AS camera_name, c.location AS camera_location FROM tracklets t"
            " LEFT JOIN cameras c ON c.id=t.camera_id WHERE t.global_id=? ORDER BY t.t_cross", (gid,))

    def _find_item(key, rs):
        gid = key[1] if key[0] == "g" else None
        journey = _journey(gid) if gid else []
        seen = [_sighting(r, "passed") for r in rs] + [_sighting(j, "tracklet") for j in journey]
        if not seen:
            return None
        last = max(seen, key=lambda s: s["epoch"] or 0)
        best = next((r for r in rs if r["plate_status"] == pv.READ), None) or \
            next((r for r in rs if r["plate_status"] == pv.PARTIAL), None) or (rs[0] if rs else None)
        newest = rs[0] if rs else None
        return {"global_id": gid, "cls": (best or {}).get("cls") or (journey[-1]["cls"] if journey else ""),
                "colour": (best or {}).get("colour", ""), "plate": (best or {}).get("plate_bn") or
                (best or {}).get("plate_en") or "", "plate_status": (best or {}).get("plate_status", ""),
                "passed_id": newest["id"] if newest else None,
                "thumb_url": f"/api/passed/{newest['id']}/thumb" if newest and newest["thumb"] else "",
                "sightings": max(len(rs), len(journey)),
                "cameras": len({s["camera_id"] for s in seen}), "last_seen": last}

    @router.get("/api/passed/{pid}/thumb")
    def thumb(pid: int, user=Depends(staff)):
        r = store.q("SELECT thumb FROM passed_vehicles WHERE id=?", (pid,))
        if not r or not r[0]["thumb"] or root is None:
            raise HTTPException(404, "No picture")
        p = root / r[0]["thumb"]
        if p.is_symlink() or not p.is_file() or root.resolve() not in p.resolve().parents:
            raise HTTPException(404, "No picture")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @router.get("/api/passed/{pid}/similar")
    def similar(pid: int, k: int = Query(la.DEFAULT_K, ge=1, le=la.MAX_K), same_class: bool = True,
                user=Depends(staff)):
        r = row_or_404(pid)
        if not r["tracklet_id"]:
            return {"items": [], "suggestions_only": True,
                    "reason": "this vehicle has no stored appearance vector (it was not tracked long enough to be "
                              "summarised, or it was parked)"}
        out = index.similar(r["tracklet_id"], k, same_class)
        for it in out["items"]:
            it["thumb_url"] = f"/api/passed/{it['passed_id']}/thumb" if it.get("passed_id") and it.get("thumb") else ""
            it.pop("thumb", None)
        out["suggestions_only"] = True
        return out

    @router.get("/api/passed/{pid}")
    def detail(pid: int, user=Depends(staff)):
        r = row_or_404(pid)
        out = public(r)
        out["journey"] = _journey(r["global_id"]) if r["global_id"] else []
        return out

    return router


def build(store, data_dir, current_user, require, controller: str = "controller", admin: str = "admin", *,
          web_dir: Optional[Path] = None, purge_interval_s: float = pv.DAY):
    """The router, the look-alike index and the daily purge (call `.purge.start()` / `.purge.stop()`)."""
    index = la.LookalikeIndex(store)
    router = make_router(store, current_user, require, controller, admin, data_dir=data_dir, web_dir=web_dir,
                         index=index)
    return SimpleNamespace(router=router, index=index,
                           purge=pv.PurgeLoop(store, data_dir, interval_s=purge_interval_s))
