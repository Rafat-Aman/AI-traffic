"""HTTP surface of the camera map (WP9, requirements R-2 and R-8): a feature module with its own router.

Wiring (the only edits server.py needs):
    from . import api_map
    api_map.ensure_schema(store)
    app.include_router(api_map.make_router(store, current_user, require, CONTROLLER, ADMIN,
                                           status_fn=manager.status))
and, optionally, `network_fn=` a callable returning the optimiser Network built from the map (nodes named
geo.node_id(camera id)) so section distances use road length instead of straight line x circuity.

Endpoints (everything under /api/ is already behind the server's fail-shut guard; drivers are refused there
and again here, because staff = controller or admin and `require` does not admit a driver)
    GET    /map                               page (redirects to /login, drivers to /portal)
    GET    /api/map/cameras                   staff   cameras + placement + running state
    PUT    /api/map/cameras/{cid}             admin   place / move / aim a camera (audited)
    DELETE /api/map/cameras/{cid}             admin   take it off the map (audited)
    GET    /api/map/section-speeds            staff   ?from=&to=&since=&until= per camera pair
    GET    /api/map/vehicle/{gid}/segments    staff   one vehicle's camera-to-camera speeds
    GET    /api/topology                      staff   gates per camera, links with windows, two-link routes (WP19)
    PUT    /api/topology/cameras/{cid}/gates  admin   replace a camera's gates (audited); drops links to removed gates
    GET    /api/topology/suggest?from=&to=    staff   the window the map suggests for a link
    POST   /api/topology/links                admin   add a link; no min_s / max_s = the map's suggestion (audited)
    PUT    /api/topology/links/{id}           admin   edit it (audited)
    DELETE /api/topology/links/{id}           admin   remove it (audited)
    GET    /api/topology/vehicle?q=DV-000123  staff   a display id (or a merged-away one) -> the vehicle it names
    GET    /api/map/config                    staff   map provider + the BROWSER key only
    GET    /api/settings/maps                 admin   which keys are set (never their value)
    PUT    /api/settings/maps                 admin   set / clear the Google keys (audited, key not logged)

Keys. Environment variables GOOGLE_MAPS_BROWSER_KEY / GOOGLE_MAPS_SERVER_KEY win; otherwise the value stored in
the `map_settings` table of the (gitignored) database is used. The browser key must reach the browser to load
the Maps JavaScript API, so staff get it from /api/map/config (restrict it by HTTP referrer in Google Cloud).
The SERVER key is write-only over HTTP: no response, error message or audit row ever contains it; other server
code reads it with `server_key(store)`.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse

from . import geo as geo_mod
from . import section_speed as ss
from . import topology as topo_mod

WEB = Path(__file__).resolve().parents[1] / "web"
ENV_KEYS = {"browser_key": "GOOGLE_MAPS_BROWSER_KEY", "server_key": "GOOGLE_MAPS_SERVER_KEY"}
KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{20,200}$")
SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS map_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT DEFAULT ''
);
"""
DHAKA_CENTRE = (23.7806, 90.4075)
OSM_TILES = {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "max_zoom": 19,
             "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}


def ensure_schema(store) -> None:
    """Create this module's tables (camera_geo, map_settings). Idempotent; never touches existing tables."""
    geo_mod.ensure_schema(store)
    with store.lock:
        store.db.executescript(SETTINGS_SCHEMA)
        store.db.commit()
    topo_mod.ensure_schema(store)


# ---------------------------------------------------------------- keys

def _stored(store, which: str) -> Optional[str]:
    rows = store.q("SELECT value FROM map_settings WHERE key=?", (which,))
    return rows[0]["value"] if rows else None


def get_key(store, which: str, env=None) -> tuple:
    """(key or None, source) where source is "env", "stored" or "none". Environment wins."""
    env = os.environ if env is None else env
    v = (env.get(ENV_KEYS[which]) or "").strip()
    if v:
        return v, "env"
    v = _stored(store, which)
    return (v, "stored") if v else (None, "none")


def server_key(store, env=None) -> Optional[str]:
    """For server-side callers only (e.g. the Routes API provider). Never put this in a response."""
    return get_key(store, "server_key", env)[0]


def browser_key(store, env=None) -> Optional[str]:
    return get_key(store, "browser_key", env)[0]


def _hint(key: Optional[str]) -> Optional[str]:
    return None if not key else (f"set, ends …{key[-4:]}" if len(key) >= 12 else "set")


def key_status(store, env=None) -> dict:
    out = {}
    for which in ENV_KEYS:
        key, source = get_key(store, which, env)
        out[which] = {"set": key is not None, "source": source, "hint": _hint(key)}
    return out


# ---------------------------------------------------------------- helpers

def _when(value: Optional[str], what: str) -> Optional[float]:
    """Epoch seconds from a number or an ISO date-time (local time if it has no zone)."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        raise HTTPException(400, f"{what} must be epoch seconds or an ISO date-time") from None


def make_router(store, current_user, require, controller: str = "controller", admin: str = "admin", *,
                status_fn: Optional[Callable] = None, network_fn: Optional[Callable] = None,
                web_dir: Optional[Path] = None, env=None) -> APIRouter:
    """The map's router. `require(role)` is the server's dependency factory; `status_fn(camera id)` returns the
    worker status dict ({"state": ...}); `network_fn()` returns an optimiser Network or None."""
    ensure_schema(store)
    router = APIRouter()
    staff, admin_only = require(controller), require(admin)
    page = (web_dir or WEB) / "map.html"

    def state_of(cid):
        if status_fn is None:
            return "unknown"
        try:
            return (status_fn(cid) or {}).get("state", "unknown")
        except Exception:
            return "unknown"

    def road_fn():
        if network_fn is None:
            return None
        try:
            net = network_fn()
        except Exception:
            return None
        return None if net is None else ss.network_road_fn(net)

    def names():
        return {c["id"]: c["name"] for c in store.cameras()}

    # ---- the page
    @router.get("/map")
    def map_page(user=Depends(current_user)):
        if user is None:
            return RedirectResponse("/login", status_code=302)
        if user["role"] not in (controller, admin):
            return RedirectResponse("/portal", status_code=302)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    # ---- cameras
    @router.get("/api/map/cameras")
    def map_cameras(user=Depends(staff)):
        geos = geo_mod.all_geo(store)
        items = []
        for c in store.cameras():
            g = geos.get(c["id"])
            items.append({"id": c["id"], "name": c["name"], "location": c.get("location", ""), "kind": c["kind"],
                          "state": state_of(c["id"]), "placed": g is not None,
                          "geo": geo_mod.view(g) if g else None})
        return {"cameras": items, "can_edit": user["role"] == admin}

    @router.put("/api/map/cameras/{cid}")
    async def put_camera(cid: int, request: Request, user=Depends(admin_only)):
        if not store.camera(cid):
            raise HTTPException(404, "Camera not found")
        try:
            data = await request.json()
            row, warnings = geo_mod.save(store, cid, data, user.get("username") or user.get("name") or "")
        except geo_mod.GeoError as e:
            raise HTTPException(400, str(e)) from None
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None
        store.audit(user, "camera.geo", "camera", cid,
                    f"lat={row['lat']} lon={row['lon']} heading={row['heading_deg']} fov={row['fov_deg']} "
                    f"range={row['range_m']} clock_offset={row['clock_offset_s']}", user.get("address", ""))
        return {"camera_id": cid, "geo": geo_mod.view(row), "warnings": warnings}

    @router.delete("/api/map/cameras/{cid}")
    def delete_camera(cid: int, user=Depends(admin_only)):
        if not geo_mod.remove(store, cid):
            raise HTTPException(404, "That camera is not on the map")
        store.audit(user, "camera.geo.remove", "camera", cid, "taken off the map", user.get("address", ""))
        return {"ok": True}

    # ---- speeds
    @router.get("/api/map/section-speeds")
    def section_speeds(cam_from: Optional[int] = Query(None, alias="from"), cam_to: Optional[int] = Query(None, alias="to"),
                       since: Optional[str] = None, until: Optional[str] = None, user=Depends(staff)):
        t0, t1 = _when(since, "since"), _when(until, "until")
        geos = geo_mod.all_geo(store)
        rows = ss.fetch_rows(store, t0, t1)
        out = ss.compute(rows, geos, road_fn=road_fn(), since=t0, until=t1, cam_from=cam_from, cam_to=cam_to)
        nm = names()
        for p in out["pairs"]:
            p["from_name"], p["to_name"] = nm.get(p["from"], ""), nm.get(p["to"], "")
        out["generated_at"] = time.time()
        return out

    @router.get("/api/map/vehicle/{global_id}/segments")
    def vehicle_segments(global_id: int, user=Depends(staff)):
        rows = store.tracklets(global_id=global_id)
        nm = names()
        segs = ss.segments_of_vehicle(rows, geo_mod.all_geo(store), road_fn())
        for s in segs:
            s["from_name"], s["to_name"] = nm.get(s["from"], ""), nm.get(s["to"], "")
        return {"global_id": global_id, "sightings": len(rows), "segments": segs, "bounds": ss.BOUNDS,
                "clock_note": ss.CLOCK_NOTE}

    # ---- camera-network topology (WP19): gates, links, windows. Staff read, admin writes, drivers nothing.
    def link_view(l):
        return {"id": l["id"], "from_camera": l["from_camera"], "from_gate": l["from_gate"],
                "from_gate_name": l["from_gate_name"], "to_camera": l["to_camera"], "to_gate": l["to_gate"],
                "to_gate_name": l["to_gate_name"], "min_s": l["min_s"], "max_s": l["max_s"],
                "distance_m": l["distance_m"], "window_source": l["window_source"]}

    @router.get("/api/topology")
    def topology_view(user=Depends(staff)):
        gates = {}
        for c in store.cameras():
            gates[c["id"]] = topo_mod.gates_of(store, c["id"])
        routes = store.q("SELECT * FROM camera_routes WHERE hops=2 ORDER BY from_camera, to_camera")
        return {"gates": gates, "links": [link_view(l) for l in topo_mod.list_links(store)],
                "two_link_routes": routes, "dwell_s": topo_mod.SKIP_DWELL_S, "can_edit": user["role"] == admin,
                "window_basis": {"fast_kph": geo_mod.WINDOW_FAST_KPH, "slow_kph": geo_mod.WINDOW_SLOW_KPH,
                                 "circuity": geo_mod.WINDOW_CIRCUITY, "slack_s": geo_mod.WINDOW_SLACK_S,
                                 "max_s": geo_mod.WINDOW_MAX_S}}

    @router.put("/api/topology/cameras/{cid}/gates")
    async def put_gates(cid: int, request: Request, user=Depends(admin_only)):
        if not store.camera(cid):
            raise HTTPException(404, "Camera not found")
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "send a JSON object")
        try:
            out = topo_mod.save_gates(store, cid, data.get("gates"), user.get("username") or user.get("name") or "")
        except topo_mod.TopologyError as e:
            raise HTTPException(400, str(e)) from None
        store.audit(user, "topology.gates", "camera", cid,
                    f"{len(out['gates'])} gate(s): " + ", ".join(f"{g['name']} ({g['kind']})" for g in out["gates"])
                    + (f"; {out['removed_links']} link(s) removed with their gates" if out["removed_links"] else ""),
                    user.get("address", ""))
        return out

    @router.get("/api/topology/suggest")
    def topology_suggest(cam_from: int = Query(..., alias="from"), cam_to: int = Query(..., alias="to"),
                         user=Depends(staff)):
        out = topo_mod.suggest_window(store, cam_from, cam_to)
        if out is None:
            raise HTTPException(404, "Place both cameras on the map first")
        return out

    def link_detail(l):
        return (f"camera {l['from_camera']} '{l['from_gate_name']}' -> camera {l['to_camera']} '{l['to_gate_name']}', "
                f"{l['min_s']:g}-{l['max_s']:g} s ({l['window_source']})")

    async def link_body(request):
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "send a JSON object")
        return data

    @router.post("/api/topology/links")
    async def post_link(request: Request, user=Depends(admin_only)):
        data = await link_body(request)
        try:
            row = topo_mod.save_link(store, data, user.get("username") or user.get("name") or "")
        except topo_mod.TopologyError as e:
            raise HTTPException(400, str(e)) from None
        store.audit(user, "topology.link.add", "link", row["id"], link_detail(row), user.get("address", ""))
        return link_view(row)

    @router.put("/api/topology/links/{lid}")
    async def put_link(lid: int, request: Request, user=Depends(admin_only)):
        data = await link_body(request)
        try:
            row = topo_mod.save_link(store, data, user.get("username") or user.get("name") or "", link_id=lid)
        except topo_mod.TopologyError as e:
            raise HTTPException(404 if str(e) == "no such link" else 400, str(e)) from None
        store.audit(user, "topology.link.edit", "link", lid, link_detail(row), user.get("address", ""))
        return link_view(row)

    @router.delete("/api/topology/links/{lid}")
    def del_link(lid: int, user=Depends(admin_only)):
        if not topo_mod.delete_link(store, lid):
            raise HTTPException(404, "No such link")
        store.audit(user, "topology.link.remove", "link", lid, "link removed", user.get("address", ""))
        return {"ok": True}

    @router.get("/api/topology/vehicle")
    def topology_vehicle(q: str = Query(..., min_length=1, max_length=40), user=Depends(staff)):
        gid = topo_mod.parse_display_id(q)
        if gid is None:
            raise HTTPException(400, "That is not a vehicle id (DV-000123, dv123 or 123)")
        now_gid = topo_mod.resolve(store, gid)
        return {"asked": topo_mod.display_id(gid), "global_id": now_gid, "display_id": topo_mod.display_id(now_gid),
                "merged": now_gid != gid, "merged_from": [topo_mod.display_id(g) for g in topo_mod.merged_from(store, now_gid)],
                "sightings": len(store.tracklets(global_id=now_gid))}

    # ---- provider config
    @router.get("/api/map/config")
    def map_config(user=Depends(staff)):
        key = browser_key(store, env)
        google = key is not None
        return {"provider": "google" if google else "osm", "google_browser_key": key,
                "traffic_layer": google,
                "osm": OSM_TILES, "centre": list(DHAKA_CENTRE), "zoom": 13, "can_edit": user["role"] == admin,
                "bounds": ss.BOUNDS, "clock_note": ss.CLOCK_NOTE}

    # ---- keys
    @router.get("/api/settings/maps")
    def get_settings(user=Depends(admin_only)):
        return key_status(store, env)

    @router.put("/api/settings/maps")
    async def put_settings(request: Request, user=Depends(admin_only)):
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "send a JSON object")
        changes = []
        for which in ENV_KEYS:
            if which not in data or data[which] is None:
                continue
            v = data[which]
            if not isinstance(v, str):
                raise HTTPException(400, f"{which} must be text")
            v = v.strip()
            if v and not KEY_RE.match(v):       # the message names the field, never the value
                raise HTTPException(400, f"{which} does not look like an API key "
                                         "(20-200 letters, digits, - or _)")
            changes.append((which, v))
        for which, v in changes:
            if v:
                store.x("INSERT INTO map_settings (key, value, updated_at, updated_by) VALUES (?,?,?,?)"
                        " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at,"
                        " updated_by=excluded.updated_by",
                        (which, v, time.time(), user.get("username") or user.get("name") or ""))
            else:
                store.x("DELETE FROM map_settings WHERE key=?", (which,))
        if changes:
            store.audit(user, "settings.maps", "settings", "maps",
                        "; ".join(f"{w.replace('_', ' ')} {'set' if v else 'cleared'}" for w, v in changes),
                        user.get("address", ""))
        return key_status(store, env)

    return router
