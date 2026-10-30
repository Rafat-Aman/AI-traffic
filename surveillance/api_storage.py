"""HTTP surface of storage management and the ingest queue (WP13, R-5): a feature module with its own router.

Wiring (the only edits server.py needs; see docs/STORAGE.md for the ingest lambdas):
    from . import api_storage
    storage = api_storage.build(store, DATA, current_user, require, CONTROLLER, ADMIN, manager)
    app.include_router(storage.router)
    storage.queue.start()                       # on startup; storage.queue.stop() on shutdown

Endpoints (staff = controller or admin; the server's /api guard already refuses drivers and anonymous callers)
    GET   /storage                      page
    GET   /api/storage                  staff   usage by category, disk, growth + days to full, policy, plans
    GET   /api/storage/policy           admin   the retention policy and its bounds
    PUT   /api/storage/policy           admin   change it (audited)
    POST  /api/storage/plan             admin   DRY RUN: what would be deleted, and why (nothing is deleted)
    GET   /api/storage/plan/{id}        admin   a saved plan (paged items)
    POST  /api/storage/apply            admin   {"plan_id": n}: delete exactly that plan, re-checked, audited
    GET   /api/ingest                   staff   queue + disk-monitor state
    POST  /api/ingest/{id}/retry        admin
    POST  /api/ingest/{id}/cancel       admin
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse

from . import ingest as ingest_mod
from . import storage_mgr as sm

WEB = Path(__file__).resolve().parents[1] / "web"
PAGE_SIZE = 500


def build(store, data_dir, current_user, require, controller="controller", admin="admin", manager=None, *,
          max_parallel: int = 1, busy_cameras_fn: Optional[Callable] = None, web_dir: Optional[Path] = None,
          queue: Optional[ingest_mod.IngestQueue] = None, **queue_kw):
    """Make the StorageManager, the IngestQueue (if `manager`, the WorkerManager, is given) and the router.
    Returns a namespace with .router, .storage and .queue."""
    ingest_mod.ensure_schema(store)
    storage = sm.StorageManager(store, data_dir, busy_cameras_fn=busy_cameras_fn or (
        (lambda: {j["camera_id"] for j in store.q("SELECT camera_id FROM ingest_jobs WHERE state IN ('queued','running')")})))
    if queue is None and manager is not None:
        queue = ingest_mod.IngestQueue(
            store, start_fn=lambda cid: _start(store, manager, cid),
            is_running_fn=lambda cid: manager.running(cid) or manager.learning(cid),
            status_fn=manager.status, stop_fn=manager.stop, paused_fn=storage.ingest_paused,
            max_parallel=max_parallel, **queue_kw)
    router = make_router(store, storage, queue, current_user, require, controller, admin, web_dir=web_dir)
    return SimpleNamespace(router=router, storage=storage, queue=queue)


def _start(store, manager, cid):
    """Start one camera's analysis the way the upload endpoints used to."""
    if manager.learning(cid):
        raise ingest_mod.NotReady("still learning its zones")
    cam = store.camera(cid)
    if cam is None:
        raise KeyError(f"camera {cid} no longer exists")
    if not store.zones(cid) and cam["settings"].get("zone_mode") == "auto":
        manager.start_autozone(cid, apply=True, start_after=True)     # starts the analysis itself when done
    else:
        manager.start(cid)


def make_router(store, storage: sm.StorageManager, queue: Optional[ingest_mod.IngestQueue], current_user, require,
                controller="controller", admin="admin", *, web_dir: Optional[Path] = None) -> APIRouter:
    sm.ensure_schema(store)
    ingest_mod.ensure_schema(store)
    router = APIRouter()
    staff, admin_only = require(controller), require(admin)
    page = (web_dir or WEB) / "storage.html"

    def who(user):
        return user.get("address", "")

    @router.get("/storage")
    def storage_page(user=Depends(current_user)):
        if user is None:
            return RedirectResponse("/login", status_code=302)
        if user["role"] not in (controller, admin):
            return RedirectResponse("/portal", status_code=302)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    @router.get("/api/storage")
    def get_storage(user=Depends(staff)):
        out = storage.summary()
        out["plans"] = storage.plans()
        out["can_edit"] = user["role"] == admin
        out["bounds"] = sm.BOUNDS
        out["ingest"] = queue.stats() if queue else None
        return out

    @router.get("/api/storage/policy")
    def get_policy(user=Depends(admin_only)):
        return {"policy": storage.policy(), "defaults": sm.DEFAULTS, "bounds": sm.BOUNDS}

    @router.put("/api/storage/policy")
    async def put_policy(request: Request, user=Depends(admin_only)):
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "send a JSON object")
        try:
            return {"policy": storage.set_policy(data, user)}
        except sm.PolicyError as e:
            raise HTTPException(400, str(e)) from None

    def paged(plan, offset=0):
        items = plan["items"]
        out = dict(plan)
        out["items"] = items[offset:offset + PAGE_SIZE]
        out["items_total"] = len(items)
        out["offset"] = offset
        return out

    @router.post("/api/storage/plan")
    def make_plan(user=Depends(admin_only)):
        plan = storage.plan(user)
        store.audit(user, "storage.plan", "plan", plan["id"],
                    f"dry run: {plan['totals']['files']} files, {plan['totals']['bytes']} bytes", who(user))
        return paged(plan)

    @router.get("/api/storage/plan/{pid}")
    def read_plan(pid: int, offset: int = Query(0, ge=0), user=Depends(admin_only)):
        plan = storage.get_plan(pid)
        if plan is None:
            raise HTTPException(404, "No such plan")
        return paged(plan, offset)

    @router.post("/api/storage/apply")
    async def apply_plan(request: Request, user=Depends(admin_only)):
        try:
            data = await request.json()
            pid = int(data["plan_id"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, 'send {"plan_id": <number>}') from None
        try:
            return storage.apply(pid, user)
        except KeyError:
            raise HTTPException(404, "No such plan") from None
        except sm.PolicyError as e:
            raise HTTPException(409, str(e)) from None

    # ---- ingest queue
    def need_queue():
        if queue is None:
            raise HTTPException(503, "The ingest queue is not running on this server")
        return queue

    @router.get("/api/ingest")
    def get_ingest(state: Optional[str] = None, user=Depends(staff)):
        q = need_queue()
        names = {c["id"]: c["name"] for c in store.cameras()}
        jobs = q.jobs(state)
        for j in jobs:
            j["camera_name"] = names.get(j["camera_id"], "(deleted)")
        return {"stats": q.stats(), "jobs": jobs, "disk": storage.disk(), "can_edit": user["role"] == admin}

    @router.post("/api/ingest/{jid}/retry")
    def retry_job(jid: int, user=Depends(admin_only)):
        j = need_queue().retry(jid)
        if j is None:
            raise HTTPException(409, "Only a failed or cancelled job whose camera has no active job can be retried")
        store.audit(user, "ingest.retry", "ingest", jid, f"camera {j['camera_id']}", who(user))
        return j

    @router.post("/api/ingest/{jid}/cancel")
    def cancel_job(jid: int, user=Depends(admin_only)):
        j = need_queue().cancel(jid)
        if j is None:
            raise HTTPException(409, "Only a queued or running job can be cancelled")
        store.audit(user, "ingest.cancel", "ingest", jid, f"camera {j['camera_id']}", who(user))
        return j

    return router
