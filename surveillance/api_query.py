"""HTTP surface of the natural-language query panel (WP3): a feature module with its own router.

Wiring (the only edits server.py needs; see docs/QUERY.md):
    from . import api_query
    app.include_router(api_query.make_router(store, current_user, require, CONTROLLER, ADMIN, data_dir=DATA,
                                             backend_fn=lambda: vlm_mod.resolve_backend(vlm_settings())))
`backend_fn` returns a vlm.py backend (anything with ask(system, user_text, images, max_tokens, schema)) or None;
it is asked per request, so a reasoner that was started later is picked up. Tests inject `reason_fn` / `llm_fn`
directly instead. With neither, descriptive questions work fully and causal ones answer "insufficient evidence:
the visual reasoner is not available" together with the retrieved segments.

Endpoints (staff = controller or admin; a driver is refused here as well as by the server's /api/ guard)
    GET  /query                                  page
    POST /api/query                              {question, parsed?, limit?, use_llm?} -> parse, retrieve, answer
    GET  /api/query/segments/{id}                one segment: counts, vehicle sentences, links, frame availability
    GET  /api/query/segments/{id}/frame?i=&n=    one JPEG frame pulled from the stored video now
    GET  /api/query/segments/{id}/thumb          a small mid-segment JPEG (cached)
    POST /api/query/index/rebuild                admin; background, one job at a time, time-boxed
    GET  /api/query/index/status                 index size, window length, FTS, running job
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

from . import query as q_mod
from . import segment_index as si

WEB = Path(__file__).resolve().parents[1] / "web"
MAX_REBUILD_SECONDS = 600
THUMB_CACHE = 128


def ensure_schema(store) -> None:
    si.ensure_schema(store)


def make_router(store, current_user, require, controller: str = "controller", admin: str = "admin", *,
                data_dir=None, reason_fn: Optional[Callable] = None, llm_fn: Optional[Callable] = None,
                backend_fn: Optional[Callable] = None, web_dir: Optional[Path] = None,
                now_fn: Optional[Callable] = None) -> APIRouter:
    ensure_schema(store)
    router = APIRouter()
    staff, admin_only = require(controller), require(admin)
    page = (web_dir or WEB) / "query.html"
    job = {"state": "idle", "started": None, "finished": None, "result": None, "error": None}
    job_lock = threading.Lock()
    thumbs: dict = {}

    def reasoner():
        if reason_fn is not None:
            return reason_fn
        b = backend_fn() if backend_fn else None
        return q_mod.make_reason_fn(b) if b is not None else None

    def parser_llm(use):
        if not use:
            return None
        if llm_fn is not None:
            return llm_fn
        b = backend_fn() if backend_fn else None
        return q_mod.make_llm_fn(b) if b is not None else None

    def cleaned(result: dict) -> dict:
        """Nothing binary or server-side in a response: frames are fetched through /frame, never inlined, and
        a camera's source (which may hold RTSP credentials) is never included."""
        return result

    @router.get("/query")
    def query_page(user=Depends(current_user)):
        if user is None:
            return RedirectResponse("/login", status_code=302)
        if user["role"] not in (controller, admin):
            return RedirectResponse("/portal", status_code=302)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    @router.post("/api/query")
    async def ask(request: Request, user=Depends(staff)):
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "send a JSON object")
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(400, "question must be non-empty text")
        if len(question) > q_mod.MAX_QUESTION_CHARS:
            raise HTTPException(400, f"question is longer than {q_mod.MAX_QUESTION_CHARS} characters")
        parsed = body.get("parsed")
        if parsed is not None and not isinstance(parsed, dict):
            raise HTTPException(400, "parsed must be an object")
        try:
            limit = int(body.get("limit") or q_mod.DEFAULT_LIMIT)
        except (TypeError, ValueError):
            raise HTTPException(400, "limit must be a number") from None
        intent = body.get("intent")
        if intent not in (None, "", "auto", "descriptive", "causal"):
            raise HTTPException(400, "intent must be auto, descriptive or causal")
        result = q_mod.answer_question(
            store, question.strip(), now=now_fn() if now_fn else None,
            llm_fn=parser_llm(bool(body.get("use_llm", False))), reason_fn=reasoner(), data_dir=data_dir,
            parsed=parsed, limit=limit, intent=intent if intent in ("descriptive", "causal") else None)
        store.audit(user, "query.ask", "query", "", question.strip()[:200], user.get("address", ""))
        return cleaned(result)

    def window_or_404(sid: int) -> dict:
        w = si.window(store, sid)
        if w is None:
            raise HTTPException(404, "Segment not found")
        return w

    @router.get("/api/query/segments/{sid}")
    def segment(sid: int, user=Depends(staff)):
        w = window_or_404(sid)
        cam = store.camera(w["camera_id"]) or {}
        vehicles = si.vehicles_in(store, sid)
        vp = q_mod._video_path(cam, data_dir)
        has_clip = any(v["clip"] for v in vehicles)
        return {"segment": {k: w[k] for k in ("id", "camera_id", "run_id", "w_start", "w_end", "wall", "video_t0", "video_t1",
                                              "n_pass", "n_in", "n_out", "n_unknown", "counts", "mean_kph", "min_kph",
                                              "n_stopped", "events", "impact_max", "state", "zones", "summary")},
                "camera": {"id": cam.get("id"), "name": cam.get("name"), "location": cam.get("location", "")},
                "start": si.day_clock(w["w_start"]), "end": si.clock(w["w_end"]),
                "vehicles": [{**{k: v[k] for k in ("id", "kind", "text", "cls", "colour", "direction", "kph", "t", "stopped",
                                                    "dwell_s", "plate", "global_id", "violation_id")},
                              "codes": v["codes"].split(),
                              "links": {"review": f"/#/review/{v['violation_id']}" if v["violation_id"] else None,
                                        "vehicle": f"/#/vehicles/{v['global_id']}" if v["global_id"] else None}}
                             for v in vehicles],
                "frames": {"video": vp is not None and w["video_t0"] is not None, "evidence_clip": has_clip,
                           "note": "" if vp is not None and w["video_t0"] is not None else
                           ("the stored video is no longer available; evidence clips/snapshots are used where they exist"
                            if w["video_t0"] is not None else "this segment cannot be located in a video")}}

    def jpeg_response(data: bytes):
        return Response(data, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer"})

    @router.get("/api/query/segments/{sid}/frame")
    def frame(sid: int, i: int = Query(0, ge=0, le=7), n: int = Query(3, ge=1, le=8), user=Depends(staff)):
        window_or_404(sid)
        got = q_mod.pull_frames(store, sid, n, data_dir)
        if not got["frames"]:
            raise HTTPException(404, got["note"] or "No frame available")
        return jpeg_response(got["frames"][min(i, len(got["frames"]) - 1)]["jpeg"])

    @router.get("/api/query/segments/{sid}/thumb")
    def thumb(sid: int, user=Depends(staff)):
        w = window_or_404(sid)
        key = (sid, w["built_at"])
        if key not in thumbs:
            got = q_mod.pull_frames(store, sid, 1, data_dir, max_side=240)
            if not got["frames"]:
                raise HTTPException(404, got["note"] or "No frame available")
            if len(thumbs) >= THUMB_CACHE:
                thumbs.pop(next(iter(thumbs)))
            thumbs[key] = got["frames"][0]["jpeg"]
        return jpeg_response(thumbs[key])

    def run_rebuild(camera_id, since):
        t0 = time.time()
        try:
            total = {"windows": 0, "vehicles": 0, "cameras": 0}
            for c in store.cameras():
                if camera_id is not None and c["id"] != camera_id:
                    continue
                if time.time() - t0 > MAX_REBUILD_SECONDS:
                    total["stopped_early"] = True
                    break
                r = si.build_for_camera(store, c["id"], since)
                total["windows"] += r["windows"]
                total["vehicles"] += r["vehicles"]
                total["cameras"] += 1
            job.update(state="done", result=total, error=None)
        except Exception as e:                     # noqa: BLE001
            job.update(state="failed", error=f"{type(e).__name__}: {e}")
        finally:
            job["finished"] = time.time()

    @router.post("/api/query/index/rebuild")
    async def rebuild(request: Request, user=Depends(admin_only)):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(400, "send a JSON object")
        cid, since = body.get("camera_id"), body.get("since")
        if cid is not None and not (isinstance(cid, int) and not isinstance(cid, bool) and store.camera(cid)):
            raise HTTPException(404, "Camera not found")
        if since is not None:
            since = q_mod._epoch_from(since, q_mod.si.now_local())
            if since is None:
                raise HTTPException(400, "since must be epoch seconds or an ISO date-time within the last 400 days")
        with job_lock:
            if job["state"] == "running":
                raise HTTPException(409, "A rebuild is already running")
            job.update(state="running", started=time.time(), finished=None, result=None, error=None)
        threading.Thread(target=run_rebuild, args=(cid, since), daemon=True, name="segment-index-rebuild").start()
        store.audit(user, "query.index.rebuild", "index", "", f"camera={cid} since={since}", user.get("address", ""))
        return Response(status_code=202, media_type="application/json",
                        content='{"started": true}')

    @router.get("/api/query/index/status")
    def status(user=Depends(staff)):
        return {**si.status(store), "job": dict(job), "reasoner": bool(reason_fn or (backend_fn and _safe(backend_fn))),
                "max_rebuild_seconds": MAX_REBUILD_SECONDS}

    return router


def _safe(fn):
    try:
        return fn() is not None
    except Exception:
        return False
