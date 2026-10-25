"""BSAI-ComfyUI-H3-Accel-Selector -- pick the right MiniMax-H3 acceleration LoRA.

Why this exists
---------------
MiniMax-H3 ships two trunks, fl2va and ref2va, with IDENTICAL tensor key sets.
Every published acceleration LoRA is distilled for exactly one of them, but
applying the wrong one patches cleanly and renders silently wrong -- no
error, no console warning, just the wrong result.

This pack identifies the loaded base from its own weights
(final_layer.video_out fingerprint + the curve-form adaln flag), classifies
every acceleration file on disk from its safetensors HEADER alone, and
filters them against the base. It also carries the sampling recipe for each
family, because a distilled LoRA without its sigma schedule is a recipe for
noise.
"""

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"


# --------------------------------------------------------------------------
# backend route: the frontend asks it to re-filter the picker dropdown
# --------------------------------------------------------------------------

def _register_routes():
    try:
        from server import PromptServer
    except Exception:
        return

    @PromptServer.instance.routes.post("/h3_accel/scan")
    async def h3_accel_scan(request):
        import json as _json
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            body = {}

        base_hint = (body.get("base_name") or "").strip()
        show_all = bool(body.get("show_incompatible"))

        from . import h3_accel_core as core
        from . import nodes as _nodes

        base = core.base_from_name(base_hint) if base_hint else core.BaseInfo()
        items = _nodes.scan_candidates()
        rows = []
        for info in items:
            status, reason = core.judge(base, info)
            if not show_all and status == core.BAD:
                continue
            rows.append({
                "key": info.key,
                "name": info.display,
                "folder": info.folder,
                "status": status,
                "label": core.STATUS_LABEL.get(status, status),
                "reason": reason,
                "kind": info.kind,
                "trunk": info.trunk,
                "build": info.build,
                "nfe": info.nfe,
                "loader": info.loader,
                "recipe": info.recipe,
            })
        rows.sort(key=lambda r: ({"compatible": 0, "warning": 1}.get(r["status"], 2),
                                 r["name"].lower()))
        return web.json_response({
            "base": base.to_dict(),
            "items": rows,
            "total": len(items),
        })

    log_msg = "H3 Accel Selector: /h3_accel/scan ready"
    try:
        PromptServer.instance.logger.info(log_msg)
    except Exception:
        pass


try:
    _register_routes()
except Exception:
    pass
