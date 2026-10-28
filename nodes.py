"""ComfyUI nodes: pick the right MiniMax-H3 acceleration LoRA for the base you loaded.

The whole point of this pack is that the two published MiniMax-H3 trunks --
fl2va and ref2va -- ship IDENTICAL tensor key sets. A mismatched distill
therefore patches cleanly and renders silently wrong, with nothing in the
console to warn you. This pack closes that hole by identifying the loaded
base from its own weights and filtering the acceleration files against it.

Four nodes:

  H3 Accel Base Info      -- what base is this, really?
  H3 Accel LoRA Picker    -- pick + apply a matching acceleration LoRA
  H3 Accel Report         -- full compatibility report for the loaded base
  H3 Accel Fingerprint Bake -- register a custom base so it can be identified
"""

from __future__ import annotations

import os
import time

import folder_paths
import logging

try:
    from . import h3_accel_core as core
except ImportError:                      # loaded as a plain module directory
    import h3_accel_core as core

log = logging.getLogger("h3_accel")

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Model folders worth scanning for acceleration artifacts. ComfyUI's
# get_filename_list walks sub-folders, so a file parked at
# models/loras/H3加速lora/ref2va/pdd_acc/xxx.safetensors is found too --
# which matters, because the PDD Acc node only ever lists models/pdd_acc/.
SCAN_FOLDERS = ("loras", "pdd_acc", "veda", "vdn", "unet")

# Fingerprint sources, most specific first. The plugin ships its own copy so
# it works standalone, and also picks up a sibling PDD-Acc install so an
# updated fingerprint there is used without shipping a second one.
FINGERPRINT_DIRS = (
    os.path.join(PLUGIN_DIR, "fingerprints"),
    os.path.join(os.path.dirname(PLUGIN_DIR),
                 "BSAI-ComfyUI-MiniMax-H3-PDD-Acc", "partition_fingerprints"),
    os.path.join(os.path.dirname(PLUGIN_DIR),
                 "ComfyUI-MiniMax-H3-PDD-Acc", "partition_fingerprints"),
)

_fingerprints = None
_scan_cache = {}        # path -> (mtime, LoraInfo)
_lora_cache = {}        # path -> (mtime, state_dict, metadata)


# --------------------------------------------------------------------------
# folders
# --------------------------------------------------------------------------

def _register_extra_folders():
    for name in SCAN_FOLDERS:
        if name == "loras":
            continue
        path = os.path.join(folder_paths.models_dir, name)
        if os.path.isdir(path):
            try:
                folder_paths.add_model_folder_path(name, path, is_default=True)
            except Exception:
                pass


_register_extra_folders()


def resolve_key(key):
    """Turn a "folder/relative/path" combo value into an absolute path.

    Combo keys always use forward slashes; the filesystem gets the native
    separator back.
    """
    if "/" in key:
        folder, rel = key.split("/", 1)
    else:
        folder, rel = "loras", key
    rel = rel.replace("/", os.sep)
    try:
        return folder_paths.get_full_path_or_raise(folder, rel)
    except Exception:
        direct = os.path.join(folder_paths.models_dir, folder, rel)
        if os.path.exists(direct):
            return direct
        raise


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------

def load_fingerprints():
    """{trunk: fp16 video_out weight} from the shipped / sibling directories."""
    global _fingerprints
    if _fingerprints is not None:
        return _fingerprints
    found = {}
    try:
        from safetensors.torch import load_file
    except Exception:
        load_file = None
    if load_file is not None:
        for directory in FINGERPRINT_DIRS:
            if not os.path.isdir(directory):
                continue
            for name in sorted(os.listdir(directory)):
                if not (name.startswith("video_out_") and name.endswith(".safetensors")):
                    continue
                trunk = name[len("video_out_"):-len(".safetensors")]
                if trunk in found:
                    continue
                try:
                    found[trunk] = load_file(os.path.join(directory, name))["video_out_weight"]
                except Exception as exc:
                    log.warning("h3_accel: cannot read fingerprint %s: %s", name, exc)
    _fingerprints = found
    return _fingerprints


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def _list_files(folder):
    try:
        return folder_paths.get_filename_list(folder)
    except Exception:
        return []


def scan_candidates(accel_only=True):
    """Every MiniMax-H3 file found in the scan folders, classified.

    accel_only=True  -> only the acceleration artifacts (what the picker
                        offers). accel_only=False -> also the style/functional
                        H3 LoRAs and any stray checkpoint, so the report can
                        point them out.
    """
    catalog = core.load_catalog()
    out = []
    for folder in SCAN_FOLDERS:
        for rel in _list_files(folder):
            if not rel.lower().endswith(".safetensors"):
                continue
            try:
                path = resolve_key(f"{folder}/{rel}")
            except Exception:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            cached = _scan_cache.get(path)
            if cached is None or cached[0] != mtime:
                info = core.classify_lora_file(path, folder, rel, catalog)
                _scan_cache[path] = (mtime, info)
            else:
                info = cached[1]
            if not info.is_h3:
                continue
            if accel_only and not core.is_accel_candidate(info):
                continue
            out.append(info)
    return out


def _catalog_for_base(base):
    """The same catalog the core uses, resolved relative to this plugin."""
    return core.load_catalog()


# --------------------------------------------------------------------------
# base identification
# --------------------------------------------------------------------------

def identify_base(model, base_hint=""):
    """Identify the loaded base: live weights first, filename as the fallback."""
    fingerprints = load_fingerprints()
    base = core.base_from_live_model(model, fingerprints)
    hint = (base_hint or "").strip()
    if hint:
        named = core.base_from_name(hint)
        base.name = os.path.basename(hint)
        if base.trunk in ("unknown",) and named.trunk != "unknown":
            base.trunk = named.trunk
            base.notes.append("trunk from the upstream loader's filename")
        if base.build == "unknown" and named.build != "unknown":
            base.build = named.build
            base.notes.append("build from the upstream loader's filename")
        if named.trunk == "hybrid":
            base.trunk = "hybrid"
    return base


def _judge_all(base, items):
    rows = []
    for info in items:
        status, reason = core.judge(base, info)
        rows.append((status, info, reason))
    return rows


# --------------------------------------------------------------------------
# LoRA application
# --------------------------------------------------------------------------

def _load_lora_state(path):
    """Load a LoRA state dict, cached on (path, mtime)."""
    import comfy.utils
    mtime = os.path.getmtime(path)
    cached = _lora_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    _lora_cache[path] = (mtime, sd, metadata)
    return sd, metadata


def _apply_plain_lora(model, path, strength):
    """Apply a plain LoRA through ComfyUI's own quant-aware patch path."""
    import comfy.sd
    sd, metadata = _load_lora_state(path)
    patched, _ = comfy.sd.load_lora_for_models(model, None, sd, strength, 0.0,
                                               lora_metadata=metadata)
    return patched


def _delegate_pdd(model, lora, nfe, partition, check_mode):
    """Hand a PDD Acc file to the BSAI PDD-Acc plugin and return its outputs."""
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3PDDAccApply")
    if cls is None:
        raise ValueError(
            "This file is a PDD acceleration file (with a parallel-decode distillation head), which a plain LoRA loader cannot read.\n"
            "Install the BSAI-ComfyUI-MiniMax-H3-PDD-Acc pack; this node delegates the load to it automatically.")
    # The PDD Acc node's file list comes from models/pdd_acc/ only.
    available = folder_paths.get_filename_list("pdd_acc")
    name = os.path.basename(lora.relpath)
    if name not in available:
        raise ValueError(
            f"PDD file {name} is not in models/pdd_acc/, so the PDD Acc node cannot see it.\n"
            f"Move it to ComfyUI/models/pdd_acc/ and retry (current location: {lora.key}).")
    pdd_check = "strict" if check_mode == "strict" else ("warn" if check_mode == "warn" else "off")
    out = cls().apply(model, name, nfe, 1.0, 1.0, "raise", partition, True, None, pdd_check)
    return out  # (model, sigmas, info)


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

class H3AccelBaseInfo:
    """Identify the loaded MiniMax-H3 base: which trunk, which build, PDD state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Loaded MiniMax-H3 diffusion model"}),
            },
            "optional": {
                "base_hint": ("STRING", {"default": "",
                                         "tooltip": "Base filename from the upstream loader (auto-filled by the frontend; used as a fallback when the fingerprint misses)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("trunk", "build", "pdd_state", "info")
    FUNCTION = "inspect"
    CATEGORY = "model/minimax"
    DESCRIPTION = ("Identify the loaded MiniMax-H3 base: fl2va / ref2va trunk, full / pruned build, "
                   "and whether it already ships a baked PDD head group. The trunk is identified by the final_layer.video_out fingerprint.")

    def inspect(self, model, base_hint=""):
        base = identify_base(model, base_hint)
        pdd = "baked" if base.pdd_steps > 1 else "none"
        info = "\n".join(
            [f"Base: {base.name or '(unnamed)'}",
             f"Trunk: {base.trunk}  (source: {base.source})",
             f"Build: {base.build}",
             f"Baked PDD head group: {pdd} (pdd_steps={base.pdd_steps})"]
            + [f"· {n}" for n in base.notes])
        return (base.trunk, base.build, pdd, info)


class H3AccelLoRAPicker:
    """Pick an acceleration LoRA that matches the loaded base, and apply it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Loaded MiniMax-H3 diffusion model"}),
                "accel_lora": (accel_lora_choices(),
                               {"tooltip": "Candidate acceleration LoRA. The frontend filters the dropdown by base; "
                                           "PDD files are delegated to the BSAI PDD-Acc node."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                             "tooltip": "LoRA strength. Distilled LoRAs are trained at 1.0; "
                                                        "a PDD file's strength is controlled by the PDD node's own lora_strength, so this value does not apply to PDD."}),
                "check_mode": (["strict", "warn", "off"], {"default": "strict",
                                "tooltip": "strict = hard error on a trunk/build mismatch (prevents silently wrong output); "
                                           "warn = warn and continue; off = no check."}),
            },
            "optional": {
                "pdd_nfe": (["8", "4", "6"], {"default": "8",
                              "tooltip": "PDD files only. Model evaluation count: 8 = training block size, 4 = the officially endorsed merge, "
                                         "6 = non-uniform partition 8,8,4,4,4,4."}),
                "pdd_partition": ("STRING", {"default": "",
                              "tooltip": "PDD files only. Custom block sizes (fine steps), comma-separated, summing to 32, "
                                         "e.g. 8,4,4,4,4,4,4 = 7 steps."}),
                "base_hint": ("STRING", {"default": "",
                              "tooltip": "Base filename from the upstream loader (auto-filled by the frontend)"}),
                "show_incompatible": ("BOOLEAN", {"default": False,
                              "tooltip": "Also show non-matching files in the dropdown (only usable items are shown by default)"}),
            },
        }

    RETURN_TYPES = ("MODEL", "SIGMAS", "STRING", "BOOLEAN")
    RETURN_NAMES = ("model", "sigmas", "info", "compatible")
    FUNCTION = "pick"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = ("Filter and load the MiniMax-H3 acceleration LoRA by base. The fl2va and ref2va trunks share identical tensor keys, "
                   "so a mismatch raises no error and only paints the wrong thing silently - this node fingerprints the base and blocks it.")

    def pick(self, model, accel_lora, strength_model, check_mode,
             pdd_nfe="8", pdd_partition="", base_hint="", show_incompatible=False):
        base = identify_base(model, base_hint)
        items = scan_candidates()
        by_key = {i.key: i for i in items}
        lora = by_key.get(accel_lora)

        if lora is None:
            # The file may not be in the cache yet (new file, or the combo was
            # populated by the frontend). Classify it on the spot.
            try:
                path = resolve_key(accel_lora)
                folder, rel = accel_lora.split("/", 1) if "/" in accel_lora else ("loras", accel_lora)
                lora = core.classify_lora_file(path, folder, rel, _catalog_for_base(base))
            except Exception as exc:
                raise ValueError(f"Acceleration file not found {accel_lora}: {exc}")

        status, reason = core.judge(base, lora)

        if status == core.BAD:
            msg = (f"H3 acceleration LoRA mismatch: {reason}\n"
                   f"File: {lora.key}\n"
                   f"Base: {base.name or '(unnamed)'}  trunk={base.trunk}  build={base.build}\n"
                   f"If this is an intentional experiment, set check_mode to warn.")
            if check_mode == "strict":
                raise ValueError(msg)
            log.warning("H3 Accel LoRA Picker: %s", msg)

        sigmas = None
        if lora.loader == "pdd":
            out_model, sigmas, pdd_info = _delegate_pdd(
                model, lora, pdd_nfe, pdd_partition, check_mode)
            out_model = out_model
            extra = pdd_info
        elif lora.loader == "veda":
            raise ValueError(
                f"{lora.key} is Veda sparse-attention weights, not a plain LoRA.\n"
                "Load it with the BSAI-ComfyUI-VedaSparse nodes instead.")
        else:
            out_model = _apply_plain_lora(model, lora.path, strength_model)
            extra = ""

        info = "\n".join(filter(None, [
            f"Base: {base.name or '(unnamed)'}  trunk={base.trunk}  build={base.build}",
            f"File: {lora.key}",
            f"Kind: {lora.kind}  steps: {lora.nfe or '-'}  verdict: {core.STATUS_LABEL.get(status, status)}",
            f"Conclusion: {reason}",
            f"Recipe: {lora.recipe}" if lora.recipe else "",
            extra,
        ]))
        return (out_model, sigmas, info, status == core.OK)


class H3AccelReport:
    """Full compatibility report for the loaded base: every acceleration file, judged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Loaded MiniMax-H3 diffusion model"}),
            },
            "optional": {
                "base_hint": ("STRING", {"default": "",
                                         "tooltip": "Base filename from the upstream loader (auto-filled by the frontend)"}),
                "include_incompatible": ("BOOLEAN", {"default": True,
                                        "tooltip": "Also list non-matching / non-acceleration files"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = "model/minimax"
    DESCRIPTION = ("Generate an acceleration-LoRA compatibility report for the current base: identify the base, judge each acceleration file, "
                   "and give the matching sampling recipe.")

    def report(self, model, base_hint="", include_incompatible=True):
        base = identify_base(model, base_hint)
        everything = scan_candidates(accel_only=False)
        accel = [i for i in everything if core.is_accel_candidate(i)]
        rows = _judge_all(base, accel)
        if not include_incompatible:
            rows = [r for r in rows if r[0] in (core.OK, core.WARN)]
        text = core.format_report(base, rows)

        # Files that are in a LoRA folder but are not LoRAs at all -- usually a
        # full checkpoint parked in models/loras/ by mistake. Worth saying out
        # loud: a LoRA loader on a 20 GB checkpoint is a bad afternoon.
        strays = [i for i in everything if not i.is_lora]
        if strays:
            text += "\n" + "-" * 78 + "\n"
            text += "These files are in the LoRA folder but are not LoRAs (0 adapter pairs) and were excluded:\n"
            for i in strays:
                text += f"  · {i.key}  ({i.size / (1 << 30):.2f} GiB)\n"
            text += "  These are most likely full checkpoints in the wrong folder; load them with UNETLoader.\n"

        if include_incompatible:
            others = [i for i in everything
                      if i.is_lora and not core.is_accel_candidate(i)]
            if others:
                text += "\n" + "-" * 78 + "\n"
                text += f"{len(others)} more H3 LoRAs are not acceleration/few-step distillation (style or utility): "
                text += "not listed above:\n"
                for i in sorted(others, key=lambda x: x.display.lower()):
                    text += f"  · {i.display}\n"

        return {"ui": {"report": [text, {}]}, "result": (text,)}


class H3AccelFingerprintBake:
    """Register a custom MiniMax-H3 base so the picker can identify it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_file": (folder_paths.get_filename_list("diffusion_models"),
                              {"tooltip": "Base to fingerprint (reads only the single final_layer.video_out tensor)"}),
                "trunk_name": ("STRING", {"default": "",
                              "tooltip": "Give it a trunk name, e.g. fl2va / ref2va / xueluo_ref2va. "
                                         "Leave empty to auto-detect (uses the matching fingerprint when one exists)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "bake"
    CATEGORY = "model/minimax"
    DESCRIPTION = ("Bake a trunk fingerprint for a custom / fine-tuned base. Reads final_layer.video_out (~1MB), "
                   "writes an fp16 fingerprint into the plugin's fingerprints/ folder, after which this plugin recognises the base.")

    def bake(self, base_file, trunk_name=""):
        import torch
        from safetensors.torch import save_file
        try:
            path = folder_paths.get_full_path_or_raise("diffusion_models", base_file)
        except Exception as exc:
            return (f"Base file not found {base_file}: {exc}",)

        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = [k for k in f.keys() if k.endswith("final_layer.video_out.weight")]
            if len(keys) != 1:
                return (f"No unique final_layer.video_out.weight in {base_file}"
                        f"(found {len(keys)}) - this may not be a MiniMax-H3 base.",)
            w = f.get_tensor(keys[0]).to(torch.float32)

        out_dir = os.path.join(PLUGIN_DIR, "fingerprints")
        os.makedirs(out_dir, exist_ok=True)

        name = (trunk_name or "").strip()
        if name:
            # A user-supplied trunk name becomes part of the output file name, so it
            # must be a plain name (same rule as the fallback below); refuse anything
            # else instead of letting it escape the fingerprints/ folder.
            plain = "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch in "-_"))
            if plain != name or len(name) > 48:
                return (f"Trunk name {trunk_name!r} is not a plain name: only ASCII letters, "
                        f"digits, '-' and '_' are allowed (max 48 characters).",)
        else:
            matched, dists = core.identify_trunk(w, load_fingerprints())
            if matched:
                name = matched
            else:
                name = os.path.splitext(os.path.basename(base_file))[0]
                name = "".join(ch for ch in name if ch.isalnum() or ch in "-_")[:48]
        dst = os.path.join(out_dir, f"video_out_{name}.safetensors")
        save_file({"video_out_weight": w.to(torch.float16)}, dst, metadata={
            "partition": name,
            "tensor": keys[0],
            "source_file": os.path.basename(base_file),
            "note": "fp16 copy of final_layer.video_out.weight; compared by relative Frobenius distance",
        })
        global _fingerprints
        _fingerprints = None
        err = float((w - w.half().float()).norm() / w.norm())
        return (f"Wrote {dst}\n"
                f"Trunk name: {name}\n"
                f"Tensor: {list(w.shape)} fp16 storage error {err:.2e}\n"
                f"Restart ComfyUI to take effect.",)


# --------------------------------------------------------------------------
# combo helper
# --------------------------------------------------------------------------

def accel_lora_choices():
    """The combo list: every acceleration file on disk, keyed folder/relpath."""
    items = scan_candidates()
    keys = sorted({i.key for i in items})
    return keys or ["(No matching acceleration LoRA - put files in models/loras/ or models/pdd_acc/)"]


NODE_CLASS_MAPPINGS = {
    "H3AccelBaseInfo": H3AccelBaseInfo,
    "H3AccelLoRAPicker": H3AccelLoRAPicker,
    "H3AccelReport": H3AccelReport,
    "H3AccelFingerprintBake": H3AccelFingerprintBake,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AccelBaseInfo": "H3 Accel Base Info (trunk/build detection)",
    "H3AccelLoRAPicker": "H3 Accel LoRA Picker (filter and load by base)",
    "H3AccelReport": "H3 Accel Report (acceleration LoRA compatibility report)",
    "H3AccelFingerprintBake": "H3 Accel Fingerprint Bake (register a custom base)",
}
