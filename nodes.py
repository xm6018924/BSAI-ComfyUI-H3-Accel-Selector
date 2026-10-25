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
            "这个文件是 PDD 加速文件（带并行解码蒸馏头组），普通 LoRA 加载器读不了。\n"
            "请先安装 BSAI-ComfyUI-MiniMax-H3-PDD-Acc 插件，本节点会自动把加载委托给它。")
    # The PDD Acc node's file list comes from models/pdd_acc/ only.
    available = folder_paths.get_filename_list("pdd_acc")
    name = os.path.basename(lora.relpath)
    if name not in available:
        raise ValueError(
            f"PDD 文件 {name} 不在 models/pdd_acc/ 目录里，PDD Acc 节点扫描不到它。\n"
            f"请把它移动到 ComfyUI/models/pdd_acc/ 后重试（当前位置：{lora.key}）。")
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
                "model": ("MODEL", {"tooltip": "已加载的 MiniMax-H3 扩散模型"}),
            },
            "optional": {
                "base_hint": ("STRING", {"default": "",
                                         "tooltip": "上游加载器里的基座文件名（前端自动填写，用于指纹未命中时兜底）"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("trunk", "build", "pdd_state", "info")
    FUNCTION = "inspect"
    CATEGORY = "model/minimax"
    DESCRIPTION = ("识别已加载的 MiniMax-H3 基座：fl2va / ref2va 主干、完整版 / 剪枝版构建、"
                   "以及是否已自带烘焙好的 PDD 头组。主干靠 final_layer.video_out 指纹判定。")

    def inspect(self, model, base_hint=""):
        base = identify_base(model, base_hint)
        pdd = "baked" if base.pdd_steps > 1 else "none"
        info = "\n".join(
            [f"基座: {base.name or '(未命名)'}",
             f"主干: {base.trunk}  (判定来源: {base.source})",
             f"构建: {base.build}",
             f"内置 PDD 头组: {pdd} (pdd_steps={base.pdd_steps})"]
            + [f"· {n}" for n in base.notes])
        return (base.trunk, base.build, pdd, info)


class H3AccelLoRAPicker:
    """Pick an acceleration LoRA that matches the loaded base, and apply it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已加载的 MiniMax-H3 扩散模型"}),
                "accel_lora": (accel_lora_choices(),
                               {"tooltip": "候选加速 LoRA。下拉框由前端按基座自动筛选；"
                                           "PDD 文件会委托给 BSAI PDD-Acc 节点加载。"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                             "tooltip": "LoRA 强度。蒸馏 LoRA 训练于 1.0；"
                                                        "PDD 文件的强度由 PDD 节点自己的 lora_strength 控制，此值对 PDD 无效。"}),
                "check_mode": (["strict", "warn", "off"], {"default": "strict",
                                "tooltip": "strict = 主干/构建不匹配直接报错（防静默画错）；"
                                           "warn = 只警告继续；off = 不检查。"}),
            },
            "optional": {
                "pdd_nfe": (["8", "4", "6"], {"default": "8",
                              "tooltip": "仅 PDD 文件有效。模型评估次数：8 = 训练块大小，4 = 官方认可的合并，"
                                         "6 = 非均匀分区 8,8,4,4,4,4。"}),
                "pdd_partition": ("STRING", {"default": "",
                              "tooltip": "仅 PDD 文件有效。自定义块大小（细步），逗号分隔，合计 32，"
                                         "如 8,4,4,4,4,4,4 = 7 步。"}),
                "base_hint": ("STRING", {"default": "",
                              "tooltip": "上游加载器里的基座文件名（前端自动填写）"}),
                "show_incompatible": ("BOOLEAN", {"default": False,
                              "tooltip": "下拉框里是否也显示不匹配的文件（默认只显示可用项）"}),
            },
        }

    RETURN_TYPES = ("MODEL", "SIGMAS", "STRING", "BOOLEAN")
    RETURN_NAMES = ("model", "sigmas", "info", "compatible")
    FUNCTION = "pick"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = ("按基座自动筛选并加载 MiniMax-H3 加速 LoRA。fl2va 与 ref2va 两棵主干张量 key "
                   "完全相同，配错不会报错、只会安静地画错 —— 本节点用指纹识别基座并拦截。")

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
                raise ValueError(f"找不到加速文件 {accel_lora}: {exc}")

        status, reason = core.judge(base, lora)

        if status == core.BAD:
            msg = (f"H3 加速 LoRA 配对错误：{reason}\n"
                   f"文件: {lora.key}\n"
                   f"基座: {base.name or '(未命名)'}  主干={base.trunk}  构建={base.build}\n"
                   f"如确为故意实验，把 check_mode 改成 warn。")
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
                f"{lora.key} 是 Veda 稀疏注意力权重，不是普通 LoRA。\n"
                "请改用 BSAI-ComfyUI-VedaSparse 插件的节点加载它。")
        else:
            out_model = _apply_plain_lora(model, lora.path, strength_model)
            extra = ""

        info = "\n".join(filter(None, [
            f"基座: {base.name or '(未命名)'}  主干={base.trunk}  构建={base.build}",
            f"文件: {lora.key}",
            f"类型: {lora.kind}  步数: {lora.nfe or '-'}  判定: {core.STATUS_LABEL.get(status, status)}",
            f"结论: {reason}",
            f"配方: {lora.recipe}" if lora.recipe else "",
            extra,
        ]))
        return (out_model, sigmas, info, status == core.OK)


class H3AccelReport:
    """Full compatibility report for the loaded base: every acceleration file, judged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "已加载的 MiniMax-H3 扩散模型"}),
            },
            "optional": {
                "base_hint": ("STRING", {"default": "",
                                         "tooltip": "上游加载器里的基座文件名（前端自动填写）"}),
                "include_incompatible": ("BOOLEAN", {"default": True,
                                        "tooltip": "是否也列出不匹配 / 非加速的文件"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = "model/minimax"
    DESCRIPTION = ("生成当前基座的加速 LoRA 适配报告：识别基座，逐个判定每个加速文件，"
                   "并给出对应采样配方。")

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
            text += "以下文件在 LoRA 目录里，但不是 LoRA（0 个适配器对），已排除：\n"
            for i in strays:
                text += f"  · {i.key}  ({i.size / (1 << 30):.2f} GiB)\n"
            text += "  这些多半是完整 checkpoint 被放错了目录，请用 UNETLoader 加载。\n"

        if include_incompatible:
            others = [i for i in everything
                      if i.is_lora and not core.is_accel_candidate(i)]
            if others:
                text += "\n" + "-" * 78 + "\n"
                text += f"另有 {len(others)} 个 H3 LoRA 不属于加速/少步蒸馏（风格、功能类），"
                text += "未列入上表：\n"
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
                              {"tooltip": "要登记指纹的基座（只读取 final_layer.video_out 这一个张量）"}),
                "trunk_name": ("STRING", {"default": "",
                              "tooltip": "给它起一个主干名，例如 fl2va / ref2va / xueluo_ref2va。"
                                         "留空则自动判断（能匹配已有指纹就用匹配的）。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "bake"
    CATEGORY = "model/minimax"
    DESCRIPTION = ("为自定义 / 微调基座烘焙主干指纹。读取 final_layer.video_out（约 1MB），"
                   "写成 fp16 指纹放进插件的 fingerprints/ 目录，之后本插件就能认出这个基座。")

    def bake(self, base_file, trunk_name=""):
        import torch
        from safetensors.torch import save_file
        try:
            path = folder_paths.get_full_path_or_raise("diffusion_models", base_file)
        except Exception as exc:
            return (f"找不到基座文件 {base_file}: {exc}",)

        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = [k for k in f.keys() if k.endswith("final_layer.video_out.weight")]
            if len(keys) != 1:
                return (f"{base_file} 里找不到唯一的 final_layer.video_out.weight"
                        f"（找到 {len(keys)} 个）—— 这可能不是 MiniMax-H3 基座。",)
            w = f.get_tensor(keys[0]).to(torch.float32)

        out_dir = os.path.join(PLUGIN_DIR, "fingerprints")
        os.makedirs(out_dir, exist_ok=True)

        name = (trunk_name or "").strip()
        if not name:
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
        return (f"已写入 {dst}\n"
                f"主干名: {name}\n"
                f"张量: {list(w.shape)} fp16 存储误差 {err:.2e}\n"
                f"重启 ComfyUI 后生效。",)


# --------------------------------------------------------------------------
# combo helper
# --------------------------------------------------------------------------

def accel_lora_choices():
    """The combo list: every acceleration file on disk, keyed folder/relpath."""
    items = scan_candidates()
    keys = sorted({i.key for i in items})
    return keys or ["（未找到任何加速 LoRA —— 请把文件放进 models/loras/ 或 models/pdd_acc/）"]


NODE_CLASS_MAPPINGS = {
    "H3AccelBaseInfo": H3AccelBaseInfo,
    "H3AccelLoRAPicker": H3AccelLoRAPicker,
    "H3AccelReport": H3AccelReport,
    "H3AccelFingerprintBake": H3AccelFingerprintBake,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AccelBaseInfo": "H3 Accel Base Info (主干/构建识别)",
    "H3AccelLoRAPicker": "H3 Accel LoRA Picker (按基座筛选并加载)",
    "H3AccelReport": "H3 Accel Report (加速 LoRA 适配报告)",
    "H3AccelFingerprintBake": "H3 Accel Fingerprint Bake (登记自定义基座)",
}
