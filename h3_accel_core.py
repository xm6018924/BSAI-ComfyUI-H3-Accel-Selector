"""Core logic for the MiniMax-H3 acceleration-LoRA selector.

Deliberately free of ComfyUI imports at module scope (torch + stdlib only) so
the classification engine can be unit-tested standalone:

    python tests/test_h3_accel.py

Three questions are answered here, in this order of authority:

1. WHICH TRUNK is the loaded base?  fl2va / ref2va / hybrid / unknown.
   The two published trunks ship IDENTICAL tensor key sets, so a mismatched
   distill applies cleanly and renders silently wrong. They are told apart by
   final_layer.video_out.weight -- fp32-unquantized in every published build,
   bit-identical across the int8_convrot / fp8_scaled / pruned / rebased
   variants of one trunk -- while the two trunks sit 0.0503 apart in relative
   Frobenius distance. Compared by DISTANCE, never by hash.
   Falls back to filename parsing when no fingerprint matches (finetunes,
   hybrids, hand-merged checkpoints).

2. WHICH BUILD is the loaded base?  full (dense adaln, 2688) / pruned
   (curve-form adaln, 8-dim shared table) / unknown. Read straight off the
   live DiT: MiniMaxH3Model.use_adaln_curves is True exactly on curve-form
   builds.

3. WHICH LoRA files fit this base?  Every candidate is classified from its
   own safetensors HEADER (never the weights -- a 2 GB LoRA is read in
   milliseconds), because the header already carries the decisive facts:
   which trunk the file was distilled for, whether it carries dense adaln
   adapters (full-only) or none (pruned-compatible), whether it is a PDD
   head-bank file (needs the PDD loader, not a plain LoRA loader), and how
   many adapter pairs it has.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

# Relative Frobenius tolerance for a trunk match. >> cast/storage noise
# (~2e-3) and << the fl2va<->ref2va gap (0.0503).
PARTITION_TOLERANCE = 0.015

# MiniMax-H3 adaln projection input width. The dense (full) build embeds a
# 2688-wide time embedding; the pruned (curve-form) build replaces it with a
# shared 8-dim basis table and the adaln adapters collapse to width 8.
ADALN_DIM_FULL = 2688
ADALN_DIM_CURVE = 8

# final_layer.video_out emits latents_dim(24) * patch(1,2,2) = 96 channels per
# head; a PDD-baked checkpoint stacks pdd_steps heads into one weight.
VIDEO_DIM = 96

# The four top-level tensors that make a file a PDD Acc head bank rather than
# a plain LoRA (alibaba-pai/MiniMax-H3-Acc-LoRAs).
PDD_HEAD_KEYS = ("proj_out.weight", "proj_out.bias",
                 "audio_proj_out.weight", "audio_proj_out.bias")

# Longest-first so "ref2va" wins over "ref2v" and "fl2va" over "fl2v".
TRUNK_TOKENS = (
    ("ref2va", "ref2va"), ("ref2v", "ref2va"), ("r2va", "ref2va"), ("r2v", "ref2va"),
    ("omniref", "ref2va"),
    ("fl2va", "fl2va"), ("fl2v", "fl2va"), ("t2va", "fl2va"), ("t2v", "fl2va"),
    ("i2v", "fl2va"),
)

_HYBRID_RE = re.compile(r"hybrid|b25[-_]?49", re.I)
_NFE_RE = re.compile(r"(?<![0-9a-z])(\d{1,2})\s*[-_]?step", re.I)
_ADALN_RE = re.compile(r"adaln_proj\.linear\.lora_([AB])\.weight$")
_HYBRID_BUILD_RE = re.compile(r"curveproj|curve_?proj|hybrid.*pruned|pruned.*hybrid", re.I)
# A pruned-compatibility conversion announces itself in the filename; the
# stripped adaln pairs are invisible in the header.
_PRUNED_HINT_RE = re.compile(r"pruned|curveproj|curve_?proj", re.I)


# --------------------------------------------------------------------------
# safetensors header (no weight load)
# --------------------------------------------------------------------------

def read_safetensors_header(path):
    """Return the safetensors header dict of `path` WITHOUT reading weights.

    Layout: u64 little-endian header length, then that many bytes of JSON.
    Reading only the header is what makes scanning a 2 GB LoRA instant.
    """
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (n,) = struct.unpack("<Q", raw_len)
        if n <= 0 or n > 64 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header is not an object")
    return header


def _tensor_entries(header):
    """{name: shape} for real tensors, dropping the __metadata__ pseudo-key."""
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if isinstance(v, dict) and "shape" in v:
            out[k] = list(v["shape"])
    return out


def _header_metadata(header):
    meta = header.get("__metadata__") or {}
    return {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {}


# --------------------------------------------------------------------------
# LoRA tensor key conventions
# --------------------------------------------------------------------------

# ComfyUI's own namespace (.lora_A/.lora_B) and Kohya's (.lora_down/.lora_up)
# are both in circulation for MiniMax-H3 adapters -- the TaoMate conversion
# ships the Kohya names -- so both are recognised.
_LORA_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight")
_LORA_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight")


def _is_a_tensor(key):
    return key.endswith(_LORA_A_SUFFIXES)


def _basename(path):
    """File name of a path written with either separator."""
    return (path or "").replace("\\", "/").rsplit("/", 1)[-1]


def _is_b_tensor(key):
    return key.endswith(_LORA_B_SUFFIXES)


def _sibling_b(key):
    """The B-tensor that pairs with an A-tensor key."""
    for a, b in zip(_LORA_A_SUFFIXES, _LORA_B_SUFFIXES):
        if key.endswith(a):
            return key[: -len(a)] + b
    return None


def detect_h3(tensors):
    """Is this file a MiniMax-H3 adapter?

    H3 is the only model in its neighbourhood with a token refiner, and the
    only one whose final layer splits into video_out / audio_out. Those two
    are decisive. The weaker markers (fused qkv_proj, context_embedder) only
    count on their own: LTX, Krea2, Qwen-Image and MOSS all have a
    final_layer / transformer_blocks of their own and would otherwise slip
    into an H3 picker.
    """
    joined = "\n".join(tensors)
    if ("token_refiner" in joined
            or "final_layer.video_out" in joined
            or "final_layer.audio_out" in joined):
        return True
    if "attn.qkv_proj" in joined and ("mlp.fc1" in joined or "mlp.fc2" in joined):
        return True
    if "context_embedder" in joined and "audio_proj_out" in joined:
        return True
    return False


# --------------------------------------------------------------------------
# LoRA classification
# --------------------------------------------------------------------------

@dataclass
class LoraInfo:
    """Everything we know about one candidate acceleration file."""

    key: str = ""               # "<folder>/<relative path>" -- the combo value
    folder: str = ""            # "loras" | "pdd_acc" | "veda" | ...
    relpath: str = ""           # path relative to that folder
    path: str = ""              # absolute path on disk
    size: int = 0
    kind: str = "unknown"       # pdd | turbo | hyperflow | dmad | pdmd | taomate |
                                # flashgen | faststep | veda | elm | style | unknown
    trunk: str = "unknown"      # fl2va | ref2va | any | hybrid | unknown
    build: str = "unknown"      # full | pruned | any | unknown
    loader: str = "lora"        # lora | pdd | veda | none
    nfe: int | None = None
    has_adaln: bool = False
    adaln_dim: int | None = None
    n_pairs: int = 0
    is_pdd: bool = False
    is_h3: bool = False
    is_lora: bool = True       # has LoRA adapter pairs (a checkpoint has none)
    metadata: dict = field(default_factory=dict)
    recipe: str = ""
    notes: list = field(default_factory=list)
    source: str = "auto"        # auto | catalog

    @property
    def display(self):
        return _basename(self.relpath)

    def to_dict(self):
        return {
            "key": self.key, "name": self.display, "folder": self.folder,
            "relpath": self.relpath, "size": self.size, "kind": self.kind,
            "trunk": self.trunk, "build": self.build, "loader": self.loader,
            "nfe": self.nfe, "has_adaln": self.has_adaln,
            "adaln_dim": self.adaln_dim, "n_pairs": self.n_pairs,
            "is_pdd": self.is_pdd, "is_h3": self.is_h3,
            "recipe": self.recipe,
            "notes": list(self.notes), "source": self.source,
        }


def _infer_rank_and_width(a_shape, b_shape):
    """Return (rank, in_features) from a LoRA A/B pair, layout-agnostic.

    ComfyUI / Kohya layout: A=[rank, in], B=[out, rank].
    Diffusers layout:        A=[in, rank],   B=[rank, out].

    Both readings are arithmetically valid for the same pair, so the tie is
    broken by the rank being the SMALL dimension: a LoRA rank is 8..128,
    an in_features width is 1024..28672. Getting this backwards silently
    reports a rank as the adaln width (64, 16, 2 ...) and would classify a
    full-build file as pruned.
    """
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None, None
    a0, a1 = a_shape
    b0, b1 = b_shape
    cands = []
    if a0 == b1:      # A=[rank, in], B=[out, rank]
        cands.append((a0, a1))
    if a1 == b0:      # A=[in, rank], B=[rank, out]
        cands.append((a1, a0))
    if not cands:
        return None, None
    return min(cands, key=lambda t: t[0])


def detect_trunk_from_name(name):
    """Which trunk a filename names, or None when it names neither/both."""
    low = (name or "").lower()
    hits = set()
    for token, trunk in TRUNK_TOKENS:
        if token in low:
            hits.add(trunk)
    if len(hits) == 1:
        return hits.pop()
    return None


def detect_nfe_from_name(name):
    m = _NFE_RE.search(name or "")
    if not m:
        return None
    n = int(m.group(1))
    return n if 2 <= n <= 32 else None


def classify_lora_file(path, folder, relpath, catalog=None):
    """Build a LoraInfo for one file, from its safetensors header only."""
    # The combo key always uses forward slashes; the OS-native relative path
    # is kept separately for path resolution.
    rel_native = relpath
    rel_key = relpath.replace("\\", "/")
    key = f"{folder}/{rel_key}" if folder else rel_key
    info = LoraInfo(key=key, folder=folder, relpath=rel_native, path=path)
    try:
        info.size = os.path.getsize(path)
    except OSError:
        pass

    try:
        header = read_safetensors_header(path)
    except Exception as exc:
        info.notes.append(f"unreadable header: {exc}")
        return info

    tensors = _tensor_entries(header)
    info.metadata = _header_metadata(header)

    a_keys = [k for k in tensors if _is_a_tensor(k)]
    info.n_pairs = len(a_keys)
    info.is_pdd = all(k in tensors for k in PDD_HEAD_KEYS)
    info.is_h3 = detect_h3(tensors)

    # The adaln adapters decide which build the file was distilled for:
    # dense adaln means the FULL build (in_features 2688), a curve-projected
    # adaln means the PRUNED build (in_features 8).
    for k in a_keys:
        if "adaln_proj" not in k:
            continue
        b_key = _sibling_b(k)
        if not b_key or b_key not in tensors:
            continue
        _, in_features = _infer_rank_and_width(tensors[k], tensors[b_key])
        if in_features:
            info.has_adaln = True
            info.adaln_dim = in_features
            break

    if info.is_pdd:
        info.kind = "pdd"
        info.loader = "pdd"
        info.build = "any"
    elif info.has_adaln and info.adaln_dim == ADALN_DIM_CURVE:
        info.build = "pruned"
    elif info.has_adaln and info.adaln_dim == ADALN_DIM_FULL:
        info.build = "full"
    elif not info.has_adaln:
        # No adaln adapters at all: the file loads on either build, but the
        # adaln half of the distill (if the family had one) is simply absent.
        info.build = "any"

    # ---- catalog pass: family rules refine kind/trunk/nfe/recipe ----
    if catalog:
        for rule in catalog:
            if not _rule_matches(rule, info):
                continue
            info.source = "catalog"
            info.kind = rule.get("kind", info.kind)
            info.trunk = rule.get("trunk", info.trunk)
            info.build = rule.get("build", info.build)
            info.loader = rule.get("loader", info.loader)
            if "h3" in rule:
                info.is_h3 = bool(rule["h3"])
            if rule.get("nfe") is not None:
                info.nfe = int(rule["nfe"])
            if rule.get("recipe"):
                info.recipe = rule["recipe"]
            if rule.get("note"):
                info.notes.append(rule["note"])
            break

    # A file with no adapter pairs at all is not a LoRA -- it is a checkpoint
    # (a full H3 model parked in the loras folder) or a bespoke format. Those
    # must never reach a LoRA loader.
    info.is_lora = bool(info.n_pairs) or info.is_pdd or info.loader == "veda"

    # ---- fallbacks when the catalog said nothing ----
    # A filename that explicitly names a trunk always wins over the catalog's
    # "any": larryvrh's v4 turbo is documented for every trunk, but a file
    # called "...REF2VAHybrid..." is a ref2va artifact and must be judged as
    # one.
    named = detect_trunk_from_name(_basename(relpath))
    if named and info.trunk in (None, "unknown", "any"):
        if info.trunk == "any":
            info.notes.append("trunk narrowed from filename")
        info.trunk = named
    # A file with NO adaln adapters at all is often a *pruned-compatibility
    # conversion* of a full-model LoRA (the converters strip the 51 dense
    # adaln pairs). The name says so; the header cannot. Only downgrade the
    # build when the header left it undecided.
    if info.build == "any" and _PRUNED_HINT_RE.search(_basename(relpath)):
        info.build = "pruned"
        info.notes.append("pruned-compatible conversion (adaln pairs stripped)")
    if info.nfe is None:
        n = detect_nfe_from_name(_basename(relpath))
        if n:
            info.nfe = n
            info.notes.append("step count from filename")
    if not info.recipe:
        info.recipe = default_recipe(info)
    return info


def _rule_matches(rule, info):
    """A catalog rule matches on folder, key regex, name regex and/or kind.

    `kind` is a filter only once the header already told us the kind; while
    the kind is still "unknown" the rule's kind is treated as a hint and
    assigned by the caller. Without that, every rule would be skipped for
    exactly the files the catalog exists to describe.
    """
    if rule.get("folder") and rule["folder"] != info.folder:
        return False
    rx = rule.get("name_regex")
    if rx:
        if not re.search(rx, _basename(info.relpath), re.I):
            return False
    kx = rule.get("key_regex")
    if kx:
        if not re.search(kx, info.key, re.I):
            return False
    elif not rx and not rule.get("folder"):
        return False
    if rule.get("kind") and info.kind != "unknown" and rule["kind"] != info.kind:
        return False
    return True


def default_recipe(info):
    """Fallback sampling recipe when the catalog has nothing for this file."""
    if info.kind == "pdd":
        return ("PDD 8/4 步 · sampler=euler · CFG=1.0 · SigmaShift 12.0/3.0 · "
                "sigmas 走 PDD 节点的输出 · 强度 1.0 · 不要与其它蒸馏 LoRA 叠加")
    if info.kind == "turbo":
        return ("少步蒸馏 · CFG=1.0 · 强度 1.0（噪声大降到 0.6–0.8）· "
                "步数按文件标称 · 不要与其它蒸馏 LoRA 叠加")
    if info.kind == "hyperflow":
        return ("8 步 · 手动 sigma: 1.0, 0.931506, 0.839236, 0.703462, 0.5, "
                "0.296538, 0.160764, 0.068494, 0.0 · 或 euler/normal")
    if info.kind in ("taomate", "dmad", "pdmd", "elm", "flashgen", "faststep"):
        return f"{info.kind} 少步蒸馏 · CFG=1.0（>1.0 会烧）· 强度 1.0 · 按文件标称步数"
    if info.kind == "veda":
        return "Veda 稀疏注意力权重 · 配合 VedaSparse 节点使用"
    return "未识别配方：请查该文件模型卡"


# --------------------------------------------------------------------------
# candidate selection
# --------------------------------------------------------------------------

# The families that are actually step-distillation / acceleration artifacts.
ACCEL_KINDS = frozenset({
    "pdd", "turbo", "hyperflow", "dmad", "pdmd", "taomate",
    "flashgen", "faststep", "veda", "elm",
})

# Filename smell test, so a style LoRA never lands in the acceleration picker.
_ACCEL_HINT_RE = re.compile(
    r"turbo|[23468]_?steps?\b|dmad|\bdmd\b|pdd|pdmd|taomate|hyperflow|flashgen|"
    r"faststep|fasth3|elm[_-]|longlive|veda|accel|distill|加速", re.I)


def is_accel_candidate(info):
    """Is this file an H3 acceleration artifact (as opposed to a style LoRA)?

    Three gates, in order:
      1. it must be a MiniMax-H3 adapter at all -- Krea2, MOSS-Image,
         Qwen-Image and LTX all ship turbo/4step LoRAs that would otherwise
         sit in the picker looking like valid H3 accelerators;
      2. it must belong to a known acceleration family, or at least smell
         like one in its filename -- realism LoRAs, film looks and character
         swaps are filtered out so the list stays short enough to read;
      3. a PDD head-bank file always counts, whatever its name.
    """
    if not info.is_h3:
        return False
    if not info.is_lora:
        return False
    if info.kind in ACCEL_KINDS:
        return True
    return bool(_ACCEL_HINT_RE.search(_basename(info.relpath)))


_CATALOG_CACHE = {}


def load_catalog(path=None):
    """Load the family rules from catalog.json (cached)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")
    key = os.path.abspath(path)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    rules = []
    try:
        with open(key, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = data.get("rules") or []
    except Exception:
        rules = []
    _CATALOG_CACHE[key] = rules
    return rules


# --------------------------------------------------------------------------
# base model identification
# --------------------------------------------------------------------------

@dataclass
class BaseInfo:
    """What we know about the loaded MiniMax-H3 base."""

    name: str = ""
    trunk: str = "unknown"     # fl2va | ref2va | hybrid | unknown
    build: str = "unknown"     # full | pruned | unknown
    pdd_steps: int = 1         # >1 = the checkpoint already carries a PDD head bank
    source: str = "unknown"    # fingerprint | filename | live | unknown
    video_out_shape: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def is_h3(self):
        return self.trunk != "unknown" or self.build != "unknown" or bool(self.notes)

    def to_dict(self):
        return {
            "name": self.name, "trunk": self.trunk, "build": self.build,
            "pdd_steps": self.pdd_steps, "source": self.source,
            "video_out_shape": list(self.video_out_shape),
            "notes": list(self.notes),
        }


def base_from_name(name):
    """Trunk/build guess parsed out of a checkpoint filename."""
    info = BaseInfo(name=os.path.basename(name or ""))
    low = (name or "").lower()
    trunk = detect_trunk_from_name(name)
    if _HYBRID_RE.search(low):
        info.trunk = "hybrid"
    elif trunk:
        info.trunk = trunk
    if "pruned" in low or "curveproj" in low:
        info.build = "pruned"
    elif any(t in low for t in ("int8_convrot", "fp8", "bf16", "int8", "w4a8", "nvfp4")):
        info.build = "full"
    info.source = "filename"
    return info


def base_from_live_model(model, fingerprints=None):
    """Identify the loaded base from the LIVE weights -- authoritative.

    Reads `use_adaln_curves` (pruned vs full) and `final_layer.video_out`
    (trunk by fingerprint, pdd_steps by channel count) off the DiT itself.
    """
    info = BaseInfo()
    try:
        dm = model.get_model_object("diffusion_model")
    except Exception as exc:
        info.notes.append(f"cannot reach the diffusion model: {exc}")
        return info

    info.build = "pruned" if getattr(dm, "use_adaln_curves", False) else "full"

    try:
        fl = model.get_model_object("diffusion_model.final_layer")
        weight = fl.video_out.weight
        info.video_out_shape = list(weight.shape)
        info.pdd_steps = max(1, int(weight.shape[0]) // VIDEO_DIM)
        if info.pdd_steps > 1:
            info.notes.append(
                f"checkpoint already carries a baked PDD head bank (pdd_steps="
                f"{info.pdd_steps}) -- an external PDD head bank is redundant")
    except Exception as exc:
        info.notes.append(f"final_layer.video_out not readable: {exc}")
        return info

    if fingerprints:
        trunk, dists = identify_trunk(weight, fingerprints)
        if trunk:
            info.trunk = trunk
            info.source = "fingerprint"
            info.notes.append(
                "trunk by fingerprint: " +
                ", ".join(f"{k} {v:.4f}" for k, v in sorted(dists.items())))
        else:
            info.source = "live"
            info.notes.append(
                "trunk fingerprint INCONCLUSIVE (" +
                (", ".join(f"{k} {v:.4f}" for k, v in sorted(dists.items())) or "no shape match") +
                f"; tol {PARTITION_TOLERANCE}) -- finetune or hand-merge?")
    return info


def identify_trunk(video_out_weight, fingerprints, tol=PARTITION_TOLERANCE):
    """Match a live final_layer.video_out.weight against shipped fingerprints.

    Returns (trunk or None, {trunk: relative Frobenius distance}).
    Advisory only -- every failure degrades to "no match", never an abort.
    """
    try:
        import torch
    except Exception:
        return None, {}
    try:
        live = video_out_weight.detach().to(torch.float32).cpu()
    except Exception:
        return None, {}
    dists = {}
    for name, ref in (fingerprints or {}).items():
        try:
            ref = ref.to(torch.float32)
            if ref.shape != live.shape:
                continue
            dists[name] = float((live - ref).norm() / ref.norm())
        except Exception:
            continue
    if not dists:
        return None, dists
    best = min(dists, key=dists.get)
    return (best, dists) if dists[best] <= tol else (None, dists)


# --------------------------------------------------------------------------
# compatibility verdict
# --------------------------------------------------------------------------

# status values, best first
OK = "compatible"
WARN = "warning"
BAD = "incompatible"
NA = "not-applicable"

STATUS_LABEL = {OK: "OK", WARN: "WARN", BAD: "MISMATCH", NA: "N/A"}


def judge(base, lora):
    """Decide whether `lora` may be applied to `base`.

    Returns (status, reason). The rules are deliberately ordered so the most
    decisive one wins: a wrong trunk beats a build mismatch, because a wrong
    trunk renders silently wrong while a build mismatch at least errors out.
    """
    if lora.kind == "pdd":
        if base.trunk in ("fl2va", "ref2va") and lora.trunk in ("fl2va", "ref2va") \
                and base.trunk != lora.trunk:
            return BAD, (f"PDD 文件是 {lora.trunk.upper()} 蒸馏，基座是 "
                         f"{base.trunk.upper()} 主干 -- 两者 key 集合相同，会安静地画错。"
                         f"请改用 {base.trunk.upper()} 版 PDD 文件。")
        if base.pdd_steps > 1:
            return WARN, ("基座已自带烘焙好的 PDD 头组（pdd_steps>1），外部 PDD 头组冗余；"
                          "若基座是烘焙主干版，把 lora_strength 设为 0.0。")
        return OK, "PDD 头组 + 主干 LoRA，匹配。"

    if base.trunk in ("fl2va", "ref2va") and lora.trunk in ("fl2va", "ref2va") \
            and base.trunk != lora.trunk:
        return BAD, (f"文件蒸馏于 {lora.trunk.upper()}，基座是 {base.trunk.upper()} 主干。"
                     f"两棵主干张量 key 完全相同，套上去不会报错，只会安静地画错。")

    if base.trunk == "hybrid":
        return WARN, ("基座是 fl2va+ref2va 混合主干，PDD 对混合主干属 off-label"
                      "（训练在纯主干上）-- 建议先与匹配的纯主干 A/B 对比。")

    # A style / functional LoRA is not an acceleration artifact at all --
    # judged before the build checks, which would otherwise call it "OK" on
    # the strength of a matching adaln width.
    if lora.kind in ("style", "unknown"):
        return NA, "这不是加速/少步蒸馏文件，是风格或功能 LoRA。"

    if lora.trunk == "unknown":
        return WARN, "无法从文件名/权重确定该文件的蒸馏目标主干，兼容性未验证。"

    # ---- build compatibility ----
    # The asymmetry is deliberate:
    #   dense-adaln LoRA on a curve-form base -> shape mismatch, ComfyUI logs a
    #     wall of adaln_proj errors and DROPS that half of the distill. Hard fail.
    #   adaln-free (pruned-conversion) LoRA on a full base -> applies cleanly,
    #     but the adaln half of the distill was stripped by the converter, so
    #     the result is weaker than the original full-model file. Warn, not fail.
    if base.build == "pruned" and lora.build == "full":
        return BAD, ("基座是剪枝（曲线形式 adaln）版，而该文件带 dense adaln 适配器"
                     "（输入宽度 2688）-- 形状对不上，ComfyUI 会刷一串 adaln_proj 报错并"
                     "静默丢掉这部分蒸馏。请改用剪枝版/无 adaln 版文件。")
    if base.build == "full" and lora.build == "pruned":
        return WARN, ("该文件是为剪枝基座做的兼容转换版（dense adaln 适配器已被剥掉），"
                      "在完整版基座上能加载，但蒸馏强度弱于原版完整文件。")
    if base.build == "pruned" and lora.build == "any":
        return OK, "该文件不含 dense adaln 适配器，在剪枝基座上可直接使用。"
    if base.build == "full" and lora.build == "any":
        return OK, "该文件不含 dense adaln 适配器，完整版基座上可直接使用。"

    return OK, "主干与构建均匹配。"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def format_report(base, items, max_rows=200):
    """Human-readable report: the base on top, then every candidate."""
    lines = []
    lines.append("=" * 78)
    lines.append("MiniMax-H3 加速 LoRA 适配报告")
    lines.append("=" * 78)
    lines.append(f"基座        : {base.name or '(未命名)'}")
    lines.append(f"主干        : {base.trunk}    (判定来源: {base.source})")
    lines.append(f"构建        : {base.build}"
                 f"{'  (剪枝=曲线形式 adaln)' if base.build == 'pruned' else ''}")
    if base.pdd_steps > 1:
        lines.append(f"内置 PDD    : pdd_steps={base.pdd_steps}  (已烘焙头组)")
    for n in base.notes:
        lines.append(f"  · {n}")
    lines.append("")

    counts = {}
    for it in items:
        counts[it[0]] = counts.get(it[0], 0) + 1
    summary = "  ".join(f"{STATUS_LABEL.get(k, k)}={v}" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1]))
    lines.append(f"候选加速文件 {len(items)} 个   {summary}")
    lines.append("-" * 78)

    order = {OK: 0, WARN: 1, NA: 2, BAD: 3}
    rows = sorted(items, key=lambda it: (order.get(it[0], 9), it[1].display.lower()))
    for status, lora, reason in rows[:max_rows]:
        tag = STATUS_LABEL.get(status, status)
        head = f"[{tag:^8}] {lora.display}"
        if lora.nfe:
            head += f"  ({lora.nfe} 步)"
        lines.append(head)
        lines.append(f"           目录={lora.folder}  类型={lora.kind}  "
                     f"主干={lora.trunk}  构建={lora.build}"
                     f"{'  需PDD节点' if lora.loader == 'pdd' else ''}")
        lines.append(f"           {reason}")
        if lora.recipe:
            lines.append(f"           配方: {lora.recipe}")
        lines.append("")
    if len(rows) > max_rows:
        lines.append(f"... 还有 {len(rows) - max_rows} 个未显示")
    return "\n".join(lines)
