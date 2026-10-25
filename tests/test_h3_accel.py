"""Standalone tests for the H3 Accel Selector classification engine.

    python tests/test_h3_accel.py

torch is not required for most of these; the live-scan section is skipped
when the models directory is not where it is expected.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_accel_core as core

FAILED = []


def check(label, got, want):
    if got != want:
        FAILED.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok    {label}")


# --------------------------------------------------------------------------
print("rank / in_features inference")
# --------------------------------------------------------------------------
# ComfyUI layout: A=[rank, in], B=[out, rank]
check("comfyui layout", core._infer_rank_and_width([64, 2688], [96768, 64]), (64, 2688))
# diffusers layout: A=[in, rank], B=[rank, out]
check("diffusers layout", core._infer_rank_and_width([2688, 64], [64, 96768]), (64, 2688))
# a rank-16 style LoRA must not report 16 as the adaln width
check("rank 16 not confused for width", core._infer_rank_and_width([16, 2688], [96768, 16]), (16, 2688))
# curve-projected adaln is width 8
check("curve width 8", core._infer_rank_and_width([2, 8], [96768, 2]), (2, 8))
check("garbage shapes", core._infer_rank_and_width([1, 2, 3], [4]), (None, None))

# --------------------------------------------------------------------------
print("trunk detection from names")
# --------------------------------------------------------------------------
check("ref2va token", core.detect_trunk_from_name("minimax_h3_ref2va_int8_convrot.safetensors"), "ref2va")
check("ref2v token", core.detect_trunk_from_name("minimax_h3_ref2v_turbo_4step_v0.1.safetensors"), "ref2va")
check("fl2va token", core.detect_trunk_from_name("minimax_h3_fl2va_zs05_int8_convrot.safetensors"), "fl2va")
check("fl2v token", core.detect_trunk_from_name("minimax_h3_fl2v_turbo_8step_v1.0.safetensors"), "fl2va")
check("longest match wins", core.detect_trunk_from_name("x_ref2va_y"), "ref2va")
check("no trunk token", core.detect_trunk_from_name("minimax_h3_hyperflow_8step_v1.0.safetensors"), None)

# --------------------------------------------------------------------------
print("base from filename")
# --------------------------------------------------------------------------
b = core.base_from_name("minimax_h3_ref2va_int8_convrot.safetensors")
check("base trunk", b.trunk, "ref2va")
check("base build", b.build, "full")
b = core.base_from_name("fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors")
check("pruned base", b.build, "pruned")
b = core.base_from_name("minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors")
check("hybrid base", b.trunk, "hybrid")

# --------------------------------------------------------------------------
print("H3 detection from key sets")
# --------------------------------------------------------------------------
check("h3 by token_refiner", core.detect_h3(["diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight"]), True)
check("h3 by video_out", core.detect_h3(["diffusion_model.final_layer.video_out.lora_A.weight"]), True)
check("h3 by qkv+mlp", core.detect_h3(["blocks.0.attn.qkv_proj.lora_A.weight",
                                       "blocks.0.mlp.fc2.lora_A.weight"]), True)
check("not h3: krea2", core.detect_h3(["transformer_blocks.9.ff.up.lora_A.weight",
                                        "final_layer.linear.lora_A.weight"]), False)
check("not h3: qwen-image", core.detect_h3(["transformer.transformer_blocks.9.img_mlp.proj.lora_A.weight"]), False)
check("not h3: ltx", core.detect_h3(["diffusion_model.adaln_single.linear.lora_A.weight",
                                     "diffusion_model.transformer_blocks.9.video_to_audio_attn.to_q.lora_A.weight"]), False)
check("not h3: moss", core.detect_h3(["diffusion_model.layers.0.proj.ffn_down.lora_down.weight"]), False)

# --------------------------------------------------------------------------
print("compatibility verdicts")
# --------------------------------------------------------------------------
def lora(trunk="ref2va", build="any", kind="turbo", adaln=None, h3=True):
    return core.LoraInfo(key="loras/x.safetensors", folder="loras", relpath="x.safetensors",
                         kind=kind, trunk=trunk, build=build, has_adaln=adaln is not None,
                         adaln_dim=adaln, is_h3=h3)


base = core.BaseInfo(trunk="ref2va", build="full", source="fingerprint")

s, _ = core.judge(base, lora(trunk="fl2va"))
check("wrong trunk is a hard mismatch", s, core.BAD)
s, _ = core.judge(base, lora(trunk="ref2va"))
check("matching trunk is ok", s, core.OK)
s, _ = core.judge(base, lora(trunk="any"))
check("trunk-agnostic file is ok", s, core.OK)
s, _ = core.judge(core.BaseInfo(trunk="ref2va", build="pruned"),
                  lora(trunk="ref2va", build="full", adaln=2688))
check("dense adaln on a pruned base is a hard mismatch", s, core.BAD)
s, _ = core.judge(core.BaseInfo(trunk="ref2va", build="pruned"),
                  lora(trunk="ref2va", build="pruned", adaln=8))
check("curve adaln on a pruned base is ok", s, core.OK)
s, _ = core.judge(core.BaseInfo(trunk="ref2va", build="full"),
                  lora(trunk="ref2va", build="pruned"))
check("pruned conversion on a full base warns", s, core.WARN)
s, _ = core.judge(base, lora(trunk="fl2va", kind="pdd"))
check("wrong-trunk PDD is a hard mismatch", s, core.BAD)
s, _ = core.judge(base, lora(trunk="ref2va", kind="pdd"))
check("matching PDD is ok", s, core.OK)
s, _ = core.judge(core.BaseInfo(trunk="hybrid", build="full"), lora(trunk="any"))
check("hybrid base warns", s, core.WARN)
s, _ = core.judge(base, lora(trunk="ref2va", kind="style"))
check("style file is not-applicable", s, core.NA)

# --------------------------------------------------------------------------
print("candidate gating")
# --------------------------------------------------------------------------
check("h3 turbo is a candidate",
      core.is_accel_candidate(lora(trunk="ref2va", kind="turbo")), True)
check("non-h3 turbo is not a candidate",
      core.is_accel_candidate(lora(trunk="ref2va", kind="turbo", h3=False)), False)
check("h3 style file is not a candidate",
      core.is_accel_candidate(lora(trunk="ref2va", kind="style")), False)

# --------------------------------------------------------------------------
print("safetensors header parsing")
# --------------------------------------------------------------------------
import json
import struct
import tempfile


def write_fake_safetensors(path, tensors, metadata=None):
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, shape in tensors.items():
        size = 1
        for s in shape:
            size *= s
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size * 4]}
        offset += size * 4
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "fake.safetensors")
    write_fake_safetensors(p, {
        "diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight": [16, 2688],
        "diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight": [96768, 16],
        "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight": [16, 5376],
        "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight": [28672, 16],
    }, {"ss_model_name": "test"})
    info = core.classify_lora_file(p, "loras", "fake.safetensors", core.load_catalog())
    check("header: n_pairs", info.n_pairs, 2)
    check("header: adaln dim", info.adaln_dim, 2688)
    check("header: has adaln", info.has_adaln, True)
    check("header: build", info.build, "full")
    check("header: is h3", info.is_h3, True)
    check("header: metadata", info.metadata.get("ss_model_name"), "test")

# --------------------------------------------------------------------------
print("live scan (skipped when the models dir is absent)")
# --------------------------------------------------------------------------
models = os.environ.get("H3_ACCEL_MODELS_DIR")
if models and os.path.isdir(models):
    n = 0
    for folder in ("loras", "pdd_acc", "veda"):
        root = os.path.join(models, folder)
        for dirpath, _d, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".safetensors"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
                info = core.classify_lora_file(os.path.join(dirpath, fn), folder, rel,
                                              core.load_catalog())
                if core.is_accel_candidate(info):
                    n += 1
                    check(f"live: {fn[:40]} is H3", info.is_h3, True)
    print(f"  ({n} acceleration candidates found under {models})")
else:
    print("  skipped (set H3_ACCEL_MODELS_DIR to enable)")

# --------------------------------------------------------------------------
print("end-to-end: base identification from a real fingerprint tensor")
# --------------------------------------------------------------------------
# Feed the shipped fp16 fingerprint back in as if it were the live
# final_layer.video_out.weight. That exercises the whole identification path
# -- model object walk, curve-form flag, fingerprint distance -- without
# loading a 31 GB checkpoint.
try:
    import torch
    from safetensors.torch import load_file

    fp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "fingerprints")

    class _FakeFinalLayer:
        def __init__(self, weight):
            self.video_out = type("vo", (), {"weight": weight})()

    class _FakeDiT:
        use_adaln_curves = False

        def __init__(self, weight, curves=False):
            self.final_layer = _FakeFinalLayer(weight)
            self.use_adaln_curves = curves

    class _FakeModel:
        def __init__(self, diT):
            self._dit = diT

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self._dit
            if name == "diffusion_model.final_layer":
                return self._dit.final_layer
            raise KeyError(name)

    fps = {}
    for trunk in ("fl2va", "ref2va"):
        p = os.path.join(fp_dir, f"video_out_{trunk}.safetensors")
        if os.path.exists(p):
            fps[trunk] = load_file(p)["video_out_weight"]
    if len(fps) == 2:
        m = _FakeModel(_FakeDiT(fps["ref2va"]))
        b = core.base_from_live_model(m, fps)
        check("live model trunk", b.trunk, "ref2va")
        check("live model build", b.build, "full")
        check("live model pdd_steps", b.pdd_steps, 1)
        check("live model source", b.source, "fingerprint")

        m2 = _FakeModel(_FakeDiT(fps["fl2va"]))
        check("live model trunk (fl2va)", core.base_from_live_model(m2, fps).trunk, "fl2va")

        m3 = _FakeModel(_FakeDiT(fps["ref2va"], curves=True))
        check("curve-form base", core.base_from_live_model(m3, fps).build, "pruned")

        # A PDD-baked checkpoint stacks 8 heads into the same weight.
        m4 = _FakeModel(_FakeDiT(fps["ref2va"].repeat(8, 1)))
        check("baked pdd detection", core.base_from_live_model(m4, fps).pdd_steps, 8)
    else:
        print("  skipped (fingerprints not found)")
except Exception as exc:
    FAILED.append(f"end-to-end: {exc}")
    print(f"  FAIL  end-to-end: {exc}")

# --------------------------------------------------------------------------
print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("all tests passed")
