# BSAI-ComfyUI-H3-Accel-Selector

**按基座自动筛选 MiniMax-H3 加速 LoRA 的插件。**

## 为什么需要它

MiniMax-H3 有两棵主干：**fl2va**（文生/首尾帧）和 **ref2va**（参考生）。这两棵主干的
**张量 key 集合完全相同**。所有公开的加速 LoRA 都是针对其中一棵蒸馏的，于是：

> 把 fl2va 的加速 LoRA 套到 ref2va 的基座上 —— **不会报错，不会警告，只会安静地画错。**

这不是"效果差一点"，是主干语义完全错位。剪枝构建还有第二层坑：完整版的 dense adaln
适配器（输入宽度 2688）套到剪枝版（曲线形式 adaln，宽度 8）上，形状对不上，ComfyUI
会刷一串 `adaln_proj` 报错，然后**静默丢掉那部分蒸馏**。

本插件做三件事：

1. **认出基座** —— 从模型自己的权重判断它是 fl2va 还是 ref2va、是完整版还是剪枝版、
   是否已自带烘焙好的 PDD 头组。
2. **认出文件** —— 只读每个候选文件的 safetensors **头部**（不读权重，2GB 的 LoRA 也是
   毫秒级），判断它是给哪棵主干蒸馏的、带不带 dense adaln、是不是带 PDD 头组。
3. **过滤 + 给配方** —— 只留下能用的文件，并附上这个加速系族正确的采样配方。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <this repo>
```

无需额外依赖（只用 ComfyUI 自带的 torch / safetensors）。

## 四个节点

节点在 `model/minimax` 分类下。

### H3 Accel Base Info（主干/构建识别）

接上模型，输出 `trunk` / `build` / `pdd_state` / `info`。想搞清楚"我手上这个基座到底是
什么"的时候用它。

### H3 Accel LoRA Picker（按基座筛选并加载）

主节点。接模型 → 选加速 LoRA → 接采样器。

- `accel_lora` 下拉框**由前端按基座实时筛选**：上游换了模型，列表自动变成能用的那些。
  勾上 `show_incompatible` 可以把不匹配的文件也显示出来（带判定理由）。
- `check_mode`：`strict`（默认）主干/构建不匹配直接报错；`warn` 只警告继续；`off` 不检查。
- 选中 **PDD** 文件时，本节点**自动委托**给 `BSAI-ComfyUI-MiniMax-H3-PDD-Acc` 的
  `MiniMax H3 PDD Acc LoRA (Apply)` 节点加载，并把它的 **SIGMAS** 直接输出出来 ——
  接 `SamplerCustomAdvanced` 的 `sigmas` 输入即可，不需要再单独摆一个 PDD 节点。
- 普通 LoRA 走 ComfyUI 自己的量化感知补丁路径，和 `LoraLoaderModelOnly` 等价。

### H3 Accel Report（加速 LoRA 适配报告）

输出一份完整报告：基座判定 + 逐个文件的判定（OK / WARN / MISMATCH）+ 每个文件对应的
采样配方。选型时先跑这个。

另外会单独列出**放在 LoRA 目录里但根本不是 LoRA 的文件**（0 个适配器对，通常是放错
目录的完整 checkpoint）—— 拿 LoRA 加载器去怼 20GB checkpoint 是个很糟糕的下午。

### H3 Accel Fingerprint Bake（登记自定义基座）

你的自定义微调基座（`minimax_h3_ref2va_XUELUO_int8_convrot`、`Minimax-h3_Singularity_*`、
`PinkCherry_h3_fl2va_*` 之类）不在官方指纹里，指纹判定会 inconclusive。这个节点读取
基座的 `final_layer.video_out`（约 1MB，用 mmap，不加载整个模型），写成 fp16 指纹放进
插件的 `fingerprints/`，重启后本插件就能认出它。

## 判定原理

### 主干：final_layer.video_out 指纹

`final_layer.video_out.weight` 在所有公开构建里都是 fp32 不量化区，并且**同一棵主干**的
int8_convrot / fp8_scaled / pruned / adaln-rebased 变体之间**逐位相同**；两棵主干之间
相对 Frobenius 距离约 **0.0503**。所以按**距离**比对，不按哈希：

| 距离 | 含义 |
|---|---|
| ≤ 0.015 | 匹配（容差 >> 转换/存储噪声 ~2e-3，<< 主干间距 0.0503） |
| > 0.015 | 不匹配 → 判定 inconclusive（微调？手工合并？） |

指纹来自 `fingerprints/`，插件自带一份，同时也**自动发现**同目录下的
`BSAI-ComfyUI-MiniMax-H3-PDD-Acc/partition_fingerprints/`，那边的更新会被优先使用。

### 构建：use_adaln_curves

ComfyUI 核心的 `MiniMaxH3Model.use_adaln_curves` 在剪枝（曲线形式）构建上为 True。
直接读，不需要猜。

### 文件：只读 safetensors 头部

头部里就有决定性事实：

- **dense adaln 的输入宽度** —— `lora_A`/`lora_B`（或 Kohya 的 `lora_down`/`lora_up`）
  的形状反解出 in_features。2688 = 完整版蒸馏、8 = 曲线形式蒸馏。
  注意 rank 和 in_features 必须按**小的那一维**判定，否则 64/16/2 会被误当成宽度。
- **PDD 头组** —— 顶层有 `proj_out.weight` / `audio_proj_out.weight` 等 4 个张量，
  那就不是普通 LoRA，普通加载器读不了。
- **是不是 H3 的文件** —— `token_refiner`（H3 独有）、`final_layer.video_out`、
  `attn.qkv_proj` + `mlp.fc1/fc2`。这样 Krea2 / MOSS-Image / Qwen-Image / LTX 的
  turbo、4step LoRA 就不会混进 H3 的选择器。

## 扫描范围

`models/loras/`、`models/pdd_acc/`、`models/veda/`、`models/vdn/`、`models/unet/`，
**递归**扫描子目录。

> 注意：PDD Acc 节点只列 `models/pdd_acc/`。如果把 PDD 文件放在别处（比如
> `models/loras/H3加速lora/ref2va/pdd_acc/`），本插件能看见它，但委托给 PDD 节点时
> 会明确提示"请移到 models/pdd_acc/"。

## 配方速查

| 系族 | 步数 | 采样器 | CFG | SigmaShift | 备注 |
|---|---|---|---|---|---|
| PDD Acc（官方） | 8 / 4 / 6 | euler | 1.0 | 12.0 / 3.0 | sigmas 走 PDD 节点输出；不叠其它蒸馏 |
| LightX2V Ref2V turbo 4-step v0.1 | 4 | — | 1.0 | 12.0 / 3.0 | 强度 1.0（Kijai 建议 0.75） |
| LightX2V Ref2V turbo 8-step v1.0 768p | 8 | — | 1.0 | 6.0 / 3.0 | |
| larryvrh Turbo v4（step600 EMA） | 6 | simple | 1.0 | 12.0 / 4–6 | 强度 1.0（社区 0.6–0.8） |
| HyperFlow 8-step | 8 | euler/normal | — | — | 手动 9 点 sigma，见下 |
| TaoMate 3-step | 3 | euler | 1.0 | — | 仅 FL2VA；CFG>1.0 会烧 |
| FastH3 4-step | 4 | — | — | — | 仅 T2V/首尾帧；Ref2VA 未蒸馏 |

HyperFlow 手动 sigma（9 点 = 8 个间隔）：

```
1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0
```

**通用铁律**

- 蒸馏 LoRA **不叠加** —— PDD / turbo / HyperFlow 三选一。角色/风格 LoRA 可以叠。
- 不要叠 step-cache（blockcache / EasyCache）—— PDD 的 final_layer 补丁会 fail-closed。
- 步数越少**贴参考越弱**，Ref2VA 尤其明显。要紧贴参考就 20–25 步关掉 turbo。
- Turbo 缩短的是 schedule，不是单步耗时；4 步 LoRA 跑 20 步 ≈ 基线 20 步的耗时。

## 以 `minimax_h3_ref2va_int8_convrot` 为例

这是 **Ref2VA 主干 + INT8 ConvRot 完整版**（约 31.7 GiB，dense adaln 2688）。

| 判定 | 文件 |
|---|---|
| ✅ 可用 | `minimax_h3_ref2va_pdd_acc_8step_comfyui`（PDD 8 步） |
| ✅ 可用 | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16`（4 步） |
| ✅ 可用 | `minimax_h3_turbo_v4_step600_ema`（6 步，跨主干） |
| ✅ 可用 | `minimax_h3_hyperflow_8step_v1.0`（8 步，跨主干） |
| ✅ 可用 | `minimax_h3_pdmd_2step` / `4step`、`minimax_h3_ELM_longlive_plug_4step` |
| ⚠️ 可用但更弱 | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui`（剪枝转换版，adaln 被剥掉） |
| ❌ 拒绝 | `minimax_h3_fl2va_pdd_acc_8step_comfyui`（FL2VA 蒸馏） |
| ❌ 拒绝 | 全部 `minimax_h3_fl2v_turbo_*`、`TaoMate-H3-step3000`（仅 FL2VA） |
| ❌ 拒绝 | `minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_*`（FL2VA + 剪枝曲线基） |

完整清单用 **H3 Accel Report** 节点出。

## 示例工作流

[`example_workflows/BSAI_H3_Film_Factory_直播带货_手动档_ref2va_int8_accel-selector.json`](example_workflows/BSAI_H3_Film_Factory_直播带货_手动档_ref2va_int8_accel-selector.json)
— 以 BSAI H3 Film Factory 电影工厂（直播带货手动档、ref2va int8）为底，把四个节点全部接上：

- **LoRA Picker** 串在模型链里（`ComfySwitchNode → Picker → H3LoraStack → Film Factory`）：
  `minimax_h3_ref2va_int8_convrot` 基座上只列出 ref2va 蒸馏文件，示例选 LightX2V
  Ref2V turbo 4-step（Film Factory 步数相应改为 4）。同时把上游 `BSAI_SolH3_Loader`
  的 `lora_name` 设为 `none`，避免两个蒸馏叠加。
- **Base Info** 与 **Report** 接在 Switch 的 MODEL 输出上，每次运行打印基座判定与
  全部候选文件的 OK / WARN / MISMATCH 结论（结果用 PreviewAny 显示）。
- **Fingerprint Bake** 默认旁路，换自定义基座时取消旁路跑一次即可登记指纹。

拖进 ComfyUI 即用；节点组内附中文说明（含换回 PDD 8 步的做法）。

## 目录

```
BSAI-ComfyUI-H3-Accel-Selector/
├── __init__.py              节点注册 + /h3_accel/scan 路由
├── nodes.py                 四个节点
├── h3_accel_core.py         分类引擎（纯 torch/stdlib，可独立测试）
├── catalog.json             加速系族规则 + 配方
├── fingerprints/            主干指纹（fp16 video_out）
├── web/h3_accel.js          下拉框实时筛选
├── example_workflows/       示例工作流（H3 Film Factory 电影工厂）
└── tests/test_h3_accel.py   自测
```

## 测试

```bash
python tests/test_h3_accel.py
# 加上真实文件扫描：
set H3_ACCEL_MODELS_DIR=/path/to/ComfyUI/models && python tests/test_h3_accel.py
```

## 已知边界

- **混合主干**（`minimax_h3_hybrid_fl2va_ref2va_b25-49`）：指纹必然不匹配，只能靠文件名
  判定，PDD 对混合主干属 off-label。插件对混合基座一律给 WARN 而不是拦截。
- **Veda 权重**（`models/veda/`）不是普通 LoRA，是稀疏注意力替换件，要用
  `BSAI-ComfyUI-VedaSparse` 加载。插件识别出来并拒绝用普通 LoRA 流程加载。
- **FastH3 是 checkpoint 不是 LoRA**：`fastvideo_fasth3_8step_v2_pruned_int8_convrot` 之类
  放进 `models/loras/` 会被报告为"不是 LoRA"。
- 指纹判定是**建议性**的：匹配不上就降级成 inconclusive，绝不因此中断渲染。

---

# English

**Auto-filter MiniMax-H3 acceleration LoRAs against the base model you loaded.**

## Why this exists

MiniMax-H3 ships two trunks — **fl2va** (text-to-video / first-last-frame) and **ref2va**
(reference-to-video). Their **tensor key sets are identical**. Every public acceleration
LoRA is distilled against one of them, so:

> Applying a fl2va accel LoRA to a ref2va base — **no error, no warning, just silently
> wrong output.**

Not "slightly worse": the trunk semantics are completely misaligned. Pruned builds add a
second trap: a full-build dense adaln adapter (input width 2688) on a pruned, curve-form
base (width 8) makes ComfyUI log a wall of `adaln_proj` errors and then **silently drop
that half of the distillation**.

This pack does three things:

1. **Identify the base** — from the model's own weights: fl2va or ref2va, full or pruned
   build, and whether it already carries baked PDD heads.
2. **Classify each file** — reads only the safetensors **header** (no weights, so even a
   2 GB LoRA is scanned in milliseconds): which trunk it was distilled for, dense adaln
   or not, PDD head bank or not.
3. **Filter + prescribe** — keeps only the usable files and attaches the correct
   sampling recipe for that acceleration family.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo>
```

No extra dependencies (uses only ComfyUI's bundled torch / safetensors).

## The four nodes

All nodes live under the `model/minimax` category.

### H3 Accel Base Info (trunk / build identification)

Attach a model, get `trunk` / `build` / `pdd_state` / `info`. Use it whenever you wonder
"what exactly is this base I just downloaded?"

### H3 Accel LoRA Picker (filter against the base and load)

The main node. Model in → pick an accel LoRA → feed the sampler.

- The `accel_lora` dropdown is **filtered live by the frontend against the base**: switch
  the upstream model and the list becomes the usable files automatically. Tick
  `show_incompatible` to also list mismatched files (with the reason).
- `check_mode`: `strict` (default) hard-fails on trunk/build mismatch; `warn` logs and
  continues; `off` skips checks.
- Selecting a **PDD** file automatically **delegates** loading to the
  `BSAI-ComfyUI-MiniMax-H3-PDD-Acc` node (`MiniMax H3 PDD Acc LoRA (Apply)`) and passes
  its **SIGMAS** straight through — wire that into `SamplerCustomAdvanced`'s `sigmas`
  input; no separate PDD node needed.
- Ordinary LoRAs go through ComfyUI's own quantization-aware patch path, equivalent to
  `LoraLoaderModelOnly`.

### H3 Accel Report (acceleration LoRA compatibility report)

Full report: base verdict + per-file verdicts (OK / WARN / MISMATCH) + each file's
sampling recipe. Run this first when choosing.

It also lists files sitting in the LoRA folders that **aren't LoRAs at all** (zero
adapter pairs — usually a full checkpoint dumped in the wrong directory). Feeding a
20 GB checkpoint to a LoRA loader makes for a very bad afternoon.

### H3 Accel Fingerprint Bake (register a custom base)

Your own fine-tuned bases (`minimax_h3_ref2va_XUELUO_int8_convrot`,
`Minimax-h3_Singularity_*`, `PinkCherry_h3_fl2va_*`, ...) aren't in the shipped
fingerprints and would be judged inconclusive. This node reads the base's
`final_layer.video_out` (about 1 MB, via mmap — never loads the whole model), writes an
fp16 fingerprint into the plugin's `fingerprints/`, and after a restart the plugin
recognizes it.

## How the matching works

### Trunk: the final_layer.video_out fingerprint

`final_layer.video_out.weight` stays fp32 (never quantized) in every public build, and is
**bit-identical** across int8_convrot / fp8_scaled / pruned / adaln-rebased variants of
the same trunk; the relative Frobenius distance between the two trunks is about
**0.0503**. So matching is by **distance**, not hash:

| Distance | Meaning |
|---|---|
| ≤ 0.015 | Match (tolerance ≫ conversion/storage noise ~2e-3, ≪ trunk gap 0.0503) |
| > 0.015 | No match → inconclusive (fine-tune? manual merge?) |

Fingerprints come from `fingerprints/` (shipped) — the plugin also **auto-discovers**
`BSAI-ComfyUI-MiniMax-H3-PDD-Acc/partition_fingerprints/` next to it and prefers
whatever it finds there.

### Build: use_adaln_curves

ComfyUI core's `MiniMaxH3Model.use_adaln_curves` is True on pruned (curve-form) builds.
Read directly, no guessing.

### Files: safetensors header only

The header alone contains the decisive facts:

- **Dense adaln input width** — back-solved from `lora_A`/`lora_B` (or Kohya
  `lora_down`/`lora_up`) shapes. 2688 = full-build distill, 8 = curve-form distill.
  Rank and in_features must be judged on the **smaller** dimension, otherwise
  64/16/2 get mistaken for widths.
- **PDD head bank** — top-level `proj_out.weight` / `audio_proj_out.weight` and two
  siblings means it's not an ordinary LoRA; ordinary loaders can't read it.
- **Is it even an H3 file** — `token_refiner` (H3-only), `final_layer.video_out`,
  `attn.qkv_proj` + `mlp.fc1/fc2`. Keeps Krea2 / MOSS-Image / Qwen-Image / LTX turbo and
  4-step LoRAs out of the H3 picker.

## Scan scope

`models/loras/`, `models/pdd_acc/`, `models/veda/`, `models/vdn/`, `models/unet/` —
scanned **recursively**.

> Note: the PDD Acc node only lists `models/pdd_acc/`. If a PDD file lives elsewhere
> (say `models/loras/H3加速lora/ref2va/pdd_acc/`), this plugin can see it, but when it
> delegates, the PDD node will tell you to move it to `models/pdd_acc/`.

## Recipe cheat sheet

| Family | Steps | Sampler | CFG | SigmaShift | Notes |
|---|---|---|---|---|---|
| PDD Acc (official) | 8 / 4 / 6 | euler | 1.0 | 12.0 / 3.0 | sigmas come from the PDD node; don't stack other distills |
| LightX2V Ref2V turbo 4-step v0.1 | 4 | — | 1.0 | 12.0 / 3.0 | strength 1.0 (Kijai suggests 0.75) |
| LightX2V Ref2V turbo 8-step v1.0 768p | 8 | — | 1.0 | 6.0 / 3.0 | |
| larryvrh Turbo v4 (step600 EMA) | 6 | simple | 1.0 | 12.0 / 4–6 | strength 1.0 (community 0.6–0.8) |
| HyperFlow 8-step | 8 | euler/normal | — | — | manual 9-point sigma, below |
| TaoMate 3-step | 3 | euler | 1.0 | — | FL2VA only; CFG > 1.0 burns it |
| FastH3 4-step | 4 | — | — | — | T2V/FLF only; never distilled for Ref2VA |

HyperFlow manual sigma (9 points = 8 intervals):

```
1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0
```

**Universal rules**

- Distill LoRAs **don't stack** — PDD / turbo / HyperFlow, pick one. Character/style
  LoRAs can stack on top.
- Don't stack step-cache (blockcache / EasyCache) — the PDD final_layer patch
  fails closed.
- Fewer steps = weaker reference adherence, most visible on Ref2VA. For tight reference
  adherence run 20–25 steps with turbo off.
- Turbo shortens the schedule, not per-step cost; a 4-step LoRA run for 20 steps costs
  about the same as the 20-step baseline.

## Worked example: `minimax_h3_ref2va_int8_convrot`

A **Ref2VA trunk + INT8 ConvRot full build** (~31.7 GiB, dense adaln 2688).

| Verdict | File |
|---|---|
| ✅ usable | `minimax_h3_ref2va_pdd_acc_8step_comfyui` (PDD 8-step) |
| ✅ usable | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` (4-step) |
| ✅ usable | `minimax_h3_turbo_v4_step600_ema` (6-step, cross-trunk) |
| ✅ usable | `minimax_h3_hyperflow_8step_v1.0` (8-step, cross-trunk) |
| ✅ usable | `minimax_h3_pdmd_2step` / `4step`, `minimax_h3_ELM_longlive_plug_4step` |
| ⚠️ usable but weaker | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui` (pruned conversion, adaln stripped) |
| ❌ rejected | `minimax_h3_fl2va_pdd_acc_8step_comfyui` (FL2VA distill) |
| ❌ rejected | all `minimax_h3_fl2v_turbo_*`, `TaoMate-H3-step3000` (FL2VA only) |
| ❌ rejected | `minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_*` (FL2VA + pruned curve base) |

Run **H3 Accel Report** for the full list against your own base.

## Example workflow

[`example_workflows/BSAI_H3_Film_Factory_直播带货_手动档_ref2va_int8_accel-selector.json`](example_workflows/BSAI_H3_Film_Factory_直播带货_手动档_ref2va_int8_accel-selector.json)
— the BSAI H3 Film Factory live-streaming (manual mode, ref2va int8) workflow with all
four nodes wired in:

- **LoRA Picker** sits inline in the model chain (`ComfySwitchNode → Picker →
  H3LoraStack → Film Factory`): on the `minimax_h3_ref2va_int8_convrot` base it lists
  only ref2va-distilled files. The example picks LightX2V Ref2V turbo 4-step
  (Film Factory steps are set to 4 accordingly), and the upstream `BSAI_SolH3_Loader`
  gets `lora_name = none` so two distillations don't stack.
- **Base Info** and **Report** tap the Switch's MODEL output and print the base verdict
  plus every candidate's OK / WARN / MISMATCH conclusion on each run (shown via
  PreviewAny).
- **Fingerprint Bake** is bypassed by default; un-bypass it once when you switch to a
  custom base to register its fingerprint.

Drag it into ComfyUI and run. The node group carries Chinese notes, including how to
switch back to the PDD 8-step path.

## Layout

```
BSAI-ComfyUI-H3-Accel-Selector/
├── __init__.py              node registration + /h3_accel/scan route
├── nodes.py                 the four nodes
├── h3_accel_core.py         classification engine (pure torch/stdlib, testable standalone)
├── catalog.json             acceleration family rules + recipes
├── fingerprints/            trunk fingerprints (fp16 video_out)
├── web/h3_accel.js          live dropdown filtering
├── example_workflows/       example workflow (H3 Film Factory)
└── tests/test_h3_accel.py   self-test
```

## Tests

```bash
python tests/test_h3_accel.py
# with a real model scan:
set H3_ACCEL_MODELS_DIR=/path/to/ComfyUI/models && python tests/test_h3_accel.py
```

## Known limits

- **Hybrid trunks** (`minimax_h3_hybrid_fl2va_ref2va_b25-49`): the fingerprint cannot
  match by construction; filename rules are all that's left, and PDD on a hybrid base is
  off-label. The plugin always gives WARN on hybrid bases instead of blocking.
- **Veda weights** (`models/veda/`) aren't ordinary LoRAs — sparse-attention replacement
  modules loaded via `BSAI-ComfyUI-VedaSparse`. The plugin identifies them and refuses
  the ordinary LoRA path.
- **FastH3 is a checkpoint, not a LoRA**: `fastvideo_fasth3_8step_v2_pruned_int8_convrot`
  and friends dropped into `models/loras/` get reported as "not a LoRA".
- Fingerprint verdicts are **advisory**: a miss degrades to inconclusive and never
  interrupts a render.
