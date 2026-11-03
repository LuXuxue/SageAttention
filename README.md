# SageAttention-HIP

A [SageAttention](https://github.com/thu-ml/SageAttention) fork for **AMD RDNA2/RDNA3 on Windows**:
native HIP/WMMA attention kernels for `gfx103x` and `gfx110x`, a stable-ABI wheel layout, and a set of
tuning changes measured on an **RX 7800 XT (`gfx1101`, 60 CU)**.

Drop-in replacement for `sageattention.sageattn` — same Python API, no Triton required for the
`gfx110x` path, and it is what `--use-sage-attention` uses in ComfyUI.

*Maintained by [@aksugat](https://github.com/aksugat). Based on the HIP port by
[@LuXuxue](https://github.com/LuXuxue), which builds on the Windows/ABI3 work by
[@woct0rdho](https://github.com/woct0rdho) and the original SageAttention kernels by
[thu-ml](https://github.com/thu-ml/SageAttention).*

---

## What this fork changes

Everything below was measured on a 7800 XT (ROCm 10.3, torch 2.16.0a0+rocm10.3). Methodology: same
shapes, clock-warmed, 6 alternating measurements per configuration; "best" is the median of those.

| Change | Effect |
|---|---|
| `INT8_BM128_THR` default 2048 → **16384** (BM=128 stages V in LDS, but on 60 CUs it only wins from ~16k up) | kv=12288: **45.70 → 41.68 ms (−8.8%)**; kv=16384 keeps the faster BM=128 kernel |
| D128 direct fp16/bf16 `BLOCK_M` 64 → **128** (the BM=64 instantiation needs 226 VGPRs and spills 48 B of scratch) | 4096/128/fp16: **4.87 → 4.66 ms** |
| Cross-attention `D128_CROSS` 0 → **8192**: direct wins when `q_len*2 < kv_len` (no quantization prepass, V_T stays cache-resident) | q=512 kv=4096: **1.33 → 0.97 ms (−27%)**; q=1024 kv=8192: 3.11 → 2.76 ms (−11%) |
| `load_i8_frag`: one 16 B LDS read instead of four scalar int loads (which compiled to two `ds_load_2addr_b32`) | part of the 16k win below |
| Direct path: K row stride `HeadDim+16` → `+8` halfs → 2-way instead of 4-way LDS bank conflicts | direct path, small kv |
| Epilogue: dtype-independent pack + register select instead of an `hw*4` indexed local array (was 48 B private scratch + 48 scratch ops) | all paths |
| Direct path: `col >= kv_len` hoisted out of the 8-element loop | direct path |
| Dispatch defaults for `gfx1101`: D128 6144, D64 causal 8192, D128 cross 8192 (see the knob table) | best path picked without any env vars |
| End-to-end result at 16384 tokens, 24 heads, bf16 | **82.4 → 78.8 ms (−4.5%)** |

Same numerics as before: relL2 vs a float32 reference is **1.2e-2** on the int8 paths and **3e-4** on
the direct paths, identical to upstream.

## Requirements

- Windows 10/11 with the **ROCm SDK** (`rocm-sdk init` / `rocm-sdk path --root`) or a ROCm-enabled PyTorch
- Python ≥ 3.10 (the wheel uses the stable ABI, `cp310-abi3`)
- An RDNA2 or RDNA3 GPU: `gfx103x` (Radeon 6000) or `gfx110x` (Radeon 7000 / 780M)
- Optional: Ninja + MSVC (only to build from source)

## Install

### From source on Windows (recommended)

```bat
git clone https://github.com/aksugat/SageAttention-HIP.git
cd SageAttention-HIP
scripts\sagehip-build.bat
```

`scripts\sagehip-build.bat` builds for the arch set at the top of the file (`GPU_ARCH=gfx1101` by
default), installs into the Python it points at (`PY=H:\rocm313\python_embeded\python.exe` by default)
and clears the import cache. The first build is slow; later ones reuse `build/` and take seconds.

Or do it by hand:

```bat
set "GPU_ARCHS=gfx1101"
set "CMAKE_GENERATOR=Ninja"
python -m pip install . --no-build-isolation --no-deps --force-reinstall
```

Set `GPU_ARCHS` to the arch of your card (`gfx1100`/`gfx1101`/`gfx1102`/`gfx1030`/`gfx1031`/`gfx1032`).

### Use

```python
import torch
from sageattention import sageattn

q = torch.randn(1, 24, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = sageattn(q, k, v, tensor_layout="HND", is_causal=False)  # [1, 24, 4096, 128]
```

* `tensor_layout="HND"` → `[B, H, N, D]`; `"NHD"` → `[B, N, H, D]` (strides must keep `D` contiguous).
* `head_dim` must be **64 or 128**.
* `is_causal=True` only for `qo_len == kv_len` (video models).
* `return_lse=True` returns a zero placeholder — LSE is not computed by these kernels.
* No attention-mask argument: for masked attention keep using ComfyUI's PyTorch path or wrap it
  outside of SageAttention.

In ComfyUI: launch with `--use-sage-attention`.

## Performance (RX 7800 XT, `gfx1101`, 60 CU)

End-to-end `sageattn`, bf16/fp16, no env overrides, 1 batch:

| shape (B×H×S×D) | dtype | causal | Sage (ms) | torch SDPA (ms) | speedup |
|---|---|---|---|---|---|
| 1×16×2048×64 (SDXL 2k) | fp16 | | 0.43 | 13.4 | ~31× |
| 1×16×4096×64 (SDXL 4k) | fp16 | | 1.53 | 49.9 | ~33× |
| 1×16×4096×77×64 (SDXL cross) | fp16 | | 0.047 | 1.66 | ~35× |
| 1×24×2048×128 | bf16 | | 1.16 | 28.4 | ~24× |
| 1×24×4096×128 | bf16 | | 4.42 | 105.1 | ~24× |
| 1×24×8192×128 | bf16 | | 17.6–20.3 † | — | |
| 1×24×12288×128 | bf16 | | 41.4 | — | |
| 1×24×16384×128 | bf16 | | 78.8 | — | |
| 1×24×8192×128 | bf16 | ✓ | 9.84 | — | |
| 1×24×16384×128 | bf16 | ✓ | 42.9 | — | |
| 1×24×1024×8192×128 (cross) | bf16 | | 3.14 | — | |

† The 8k shape is bimodal on this card (17.6 vs 20.3 ms depending on the memory clock state), so treat
single measurements there as ±10%.

Notes:

* SDPA baseline was taken with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset (the default in the
  usual launcher). With AOTriton enabled the baseline gets much better and the speedup shrinks.
* `comfy-kitchen`'s HIP int8 attention is still ~10–30% faster than this build at 12k–16k, because it
  also quantizes the P·V product (int8 PV). Closing that is the main item in the roadmap.

## Accuracy

relL2 versus a chunked float32 reference on the same inputs:

| path | relL2 | max abs |
|---|---|---|
| direct fp16/bf16 (small kv, cross, ≤ D128 4k) | 2.6e-4 … 1.7e-3 | ≤ 1e-3 |
| int8 QK + fp16 PV (kv ≥ 6144) | 1.1e-2 … 1.2e-2 | ≤ 5e-3 |

That matches upstream SageAttention: the int8 QK path trades ~1% relative error for the throughput.

## Tuning knobs

All of these are optional environment variables. The defaults below are what this fork is tuned to;
they work unmodified on `gfx1101`, and the `gfx103x` path keeps its own defaults.

| Variable | Default | Meaning |
|---|---|---|
| `SAGEATTN_BACKEND` | `native` | `native` = HIP/WMMA kernels, `triton` = upstream Triton path. Read at import. |
| `SAGEATTN_DIRECT_THRESHOLD_D128` | `6144` | use the non-quantized direct kernel up to this kv length (D=128) |
| `SAGEATTN_DIRECT_THRESHOLD_D128_CROSS` | `8192` | same, for `q_len * 2 < kv_len` (0 = always int8) |
| `SAGEATTN_DIRECT_THRESHOLD_D64` | `2048` HND / `3072` NHD | direct/int8 split for D=64 self-attention |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL` | `8192` | direct kernel limit for D=64 causal |
| `SAGEATTN_DIRECT_THRESHOLD_D64_CROSS` | `0` | D=64 cross-attention |
| `SAGEATTN_INT8_BM128_THR` | `16384` | kv length from which int8 uses `BLOCK_M=128` (LDS-staged V). Values are tokens. |
| `SAGEATTN_INT8_BN128` | auto (`32` below 5k, `64` above) | force int8 `BLOCK_N` (16/32/64/128; ignored while BM=128 is active) |
| `SAGEATTN_INT8_WPE` | `1` | occupancy variant. D=128: `1`/`2` → `wpe2_t`, `4` → `wpe4_t` |
| `SAGEATTN_FP16_BM` / `SAGEATTN_BF16_BM` | D128 `128`, D64 `64` | direct-path `BLOCK_M` (`32`/`64`/`128`) |
| `SAGEATTN_FP16_BN` / `SAGEATTN_BF16_BN` | `16` | direct-path `BLOCK_N` (`16`/`32`/`64`/`128`) |
| `SAGEATTN_INT8_32` | `1` | D=64: 32 query rows per warp |
| `SAGEATTN_D64_32W_BM`, `SAGEATTN_D64_BN`, `SAGEATTN_D64_32W_WPE`, `SAGEATTN_D64_X_BN` | `128`, `32`, `1`, `16` | D=64 int8 tile variants (`D64_32W_BM` must stay ≤ 128) |
| `SAGEATTN_QUANT_BLK`, `SAGEATTN_QUANT_GPB`, `SAGEATTN_VT_GRID` | auto | quantization prepass tiling / V-transpose grid |

## How the kernels work

`csrc/qattn/attn_gfx110x.cu` (RDNA3) and `attn_gfx103x.cu` (RDNA2) implement two paths:

* **Direct** — fp16/bf16 WMMA (`v_wmma_f32_16x16x16_f16` / `_bf16`), QK and PV both in the input
  dtype, one warp per 16 query rows. Used for small kv and cross-attention.
* **int8 QK + fp16 PV** — per-block quantization prepass (`quant_qk_int8`), QK on
  `v_wmma_i32_16x16x16_iu8`, softmax in fp32, PV on fp16 WMMA with a hand-assembled P fragment. V is
  read from a transposed `V_T[B,H,D,N]` copy so the PV operand is contiguous; for `BLOCK_M=128` the
  V tile is staged in LDS instead of being re-read per warp.

Both are wave32, keep K in LDS (padded against bank conflicts) and double-buffer the K tile through
registers. `gfx1101` has no `cp.async`, so prefetching is register-based.

## Roadmap

- [ ] **int8 PV** — quantize P (fixed `1/127`, it is in `[0,1]` after `exp2`) and V_T (per-`d` scale,
      produced for free inside the V-transpose kernel), accumulate in int32, dequantize once in the
      epilogue. Projected ~1.3× at 12k–16k; needs an accuracy pass.
- [ ] 256-query-row CTA (fewer K/V reloads per query row), currently 64/128.
- [ ] Double-buffered `v_tile` for the BM=128 path.
- [ ] `VTileStride` bank-conflict rework (72 halfs needs 16 B split of the 32 B V reads first).
- [ ] Remove one of the two `__syncthreads` per KV tile via a ping-pong `k_tile`.
- [ ] Reverse causal block order (cheapest CTAs first).

Contributions and measurements on other RDNA3 cards are welcome — please include the exact shape,
`rocm-sdk` / torch version and a baseline.

## Credits & license

* [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — original algorithm and CUDA kernels
* [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention) — Windows wheels, stable ABI, build plumbing
* [LuXuxue/SageAttention](https://github.com/LuXuxue/SageAttention) — HIP `gfx103x`/`gfx110x` port, dispatch and knobs

Apache-2.0, see [LICENSE](LICENSE).

