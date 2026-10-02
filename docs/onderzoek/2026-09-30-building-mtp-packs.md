# Building MTPLX packs with a BF16 MTP head (30 September 2026)

How we built four MTPLX packs whose draft head is the model's own MTP layer in BF16, what went wrong on the way, and how we measured them. Written as a recipe for the next model.

| Pack | Body | MTP head | Outcome |
|---|---|---|---|
| Apodex-1.1-mini Balance | own Forge build from BF16, 6-bit g64 | BF16 from the source | community MLX quants had dropped the MTP layer; now D2 1.72x AR |
| Qwen3.6-35B-A3B Balance, BF16 MTP (`-yb`) | Youssofal Balance body, unchanged | BF16 from `Qwen/Qwen3.6-35B-A3B` | decode +12%, short replies +19%, label scoring identical |
| Qwen3.6-35B-A3B Speed, BF16 MTP (`-yb`) | Youssofal Speed body, unchanged | BF16 from `Qwen/Qwen3.6-35B-A3B` | decode +32%, short replies +23%, label scoring identical |

All numbers: M5 Pro 64 GB, turbo profile, depth 2, compared with the same body carrying its original int4 head. Details in [qwen38](2026-09-30-qwen38.md) (finding 90) and the model cards.

## Why a BF16 head

MTPLX drafts with the model's native MTP layer. Published packs often ship that head quantized to int4 to save ~1 GB. On MoE models this costs a lot of acceptance: Qwen3.6 Balance went from 0.77 / 0.42 (int4) to 0.91 / 0.72 (BF16) mean acceptance at D2. Speculative decoding with exact acceptance keeps the output distribution, so a better head is pure speed; the only cost is memory (+0.6 to +1.1 GB).

Check first what the pack really ships: the `mtp_policy` label is not enough. The Qwen3.8-27B Optimized-Speed pack says `keep_bf16` in `mtplx_runtime.json`, but its `mtp.safetensors` holds a 4-bit g64 head (239 MB, `U32` payload with scales, `mtplx_mtp_quantization.prequantized: true` in `config.json`). Look at the size of `mtp.safetensors` and its dtypes. On that dense model a BF16 head did not help (2 to 8% slower, equal acceptance; see [qwen38-speed-options](2026-10-02-qwen38-speed-options.md)), so the gain from a BF16 head is model-specific: measure it.

## Step 1: find a source that still has the MTP layer

- The BF16 release of the model usually has it; MLX community quants often drop it ("MTP layer dropped in this build").
- Check without downloading the weights: fetch `model.safetensors.index.json` and count keys starting with `mtp.`, or run `mtplx forge probe <repo>`. "BF16/native MTP source detected; Forge will convert the body and preserve MTP" is what you want.
- Note the expert layout of the MTP layer. Numbered experts (`mtp.layers.0.mlp.experts.N.gate_proj.weight`, 785 keys for a 256-expert layer) load directly. Fused experts (`experts.gate_up_proj`, `experts.down_proj`, 19 keys in total) need step 3 on MTPLX 2.12.0.

## Step 2: Forge build

Recipe as a JSON file, mirroring the target pack (here Balance: 6-bit g64 body, router and shared-expert gates in 8-bit; Speed: `body_bits 4`):

```json
{
  "body_bits": 6,
  "body_group_size": 64,
  "body_mode": "affine",
  "mtp_policy": "keep_bf16",
  "module_overrides": [
    {"suffix": "mlp.gate", "bits": 8, "group_size": 64},
    {"suffix": "mlp.shared_expert_gate", "bits": 8, "group_size": 64}
  ]
}
```

```bash
mtplx forge build --repo <hf-repo or local dir> --out <runs dir> --run-id <id> \
  --recipe "$(python3 -c 'import json;print(json.dumps(json.load(open("recipe.json"))))')" \
  --branded-name <Pack-Name>
```

- `--recipe` takes inline JSON or a preset name, not a file path.
- The download goes to the model root as `<org>--<name>`; a local directory as `--repo` skips the download.
- On an M5 Pro 64 GB: download 72 GB at 35 MB/s, conversion a few minutes, then Forge verifies with `mtplx tune` on a free port. Nothing else may use the GPU meanwhile.
- Check the result in `<runs>/<id>/verify.json` and `forge.json` (`mtp_contract_calibration`). A calibration that reports very few accepted draft tokens per round (0.36 for us, against 1.80 for a healthy head) means the head did not load properly; see step 3.

## Step 3: fused MTP experts (MTPLX 2.12.0)

On 2.12.0 a sidecar with fused experts loads without error, but the routed experts keep their random init: `_finalize_mtp_weights` only stacks numbered experts and `load_weights(strict=False)` drops the fused keys. Fixed upstream in [youssofal/MTPLX#574](https://github.com/youssofal/MTPLX/pull/574). Until that ships, split the sidecar into numbered experts (gate is the first half of `gate_up_proj`; hub layout `[E, 2*inter, hidden]`):

```python
import mlx.core as mx
w = mx.load("mtp.safetensors"); out = {}
for k, v in w.items():
    if k.endswith("mlp.experts.gate_up_proj"):
        pre, half = k[: -len("gate_up_proj")], v.shape[1] // 2
        for e in range(v.shape[0]):
            out[f"{pre}{e}.gate_proj.weight"] = v[e, :half, :]
            out[f"{pre}{e}.up_proj.weight"] = v[e, half:, :]
    elif k.endswith("mlp.experts.down_proj"):
        pre = k[: -len("down_proj")]
        for e in range(v.shape[0]):
            out[f"{pre}{e}.down_proj.weight"] = v[e]
    else:
        out[k] = v
mx.eval(out)
mx.save_safetensors("mtp.safetensors", out, metadata={"format": "mlx"})
```

Keep the original sidecar; a split pack works with and without the fix.

## Step 4: runtime metadata

Forge writes `mtplx_runtime.json` without a served model id or recommended depth. Set both, so a server started with only `--model <dir>` serves a stable id (a launcher that checks `/health` against a configured id needs this) and the right depth:

```python
import json
r = json.load(open("mtplx_runtime.json"))
r["public_model_id"] = "mtplx-<name>"
r["recommended_mtp_depth"] = 2
json.dump(r, open("mtplx_runtime.json", "w"), indent=2)
```

## Step 5 (optional): keep an existing body, swap only the head

A Forge body quantized from BF16 is not bit-identical to a published pack's body. We saw it on label scoring (first-token probabilities over fixed labels): 1 of 60 prompts differed at 6-bit, 3 of 60 at 4-bit. To get the new head without any change to the body, combine:

1. Clone the Forge pack from step 2 to 4 (APFS clone, no extra disk): `cp -cR <forge-pack> <combo-pack>`.
2. Remove its body: `model-*-of-*.safetensors`, `model.safetensors.index.json`, and `model-vision.safetensors` if the original pack keeps the vision tower inside its shards (check the index).
3. Clone in the original pack's `model-*.safetensors` and `model.safetensors.index.json`.
4. Keep the Forge pack's `config.json` (it describes the BF16 head; compare the `quantization` blocks first, they must match apart from notation) and `mtp.safetensors`; set a new `public_model_id`.

Result: label scoring identical to the original pack, decode speed from the new head. Credit the body's author on the model card.

## Step 6: measure

- `mtplx tune --model <dir> --depths 1,2,3 --retune --no-save --json --output <file>`: AR and D1 to D3 with acceptance per position, no server needed (code suite, thinking off, optimistic).
- Server workload on the same build for both packs: decode over 256 tokens, short replies, mean acceptance from `/v1/mtplx/snapshot` (`latest`), label scoring, and an agent task with real tools. Start a fresh server per variant.
- Compare against the same body with the old head, not across bodies.

## Pitfalls we hit

- One GPU, 64 GB: never load a second model while a server runs, and watch swap. A variant with 8,192-token prefill chunks on a dense 27B model pushed the machine into 9.4 GB of swap and slowed every later measurement for hours.
- Anything that can call the server (a platform, chat titles) disturbs measurements on the same port; point it elsewhere during a run, and check `requests_completed` in `/health`.
- A task runner with a hard time limit can cut off a slow local model right after it finished, and a blind retry can overwrite the result.
- `mtplx forge publish --path <dir> --repo <owner/name> --visibility public --license apache-2.0 --readme-path <card> --token stdin` uploads a pack; remove backup files (`*.orig`) from the pack directory first.
