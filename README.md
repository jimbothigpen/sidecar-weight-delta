# sidecar-weight-delta

Out-of-tree [frankenturbo2](https://github.com/jimbothigpen/frankenturbo2) sidecar
handler plugin that registers `weight_delta` as a sidecar type. Applies
rank-r per-tensor weight perturbations (Heretic-style rank-1 surgery,
Abliterix, generic LoRA-shaped weight deltas) to a loaded base model at
attach time, without re-quantizing the underlying weights.

## How it works

A `.wd.gguf` file is dual-schema: it satisfies both the engine's existing
LoRA adapter format AND carries a `sidecar.type = "weight_delta"` tag for
plugin dispatch.

- The plugin's `load()` validates `wd.arch` against the model and stashes
  the path.
- The plugin's `apply_to_weights()` calls
  `llama_adapter_lora_init(model, path)` and returns the resulting
  adapter to the engine.
- The engine inserts the adapter into `ctx->loras`; the existing
  `build_lora_mm()` graph helper picks it up at every matmul site, so no
  new graph code is needed.

## On-disk schema

```
sidecar.type            str    "weight_delta"
wd.arch                 str    target arch        (must match the model's general.architecture)
general.type            str    "adapter"          (parsed by llama_adapter_lora_init)
general.architecture    str    target arch        (parsed by llama_adapter_lora_init)
adapter.type            str    "lora"
adapter.lora.alpha      f32    LoRA alpha
tensors:                       <base_name>.lora_a [d_in, r]
                               <base_name>.lora_b [r, d_out]
```

## Build

```bash
cmake -S /usr/src/llama-forks/sidecar-weight-delta \
      -B /home/builduser/sidecar-builds/sidecar-weight-delta \
      -DLLAMA_INSTALL_PREFIX=/opt/llama-frankenturbo2-vulkan
cmake --build /home/builduser/sidecar-builds/sidecar-weight-delta -j12
sudo cmake --install /home/builduser/sidecar-builds/sidecar-weight-delta \
            --prefix /opt/llama-frankenturbo2-vulkan
sudo cmake --install /home/builduser/sidecar-builds/sidecar-weight-delta \
            --prefix /opt/llama-frankenturbo2-rocm
```

## Use

```bash
llama-cli --sidecar-load-plugin /opt/llama-frankenturbo2-vulkan/lib/sidecars/libsidecar_weight_delta.so \
          --sidecar-vectors /path/to/your.wd.gguf \
          -m base_model.gguf -p "..."
```

## Producing a `.wd.gguf`

See the sibling `heretic-to-sidecar` repo for the Heretic-driven flow.
For arbitrary peft LoRA adapters, run llama.cpp's
`convert_lora_to_gguf.py` against the adapter dir, then post-process the
output to add the `sidecar.type` and `wd.arch` KVs (see
`heretic-to-sidecar/scripts/peft_to_wd_gguf.py` for an implementation).
