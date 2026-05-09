// Standalone out-of-tree sidecar plugin: rank-r per-tensor weight deltas.
//
// Loaded into a frankenturbo2 / llama.cpp engine via:
//
//     llama-cli --sidecar-load-plugin /path/to/libsidecar_weight_delta.so \
//               --sidecar-vectors /path/to/your.wd.gguf
//
// Semantics: applies a rank-r weight perturbation W' = W + B @ A to each
// targeted weight tensor, where (A, B) live in the .wd.gguf as
// `<tensor_name>.lora_a` / `<tensor_name>.lora_b`. Heretic-style rank-1
// surgery is the trivial r=1 case (B is [d_out, 1], A is [1, d_in]).
//
// Implementation: the .wd.gguf is dual-schema — it satisfies both the
// engine's existing LoRA adapter format AND carries a sidecar.type tag so
// the plugin loader dispatches here. apply_to_weights() delegates to
// llama_adapter_lora_init(), then hands the adapter to the engine via
// the apply_to_weights return value. The engine wires the adapter into
// ctx->loras so build_lora_mm() picks it up at every matmul site.
//
// On-disk schema:
//   sidecar.type            str    "weight_delta"          (plugin dispatch tag)
//   wd.arch                 str    target arch              (matches model arch; redundant with general.architecture but checked early)
//   general.type            str    "adapter"                (required by llama_adapter_lora_init)
//   general.architecture    str    target arch              (required by llama_adapter_lora_init)
//   adapter.type            str    "lora"                   (required)
//   adapter.lora.alpha      f32    LoRA alpha               (required; emitter sets to rank when no rescaling desired)
//   tensors: <base>.lora_a, <base>.lora_b   any compatible dtype, shapes [d_in, r] and [r, d_out]

#include <llama-sidecar-plugin.h>

#include <ggml.h>
#include <gguf.h>

#include <cstdarg>
#include <cstdio>
#include <string>

namespace {

inline void log_err(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

inline void log_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

// Family-prefix arch match — see sidecar-control-vector for the rationale
// (HF transformers configs sometimes carry an attention-variant suffix that
// llama.cpp's GGUF arch tag doesn't, e.g. "gemma4-iswa" vs "gemma4").
bool model_arch_matches(const llama_model & model, const std::string & wd_arch) {
    char buf[256] = {0};
    const int n = llama_model_meta_val_str(&model, "general.architecture", buf, sizeof(buf));
    if (n < 0) {
        log_err("weight_delta: model has no 'general.architecture' meta\n");
        return false;
    }
    const std::string model_arch = buf;
    if (wd_arch == model_arch) return true;
    if (wd_arch.rfind(model_arch + "-", 0) == 0) {
        log_info("weight_delta: arch '%s' accepted as variant of model arch '%s'\n",
                 wd_arch.c_str(), model_arch.c_str());
        return true;
    }
    if (model_arch.rfind(wd_arch + "-", 0) == 0) {
        log_info("weight_delta: arch '%s' accepted as variant of model arch '%s'\n",
                 wd_arch.c_str(), model_arch.c_str());
        return true;
    }
    return false;
}

// Multiply-inherits from llama_sidecar_handler (base ABI) and
// llama_sidecar_handler_weights (the optional weight-modification interface
// added 2026-05-09). The engine probes for the weights interface via
// dynamic_cast at sidecar attach; plugins that don't implement it (including
// old binaries built against pre-2026-05-09 headers) are transparently
// skipped for the apply_to_weights dispatch.
struct weight_delta_handler
    : public llama_sidecar_handler
    , public llama_sidecar_handler_weights {
    std::string type() const override { return "weight_delta"; }

    bool load(
            const llama_model & model,
            gguf_context * gguf,
            ggml_context * /*ctx_meta*/,
            const std::string & path,
            float scale_override,
            float /*threshold_override*/) override {

        // Validate the on-disk wd.* metadata before delegating to the LoRA
        // loader. The LoRA loader will also check general.type/adapter.type
        // when apply_to_weights() runs; this early check just gives a
        // friendlier error if the file lacks our plugin's required tags.
        std::string wd_arch_str;
        {
            const int id = gguf_find_key(gguf, "wd.arch");
            if (id < 0) {
                log_err("weight_delta: missing required key 'wd.arch'\n");
                return false;
            }
            if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
                log_err("weight_delta: 'wd.arch' must be a string\n");
                return false;
            }
            wd_arch_str = gguf_get_val_str(gguf, id);
        }
        if (!model_arch_matches(model, wd_arch_str)) {
            log_err("weight_delta: arch '%s' does not match model arch\n",
                    wd_arch_str.c_str());
            return false;
        }

        sidecar_path = path;
        scale = (scale_override == 0.0f) ? 1.0f : scale_override;

        log_info("weight_delta: validated arch='%s' scale=%.4f path='%s' (deferring tensor load to apply_to_weights)\n",
                 wd_arch_str.c_str(), scale, path.c_str());
        return true;
    }

    std::vector<std::pair<llama_adapter_lora *, float>> apply_to_weights(
            llama_model & model,
            llama_context * /*lctx*/,
            const std::string & /*path*/) override {

        // Delegate the actual tensor parsing + backend allocation to the
        // engine's existing LoRA adapter loader. The .wd.gguf is dual-schema
        // (both LoRA-format AND sidecar.type-tagged), so this re-opens the
        // file and parses it as a LoRA adapter. The cost is one extra
        // gguf_init_from_file pass on a small (few-MB) file at attach time;
        // negligible compared to the model load itself.
        llama_adapter_lora * adapter = llama_adapter_lora_init(&model, sidecar_path.c_str());
        if (adapter == nullptr) {
            log_err("weight_delta: llama_adapter_lora_init failed for '%s'\n",
                    sidecar_path.c_str());
            return {};
        }
        log_info("weight_delta: registered LoRA-shaped adapter from '%s' at scale=%.4f\n",
                 sidecar_path.c_str(), scale);
        return { { adapter, scale } };
    }

private:
    std::string sidecar_path;
    float       scale = 1.0f;
};

} // namespace

LLAMA_SIDECAR_PLUGIN_INIT_DECL {
    llama_sidecar_register(
        "weight_delta",
        []() -> llama_sidecar_handler_ptr {
            return std::make_shared<weight_delta_handler>();
        });
    return 0;
}
