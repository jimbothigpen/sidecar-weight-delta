cmake_minimum_required(VERSION 3.18)
project(sidecar_weight_delta CXX)

#
# Standalone out-of-tree sidecar handler plugin: rank-r per-tensor weight
# deltas (Heretic-style rank-1 surgery, Abliterix, generic LoRA-shaped
# weight perturbations). Loaded as
#     --sidecar-load-plugin libsidecar_weight_delta.so
#     --sidecar-vectors my_delta.wd.gguf
#
# The .wd.gguf format is dual-schema: it carries both the standard LoRA
# adapter metadata + tensor naming AND a `sidecar.type = "weight_delta"`
# tag for plugin dispatch. The plugin reuses the engine's
# llama_adapter_lora_init() to load the adapter and registers it via the
# new apply_to_weights() hook so build_lora_mm() picks it up at every
# matmul site in the forward graph.
#
# Builds against an installed frankenturbo2 engine; produces
# libsidecar_weight_delta.so.
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(LLAMA_INSTALL_PREFIX "/opt/llama-frankenturbo2-vulkan"
    CACHE PATH "frankenturbo2 install prefix (contains include/ and lib/)")

add_library(sidecar_weight_delta SHARED weight_delta.cpp)

target_include_directories(sidecar_weight_delta PRIVATE
    "${LLAMA_INSTALL_PREFIX}/include")

find_library(LLAMA_LIB      llama      PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB       ggml       PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB  ggml-base  PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

target_link_libraries(sidecar_weight_delta PRIVATE
    ${LLAMA_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${LLAMA_INSTALL_PREFIX}" CACHE PATH "" FORCE)
endif()

install(TARGETS sidecar_weight_delta
        LIBRARY DESTINATION lib/sidecars)
