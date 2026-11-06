#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
pin=b7c70275f04126c647edb989270cc429776955d1
upstream=https://github.com/CMU-SAFARI/ramulator2.git
backend=ramulator2
overlay=pim_ramulator_src

if [[ ! -f "$backend/.tetris-overlay" ]]; then
    if [[ -e "$backend" ]]; then
        echo "Move the existing $backend directory aside before building." >&2
        exit 1
    fi
    git init -q "$backend"
    git -C "$backend" fetch --depth=1 "$upstream" "$pin"
    git -C "$backend" checkout -q --detach FETCH_HEAD

    cp "$overlay"/*.yaml "$backend/"
    cp "$overlay"/hbm3{,_pim}_linear_mappers.cpp "$backend/src/addr_mapper/impl/"
    cp "$overlay/HBM3-PIM.cpp" "$backend/src/dram/impl/"
    cp "$overlay"/hbm3{,_pim}_controller.cpp "$backend/src/dram_controller/impl/"
    cp "$overlay/hbm3_trace_recorder.cpp" "$backend/src/dram_controller/impl/plugin/"
    cp "$overlay"/{all_bank_refresh_hbm3,no_refresh}.cpp "$backend/src/dram_controller/impl/refresh/"
    cp "$overlay/pim_scheduler.cpp" "$backend/src/dram_controller/impl/scheduler/"
    cp "$overlay/pim_loadstore_trace.cpp" "$backend/src/frontend/impl/memory_trace/"
    cp "$overlay/PIM_DRAM_system.cpp" "$backend/src/memory_system/impl/"
    cp -r "$overlay/trace_gen" "$backend/"
    for patch_file in "$overlay"/patches/*.patch; do
        patch --batch --forward -d "$backend" -p1 < "$patch_file"
    done
    printf '%s\n' "$pin" > "$backend/.tetris-overlay"
fi
[[ $(git -C "$backend" rev-parse HEAD) == "$pin" ]]
# The pinned revision relies on a transitive declaration of uint64_t.
if ! grep -q '^#include <cstdint>' "$backend/src/base/utils.h"; then
    sed -i '/^#include <vector>/a #include <cstdint>' "$backend/src/base/utils.h"
fi
cmake -S "$backend" -B "$backend/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$backend/build" --parallel "${JOBS:-2}"