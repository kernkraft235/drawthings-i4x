#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p build/tmp build/module-cache dist
export TMPDIR="$PWD/build/tmp"
export CLANG_MODULE_CACHE_PATH="$PWD/build/module-cache"
export MACOSX_DEPLOYMENT_TARGET=13.0
target="$(uname -m)-apple-macosx13.0"
native=src/dt_i4x/native
clang -target "$target" -O3 -fno-math-errno -fno-trapping-math -ffp-contract=off -fblocks -DUSE_DISPATCH -c "$native/ccv_nnc_8i_rowwise.c" -o build/quant.o
clang -target "$target" -O3 -c native_cli/bridge.c -o build/bridge.o
clang -target "$target" -O3 -I "$native" -c "$native/depalettize.c" -o build/palette.o
swiftc -target "$target" -O -module-cache-path "$PWD/build/module-cache" -import-objc-header native_cli/bridge.h native_cli/main.swift build/quant.o build/bridge.o build/palette.o -lsqlite3 -lz -o dist/dt-i4x
if [ "${1:-}" = "--build-only" ]; then exit 0; fi
if [ "$#" -eq 0 ]; then set -- --help; fi
exec ./dist/dt-i4x "$@"
