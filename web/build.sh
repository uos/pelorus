#!/usr/bin/env bash
# Build the browser demo into this folder: the wasm module in pkg/ and the
# configuration presets in configs/. Serve the folder with any static server,
# for example: python3 -m http.server -d web
#
# Needs the wasm32-unknown-unknown target, clang with WebAssembly support for
# the zstd and lz4 C code, and wasm-bindgen-cli matching the wasm-bindgen crate
# (cargo install wasm-bindgen-cli --version 0.2.129).
set -euo pipefail

web="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$web")"
wasm_bindgen="${WASM_BINDGEN:-wasm-bindgen}"

cargo build --manifest-path "$root/Cargo.toml" -p pelorus_web --release \
  --target wasm32-unknown-unknown
"$wasm_bindgen" --target web --out-dir "$web/pkg" \
  "$root/target/wasm32-unknown-unknown/release/pelorus_web.wasm"

# The ratslang grammar of the VS Code extension highlights the config editor.
grammar="$root/../ratslang/vscode/syntaxes/rl.tmLanguage.json"
if [[ -f "$grammar" ]]; then
  mkdir -p "$web/syntaxes"
  cp "$grammar" "$web/syntaxes/ratslang.tmLanguage.json"
else
  echo "No ratslang checkout next to pelorus. The config editor stays plain." >&2
fi

# Libron, the typeface of the documentation, with its licence.
mkdir -p "$web/fonts"
cp "$root"/docs/docs/fonts/Libron-*.woff2 "$root/docs/docs/fonts/OFL.txt" "$web/fonts/"

mkdir -p "$web/configs"
for preset in pelorus-spires pelorus-grandtour pelorus-helipr-ouster pelorus-ouster pelorus-newer-college pelorus-velodyne \
  pelorus-helipr-avia pelorus-blickfeld; do
  cp "$root/$preset.rl" "$web/configs/"
done

if [[ ! -f "$web/demo/keble_college_02_30s.mcap" ]]; then
  echo "The demo bag is missing. Create it with web/make-demo.sh." >&2
fi
