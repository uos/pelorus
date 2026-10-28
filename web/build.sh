#!/usr/bin/env bash
# Build the browser demo into this folder: the wasm module in pkg/ and the
# configuration presets in configs/. Serve the folder with any static server,
# for example: python3 -m http.server -d web
#
# Needs the wasm32-unknown-unknown target, clang with WebAssembly support for
# the zstd and lz4 C code, and wasm-bindgen-cli matching the wasm-bindgen crate
# (cargo install wasm-bindgen-cli --version 0.2.129). web/deploy.sh uploads the
# result to pelorus.steado.tech.
set -euo pipefail

web="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$web")"
wasm_bindgen="${WASM_BINDGEN:-wasm-bindgen}"

cargo build --manifest-path "$root/Cargo.toml" -p pelorus_web --release --features webgpu \
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

# Share the documentation logo with the browser demo, and its tab icons with
# the script that picks one for the browser's colour scheme.
cp "$root/docs/docs/images/pelorus.svg" "$web/pelorus.svg"
cp "$root"/docs/docs/images/favicon{,-light,-dark}.svg "$root/docs/docs/javascripts/favicon.js" "$web/"

# Libron, the typeface of the documentation, with its licence.
mkdir -p "$web/fonts"
cp "$root"/docs/docs/fonts/Libron-*.woff2 "$root/docs/docs/fonts/OFL.txt" "$web/fonts/"

# The configuration presets come from a pelorus-configs checkout next to
# pelorus (https://codeberg.org/stelzo/pelorus-configs), or from PELORUS_CONFIGS.
configs="${PELORUS_CONFIGS:-$root/../pelorus-configs}"
mkdir -p "$web/configs"
for preset in oxford-spires grandtour helipr-ouster newer-college; do
  cp "$configs/$preset.rl" "$web/configs/"
done

if [[ ! -f "$web/demo/keble_college_02_30s.mcap" ]]; then
  echo "The demo bag is missing. Create it with web/make-demo.sh." >&2
fi
