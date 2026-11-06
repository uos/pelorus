#!/usr/bin/env bash
# Upload the browser demo to https://pelorus.steado.tech, or check what is
# served there. The CI runs this through .forgejo/workflows/web.yml.
#
#   web/deploy.sh [--dry-run] site          the files of web/ after web/build.sh
#   web/deploy.sh [--dry-run] bags <folder> the demo bags, found anywhere below <folder>
#   web/deploy.sh check                     fetch every file the page loads
#
# The two uploads are independent: the site upload never touches demo/*.mcap,
# docs/, which docs.yml uploads, or bench/, which the pelorus-bench CI uploads, and the bag upload touches nothing else. Before uploading, both check that
# every file main.js and benchmark.js reference is there, so a missing preset or
# bag fails here and not in a visitor's browser. --dry-run lists the changes
# without making them.
#
# The server only accepts rsync into the site folder from the pelorus-web user
# (rrsync in its authorized_keys). DEPLOY_KEY names the private key file;
# without it ssh uses its own configuration.
set -euo pipefail

web="$(cd "$(dirname "$0")" && pwd)"
host="${DEPLOY_HOST:-pelorus.steado.tech}"
target="pelorus-web@$host:"
url="https://$host"

dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=(--dry-run)
  shift
fi
mode="${1:?Usage: web/deploy.sh [--dry-run] site | bags <folder> | check}"

# The paths the page fetches, relative to web/.
referenced() {
  grep -ohE "(demo|configs)/[A-Za-z0-9_.-]+\.(mcap|rl)" "$web/main.js" "$web/benchmark.js" | sort -u
}

known_hosts="$(mktemp)"
trap 'rm -f "$known_hosts"' EXIT
echo "[$host]:2222 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBvqpiJdvk+zuDXcjOgNdA3p98+6L7MAnhOEOLrw4phX" >"$known_hosts"
ssh=(ssh -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$known_hosts")
if [[ -n "${DEPLOY_KEY:-}" ]]; then
  ssh+=(-i "$DEPLOY_KEY" -o IdentitiesOnly=yes)
fi
rsync=(rsync -rlt --checksum --delay-updates --chmod=D755,F644 --itemize-changes --stats -e "${ssh[*]}" ${dry_run[@]+"${dry_run[@]}"})

case "$mode" in
  site)
    missing=0
    for file in index.html main.js pkg/pelorus_web.js pkg/pelorus_web_bg.wasm pelorus.svg favicon.svg favicon.js; do
      [[ -f "$web/$file" ]] || { echo "Missing $file. Run web/build.sh first." >&2; missing=1; }
    done
    for file in $(referenced | grep '^configs/'); do
      [[ -f "$web/$file" ]] || { echo "Missing $file, referenced by the page." >&2; missing=1; }
    done
    compgen -G "$web/fonts/*.woff2" >/dev/null || { echo "Missing the fonts." >&2; missing=1; }
    [[ -f "$web/syntaxes/ratslang.tmLanguage.json" ]] || echo "No ratslang grammar. The config editor stays plain." >&2
    ((missing == 0)) || exit 1

    # Excluded files are also kept on the server, which keeps the bags, the
    # documentation that docs.yml uploads to docs/ and the benchmark page that
    # pelorus-bench uploads to bench/.
    "${rsync[@]}" --delete \
      --exclude='/demo/*.mcap' --exclude='/docs/' --exclude='/bench/' --exclude='/src/' --exclude='/Cargo.toml' --exclude='/*.sh' \
      --exclude='/serve.py' --exclude='/.gitignore' \
      "$web/" "$target"
    ;;

  bags)
    source="${2:?Usage: web/deploy.sh [--dry-run] bags <folder>}"
    staging="$(mktemp -d)"
    trap 'rm -f "$known_hosts"; rm -rf "$staging"' EXIT
    while IFS= read -r -d '' bag; do
      ln -s "$bag" "$staging/$(basename "$bag")"
    done < <(find -L "$source" -name '*.mcap' -print0)
    missing=0
    for file in $(referenced | grep '^demo/'); do
      [[ -e "$staging/${file#demo/}" ]] || { echo "Missing $file below $source." >&2; missing=1; }
    done
    ((missing == 0)) || exit 1

    "${rsync[@]}" --copy-links --delete --include='*.mcap' --exclude='*' "$staging/" "${target}demo/"
    ;;

  check)
    failed=0
    for file in index.html main.js worker.js viewer.js pkg/pelorus_web.js pkg/pelorus_web_bg.wasm $(referenced); do
      status="$(curl -sS -o /dev/null --head -w '%{http_code} %{content_type}' "$url/$file")" || status="error"
      echo "$status  $url/$file"
      [[ "$status" == 200* ]] || failed=1
      if [[ "$file" == *.wasm && "$status" != *application/wasm* ]]; then
        echo "  The wasm module needs the application/wasm content type." >&2
        failed=1
      fi
    done
    ((failed == 0)) || { echo "The demo on $url is incomplete." >&2; exit 1; }
    ;;

  *)
    echo "Unknown mode $mode. Use site, bags or check." >&2
    exit 1
    ;;
esac
