# Docs

## Building

Install uv if you don't have it already, then zensical:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install zensical
```

Build the site, or serve it locally with live reload:

```sh
mask docs
mask docs-serve
```

Both run `cargo run -p pelorus_docs` first. Running `zensical` on its own skips
that step and leaves the configuration pages missing or stale, so use the two
tasks above.

## Configuration schema

`pelorus_core/default.rl` is the source for defaults, prose, and constraints
(`@type`, `@enum`, `@min`, `@max`, and related annotations). `pelorus_docs`
uses Ratslang's constraint parser to render these rules on the website.

The runtime deserializes the compiled default/user layers in one call into the
`Deserialize` structs in `pelorus_core/src/config/input.rs`. Keep those structs
in step with the `.rl` layout when adding settings. The conversion in
`config.rs` assembles the runtime configuration and checks relationships between
settings. Units, ranges, and enum spellings use shared serde adapters.
