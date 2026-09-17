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

## Generated pages

Nothing the generator writes is committed. `cargo run -p pelorus_docs` produces:

| Path | From |
| --- | --- |
| `zensical.toml` | `zensical.template.toml`, with the reference entries of the `nav` table filled in |
| `docs/configuration/recommended.md` | the `@recommended` annotations in `pelorus_core/default.rl` |
| `docs/configuration/reference/` | the comments in `pelorus_core/default.rl` |

Edit `zensical.template.toml`, never `zensical.toml`. A checkout has no
`zensical.toml` until the generator runs, so calling `zensical` directly in a
fresh tree fails with `No config file found`. That is deliberate, and is why the
mask tasks above exist.

Adding, renaming or removing a top-level block in `default.rl` needs no other
step. The page, its row in the reference index, and its navigation entry all
follow from the block and its comment, and the page of a block that is gone is
deleted. The generator refuses to produce a page for an option or block that has
no comment, and the `config-docs` job in `.forgejo/workflows/ci.yml` runs it on
every push for exactly that reason.

Each option on a reference page carries a copy-to-clipboard line holding its full
key and default. The markup is generated, and the behaviour and styling live in
`docs/javascripts/config-reference.js` and `docs/stylesheets/config-reference.css`,
which are registered in the template. The script binds one listener on the
document, not one per button, so it keeps working across instant navigation.
