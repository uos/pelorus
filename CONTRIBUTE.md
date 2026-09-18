# 🔧 Development

For faster binary installs, install `cargo-binstall` first:
```sh
cargo install cargo-binstall --locked
```

We use `mask` to simplify common development commands and save you some typing.
```sh
cargo binstall mask cargo-mommy
```

You can omit cargo-mommy by removing the mommy prefixes in the Maskfile... if you are on the more serious side of life.

In the project directory (where `./maskfile.md` is located), you can see all available commands:
```sh
mask --help
```

For proper visualisations, the development environment uses `rerun`. It's recommended to run it in the background.
```sh
cargo binstall rerun-cli
rerun
```
