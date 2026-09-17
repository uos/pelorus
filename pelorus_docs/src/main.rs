//! Generates the configuration pages of the documentation from the comments in
//! `pelorus_core/default.rl`.
//!
//! ```sh
//! cargo run -p pelorus_docs            # write the pages
//! cargo run -p pelorus_docs -- --check # fail if the pages are out of date
//! ```
//!
//! Nothing it writes is committed. `docs/zensical.toml` is written from the
//! committed `docs/zensical.template.toml`, and the configuration pages are
//! written under `docs/docs/configuration/`. A docs build runs this first, which
//! is what `mask docs` and `mask docs-serve` do.

mod parse;
mod render;

use std::path::{Path, PathBuf};

/// The same file `pelorus_core` embeds and `pelorus config generate full` prints.
pub const DEFAULT_CONFIG: &str = include_str!("../../pelorus_core/default.rl");

fn main() -> std::process::ExitCode {
    let mut check = false;
    let mut docs_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--docs-dir" => match args.next() {
                Some(value) => docs_dir = Some(PathBuf::from(value)),
                None => return fail("--docs-dir needs a path"),
            },
            "-h" | "--help" => {
                println!(
                    "Usage: cargo run -p pelorus_docs [--check] [--docs-dir <path>]\n\n\
                     Writes the configuration pages of the documentation from the comments\n\
                     in pelorus_core/default.rl.\n\n\
                     --check      do not write anything; exit non-zero if a page is out of date\n\
                     --docs-dir   directory with the Markdown sources (default: docs/docs).\n\
                      Its parent directory must hold zensical.toml."
                );
                return std::process::ExitCode::SUCCESS;
            }
            other => return fail(&format!("unknown argument `{other}`")),
        }
    }

    let docs_dir = docs_dir.unwrap_or_else(|| workspace_root().join("docs/docs"));

    let parsed = match parse::parse(DEFAULT_CONFIG) {
        Ok(parsed) => parsed,
        Err(err) => return fail(&format!("cannot read default.rl: {err}")),
    };

    let problems = parsed.lint(DEFAULT_CONFIG);
    if !problems.is_empty() {
        eprintln!("default.rl is not fully documented:");
        for problem in &problems {
            eprintln!("  {problem}");
        }
        eprintln!(
            "\nEvery option and every block needs a comment above it, because that comment is \
             what the documentation shows."
        );
        return std::process::ExitCode::FAILURE;
    }

    let pages = match render::render(&parsed, DEFAULT_CONFIG) {
        Ok(pages) => pages,
        Err(err) => return fail(&err),
    };

    // Every generated file, as an absolute path. zensical.toml sits next to the
    // Markdown sources instead of under them, and is written from the committed
    // template with the reference entries of its nav table filled in.
    let mut targets: Vec<(PathBuf, String)> = pages
        .into_iter()
        .map(|(path, content)| (docs_dir.join(path), content))
        .collect();

    let Some(config_dir) = docs_dir.parent() else {
        return fail("--docs-dir has no parent to hold the zensical configuration");
    };
    let template_path = config_dir.join("zensical.template.toml");
    match std::fs::read_to_string(&template_path) {
        Ok(template) => match render::splice_nav(&template, &parsed) {
            Ok(spliced) => targets.push((config_dir.join("zensical.toml"), spliced)),
            Err(err) => return fail(&err),
        },
        Err(err) => return fail(&format!("cannot read {}: {err}", template_path.display())),
    }

    // A block removed from default.rl must take its page with it, otherwise the
    // orphan keeps building and keeps turning up in search.
    let orphans = match orphan_pages(&docs_dir, &targets) {
        Ok(orphans) => orphans,
        Err(err) => return fail(&err),
    };

    if check {
        let mut stale: Vec<String> = orphans
            .iter()
            .map(|path| format!("{} is no longer generated", path.display()))
            .collect();
        for (path, content) in &targets {
            match std::fs::read_to_string(path) {
                Ok(existing) if existing == *content => {}
                Ok(_) => stale.push(format!("{} differs", path.display())),
                Err(_) => stale.push(format!("{} is missing", path.display())),
            }
        }
        if stale.is_empty() {
            println!(
                "generated documentation is up to date ({} files)",
                targets.len()
            );
            return std::process::ExitCode::SUCCESS;
        }
        eprintln!("the generated documentation is out of date:");
        for entry in &stale {
            eprintln!("  {entry}");
        }
        eprintln!("\nRun `cargo run -p pelorus_docs` and commit the result.");
        return std::process::ExitCode::FAILURE;
    }

    for path in &orphans {
        if let Err(err) = std::fs::remove_file(path) {
            return fail(&format!("cannot remove {}: {err}", path.display()));
        }
        println!("removed {}", path.display());
    }

    for (path, content) in &targets {
        if let Some(parent) = path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            return fail(&format!("cannot create {}: {err}", parent.display()));
        }
        if let Err(err) = std::fs::write(path, content) {
            return fail(&format!("cannot write {}: {err}", path.display()));
        }
        println!("wrote {}", path.display());
    }
    std::process::ExitCode::SUCCESS
}

/// Reference pages on disk that this run does not generate, which means the
/// block they documented is gone from `default.rl`.
fn orphan_pages(docs_dir: &Path, targets: &[(PathBuf, String)]) -> Result<Vec<PathBuf>, String> {
    let reference = docs_dir.join("configuration/reference");
    let entries = match std::fs::read_dir(&reference) {
        Ok(entries) => entries,
        // Nothing generated yet, so nothing can be orphaned.
        Err(_) => return Ok(Vec::new()),
    };

    let mut orphans = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|err| format!("cannot read {}: {err}", reference.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "md")
            && !targets.iter().any(|(target, _)| *target == path)
        {
            orphans.push(path);
        }
    }
    orphans.sort();
    Ok(orphans)
}

fn fail(message: &str) -> std::process::ExitCode {
    eprintln!("error: {message}");
    std::process::ExitCode::FAILURE
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate lives in the workspace")
        .to_path_buf()
}
