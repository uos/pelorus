use core::error::Error as CoreError;
use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn CoreError>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let default_config_path = PathBuf::from(&manifest_dir).join("../default.rl");

    println!("cargo:rerun-if-changed={}", default_config_path.display());

    let default_config = std::fs::read_to_string(&default_config_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read default config at {}: {}",
            default_config_path.display(),
            e
        )
    })?;

    let default_ast = ratslang::compile_code(&default_config)
        .map_err(|e| anyhow::anyhow!("Failed to compile default config: {}", e))?;
    let empty_user_ast = ratslang::compile_code("")?;

    // Parse the config - this will resolve variables and return usage info
    let (_config, (assigned_vars, evaluated_vars)) =
        pelorus_core::config::ast_to_config_with_usage(default_ast, empty_user_ast)
            .map_err(|e| anyhow::anyhow!("Default configuration is invalid: {}", e))?;

    // After config parsing, check for unused variables
    for var in assigned_vars.iter() {
        if !evaluated_vars.contains(var) {
            println!(
                "cargo:warning=Warning: Variable '{}' is not recognized by Pelorus",
                var
            );
        }
    }

    Ok(())
}
