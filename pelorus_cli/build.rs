use core::error::Error as CoreError;

fn main() -> Result<(), Box<dyn CoreError>> {
    let default_ast = ratslang::compile_code(pelorus_core::DEFAULT_CONFIG_STR)
        .map_err(|e| anyhow::anyhow!("Failed to compile default config: {}", e))?;
    let empty_user_ast = ratslang::compile_code("")?;

    let (_config, (assigned_vars, evaluated_vars)) =
        pelorus_core::config::ast_to_config_with_usage(default_ast, empty_user_ast)
            .map_err(|e| anyhow::anyhow!("Default configuration is invalid: {}", e))?;

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
