pub(crate) fn validate_args(command: &str, args: &[String]) -> Result<(), String> {
    let universal_blocked = ["--exec", "-exec", ";", "&&", "||", "|", ">", ">>", "<"];
    for arg in args {
        for pattern in &universal_blocked {
            if arg.contains(pattern) {
                return Err(format!(
                    "Argument '{}' contains blocked pattern '{}'",
                    arg, pattern
                ));
            }
        }
    }

    match command {
        "git" => {
            let git_blocked = ["push", "reset", "clean", "rm", "checkout"];
            if let Some(subcmd) = args.first() {
                if git_blocked.contains(&subcmd.as_str()) {
                    return Err(format!("git subcommand '{}' is blocked", subcmd));
                }
            }
        }
        "find" if args.iter().any(|a| a == "-delete") => {
            return Err("find -delete is blocked".to_string());
        }
        "cargo" => {
            let cargo_allowed = [
                "build", "test", "check", "clippy", "fmt", "doc", "metadata", "tree",
            ];
            if let Some(subcmd) = args.first() {
                if !cargo_allowed.contains(&subcmd.as_str()) {
                    return Err(format!("cargo subcommand '{}' is not allowed", subcmd));
                }
            }
        }
        _ => {}
    }

    let home = std::env::var("HOME").ok();
    for arg in args {
        if arg.contains("..") {
            return Err(format!("Path '{}' contains forbidden components", arg));
        }
        if arg.starts_with('/') {
            let path = std::path::Path::new(arg);
            if arg.starts_with("/tmp/") || arg == "/tmp" {
                continue;
            }
            let canonical = if path.exists() {
                path.canonicalize().ok()
            } else if let Some(parent) = path.parent() {
                parent
                    .canonicalize()
                    .ok()
                    .map(|p| p.join(path.file_name().unwrap_or_default()))
            } else {
                None
            };
            let (Some(canonical), Some(home)) = (canonical, home.clone()) else {
                return Err("Cannot validate path: HOME not set or parent missing".to_string());
            };
            let canonical_str = canonical.to_string_lossy();
            if !canonical_str.starts_with(&home) {
                return Err(format!("Path '{}' is outside allowed directories", arg));
            }
        }
    }

    Ok(())
}
