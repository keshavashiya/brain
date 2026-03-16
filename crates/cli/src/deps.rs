//! External dependency management via Docker Compose.

use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum DepsAction {
    /// Start external service containers (SearXNG).
    Up,
    /// Stop external service containers.
    Down,
    /// Show external service container status.
    Status,
}

fn find_compose_file() -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("docker/docker-compose.yml")),
        std::env::current_exe().ok().and_then(|p| {
            p.parent()
                .map(|d| d.join("../share/brain/docker/docker-compose.yml"))
        }),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|candidate| candidate.is_file())
}

pub(crate) fn cmd_deps(action: DepsAction) -> anyhow::Result<()> {
    let compose_file = find_compose_file().ok_or_else(|| {
        anyhow::anyhow!(
            "docker/docker-compose.yml not found.\n\
             If installed from release, run from the Brain source directory."
        )
    })?;

    let docker_ok = std::process::Command::new("docker")
        .arg("info")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !docker_ok {
        anyhow::bail!(
            "Docker is not running or not installed.\n\
             Install Docker Desktop: https://docs.docker.com/get-docker/"
        );
    }

    let compose_dir = compose_file.parent().unwrap();
    let run = |args: &[&str]| -> anyhow::Result<()> {
        let status = std::process::Command::new("docker")
            .arg("compose")
            .args(["-f", compose_file.to_str().unwrap()])
            .args(["--project-directory", compose_dir.to_str().unwrap()])
            .args(args)
            .status()?;
        if !status.success() {
            anyhow::bail!("docker compose {} failed", args.join(" "));
        }
        Ok(())
    };

    match action {
        DepsAction::Up => {
            println!("Starting Brain external services...");
            run(&["up", "-d"])?;
            println!("\nServices started:");
            println!("  SearXNG → http://127.0.0.1:8888");
            println!("\nRun `brain status` to verify connectivity.");
        }
        DepsAction::Down => {
            println!("Stopping Brain external services...");
            run(&["down"])?;
            println!("Services stopped.");
        }
        DepsAction::Status => {
            run(&["ps"])?;
        }
    }
    Ok(())
}
