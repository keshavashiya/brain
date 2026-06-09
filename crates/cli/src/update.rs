//! `brain update` — check the latest GitHub release and, when newer, run the
//! official one-line installer to swap the binary in place.
//!
//! Self-update is deliberately thin: it queries the GitHub Releases API for the
//! latest tag, compares it to the running version, and (on confirmation) shells
//! out to `scripts/install.sh` pinned to that tag. The installer already owns
//! platform/arch detection and SHA-256 verification, so there is one audited
//! download path, not two.
//!
//! There is **no passive update check** anywhere else — `brain --version` and
//! daemon startup never touch the network. This command is the only place an
//! update probe happens, and only because the user asked for it. `--check`
//! reports availability without installing.

use std::io::Write;
use std::process::Command;

/// `owner/repo` the releases + installer live under. Mirrors `install.sh`.
const REPO: &str = "keshavashiya/brain";
/// Raw URL of the installer, fetched fresh so a self-update always runs the
/// latest install logic (matches the documented `curl … | sh` one-liner).
const INSTALL_URL: &str =
    "https://raw.githubusercontent.com/keshavashiya/brain/main/scripts/install.sh";

/// Entry point for `brain update`.
///
/// - `check_only`: report whether a newer release exists, then stop.
/// - `assume_yes`: skip the install confirmation prompt.
/// - `pin`: install this exact tag instead of the latest (allows reinstall /
///   downgrade); the up-to-date short-circuit is skipped when set.
pub async fn cmd_update(
    check_only: bool,
    assume_yes: bool,
    pin: Option<String>,
) -> anyhow::Result<()> {
    let current = env!("CARGO_PKG_VERSION");

    let target = match &pin {
        Some(v) => normalize(v),
        None => fetch_latest_release().await?,
    };

    // Without an explicit pin, report the comparison and bail early if current.
    if pin.is_none() {
        if is_newer(&target, current) {
            println!("A newer Brain release is available: v{current} → v{target}");
        } else {
            println!("Brain is up to date (v{current}).");
            return Ok(());
        }
    }

    if check_only {
        return Ok(());
    }

    if !assume_yes
        && !confirm(&format!(
            "Install v{target} now via the official installer ({INSTALL_URL})?"
        ))?
    {
        println!("Update cancelled.");
        return Ok(());
    }

    run_installer(&target)
}

/// GET the latest release tag from the GitHub API, normalized (no leading `v`).
async fn fetch_latest_release() -> anyhow::Result<String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    // GitHub requires a User-Agent. A short timeout keeps an offline box from
    // hanging the command.
    let client = reqwest::Client::builder()
        .user_agent(concat!("brain/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("could not reach GitHub ({e}); check your connection"))?;
    if !resp.status().is_success() {
        anyhow::bail!("GitHub API returned {} for {url}", resp.status());
    }
    let json: serde_json::Value = resp.json().await?;
    let tag = json
        .get("tag_name")
        .and_then(|t| t.as_str())
        .ok_or_else(|| anyhow::anyhow!("no tag_name in the GitHub release response"))?;
    Ok(normalize(tag))
}

/// Pipe the installer to `sh`, pinning the target tag. `install.sh` handles
/// arch detection + checksum verification and writes to `$BRAIN_PREFIX`.
fn run_installer(target: &str) -> anyhow::Result<()> {
    if cfg!(windows) {
        anyhow::bail!(
            "self-update via the shell installer is not supported on Windows; \
             reinstall with `cargo install brainos` or download v{target} from \
             https://github.com/{REPO}/releases"
        );
    }

    println!("Running the official installer for v{target}…");
    let script = format!("curl -fsSL {INSTALL_URL} | sh");
    let status = Command::new("sh")
        .arg("-c")
        .arg(&script)
        .env("BRAIN_VERSION", format!("v{target}"))
        .status()
        .map_err(|e| anyhow::anyhow!("failed to launch installer: {e}"))?;
    if !status.success() {
        anyhow::bail!("installer exited unsuccessfully ({status})");
    }
    println!("Updated to v{target}. Restart any running `brain` daemon to pick it up.");
    Ok(())
}

/// Yes/no prompt on the terminal; defaults to no on empty/EOF.
fn confirm(question: &str) -> anyhow::Result<bool> {
    print!("{question} [y/N] ");
    std::io::stdout().flush()?;
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf)?;
    Ok(matches!(
        buf.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// Strip a leading `v` and surrounding whitespace from a tag/version string.
fn normalize(v: &str) -> String {
    v.trim().trim_start_matches('v').to_string()
}

/// `candidate` is a strictly newer semver than `current`. Numeric per-component
/// (`0.10.0` > `0.9.0`); missing/junk components read as 0.
fn is_newer(candidate: &str, current: &str) -> bool {
    parse(candidate) > parse(current)
}

fn parse(v: &str) -> (u64, u64, u64) {
    let core = normalize(v);
    let core = core.split(['-', '+']).next().unwrap_or(&core);
    let mut it = core
        .split('.')
        .map(|p| p.trim().parse::<u64>().unwrap_or(0));
    (
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_v_and_whitespace() {
        assert_eq!(normalize("v0.5.0"), "0.5.0");
        assert_eq!(normalize("  0.5.0 "), "0.5.0");
        assert_eq!(normalize("0.5.0"), "0.5.0");
    }

    #[test]
    fn is_newer_compares_numerically() {
        assert!(is_newer("0.10.0", "0.9.0"), "0.10 > 0.9 numerically");
        assert!(is_newer("v0.5.0", "0.4.9"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(!is_newer("0.4.0", "0.4.0"), "equal is not newer");
        assert!(!is_newer("0.4.0", "0.5.0"), "older is not newer");
        // Pre-release suffix is dropped to the base version.
        assert!(!is_newer("0.5.0-rc1", "0.5.0"));
    }
}
