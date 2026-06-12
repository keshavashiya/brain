//! Power-source probing — the writer side of [`brain::Power`].
//!
//! The serve loop spawns one bounded probe task (see
//! `background::spawn_power_probe`) that asks the platform for the current
//! power source each round and folds the answer into the kernel's
//! `External / Battery` view. No network and no new dependency: macOS is
//! read via `pmset`, Linux via `/sys/class/power_supply`; any other
//! platform (or a machine with no battery) reports "undetectable" and the
//! state stays pinned `External`. This module owns the pieces that loop
//! composes:
//!
//! * [`probe`] — one platform query, returning the state plus a short
//!   human-readable detail (`"battery at 47%"`), or `None` when the power
//!   source cannot be determined here.
//! * [`hold_while_on_battery`] — the deferral gate the heavy maintenance
//!   loops (consolidation, graph compaction) call after each tick: logs
//!   once on entry, holds while on battery, logs once on resume.

use brain::PowerState;

/// Query the platform for the current power source. `None` means this
/// platform (or machine) can't say — the caller should stop probing and
/// leave the state pinned `External`.
pub(crate) async fn probe() -> Option<(PowerState, String)> {
    #[cfg(target_os = "macos")]
    {
        probe_macos().await
    }
    #[cfg(target_os = "linux")]
    {
        probe_linux()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
async fn probe_macos() -> Option<(PowerState, String)> {
    let batt = run_pmset(&["-g", "batt"]).await?;
    let (state, mut detail) = parse_pmset_batt(&batt)?;
    // Low Power Mode is worth surfacing in the detail (the user asked the
    // OS to go easy), but deferral keys on battery vs external only.
    if let Some(settings) = run_pmset(&["-g"]).await {
        if pmset_low_power(&settings) {
            detail.push_str(" (low power mode)");
        }
    }
    Some((state, detail))
}

#[cfg(target_os = "macos")]
async fn run_pmset(args: &[&str]) -> Option<String> {
    let out = tokio::process::Command::new("pmset")
        .args(args)
        .output()
        .await
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse `pmset -g batt` output: the first line names the source
/// (`Now drawing from 'AC Power'` / `'Battery Power'`), a battery line
/// carries the percentage.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_pmset_batt(out: &str) -> Option<(PowerState, String)> {
    let state = if out.contains("'AC Power'") {
        PowerState::External
    } else if out.contains("'Battery Power'") {
        PowerState::Battery
    } else {
        return None;
    };
    let detail = match (state, percent_token(out)) {
        (PowerState::Battery, Some(pct)) => format!("battery at {pct}%"),
        (PowerState::Battery, None) => "on battery".to_string(),
        (PowerState::External, _) => "external power".to_string(),
    };
    Some((state, detail))
}

/// Whether `pmset -g` reports Low Power Mode active (`lowpowermode  1`).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn pmset_low_power(out: &str) -> bool {
    out.lines().any(|l| {
        let l = l.trim();
        l.starts_with("lowpowermode") && l.ends_with('1')
    })
}

/// First `NN%` token in the output, as the bare number.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn percent_token(out: &str) -> Option<&str> {
    out.split_whitespace().find_map(|tok| {
        let stripped = tok.trim_end_matches([';', ',']);
        let num = stripped.strip_suffix('%')?;
        (!num.is_empty() && num.chars().all(|c| c.is_ascii_digit())).then_some(num)
    })
}

#[cfg(target_os = "linux")]
fn probe_linux() -> Option<(PowerState, String)> {
    let mut supplies = Vec::new();
    for entry in std::fs::read_dir("/sys/class/power_supply").ok()?.flatten() {
        let path = entry.path();
        let read = |name: &str| {
            std::fs::read_to_string(path.join(name))
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        supplies.push(Supply {
            kind: read("type"),
            online: read("online"),
            status: read("status"),
            capacity: read("capacity"),
        });
    }
    fold_supplies(&supplies)
}

/// One `/sys/class/power_supply/<entry>`, reduced to the fields the fold
/// reads. Missing files read as empty strings.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct Supply {
    kind: String,
    online: String,
    status: String,
    capacity: String,
}

/// Fold the supply entries into a state: any online mains-class adapter →
/// `External`; else a charging/full battery implies a powered source the
/// kernel didn't list → `External`; else a present battery → `Battery`;
/// no supplies at all → `None` (desktop, nothing to probe).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn fold_supplies(supplies: &[Supply]) -> Option<(PowerState, String)> {
    if supplies
        .iter()
        .any(|s| s.kind != "Battery" && s.online == "1")
    {
        return Some((PowerState::External, "external power".to_string()));
    }
    let battery = supplies.iter().find(|s| s.kind == "Battery")?;
    if matches!(battery.status.as_str(), "Charging" | "Full") {
        return Some((PowerState::External, "external power".to_string()));
    }
    let detail = if battery.capacity.is_empty() {
        "on battery".to_string()
    } else {
        format!("battery at {}%", battery.capacity)
    };
    Some((PowerState::Battery, detail))
}

/// Deferral gate for heavy maintenance loops. Returns immediately when
/// deferral is off or the machine is on external power; otherwise logs the
/// deferral once, holds (rechecking the shared handle) until external power
/// returns, then logs the resume — so the log tells the whole story with
/// two lines per battery stretch, never one per recheck.
pub(crate) async fn hold_while_on_battery(power: &brain::Power, defer: bool, task: &str) {
    if !defer || !power.is_battery() {
        return;
    }
    tracing::info!(task, "Heavy maintenance deferred: running on battery");
    while power.is_battery() {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    }
    tracing::info!(task, "Heavy maintenance resuming: external power restored");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pmset_on_battery() {
        let out = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=12345)\t47%; \
                   discharging; 3:42 remaining present: true\n";
        assert_eq!(
            parse_pmset_batt(out),
            Some((PowerState::Battery, "battery at 47%".to_string()))
        );
    }

    #[test]
    fn parses_pmset_on_ac() {
        let out = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345)\t100%; \
                   charged; 0:00 remaining present: true\n";
        assert_eq!(
            parse_pmset_batt(out),
            Some((PowerState::External, "external power".to_string()))
        );
    }

    #[test]
    fn pmset_garbage_is_undetectable() {
        assert_eq!(parse_pmset_batt("no battery information available"), None);
    }

    #[test]
    fn detects_low_power_mode_line() {
        assert!(pmset_low_power(
            "System-wide power settings:\n lowpowermode         1\n standby 1\n"
        ));
        assert!(!pmset_low_power(
            "System-wide power settings:\n lowpowermode         0\n"
        ));
    }

    fn supply(kind: &str, online: &str, status: &str, capacity: &str) -> Supply {
        Supply {
            kind: kind.into(),
            online: online.into(),
            status: status.into(),
            capacity: capacity.into(),
        }
    }

    #[test]
    fn sysfs_mains_online_is_external() {
        let s = [
            supply("Mains", "1", "", ""),
            supply("Battery", "", "Discharging", "80"),
        ];
        assert_eq!(
            fold_supplies(&s),
            Some((PowerState::External, "external power".to_string()))
        );
    }

    #[test]
    fn sysfs_discharging_battery_without_mains_is_battery() {
        let s = [
            supply("Mains", "0", "", ""),
            supply("Battery", "", "Discharging", "47"),
        ];
        assert_eq!(
            fold_supplies(&s),
            Some((PowerState::Battery, "battery at 47%".to_string()))
        );
    }

    #[test]
    fn sysfs_charging_battery_implies_external_even_without_a_mains_entry() {
        let s = [supply("Battery", "", "Charging", "60")];
        assert_eq!(
            fold_supplies(&s),
            Some((PowerState::External, "external power".to_string()))
        );
    }

    #[test]
    fn sysfs_no_supplies_is_undetectable() {
        assert_eq!(fold_supplies(&[]), None);
        // Desktop with a PSU entry the kernel reports as offline mains and
        // no battery: nothing to defer on either way.
        assert_eq!(fold_supplies(&[supply("Mains", "0", "", "")]), None);
    }

    #[tokio::test]
    async fn hold_returns_immediately_on_external_or_when_deferral_is_off() {
        let p = brain::Power::default();
        hold_while_on_battery(&p, true, "test").await; // external — no hold
        p.set(PowerState::Battery);
        hold_while_on_battery(&p, false, "test").await; // deferral off — no hold
    }

    #[tokio::test(start_paused = true)]
    async fn hold_blocks_on_battery_and_releases_when_external_power_returns() {
        let p = brain::Power::default();
        p.set(PowerState::Battery);
        let watcher = p.clone();
        let hold = tokio::spawn(async move { hold_while_on_battery(&watcher, true, "test").await });
        // Paused time auto-advances through the gate's recheck sleeps; the
        // hold must outlast a flip that happens mid-recheck and return on
        // the next one.
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        assert!(!hold.is_finished(), "must hold while on battery");
        p.set(PowerState::External);
        tokio::time::timeout(std::time::Duration::from_secs(120), hold)
            .await
            .expect("gate must release once external power returns")
            .unwrap();
    }
}
