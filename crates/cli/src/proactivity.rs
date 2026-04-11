//! Proactivity subcommand — quick toggle for proactive notifications.

use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum ProactivityAction {
    /// Show current proactivity settings
    Status,
    /// Enable proactive notifications
    On,
    /// Disable proactive notifications
    Off,
}

pub(crate) fn cmd_proactivity(
    config: &brain_core::BrainConfig,
    action: ProactivityAction,
) -> anyhow::Result<()> {
    match action {
        ProactivityAction::Status => {
            println!("Proactivity settings:");
            println!(
                "  Enabled:            {}",
                if config.proactivity.enabled {
                    "yes"
                } else {
                    "no"
                }
            );
            println!("  Max per day:        {}", config.proactivity.max_per_day);
            println!(
                "  Min interval:       {} minutes",
                config.proactivity.min_interval_minutes
            );
            println!(
                "  Quiet hours:        {} — {}",
                config.proactivity.quiet_hours.start, config.proactivity.quiet_hours.end
            );
            println!(
                "  Open-loop detection: {}",
                if config.proactivity.open_loop.enabled {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        ProactivityAction::On => {
            set_proactivity_enabled(true)?;
            println!("Proactive notifications enabled.");
            println!("Restart `brain serve` or `brain start` for the change to take effect.");
        }
        ProactivityAction::Off => {
            set_proactivity_enabled(false)?;
            println!("Proactive notifications disabled.");
            println!("Restart `brain serve` or `brain start` for the change to take effect.");
        }
    }
    Ok(())
}

/// Update the `proactivity.enabled` field in the user config file.
fn set_proactivity_enabled(enabled: bool) -> anyhow::Result<()> {
    let config_path = brain_core::BrainConfig::user_config_path();

    if !config_path.exists() {
        anyhow::bail!(
            "No config file found at {}. Run `brain init` first.",
            config_path.display()
        );
    }

    let yaml = std::fs::read_to_string(&config_path)?;
    let value_str = if enabled { "true" } else { "false" };

    // Try to replace existing proactivity.enabled line
    let updated = if let Some(line_start) = yaml.find("proactivity:") {
        let rest = &yaml[line_start..];
        if let Some(enabled_offset) = rest.find("enabled:") {
            let abs_offset = line_start + enabled_offset;
            // Find the end of the line
            let line_end = yaml[abs_offset..]
                .find('\n')
                .map(|i| abs_offset + i)
                .unwrap_or(yaml.len());
            // Find the indent before "enabled:"
            let indent_start = yaml[..abs_offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
            let indent = &yaml[indent_start..abs_offset];
            let new_line = format!("{indent}enabled: {value_str}");
            format!("{}{}{}", &yaml[..indent_start], new_line, &yaml[line_end..])
        } else {
            // proactivity section exists but no enabled key — insert it
            let after_section = yaml[line_start..]
                .find('\n')
                .map(|i| line_start + i + 1)
                .unwrap_or(yaml.len());
            format!(
                "{}  enabled: {}\n{}",
                &yaml[..after_section],
                value_str,
                &yaml[after_section..]
            )
        }
    } else {
        // No proactivity section at all — append it
        format!("{yaml}\nproactivity:\n  enabled: {value_str}\n")
    };

    std::fs::write(&config_path, updated)?;
    Ok(())
}
