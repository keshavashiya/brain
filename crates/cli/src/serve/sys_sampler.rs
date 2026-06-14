//! System-state sampler for the `sys` reflex.
//!
//! Composes the kernel's existing signals into a [`reflex::SysSnapshot`] on
//! every poll, with no new egress destination and no native dependency:
//!
//! * **battery + AC** — [`super::power::probe_battery`] (pmset / sysfs).
//! * **network** — the shared [`brain::Connectivity`] handle the connectivity
//!   probe already maintains (so `network_changed` reads the same
//!   online/offline view as the rest of the kernel; inert when connectivity
//!   probing is disabled, since the handle then never leaves `Online`).
//! * **lock** — the platform session: systemd-logind (`LockedHint`) on Linux,
//!   the CoreGraphics login-session dictionary (`CGSSessionScreenIsLocked`) on
//!   macOS via a small raw FFI (no extra crate). `None` where no GUI session is
//!   reachable (headless / ssh), so `lock_changed` stays inert rather than
//!   guessing.

use brain::{Connectivity, ConnectivityState, PowerState};
use reflex::{NetworkState, SysSnapshot, SysStateSampler};

/// Sampler backing the `sys` reflex. Holds a clone of the connectivity
/// handle so `network_changed` needs no probing of its own.
pub(crate) struct SysSampler {
    connectivity: Connectivity,
}

impl SysSampler {
    pub(crate) fn new(connectivity: Connectivity) -> Self {
        Self { connectivity }
    }
}

#[async_trait::async_trait]
impl SysStateSampler for SysSampler {
    async fn sample(&self) -> SysSnapshot {
        let (battery_percent, on_ac) = match super::power::probe_battery().await {
            Some((state, pct)) => (pct, Some(matches!(state, PowerState::External))),
            None => (None, None),
        };
        SysSnapshot {
            battery_percent,
            on_ac,
            network: Some(network_state(self.connectivity.state())),
            locked: probe_lock().await,
        }
    }
}

/// Map the kernel's three-state connectivity view onto the reflex's binary
/// online/offline. `Degraded` still has *some* reachability, so it counts as
/// online for the `network_changed` edge.
fn network_state(state: ConnectivityState) -> NetworkState {
    match state {
        ConnectivityState::Offline => NetworkState::Offline,
        ConnectivityState::Online | ConnectivityState::Degraded => NetworkState::Online,
    }
}

/// Best-effort screen-lock state. `None` when the platform has no
/// dependency-free way to report it (so `lock_changed` stays inert).
async fn probe_lock() -> Option<bool> {
    #[cfg(target_os = "linux")]
    {
        probe_lock_linux().await
    }
    #[cfg(target_os = "macos")]
    {
        probe_lock_macos()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Query systemd-logind for the caller session's `LockedHint`.
#[cfg(target_os = "linux")]
async fn probe_lock_linux() -> Option<bool> {
    let out = tokio::process::Command::new("loginctl")
        .args(["show-session", "self", "-p", "LockedHint", "--value"])
        .output()
        .await
        .ok()?;
    out.status
        .success()
        .then(|| parse_locked_hint(&String::from_utf8_lossy(&out.stdout)))
        .flatten()
}

/// Parse a `loginctl … LockedHint --value` line: `yes`/`no` → `Some(bool)`,
/// anything else (no graphical session, older systemd) → `None`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_locked_hint(value: &str) -> Option<bool> {
    match value.trim() {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

/// Screen-lock state from the CoreGraphics login-session dictionary.
///
/// `CGSessionCopyCurrentDictionary` returns the current GUI session's state
/// (or null when called outside one — headless, ssh, a session-less daemon),
/// keyed by `CGSSessionScreenIsLocked` when the screen is locked. Raw FFI so we
/// don't pull a CoreFoundation crate in for one boolean.
#[cfg(target_os = "macos")]
fn probe_lock_macos() -> Option<bool> {
    use std::os::raw::{c_char, c_void};

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGSessionCopyCurrentDictionary() -> *const c_void;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const c_char,
            encoding: u32,
        ) -> *const c_void;
        fn CFBooleanGetValue(boolean: *const c_void) -> u8;
        fn CFGetTypeID(cf: *const c_void) -> usize;
        fn CFBooleanGetTypeID() -> usize;
        fn CFRelease(cf: *const c_void);
    }
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    // SAFETY: standard CoreFoundation memory rules. We own the dictionary
    // (`Copy`) and the key (`Create`) and release both; the looked-up value is
    // borrowed (not released). A null dictionary means no reachable session.
    unsafe {
        let dict = CGSessionCopyCurrentDictionary();
        if dict.is_null() {
            return None;
        }
        let key = CFStringCreateWithCString(
            std::ptr::null(),
            c"CGSSessionScreenIsLocked".as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        let locked = if key.is_null() {
            None
        } else {
            let val = CFDictionaryGetValue(dict, key);
            if val.is_null() {
                // Session present, key absent → not locked.
                Some(false)
            } else if CFGetTypeID(val) == CFBooleanGetTypeID() {
                Some(CFBooleanGetValue(val) != 0)
            } else {
                None
            }
        };
        if !key.is_null() {
            CFRelease(key);
        }
        CFRelease(dict);
        locked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connectivity_maps_to_binary_network_state() {
        assert_eq!(
            network_state(ConnectivityState::Online),
            NetworkState::Online
        );
        // Degraded still has reachability → online for the edge.
        assert_eq!(
            network_state(ConnectivityState::Degraded),
            NetworkState::Online
        );
        assert_eq!(
            network_state(ConnectivityState::Offline),
            NetworkState::Offline
        );
    }

    #[test]
    fn locked_hint_parses_yes_no_and_rejects_other() {
        assert_eq!(parse_locked_hint("yes\n"), Some(true));
        assert_eq!(parse_locked_hint("no"), Some(false));
        assert_eq!(parse_locked_hint(""), None);
        assert_eq!(parse_locked_hint("unknown"), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_lock_probe_links_and_is_memory_safe() {
        // Exercises the raw CoreGraphics/CoreFoundation FFI under the harness:
        // it must link and be memory-safe. A live GUI session (unlocked)
        // returns `Some(false)`; a headless CI runner with no session returns
        // `None`. Either is valid — the point is that the unsafe block runs.
        let state = probe_lock_macos();
        assert!(matches!(state, None | Some(_)));
        eprintln!("probe_lock_macos() = {state:?}");
    }

    #[tokio::test]
    async fn sampler_reads_network_from_the_shared_handle() {
        let conn = Connectivity::default();
        let sampler = SysSampler::new(conn.clone());
        assert_eq!(sampler.sample().await.network, Some(NetworkState::Online));
        conn.set(ConnectivityState::Offline);
        assert_eq!(sampler.sample().await.network, Some(NetworkState::Offline));
    }
}
