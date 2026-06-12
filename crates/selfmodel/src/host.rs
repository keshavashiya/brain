//! Host self-model — Brain's grounded knowledge of the *machine it runs on*.
//!
//! [`super::ProductSelfModel`] grounds the SOUL in Brain-the-product; this
//! sibling grounds it in Brain-the-circumstances: OS/arch, cores, RAM,
//! GPU/VRAM (best-effort per platform), and the data-dir disk class/free
//! space. Probed once at bootstrap, then read-only.
//!
//! Three consumers, one model:
//! - the capability digest names the machine class so the reasoner sizes its
//!   suggestions to the hardware in front of it,
//! - `brain doctor` flags a configured local model that exceeds the memory
//!   the machine can actually give it,
//! - `brain init` recommends a local model size that fits.
//!
//! Everything is best-effort: a probe that fails leaves a `None`/`Unknown`
//! field, never an error — situational awareness must not block boot.

use std::path::Path;
use std::process::Command;

/// GPU memory available for local inference, by acceleration story.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuInfo {
    /// Apple Silicon unified memory: the GPU shares system RAM. `budget_bytes`
    /// is the working-set Metal will realistically grant (~75% of RAM).
    AppleUnified { budget_bytes: u64 },
    /// A discrete GPU with dedicated VRAM (detected via `nvidia-smi`).
    Discrete { name: String, vram_bytes: u64 },
    /// No GPU detected, or the platform probe isn't implemented — inference
    /// budget falls back to a share of system RAM (CPU inference).
    Unknown,
}

/// Storage class of the disk holding the Brain data dir.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskClass {
    Ssd,
    Hdd,
    Unknown,
}

/// The data-dir disk: class + free space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskInfo {
    pub class: DiskClass,
    pub free_bytes: u64,
}

/// Brain's grounded knowledge of the host machine. Built once at bootstrap
/// via [`HostModel::probe`]; tests construct it directly with known values.
#[derive(Debug, Clone)]
pub struct HostModel {
    /// `std::env::consts::OS` value, e.g. `macos` / `linux` / `windows`.
    pub os: String,
    /// `std::env::consts::ARCH` value, e.g. `aarch64` / `x86_64`.
    pub arch: String,
    /// CPU/SoC marketing name when the platform exposes one
    /// (e.g. "Apple M3 Max"), best-effort.
    pub chip: Option<String>,
    /// Logical cores visible to this process.
    pub cores: usize,
    /// Total system RAM. `None` when the probe fails.
    pub ram_bytes: Option<u64>,
    pub gpu: GpuInfo,
    /// The disk holding the data dir. `None` when no data dir was supplied
    /// or no mounted disk contains it.
    pub disk: Option<DiskInfo>,
}

const GIB: u64 = 1024 * 1024 * 1024;

/// Share of system RAM Metal will realistically grant the GPU on Apple
/// Silicon (`recommendedMaxWorkingSetSize` is ~75% of unified memory).
const APPLE_UNIFIED_GPU_SHARE: f64 = 0.75;

/// Share of system RAM treated as the inference budget when there is no
/// detected GPU — CPU inference competes with everything else on the box.
const CPU_INFERENCE_RAM_SHARE: f64 = 0.5;

/// Estimated GiB per billion parameters at Q4 quantization (weights), plus
/// the flat runtime overhead (KV cache, context, runtime) in GiB.
const Q4_GIB_PER_B_PARAMS: f64 = 0.6;
const RUNTIME_OVERHEAD_GIB: f64 = 1.5;

impl HostModel {
    /// Probe the host once. `data_dir` (when supplied) selects which disk's
    /// class/free-space to report. Never fails — unknown stays unknown.
    pub fn probe(data_dir: Option<&Path>) -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let ram_bytes = match sys.total_memory() {
            0 => None,
            bytes => Some(bytes),
        };

        let cores = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);

        let chip = probe_chip();
        let gpu = probe_gpu(ram_bytes);
        let disk = data_dir.and_then(probe_disk);

        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            chip,
            cores,
            ram_bytes,
            gpu,
            disk,
        }
    }

    /// Coarse machine class by RAM — the word the capability digest names.
    pub fn machine_class(&self) -> &'static str {
        match self.ram_bytes {
            None => "unknown",
            Some(b) if b < 8 * GIB => "constrained",
            Some(b) if b < 16 * GIB => "standard",
            Some(b) if b < 48 * GIB => "performance",
            Some(_) => "workstation",
        }
    }

    /// Memory budget for local model inference: dedicated VRAM, the unified
    /// working set on Apple Silicon, or half of RAM for CPU inference.
    /// `None` when RAM is unknown and no GPU was detected.
    pub fn inference_budget_bytes(&self) -> Option<u64> {
        match &self.gpu {
            GpuInfo::AppleUnified { budget_bytes } => Some(*budget_bytes),
            GpuInfo::Discrete { vram_bytes, .. } => Some(*vram_bytes),
            GpuInfo::Unknown => self
                .ram_bytes
                .map(|ram| (ram as f64 * CPU_INFERENCE_RAM_SHARE) as u64),
        }
    }

    /// Largest local model size (billions of parameters, Q4) the inference
    /// budget comfortably fits, as a human recommendation string.
    pub fn local_model_recommendation(&self) -> Option<&'static str> {
        let budget_gib = self.inference_budget_bytes()? as f64 / GIB as f64;
        Some(if budget_gib >= 45.0 {
            "up to ~70B (Q4)"
        } else if budget_gib >= 21.0 {
            "up to ~30B (Q4)"
        } else if budget_gib >= 10.0 {
            "up to ~13B (Q4)"
        } else if budget_gib >= 6.0 {
            "up to ~7–8B (Q4)"
        } else if budget_gib >= 3.5 {
            "up to ~3B (Q4)"
        } else {
            "1–2B (Q4)"
        })
    }

    /// Check a configured model name against the inference budget. Returns
    /// `None` when the name carries no parseable parameter count (cloud
    /// models, embedders) or the budget is unknown — only a confident
    /// mismatch is worth a warning.
    pub fn model_fit(&self, model: &str) -> Option<ModelFit> {
        let params_b = parse_params_billions(model)?;
        let budget_bytes = self.inference_budget_bytes()?;
        let estimated_bytes = estimated_inference_bytes(params_b);
        Some(ModelFit {
            params_b,
            estimated_bytes,
            budget_bytes,
            exceeds: estimated_bytes > budget_bytes,
        })
    }

    /// The "Host:" line(s) for the capability digest — a few tokens of
    /// situational grounding, always naming the machine class.
    pub fn digest_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        parts.push(match &self.chip {
            Some(chip) => format!("{} ({}) on {}", pretty_os(&self.os), self.arch, chip),
            None => format!("{} ({})", pretty_os(&self.os), self.arch),
        });
        parts.push(format!("{} cores", self.cores));
        if let Some(ram) = self.ram_bytes {
            parts.push(format!("{} RAM", fmt_gib(ram)));
        }
        match &self.gpu {
            GpuInfo::AppleUnified { budget_bytes } => {
                parts.push(format!("~{} GPU-usable (unified)", fmt_gib(*budget_bytes)));
            }
            GpuInfo::Discrete { name, vram_bytes } => {
                parts.push(format!("{} ({} VRAM)", name, fmt_gib(*vram_bytes)));
            }
            GpuInfo::Unknown => {}
        }
        if let Some(disk) = &self.disk {
            let class = match disk.class {
                DiskClass::Ssd => "SSD",
                DiskClass::Hdd => "HDD",
                DiskClass::Unknown => "disk",
            };
            parts.push(format!("{} with {} free", class, fmt_gib(disk.free_bytes)));
        }
        let mut line = format!(
            "Host machine: {} — machine class: {}",
            parts.join(", "),
            self.machine_class()
        );
        if let Some(rec) = self.local_model_recommendation() {
            line.push_str(&format!(" (local models {} fit comfortably)", rec));
        }
        line.push('.');
        line
    }

    /// One-line hardware summary for `doctor` / `init` (no digest framing).
    pub fn summary_line(&self) -> String {
        let mut parts: Vec<String> = vec![format!("{} {}", pretty_os(&self.os), self.arch)];
        if let Some(chip) = &self.chip {
            parts.push(chip.clone());
        }
        parts.push(format!("{} cores", self.cores));
        if let Some(ram) = self.ram_bytes {
            parts.push(format!("{} RAM", fmt_gib(ram)));
        }
        match &self.gpu {
            GpuInfo::AppleUnified { budget_bytes } => {
                parts.push(format!("GPU ~{} (unified)", fmt_gib(*budget_bytes)));
            }
            GpuInfo::Discrete { name, vram_bytes } => {
                parts.push(format!("{} {}", name, fmt_gib(*vram_bytes)));
            }
            GpuInfo::Unknown => {}
        }
        parts.join(" · ")
    }
}

/// A configured model checked against the host's inference budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelFit {
    pub params_b: f64,
    pub estimated_bytes: u64,
    pub budget_bytes: u64,
    pub exceeds: bool,
}

/// Estimated memory to run a `params_b`-billion-parameter model at Q4:
/// quantized weights plus a flat KV-cache/runtime overhead.
fn estimated_inference_bytes(params_b: f64) -> u64 {
    ((params_b * Q4_GIB_PER_B_PARAMS + RUNTIME_OVERHEAD_GIB) * GIB as f64) as u64
}

/// Pull a parameter count (in billions) out of a model name, e.g.
/// `llama3.1:70b` → 70, `qwen2.5:0.5b-instruct` → 0.5, `mixtral:8x7b` → 56.
/// `None` when the name carries no size token (cloud models, embedders).
fn parse_params_billions(model: &str) -> Option<f64> {
    let lower = model.to_ascii_lowercase();
    lower
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == 'x'))
        .find_map(parse_size_token)
}

/// Parse one token of the form `<n>b` or `<experts>x<n>b` (digits first, so
/// names like `embed` or `base` can never match).
fn parse_size_token(token: &str) -> Option<f64> {
    let body = token.strip_suffix('b')?;
    if body.is_empty() || !body.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    match body.split_once('x') {
        Some((experts, per)) => {
            let experts: f64 = experts.parse().ok()?;
            let per: f64 = per.parse().ok()?;
            Some(experts * per)
        }
        None => body.parse().ok(),
    }
}

fn pretty_os(os: &str) -> &str {
    match os {
        "macos" => "macOS",
        "linux" => "Linux",
        "windows" => "Windows",
        other => other,
    }
}

fn fmt_gib(bytes: u64) -> String {
    let gib = bytes as f64 / GIB as f64;
    if gib >= 10.0 {
        format!("{} GiB", gib.round() as u64)
    } else {
        format!("{:.1} GiB", gib)
    }
}

/// CPU/SoC marketing name, best-effort per platform.
#[cfg(target_os = "macos")]
fn probe_chip() -> Option<String> {
    let out = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// CPU/SoC marketing name, best-effort per platform.
#[cfg(target_os = "linux")]
fn probe_chip() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// CPU/SoC marketing name — no probe implemented for this platform.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_chip() -> Option<String> {
    None
}

/// GPU detection, best-effort per platform. On Apple Silicon the GPU shares
/// system RAM (unified memory); elsewhere `nvidia-smi` is tried for a
/// discrete card. Anything else stays `Unknown` rather than guessing.
fn probe_gpu(ram_bytes: Option<u64>) -> GpuInfo {
    if std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64" {
        if let Some(ram) = ram_bytes {
            return GpuInfo::AppleUnified {
                budget_bytes: (ram as f64 * APPLE_UNIFIED_GPU_SHARE) as u64,
            };
        }
        return GpuInfo::Unknown;
    }
    probe_nvidia().unwrap_or(GpuInfo::Unknown)
}

/// First GPU reported by `nvidia-smi`, when present on PATH.
fn probe_nvidia() -> Option<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().next()?;
    let (name, mib) = line.rsplit_once(',')?;
    let mib: u64 = mib.trim().parse().ok()?;
    Some(GpuInfo::Discrete {
        name: name.trim().to_string(),
        vram_bytes: mib * 1024 * 1024,
    })
}

/// Class + free space of the mounted disk containing `data_dir` (longest
/// mount-point prefix wins, so `/` doesn't shadow a dedicated volume).
fn probe_disk(data_dir: &Path) -> Option<DiskInfo> {
    // The data dir may not exist yet (pre-`brain init`); walk up to the
    // nearest existing ancestor so the mount lookup still resolves.
    let mut target = data_dir;
    while !target.exists() {
        target = target.parent()?;
    }
    let target = target.canonicalize().ok()?;

    let disks = sysinfo::Disks::new_with_refreshed_list();
    let best = disks
        .iter()
        .filter(|d| target.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())?;
    let class = match best.kind() {
        sysinfo::DiskKind::SSD => DiskClass::Ssd,
        sysinfo::DiskKind::HDD => DiskClass::Hdd,
        sysinfo::DiskKind::Unknown(_) => DiskClass::Unknown,
    };
    Some(DiskInfo {
        class,
        free_bytes: best.available_space(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with(ram_gib: u64, gpu: GpuInfo) -> HostModel {
        HostModel {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            chip: Some("Apple M3 Max".to_string()),
            cores: 14,
            ram_bytes: Some(ram_gib * GIB),
            gpu,
            disk: Some(DiskInfo {
                class: DiskClass::Ssd,
                free_bytes: 250 * GIB,
            }),
        }
    }

    #[test]
    fn parses_parameter_counts_from_model_names() {
        assert_eq!(parse_params_billions("llama3.1:70b"), Some(70.0));
        assert_eq!(parse_params_billions("qwen2.5:7b-instruct"), Some(7.0));
        assert_eq!(parse_params_billions("qwen2.5:0.5b"), Some(0.5));
        assert_eq!(parse_params_billions("mixtral:8x7b"), Some(56.0));
        assert_eq!(parse_params_billions("phi3:14b-medium-q4_K_M"), Some(14.0));
        // No size token → no claim (cloud models, embedders, bare names).
        assert_eq!(parse_params_billions("gpt-4o"), None);
        assert_eq!(parse_params_billions("claude-sonnet-4-6"), None);
        assert_eq!(parse_params_billions("nomic-embed-text"), None);
        assert_eq!(parse_params_billions("llama3"), None);
        // `b` not preceded by digits can't match.
        assert_eq!(parse_params_billions("model-base-b"), None);
    }

    #[test]
    fn machine_class_boundaries() {
        let class = |gib: u64| model_with(gib, GpuInfo::Unknown).machine_class();
        assert_eq!(class(4), "constrained");
        assert_eq!(class(8), "standard");
        assert_eq!(class(16), "performance");
        assert_eq!(class(48), "workstation");
        assert_eq!(class(64), "workstation");
        let mut unknown = model_with(8, GpuInfo::Unknown);
        unknown.ram_bytes = None;
        assert_eq!(unknown.machine_class(), "unknown");
    }

    #[test]
    fn digest_line_names_the_machine_class() {
        let m = model_with(
            36,
            GpuInfo::AppleUnified {
                budget_bytes: 27 * GIB,
            },
        );
        let line = m.digest_line();
        assert!(line.contains("machine class: performance"), "{line}");
        assert!(line.contains("Apple M3 Max"), "{line}");
        assert!(line.contains("36 GiB RAM"), "{line}");
        assert!(line.contains("SSD"), "{line}");
    }

    #[test]
    fn oversized_model_exceeds_budget_and_fitting_model_does_not() {
        let m = model_with(
            16,
            GpuInfo::AppleUnified {
                budget_bytes: 12 * GIB,
            },
        );
        let over = m.model_fit("llama3.1:70b").expect("size parsed");
        assert!(over.exceeds, "70B must not fit a 12 GiB budget");
        let fits = m.model_fit("qwen2.5:7b").expect("size parsed");
        assert!(!fits.exceeds, "7B fits a 12 GiB budget");
        // Cloud model name → no fit claim at all.
        assert!(m.model_fit("gpt-4o").is_none());
    }

    #[test]
    fn inference_budget_prefers_gpu_then_falls_back_to_ram_share() {
        let unified = model_with(
            36,
            GpuInfo::AppleUnified {
                budget_bytes: 27 * GIB,
            },
        );
        assert_eq!(unified.inference_budget_bytes(), Some(27 * GIB));

        let discrete = model_with(
            64,
            GpuInfo::Discrete {
                name: "NVIDIA RTX 4090".to_string(),
                vram_bytes: 24 * GIB,
            },
        );
        assert_eq!(discrete.inference_budget_bytes(), Some(24 * GIB));

        let cpu_only = model_with(16, GpuInfo::Unknown);
        assert_eq!(cpu_only.inference_budget_bytes(), Some(8 * GIB));
    }

    #[test]
    fn recommendation_ladder_tracks_budget() {
        let rec = |budget_gib: u64| {
            model_with(
                128,
                GpuInfo::Discrete {
                    name: "test".to_string(),
                    vram_bytes: budget_gib * GIB,
                },
            )
            .local_model_recommendation()
            .unwrap()
        };
        assert_eq!(rec(48), "up to ~70B (Q4)");
        assert_eq!(rec(24), "up to ~30B (Q4)");
        assert_eq!(rec(12), "up to ~13B (Q4)");
        assert_eq!(rec(8), "up to ~7–8B (Q4)");
        assert_eq!(rec(4), "up to ~3B (Q4)");
        assert_eq!(rec(2), "1–2B (Q4)");
    }

    #[test]
    fn probe_never_panics_and_fills_basics() {
        let m = HostModel::probe(Some(Path::new("/nonexistent/brain/data")));
        assert!(m.cores >= 1);
        assert!(!m.os.is_empty());
        assert!(!m.arch.is_empty());
        // RAM should resolve on every CI platform we build on.
        assert!(m.ram_bytes.is_some());
        let _ = m.digest_line();
        let _ = m.summary_line();
    }
}
