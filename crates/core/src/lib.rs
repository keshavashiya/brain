//! # Brain Core
//!
//! Orchestrator that wires all brain subsystems together.
//!
//! Provides:
//! - Configuration management (figment + YAML)
//! - Subsystem initialization and dependency injection
//! - Message pipeline: Thalamus → Hippocampus → Cortex → Response
//! - Error handling and graceful degradation

pub mod config;

pub use config::{AccessConfig, ApiKeyConfig, BrainConfig};
