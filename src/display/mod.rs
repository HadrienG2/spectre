//! Mechanisms for displaying the computed spectra

// FIXME: Make this and the crossterm dep conditional on a Cargo feature
mod cli;

pub use cli::CliDisplay;
