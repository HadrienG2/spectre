//! Mechanisms for displaying the computed spectra

// FIXME: Make this and the crossterm dep conditional on a Cargo feature
mod cli;

pub use cli::CliDisplay;

/// Output of the frame display hook
pub enum FrameResult {
    /// Call me back on the next frame
    Continue,

    /// It's time to exit the event loop
    Stop,
}
