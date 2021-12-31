//! Mechanisms for displaying the computed spectra

#[cfg(feature = "cli")]
mod cli;
#[cfg(feature = "gui")]
mod gui;

#[cfg(feature = "cli")]
pub use cli::CliDisplay;
#[cfg(feature = "gui")]
pub use gui::GuiDisplay;

/// Input of the frame display hook
pub struct FrameInput {
    /// New spectrum length (if any)
    pub new_spectrum_len: Option<usize>,
}

/// Output of the frame display hook
#[must_use]
#[derive(Debug, PartialEq)]
pub enum FrameResult {
    /// Call me back on the next frame
    Continue,

    /// It's time to exit the event loop
    Stop,
}
