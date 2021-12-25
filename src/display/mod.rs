//! Mechanisms for displaying the computed spectra

// FIXME: Make this and the crossterm dep conditional on a Cargo feature
#[cfg(feature = "cli")]
mod cli;
#[cfg(feature = "gui")]
mod gui;

#[cfg(feature = "cli")]
pub use cli::CliDisplay;
#[cfg(feature = "gui")]
pub use gui::GuiDisplay;

/// Output of the frame display hook
pub enum FrameResult {
    /// Call me back on the next frame
    Continue,

    /// It's time to exit the event loop
    Stop,
}
