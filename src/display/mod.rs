//! Mechanisms for displaying the computed spectra

// FIXME: Make this and the crossterm dep conditional on a Cargo feature
mod cli;
mod gui;

pub use cli::CliDisplay;
pub use gui::GuiDisplay;

/// Output of the frame display hook
pub enum FrameResult {
    /// Call me back on the next frame
    Continue,

    /// It's time to exit the event loop
    Stop,
}
