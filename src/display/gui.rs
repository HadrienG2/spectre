//! WGPU-based spectrum display

use crate::Result;
use log::{debug, info, warn};
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

/// GPU-accelerated spectrum display
pub struct GuiDisplay {
    /// Event loop
    event_loop: EventLoop<()>,
}
//
impl GuiDisplay {
    /// Set up the GPU display
    pub fn new(amp_scale: f32) -> Result<Self> {
        assert!(amp_scale > 0.0);

        // Set up event loop
        let event_loop = EventLoop::new();

        // Infer maximal window size from monitor width and height
        // TODO: Use this info later for buffer allocation, to allow resize
        let (max_width, max_height) = event_loop
            .available_monitors()
            .enumerate()
            .map(|(idx, monitor)| {
                let PhysicalSize { width, height } = monitor.size();
                let monitor_id = || {
                    format!(
                        "#{}{}",
                        idx,
                        monitor
                            .name()
                            .map(|mut name| {
                                name.insert_str(0, " (");
                                name.push_str(")");
                                name
                            })
                            .unwrap_or("".to_string())
                    )
                };
                if (width, height) == (0, 0) {
                    warn!(
                        "Size of monitor {} is unknown, assuming 1080p",
                        monitor_id()
                    );
                    (1920, 1080)
                } else {
                    debug!(
                        "Detected monitor {} with {}x{} physical pixels",
                        monitor_id(),
                        width,
                        height
                    );
                    (width, height)
                }
            })
            .fold((0, 0), |(max_width, max_height), (width, height)| {
                (width + max_width, height + max_height)
            });
        info!(
            "Inferred a maximal window size of {}x{}px",
            max_width, max_height
        );

        // Configure window
        let window = WindowBuilder::new()
            // FIXME: We do not support resizing yet
            .with_resizable(false)
            .with_title("Spectre")
            .with_visible(false)
            .with_transparent(false)
            // TODO: with_window_icon
            .build(&event_loop)?;
        let inner_size = window.inner_size();
        info!(
            "Built window with id {:?}, inner physical size {}x{}, DPI scale factor {}",
            window.id(),
            inner_size.width,
            inner_size.height,
            window.scale_factor()
        );

        // TODO: Configure GPU rendering

        Ok(Self { event_loop })
    }
}
