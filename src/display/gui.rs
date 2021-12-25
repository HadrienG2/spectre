//! WGPU-based spectrum display

use crate::{display::FrameResult, Result};
use log::{debug, info, trace, warn};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, ModifiersState, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

/// GPU-accelerated spectrum display
pub struct GuiDisplay {
    /// Event loop
    event_loop: Option<EventLoop<()>>,

    /// Window
    window: Window,

    /// Last known window inner size
    inner_size: PhysicalSize<u32>,
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

        Ok(Self {
            event_loop: Some(event_loop),
            window,
            inner_size,
        })
    }

    /// Start the event loop, run a user-provided callback on every frame
    pub fn run_event_loop(
        mut self,
        frame_callback: impl FnMut(&mut Self) -> Result<FrameResult> + 'static,
    ) -> ! {
        // TODO: Render a first frame and make the window visible
        let mut keyboard_modifiers = ModifiersState::default();
        let mut frame_callback = Some(frame_callback);
        self.event_loop
            .take()
            .expect("Event loop should be present")
            .run(move |event, _target, control_flow| {
                match event {
                    // Handle our window's events
                    Event::WindowEvent { window_id, event } if window_id == self.window.id() => {
                        match event {
                            // Handle various app termination events
                            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                                *control_flow = ControlFlow::Exit
                            }

                            // Handle keyboard input
                            WindowEvent::ModifiersChanged(modifiers) => {
                                keyboard_modifiers = modifiers
                            }
                            //
                            ref event @ WindowEvent::KeyboardInput { ref input, .. } => {
                                if input.state != ElementState::Pressed {
                                    trace!("Unhandled non-press keyboard event : {:?}", event);
                                }
                                match input.virtual_keycode {
                                    Some(VirtualKeyCode::F4) if keyboard_modifiers.alt() => {
                                        *control_flow = ControlFlow::Exit
                                    }
                                    _ => trace!("Unhandled key-press event : {:?}", event),
                                }
                            }
                            //
                            // TODO: Handle run-time settings changes via
                            //       WindowEvent::ReceivedCharacter(char)

                            // Resize and DPI changes
                            WindowEvent::Resized(new_size) => {
                                if new_size != self.inner_size {
                                    // FIXME: Implement
                                    panic!("Window resizing is not supported yet");
                                }
                            }
                            WindowEvent::ScaleFactorChanged { .. } => {
                                // FIXME: Implement
                                panic!("DPI scaling is not supported yet");
                            }

                            // Log events we don't handle yet
                            _ => trace!("Unhandled winit window event: {:?}", event),
                        }
                    }

                    // Render new frame once all events have been processed
                    Event::MainEventsCleared => {
                        self.window.request_redraw();
                    }
                    //
                    Event::RedrawRequested(window_id) if window_id == self.window.id() => {
                        match frame_callback
                            .as_mut()
                            .expect("Frame callback should be present")(
                            &mut self
                        ) {
                            Ok(FrameResult::Continue) => {}
                            Ok(FrameResult::Stop) => *control_flow = ControlFlow::Exit,
                            Err(e) => panic!("Frame processing failed: {}", e),
                        }
                    }

                    // Help out winit's event loop at handling drop
                    Event::LoopDestroyed => {
                        std::mem::drop(frame_callback.take());
                    }

                    // Log events we don't handle yet
                    _ => trace!("Unhandled winit event: {:?}", event),
                }
            })
    }
}
