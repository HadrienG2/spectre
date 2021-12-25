//! WebGPU-based spectrum display

use crate::{
    display::{FrameInput, FrameResult},
    Result,
};
use log::{debug, error, info, trace};
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, PresentMode,
    Queue, RequestAdapterOptions, Surface, SurfaceConfiguration, TextureUsages,
};
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

    /// Associated GPU surface
    surface: Surface,

    /// GPU surface configuration
    surface_config: SurfaceConfiguration,

    /// GPU device context
    device: Device,

    /// Queue for submitting work to the GPU device
    queue: Queue,

    /// Truth that the window size has changed
    resized: bool,
}
//
impl GuiDisplay {
    /// Set up the GPU display
    pub fn new(amp_scale: f32) -> Result<Self> {
        assert!(amp_scale > 0.0);

        // Set up event loop
        let event_loop = EventLoop::new();

        // Configure window
        let window = WindowBuilder::new()
            .with_resizable(true)
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

        // Initialize WebGPU adapter and presentation surface
        let instance = Instance::new(Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No compatible GPU found");

        // Describe adapter features
        let adapter_features = adapter.features();
        if adapter_features == Features::all() {
            info!("Adapter supports all standard and native WebGPU features");
        } else if adapter_features.contains(Features::all_webgpu_mask()) {
            let native_features = adapter_features.difference(Features::all_webgpu_mask());
            info!(
                "Adapter supports all standard WebGPU features and also native features {:?}",
                native_features
            );
        } else {
            info!("Adapter supports WebGPU features {:?}", adapter.features());
        }
        debug!(
            "In other words, it does NOT support features {:?}",
            Features::all().difference(adapter_features)
        );

        // Describe adapter limits
        let adapter_limits = adapter.limits();
        if adapter_limits >= Limits::default() {
            info!("Adapter supports the default WebGPU limits");
        } else if adapter_limits >= Limits::downlevel_defaults() {
            info!("Adapter supports the down-level WebGPU limits");
        } else {
            error!("Detected GPU does not even support the down-level WebGPU limits");
        }
        debug!(
            "To be more precise, adapter goes up to {:#?}",
            adapter.limits()
        );

        // Describe adapter WebGPU compliance limits, if any
        let downlevel_properties = adapter.get_downlevel_properties();
        if !downlevel_properties.is_webgpu_compliant() {
            info!(
                "Adapter is not fully WebGPU compliant, it has additional limits {:#?}",
                adapter.get_downlevel_properties(),
            );
        }

        // Describe preferred presentation surface format
        let preferred_surface_format = surface
            .get_preferred_format(&adapter)
            .expect("By the above constraint, the surface should be compatible with the adapter");
        info!(
            "Got surface with preferred format {:?} and associated features {:?}",
            preferred_surface_format,
            adapter.get_texture_format_features(preferred_surface_format),
        );

        // Configure device and queue
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some(&"GPU"),
                features: Features::empty(),
                limits: Limits::downlevel_defaults(),
            },
            None,
        ))?;
        debug!("Got a device that goes up to {:#?}", device.limits());

        // Set up device error handling
        device.on_uncaptured_error(|e| error!("Uncaptured WebGPU device error: {}", e));

        // Configure the surface for rendering:
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: preferred_surface_format,
            width: inner_size.width,
            height: inner_size.height,
            present_mode: PresentMode::Fifo,
        };
        surface.configure(&device, &surface_config);

        // ...and we're ready!
        Ok(Self {
            event_loop: Some(event_loop),
            window,
            surface,
            surface_config,
            device,
            queue,
            resized: false,
        })
    }

    /// Report terminal width in pixels
    pub fn width(&self) -> usize {
        self.surface_config.width as _
    }

    /// Start the event loop, run a user-provided callback on every frame
    pub fn run_event_loop(
        mut self,
        frame_callback: impl FnMut(&mut Self, FrameInput) -> Result<FrameResult> + 'static,
    ) -> ! {
        // TODO: Render a first frame and make the window visible
        self.window.set_visible(true);
        let mut keyboard_modifiers = ModifiersState::default();
        let mut frame_callback = Some(frame_callback);
        let mut resized = false;
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
                                if new_size
                                    != (PhysicalSize {
                                        width: self.surface_config.width,
                                        height: self.surface_config.height,
                                    })
                                {
                                    self.surface_config.width = new_size.width;
                                    self.surface_config.height = new_size.height;
                                    resized = true;
                                }
                            }
                            WindowEvent::ScaleFactorChanged { .. } => {
                                // FIXME: Implement once we have stuff to scale
                                panic!("DPI scaling is not supported yet");
                            }

                            // Ignore chatty events we don't care about
                            WindowEvent::AxisMotion { .. }
                            | WindowEvent::CursorMoved { .. }
                            | WindowEvent::CursorEntered { .. }
                            | WindowEvent::CursorLeft { .. }
                            | WindowEvent::Moved(_) => {}

                            // Log other events we don't handle yet
                            _ => trace!("Unhandled window event: {:?}", event),
                        }
                    }

                    // Render new frame once all events have been processed
                    Event::MainEventsCleared => {
                        self.window.request_redraw();
                    }
                    //
                    Event::RedrawRequested(window_id) if window_id == self.window.id() => {
                        let mut frame_input = FrameInput {
                            new_display_width: None,
                        };
                        if resized {
                            frame_input.new_display_width = Some(self.surface_config.width as _);
                            self.handle_resize();
                            resized = false;
                        }
                        match frame_callback
                            .as_mut()
                            .expect("Frame callback should be present")(
                            &mut self, frame_input
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

                    // Ignore chatty events we don't care about
                    Event::NewEvents(_)
                    | Event::RedrawEventsCleared
                    | Event::DeviceEvent { .. } => {}

                    // Log other events we don't handle yet
                    _ => trace!("Unhandled winit event: {:?}", event),
                }
            })
    }

    /// Restore the terminal to its initial state
    pub fn reset_terminal(&mut self) -> Result<()> {
        // The GUI backend does not alter the terminal state, so this is easy
        Ok(())
    }

    /// Reallocate structures that depend on the window size after a resize
    fn handle_resize(&mut self) {
        self.surface.configure(&self.device, &self.surface_config);
    }
}
