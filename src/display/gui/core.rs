//! Core context that you would find in pretty much any WGPU-based application

use crate::{
    display::gui::{Event, EventLoop},
    Result,
};
use log::{debug, error, info, trace};
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, PresentMode,
    Queue, RequestAdapterOptions, Surface, SurfaceConfiguration, SurfaceError, SurfaceTexture,
    TextureUsages,
};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, ModifiersState, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
    window::{Window, WindowBuilder},
};

/// Consequences of an event that was handled by the core context
pub enum HighLevelEvent {
    /// A resize event occurred, possibly accompanied by a DPI change
    Resized { scale_factor_ratio: Option<f32> },

    /// It is time to redraw the display
    Redraw,

    /// This is the last call before the event loop is destroyed, clean up
    Exit,
}

/// Core context that you would find in pretty much any WGPU-based application
pub struct CoreContext {
    /// Window
    window: Window,

    /// Last observed DPI scale factor
    scale_factor: f32,

    /// Associated GPU surface
    surface: Surface,

    /// GPU surface configuration (to recreate it when e.g. window is resized)
    surface_config: SurfaceConfiguration,

    /// GPU device context
    device: Device,

    /// Queue for submitting work to the GPU device
    queue: Queue,

    /// Keyboard modifier state
    keyboard_modifiers: ModifiersState,
}
//
impl CoreContext {
    /// Set up the event loop and basic GPU rendering context
    pub fn new(event_loop: &EventLoop) -> Result<Self> {
        // Configure window
        let window = WindowBuilder::new()
            .with_resizable(true)
            .with_title("Spectre")
            .with_visible(false)
            .with_transparent(false)
            // TODO: with_window_icon
            .build(&event_loop)?;
        let inner_size = window.inner_size();
        let scale_factor = window.scale_factor() as f32;
        info!(
            "Built window with id {:?}, inner physical size {}x{}, DPI scale factor {}",
            window.id(),
            inner_size.width,
            inner_size.height,
            scale_factor
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
            "In other words, it does NOT support WebGPU features {:?}",
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

        // Define minimal device requirements
        // We may need to handle screen-sized textures on any available monitor
        let mut limits = Limits::downlevel_defaults();
        for monitor in window.available_monitors() {
            match monitor.size() {
                PhysicalSize {
                    width: 0,
                    height: 0,
                } => {}
                PhysicalSize { width, height } => {
                    let max_size = width.max(height);
                    limits.max_texture_dimension_1d = limits.max_texture_dimension_1d.max(max_size);
                    limits.max_texture_dimension_2d = limits.max_texture_dimension_2d.max(max_size);
                }
            }
        }
        debug!("Want a device that goes up to {:#?}", limits);

        // Configure device and queue
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("GPU"),
                features: Features::empty(),
                limits,
            },
            None,
        ))?;

        // Configure the surface for rendering:
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: preferred_surface_format,
            width: inner_size.width,
            height: inner_size.height,
            present_mode: PresentMode::Fifo,
        };
        surface.configure(&device, &surface_config);

        // Return to caller
        Ok(Self {
            window,
            scale_factor,
            surface,
            surface_config,
            device,
            queue,
            keyboard_modifiers: ModifiersState::default(),
        })
    }

    /// Access the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Access the queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Try to access the current window surface texture
    pub fn current_surface_texture(&self) -> Result<SurfaceTexture, SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Query current display surface configuration
    pub fn surface_config(&self) -> &SurfaceConfiguration {
        &self.surface_config
    }

    /// Query current DPI scale factor
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// Show the window (which should have been previously painted)
    pub fn show_window(&mut self) {
        self.window.set_visible(true);
    }

    /// Process a winit event, tell the caller about events of particular interest
    pub fn handle_event(
        &mut self,
        event: Event,
        control_flow: &mut ControlFlow,
    ) -> Option<HighLevelEvent> {
        match event {
            // Handle our window's events
            Event::WindowEvent { window_id, event } if window_id == self.window.id() => {
                match event {
                    // Handle various app termination events
                    WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                        *control_flow = ControlFlow::Exit;
                        None
                    }

                    // Handle keyboard input
                    WindowEvent::ModifiersChanged(modifiers) => {
                        self.keyboard_modifiers = modifiers;
                        None
                    }
                    //
                    ref event @ WindowEvent::KeyboardInput { ref input, .. } => {
                        if input.state != ElementState::Pressed {
                            trace!("Unhandled non-press keyboard event : {:?}", event);
                            return None;
                        }
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::F4) if self.keyboard_modifiers.alt() => {
                                *control_flow = ControlFlow::Exit;
                            }
                            _ => {
                                trace!("Unhandled key-press event : {:?}", event);
                            }
                        }
                        None
                    }

                    // Resize and DPI changes
                    WindowEvent::Resized(new_size) => {
                        self.surface_config.width = new_size.width;
                        self.surface_config.height = new_size.height;
                        Some(HighLevelEvent::Resized {
                            scale_factor_ratio: None,
                        })
                    }
                    WindowEvent::ScaleFactorChanged {
                        scale_factor,
                        new_inner_size,
                    } => {
                        self.surface_config.width = new_inner_size.width;
                        self.surface_config.height = new_inner_size.height;
                        let new_scale_factor = scale_factor as f32;
                        let scale_factor_ratio = new_scale_factor / self.scale_factor;
                        self.scale_factor = new_scale_factor;
                        Some(HighLevelEvent::Resized {
                            scale_factor_ratio: Some(scale_factor_ratio),
                        })
                    }

                    // Ignore chatty events we don't care about
                    WindowEvent::AxisMotion { .. }
                    | WindowEvent::CursorMoved { .. }
                    | WindowEvent::CursorEntered { .. }
                    | WindowEvent::CursorLeft { .. }
                    | WindowEvent::Moved(_) => None,

                    // Log other events we don't handle yet
                    _ => {
                        trace!("Unhandled window event: {:?}", event);
                        None
                    }
                }
            }

            // Render new frame once all events have been processed
            Event::MainEventsCleared => {
                self.window.request_redraw();
                None
            }
            //
            Event::RedrawRequested(window_id) if window_id == self.window.id() => {
                Some(HighLevelEvent::Redraw)
            }

            // Help out winit's event loop at handling drop
            Event::LoopDestroyed => Some(HighLevelEvent::Exit),

            // Ignore chatty events we don't care about
            Event::NewEvents(_) | Event::RedrawEventsCleared | Event::DeviceEvent { .. } => None,

            // Log other events we don't handle yet
            _ => {
                trace!("Unhandled winit event: {:?}", event);
                None
            }
        }
    }

    /// Recreate display surface, typically after a window resize
    pub fn recreate_surface(&mut self) {
        self.surface.configure(&self.device, &self.surface_config);
    }
}
