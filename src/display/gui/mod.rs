//! WebGPU-based spectrum display

use crate::{
    display::{FrameInput, FrameResult},
    Result,
};
use log::{debug, error, info, trace};
use wgpu::{
    Backends, BlendState, ColorTargetState, ColorWrites, Device, DeviceDescriptor, Extent3d, Face,
    Features, FragmentState, FrontFace, ImageDataLayout, Instance, Limits, MultisampleState,
    PipelineLayoutDescriptor, PolygonMode, PowerPreference, PresentMode, PrimitiveState,
    PrimitiveTopology, Queue, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, Surface, SurfaceConfiguration, SurfaceError, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    VertexState,
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

    /// GPU surface configuration (to recreate it when e.g. window is resized)
    surface_config: SurfaceConfiguration,

    /// GPU device context
    device: Device,

    /// Queue for submitting work to the GPU device
    queue: Queue,

    /// Live spectrum render pipeline
    spectrum_pipeline: RenderPipeline,

    /// Live spectrum texture
    spectrum_texture: Texture,

    /// Live spectrum texture descriptor (to recreate it on window resize)
    spectrum_texture_desc: TextureDescriptor<'static>,
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

        // Define minimal device requirements
        let mut limits = Limits::downlevel_defaults();
        let monitor = window
            .current_monitor()
            .or(window.primary_monitor())
            .or(window.available_monitors().next());
        match monitor.map(|m| m.size()) {
            None
            | Some(PhysicalSize {
                width: 0,
                height: 0,
            }) => {}
            Some(PhysicalSize { width, height }) => {
                let max_size = width.max(height);
                limits.max_texture_dimension_1d = max_size;
                limits.max_texture_dimension_2d = max_size;
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

        // Load spectrum shader
        let spectrum_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Spectrum shader"),
            source: ShaderSource::Wgsl(include_str!("spectrum.wgsl").into()),
        });

        // Set up spectrum pipeline layout
        let spectrum_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spectrum pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        // Set up spectrum render pipeline
        let spectrum_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Spectrum pipeline"),
            layout: Some(&spectrum_pipeline_layout),
            vertex: VertexState {
                module: &spectrum_shader,
                entry_point: "vertex",
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            // TODO: Enable MSAA
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &spectrum_shader,
                entry_point: "fragment",
                targets: &[ColorTargetState {
                    format: surface_config.format,
                    // TODO: Enable alpha blending once we went to
                    //       render older spectra with a blur effect
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        // Set up a texture to hold live spectrum data
        let spectrum_texture_desc = TextureDescriptor {
            label: Some("Spectrum texture"),
            size: Extent3d {
                width: surface_config.height as _,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D1,
            format: TextureFormat::R32Float,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        };
        let spectrum_texture = device.create_texture(&spectrum_texture_desc);

        // ...and we're ready!
        Ok(Self {
            event_loop: Some(event_loop),
            window,
            surface,
            surface_config,
            device,
            queue,
            spectrum_pipeline,
            spectrum_texture,
            spectrum_texture_desc,
        })
    }

    /// Report desired spectrum length in bins
    pub fn spectrum_len(&self) -> usize {
        self.surface_config.height as _
    }

    /// Start the event loop, run a user-provided callback on every frame
    pub fn run_event_loop(
        mut self,
        mut frame_callback: impl FnMut(&mut Self, FrameInput) -> Result<FrameResult> + 'static,
    ) -> ! {
        // Render a first frame and make the window visible
        let first_result = frame_callback(
            &mut self,
            FrameInput {
                new_spectrum_len: None,
            },
        )
        .expect("Failed to render first frame");
        if first_result == FrameResult::Stop {
            std::mem::drop(frame_callback);
            std::process::exit(0);
        }
        self.window.set_visible(true);

        // Start the actual event loop
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
                            new_spectrum_len: None,
                        };
                        if resized {
                            frame_input.new_spectrum_len = Some(self.surface_config.height as _);
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

    /// Display a spectrum
    pub fn render(&mut self, data: &[f32]) -> Result<()> {
        // Try to access the next window texture
        let window_texture = match self.surface.get_current_texture() {
            // Succeeded
            Ok(texture) => texture,

            // Some errors will just resolve themselves, perhaps with some help
            Err(SurfaceError::Timeout | SurfaceError::Outdated) => return Ok(()),
            Err(SurfaceError::Lost) => {
                self.handle_resize();
                return Ok(());
            }

            // Other errors are presumed to be serious ones and kill the app
            #[allow(unreachable_patterns)]
            Err(e @ (SurfaceError::OutOfMemory | _)) => Err(e)?,
        };

        // Acquire a texture view (needed for render passes)
        let window_view = window_texture.texture.create_view(&TextureViewDescriptor {
            label: Some("Window surface texture view"),
            ..TextureViewDescriptor::default()
        });

        // Prepare to render on the screen texture
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Spectrum render encoder"),
            });

        // Send the new spectrum data to the device
        self.queue.write_texture(
            self.spectrum_texture.as_image_copy(),
            // FIXME: Use bytemuck
            unsafe { std::mem::transmute(data) },
            ImageDataLayout::default(),
            self.spectrum_texture_desc.size,
        );

        {
            // Set up a render pass with a black clear color
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Spectrum render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &window_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            // TODO: Bind

            // Draw the live spectrum
            // TODO: Add more instances for older spectra
            render_pass.set_pipeline(&self.spectrum_pipeline);
            render_pass.draw(0..4, 0..1);

            // TODO: Render a spectrogram too
        }

        // Submit our render command
        self.queue.submit(Some(encoder.finish()));

        // Make sure the output gets displayed on the screen
        window_texture.present();
        Ok(())
    }

    /// Restore the terminal to its initial state
    pub fn reset_terminal(&mut self) -> Result<()> {
        // The GUI backend does not alter the terminal state, so this is easy
        Ok(())
    }

    /// Reallocate structures that depend on the window size after a resize
    fn handle_resize(&mut self) {
        // Recreate display surface
        self.surface.configure(&self.device, &self.surface_config);

        // Recreate live spectrum texture
        self.spectrum_texture_desc.size.width = self.surface_config.height as _;
        self.spectrum_texture = self.device.create_texture(&self.spectrum_texture_desc);

        // TODO: Once we do spectrograms, don't just drop the old data, resample
        //       it with a compute shader for nicer UX
    }
}
