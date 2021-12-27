//! WebGPU-based spectrum display
// FIXME: This module is getting very long and should be split into smaller entities

use crate::{
    display::{FrameInput, FrameResult},
    Result,
};
use colorous::{Color, INFERNO};
use crevice::std140::{AsStd140, Std140};
use half::f16;
use log::{debug, error, info, trace};
use std::num::NonZeroU64;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AddressMode, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferBinding, BufferBindingType, BufferUsages, ColorTargetState, ColorWrites, Device,
    DeviceDescriptor, Extent3d, Face, Features, FilterMode, FragmentState, FrontFace,
    ImageDataLayout, Instance, Limits, MultisampleState, PipelineLayoutDescriptor, PolygonMode,
    PowerPreference, PresentMode, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Surface, SurfaceConfiguration,
    SurfaceError, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, ModifiersState, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

/// Default fraction of the window used by the live spectrum
const DEFAULT_SPECTRUM_WIDTH: f32 = 0.25;

/// Uniform for passing UI settings to GPU shaders
//
// NOTE: According to the Learn WGPU tutorial...
//       "To make uniform buffers portable they have to be std140 and not
//       std430. Uniform structs have to be std140. Storage structs have to be
//       std430. Storage buffers for compute shaders can be std140 or std430."
//
#[derive(AsStd140)]
struct SettingsUniform {
    /// Horizontal fraction of the window that is occupied by the live spectrum
    spectrum_width: f32,

    /// Range of amplitudes that we can display in dB
    amp_scale: f32,
}

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

    /// UI settings
    // TODO: Once we start to update this, remember to upload into settings_buffer
    settings: SettingsUniform,

    /// Buffer for holding UI settings on the device
    settings_buffer: Buffer,

    /// Bind group for things that are never rebound (sampler, UI settings)
    static_bind_group: BindGroup,

    /// Live spectrum texture
    spectrum_texture: Texture,

    /// Live spectrum texture descriptor (to recreate it on window resize)
    spectrum_texture_desc: TextureDescriptor<'static>,

    /// Live spectrum bind group that depends on the window size
    spectrum_sized_bind_group: BindGroup,

    /// Live spectrum size-sensitive bind group layout (to recreate bind group on resize)
    spectrum_sized_bind_group_layout: BindGroupLayout,

    /// Live spectrum render pipeline
    spectrum_pipeline: RenderPipeline,

    /// Transit buffer for casting spectrum magnitudes to single precision
    half_spectrum_data: Box<[f16]>,
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
                limits.max_texture_dimension_1d = limits.max_texture_dimension_1d.max(max_size);
                limits.max_texture_dimension_2d = limits.max_texture_dimension_2d.max(max_size);
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

        // Set up UI settings uniform
        let settings = SettingsUniform {
            spectrum_width: DEFAULT_SPECTRUM_WIDTH,
            amp_scale,
        };
        //
        let settings_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("UI settings uniform"),
            contents: settings.as_std140().as_bytes(),
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
        });

        // Set up spectrum and spectrogram texture sampling
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Spectrum & spectrogram sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            ..Default::default()
        });

        // Set up spectrum and spectrogram color palette
        let palette_len = device.limits().max_texture_dimension_1d;
        let palette_data = (0..palette_len as usize)
            .flat_map(|idx| {
                let Color { r, g, b } = INFERNO.eval_rational(idx, palette_len as usize);
                [r, g, b, 255]
            })
            .collect::<Box<[_]>>();
        //
        let palette_texture = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: Some("Palette texture"),
                size: Extent3d {
                    width: palette_len,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
            },
            &palette_data[..],
        );
        //
        let palette_texture_view = palette_texture.create_view(&TextureViewDescriptor {
            label: Some("Palette texture view"),
            ..Default::default()
        });

        // Set up the common bind group for things that don't need rebinding
        let static_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Spectrum & spectrogram static bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<SettingsUniform>() as u64,
                            ),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D1,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        //
        let static_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrum & spectrogram static bind group"),
            layout: &static_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&palette_texture_view),
                },
            ],
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
            format: TextureFormat::R16Float,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        };

        // TODO: Do similar stuff for the 2D spectrogram texture, except it must
        //       be usable as a storage texture so that the live spectrum shader
        //       can write new spectrum data to it.

        // Define the live spectrum size-sensitive bind group layout
        let spectrum_sized_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Spectrum size-sensitive bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D1,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // TODO: Add spectrogram as a storage image
                ],
            });

        // Load live spectrum shader
        let spectrum_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Spectrum shader"),
            source: ShaderSource::Wgsl(include_str!("spectrum.wgsl").into()),
        });

        // Set up spectrum pipeline layout
        let spectrum_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spectrum pipeline layout"),
            bind_group_layouts: &[&static_bind_group_layout, &spectrum_sized_bind_group_layout],
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

        // Set up spectrum texture and associated bind group
        let (spectrum_texture, spectrum_sized_bind_group) =
            Self::configure_spectrum_sized_bind_group(
                &device,
                &spectrum_texture_desc,
                &spectrum_sized_bind_group_layout,
            );

        // Set up half-precision spectrum data
        let half_spectrum_data = std::iter::repeat(f16::default())
            .take(surface_config.height as _)
            .collect();

        // ...and we're ready!
        Ok(Self {
            event_loop: Some(event_loop),
            window,
            surface,
            surface_config,
            device,
            queue,
            settings,
            settings_buffer,
            static_bind_group,
            spectrum_texture,
            spectrum_texture_desc,
            spectrum_sized_bind_group,
            spectrum_sized_bind_group_layout,
            spectrum_pipeline,
            half_spectrum_data,
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

            // Other errors are presumed to be serious ones and kill the app*
            Err(e @ SurfaceError::OutOfMemory) => Err(e)?,
        };

        // Acquire a texture view (needed for render passes)
        let window_view = window_texture.texture.create_view(&TextureViewDescriptor {
            label: Some("Window surface texture view"),
            ..Default::default()
        });

        // Prepare to render on the screen texture
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Spectrum render encoder"),
            });

        // Convert the new spectrum data to half precision
        for (dest, &src) in self.half_spectrum_data.iter_mut().zip(data) {
            *dest = f16::from_f32(src);
        }

        // Send the new spectrum data to the device
        self.queue.write_texture(
            self.spectrum_texture.as_image_copy(),
            bytemuck::cast_slice(&self.half_spectrum_data[..]),
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

            // Bind common data to bind group 0
            render_pass.set_bind_group(0, &self.static_bind_group, &[]);

            // Bind size-sensitive live spectrum data to bind group 1
            render_pass.set_bind_group(1, &self.spectrum_sized_bind_group, &[]);

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

        // Recreate live spectrum texture and associated bind group
        self.spectrum_texture_desc.size.width = self.surface_config.height as _;
        let (spectrum_texture, spectrum_sized_bind_group) =
            Self::configure_spectrum_sized_bind_group(
                &self.device,
                &self.spectrum_texture_desc,
                &self.spectrum_sized_bind_group_layout,
            );
        self.spectrum_texture = spectrum_texture;
        self.spectrum_sized_bind_group = spectrum_sized_bind_group;

        // TODO: Once we do spectrograms, don't just drop the old data, resample
        //       it with a compute shader for nicer UX

        // Recreate half-precision spectrum data buffer
        self.half_spectrum_data = std::iter::repeat(f16::default())
            .take(self.surface_config.height as _)
            .collect();
    }

    /// (Re)configure spectrum texture and bind group
    fn configure_spectrum_sized_bind_group(
        device: &Device,
        spectrum_texture_desc: &TextureDescriptor,
        spectrum_sized_bind_group_layout: &BindGroupLayout,
    ) -> (Texture, BindGroup) {
        let spectrum_texture = device.create_texture(&spectrum_texture_desc);
        let spectrum_texture_view = spectrum_texture.create_view(&TextureViewDescriptor {
            label: Some("Spectrum texture view"),
            ..Default::default()
        });
        let spectrum_sized_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrum size-sensitive bind group"),
            layout: &spectrum_sized_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&spectrum_texture_view),
                },
                // TODO: Add spectrogram as a storage image
            ],
        });
        (spectrum_texture, spectrum_sized_bind_group)
    }
}
