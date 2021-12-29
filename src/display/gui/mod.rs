//! WebGPU-based spectrum display
// FIXME: This module is getting very long and should be split into smaller entities

mod core;
mod settings;
mod spectrogram;
mod spectrum;

use self::{core::HighLevelEvent, settings::Settings, spectrogram::Spectrogram};
use crate::{
    display::{FrameInput, FrameResult},
    Result,
};
use colorous::{Color, INFERNO};
use half::f16;

use wgpu::{
    util::DeviceExt, AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    ColorTargetState, ColorWrites, Device, Extent3d, FilterMode, FragmentState, FrontFace,
    ImageDataLayout, MultisampleState, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, RenderPipeline, RenderPipelineDescriptor, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    SurfaceError, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::event_loop::ControlFlow;

/// Re-export core context type for child modules
pub(self) use self::core::CoreContext;

/// Custom winit event type
type CustomEvent = ();
type EventLoop = winit::event_loop::EventLoop<CustomEvent>;
type Event<'a> = winit::event::Event<'a, CustomEvent>;

/// GPU-accelerated spectrum display
pub struct GuiDisplay {
    /// Event loop
    event_loop: Option<EventLoop>,

    /// Core context
    core_context: CoreContext,

    /// UI settings
    settings: Settings,

    /// Spectrogram renderer
    spectrogram: Spectrogram,

    /// Live spectrum bind group for things that are never rebound (basic sampler, UI settings)
    spectrum_static_bind_group: BindGroup,

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
    pub fn new(amp_scale: f32, spectrogram_refresh_rate: f32) -> Result<Self> {
        assert!(amp_scale > 0.0);

        // Set up the event loop
        let event_loop = EventLoop::new();

        // Set up the core context
        let core_context = CoreContext::new(&event_loop)?;

        // Set up GPU UI settings
        let device = core_context.device();
        let (settings, settings_bind_group_layout) = Settings::new(device, amp_scale);

        // Set up spectrogram
        let (spectrogram, spectrogram_texture_view) = Spectrogram::new(
            &core_context,
            &settings_bind_group_layout,
            spectrogram_refresh_rate,
        );

        // Set up spectrum texture sampling
        let spectrum_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Spectrum sampler"),
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
            core_context.queue(),
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
        let spectrum_static_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Spectrum static bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
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
        let spectrum_static_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrum static bind group"),
            layout: &spectrum_static_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&spectrum_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&palette_texture_view),
                },
            ],
        });

        // Set up a texture to hold live spectrum data
        let surface_config = core_context.surface_config();
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
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
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
            bind_group_layouts: &[
                &settings_bind_group_layout,
                &spectrum_static_bind_group_layout,
                &spectrum_sized_bind_group_layout,
            ],
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
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
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
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        // Set up size-dependent entities
        let (half_spectrum_data, spectrum_texture, spectrum_sized_bind_group) =
            Self::configure_sized_data(
                &device,
                surface_config.height,
                &spectrum_texture_desc,
                spectrogram_texture_view,
                &spectrum_sized_bind_group_layout,
            );

        // ...and we're ready!
        Ok(Self {
            event_loop: Some(event_loop),
            core_context,
            settings,
            spectrogram,
            spectrum_static_bind_group,
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
        self.core_context.surface_config().height as _
    }

    /// Start the event loop, run a user-provided callback on every frame
    pub fn run_event_loop(
        mut self,
        mut frame_callback: impl FnMut(&mut Self, FrameInput) -> Result<FrameResult> + 'static,
    ) -> ! {
        // Display the first frame
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
        self.core_context.show_window();

        // Start the event loop
        let mut frame_callback = Some(frame_callback);
        let mut resized = false;
        self.event_loop
            .take()
            .expect("Event loop should be present")
            .run(move |event, _target, control_flow| {
                // Perform basic event handling, extract higher-level ops
                match self.core_context.handle_event(event, control_flow) {
                    // Window has been resized, DPI may have changed as well
                    Some(HighLevelEvent::Resized { scale_factor_ratio }) => {
                        resized = true;
                        if let Some(scale_factor_ratio) = scale_factor_ratio {
                            self.spectrogram
                                .handle_scale_factor_change(scale_factor_ratio);
                        }
                    }

                    // It is time to draw a new frame
                    Some(HighLevelEvent::Redraw) => {
                        let mut frame_input = FrameInput {
                            new_spectrum_len: None,
                        };
                        if resized {
                            frame_input.new_spectrum_len =
                                Some(self.core_context.surface_config().height as _);
                            self.handle_resize();
                            resized = false;
                        }
                        match frame_callback.as_mut().expect("Callback should be present")(
                            &mut self,
                            frame_input,
                        ) {
                            Ok(FrameResult::Continue) => {}
                            Ok(FrameResult::Stop) => *control_flow = ControlFlow::Exit,
                            Err(e) => panic!("Frame processing failed: {}", e),
                        }
                    }

                    // The event loop will be destroyed after this call, drop
                    // the things that need dropping for correctness
                    Some(HighLevelEvent::Exit) => std::mem::drop(frame_callback.take()),

                    // TODO: Provide some mouse controls: adjust spectrum width
                    //       via click-drag around separator, spectrum and
                    //       spectrogram zoom-in via click drag + zoom out via
                    //       right click, zoom specific to time, frequency or
                    //       magnitude scales once we have scale bars for those.

                    // This event need not concern us
                    None => {}
                }
            })
    }

    /// Display a spectrum
    pub fn render(&mut self, data: &[f32]) -> Result<()> {
        // Try to access the next window texture
        let window_texture = match self.core_context.current_surface_texture() {
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
        let mut encoder =
            self.core_context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Spectrum render encoder"),
                });

        // Convert the new spectrum data to half precision
        for (dest, &src) in self.half_spectrum_data.iter_mut().zip(data) {
            *dest = f16::from_f32(src);
        }

        // Send the new spectrum data to the device
        let queue = self.core_context.queue();
        queue.write_texture(
            self.spectrum_texture.as_image_copy(),
            bytemuck::cast_slice(&self.half_spectrum_data[..]),
            ImageDataLayout::default(),
            self.spectrum_texture_desc.size,
        );

        // Move spectrogram forward if enough time elapsed
        let spectrogram_write_idx = self.spectrogram.write_idx();

        // Update the settings
        let settings_bind_group = self.settings.updated(queue);

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

            // Draw the live spectrum
            render_pass.set_bind_group(0, settings_bind_group, &[]);
            render_pass.set_bind_group(1, &self.spectrum_static_bind_group, &[]);
            render_pass.set_bind_group(2, &self.spectrum_sized_bind_group, &[]);
            render_pass.set_pipeline(&self.spectrum_pipeline);
            render_pass.draw(0..4, spectrogram_write_idx..spectrogram_write_idx + 1);
        }
        {
            // Set up a render pass with a black clear color
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Spectrogram render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &window_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            // Draw the spectrogram
            render_pass.set_bind_group(0, settings_bind_group, &[]);
            self.spectrogram.draw(&mut render_pass);
        }

        // Submit our render command
        queue.submit(Some(encoder.finish()));

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
        // Update core context
        self.core_context.handle_resize();

        // Update spectrogram display
        let spectrogram_texture_view = self.spectrogram.handle_resize(&self.core_context);

        // Recreate live spectrum texture and associated bind group
        let surface_config = self.core_context.surface_config();
        self.spectrum_texture_desc.size.width = surface_config.height as _;
        let (half_spectrum_data, spectrum_texture, spectrum_sized_bind_group) =
            Self::configure_sized_data(
                &self.core_context.device(),
                surface_config.height,
                &self.spectrum_texture_desc,
                spectrogram_texture_view,
                &self.spectrum_sized_bind_group_layout,
            );
        self.half_spectrum_data = half_spectrum_data;
        self.spectrum_texture = spectrum_texture;
        self.spectrum_sized_bind_group = spectrum_sized_bind_group;
    }

    /// (Re)configure size-dependent textures and bind groups
    fn configure_sized_data(
        device: &Device,
        surface_height: u32,
        spectrum_texture_desc: &TextureDescriptor,
        spectrogram_texture_view: TextureView,
        spectrum_sized_bind_group_layout: &BindGroupLayout,
    ) -> (Box<[f16]>, Texture, BindGroup) {
        // Set up half-precision spectrum data input
        let half_spectrum_data = std::iter::repeat(f16::default())
            .take(surface_height as _)
            .collect();

        // Set up spectrum texture and associated bind group
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
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&spectrogram_texture_view),
                },
            ],
        });
        (
            half_spectrum_data,
            spectrum_texture,
            spectrum_sized_bind_group,
        )
    }
}
