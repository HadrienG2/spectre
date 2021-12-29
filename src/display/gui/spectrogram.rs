//! Spectrogram display

use crate::display::gui::CoreContext;
use std::time::{Duration, Instant};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    ColorTargetState, ColorWrites, Device, Extent3d, FilterMode, FragmentState, FrontFace,
    MultisampleState, PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};

/// Spectrogram display
pub struct Spectrogram {
    /// Bind group for resources that are valid forever
    static_bind_group: BindGroup,

    /// Spectrogram texture
    texture: Texture,

    /// Spectrogram texture descriptor (to recreate it on window resize)
    texture_desc: TextureDescriptor<'static>,

    /// Bind group for resources that are recreated on window resize
    sized_bind_group: BindGroup,

    /// Size-sensitive bind group layout (to recreate it on window resize)
    sized_bind_group_layout: BindGroupLayout,

    /// Render pipeline
    pipeline: RenderPipeline,

    /// Refresh period
    refresh_period: Duration,

    /// Last refresh timestamp
    last_refresh: Instant,

    /// Current texture write index
    write_idx: u32,
}
//
impl Spectrogram {
    /// Set up the spectrogram display, return spectrogram texture view that
    /// the spectrum display shader can write into
    pub fn new(
        core_context: &CoreContext,
        settings_bind_group_layout: &BindGroupLayout,
        refresh_rate: f32,
    ) -> (Self, TextureView) {
        // Set up spectrogram texture sampling & static bind group
        let device = core_context.device();
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Spectrogram sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            ..Default::default()
        });
        //
        let static_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Spectrogram static bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                }],
            });
        //
        let static_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrogram static bind group"),
            layout: &static_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(&sampler),
            }],
        });

        // Set up spectrogram texture & size-dependent bind group
        let surface_config = core_context.surface_config();
        let texture_desc = TextureDescriptor {
            label: Some("Spectrogram texture"),
            size: Extent3d {
                width: surface_config.width as _,
                height: surface_config.height as _,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        };
        //
        let sized_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Spectrogram size-sensitive bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        //
        let (texture, texture_view, sized_bind_group) =
            Self::configure_sized_data(device, &texture_desc, &sized_bind_group_layout);

        // Load shader
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Spectrogram shaders"),
            source: ShaderSource::Wgsl(include_str!("spectrogram.wgsl").into()),
        });

        // Set up pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spectrogram pipeline layout"),
            bind_group_layouts: &[
                &settings_bind_group_layout,
                &static_bind_group_layout,
                &sized_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Set up render pipeline
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Spectrogram pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
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
                module: &shader,
                entry_point: "fragment",
                targets: &[ColorTargetState {
                    format: surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        // Set up spectrogram refreshes
        let scale_factor = core_context.scale_factor();
        let refresh_period = Duration::from_secs_f32(scale_factor / refresh_rate);

        // ...and we're ready!
        (
            Self {
                texture,
                texture_desc,
                static_bind_group,
                sized_bind_group,
                sized_bind_group_layout,
                pipeline,
                refresh_period,
                last_refresh: Instant::now(),
                write_idx: 0,
            },
            texture_view,
        )
    }

    /// Handle window resize, return texture view to update spectrogram writer
    pub fn handle_resize(&mut self, new_core_context: &CoreContext) -> TextureView {
        // TODO: Don't just drop the old spectrogram texture, resample
        //       it with a compute shader for nicer UX. This will almost
        //       certainly require setting new usage flags in the descriptor.

        // Make sure the write index stays in range
        let surface_config = new_core_context.surface_config();
        self.write_idx = self.write_idx.min(surface_config.width - 1);

        // Update size-dependent GPU state
        self.texture_desc.size.width = surface_config.width as _;
        self.texture_desc.size.height = surface_config.height as _;
        let device = new_core_context.device();
        let (texture, texture_view, sized_bind_group) =
            Self::configure_sized_data(device, &self.texture_desc, &self.sized_bind_group_layout);
        self.texture = texture;
        self.sized_bind_group = sized_bind_group;

        // Bubble up spectrogram texture view to update the writer
        texture_view
    }

    /// Handle DPI scale factor change
    pub fn handle_scale_factor_change(&mut self, scale_factor_ratio: f32) {
        self.refresh_period =
            Duration::from_secs_f32(self.refresh_period.as_secs_f32() * scale_factor_ratio);
    }

    /// Query which spectrogram line should be written to by the spectrum shader
    pub fn write_idx(&mut self) -> u32 {
        if self.last_refresh.elapsed() >= self.refresh_period {
            let surface_width = self.texture_desc.size.width;
            self.write_idx = (self.write_idx + 1) % surface_width;
        }
        self.write_idx
    }

    /// Draw the spectrogram using the current render pass
    ///
    /// Assumes that UI settings are bound to bind group 0
    ///
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_bind_group(1, &self.static_bind_group, &[]);
        render_pass.set_bind_group(2, &self.sized_bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..4, self.write_idx..self.write_idx + 1);
    }

    /// (Re)configure size-dependent entities
    fn configure_sized_data(
        device: &Device,
        texture_desc: &TextureDescriptor,
        sized_bind_group_layout: &BindGroupLayout,
    ) -> (Texture, TextureView, BindGroup) {
        let texture = device.create_texture(texture_desc);
        let texture_view = texture.create_view(&TextureViewDescriptor {
            label: Some("Spectrogram texture view"),
            ..Default::default()
        });
        let sized_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrogram size-sensitive bind group"),
            layout: sized_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&texture_view),
            }],
        });
        (texture, texture_view, sized_bind_group)
    }
}
