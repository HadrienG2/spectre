//! Live spectrum display

use crate::display::gui::CoreContext;
use colorous::{Color, INFERNO};
use half::f16;
use wgpu::{
    util::DeviceExt, AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    ColorTargetState, ColorWrites, Device, Extent3d, FilterMode, FragmentState, FrontFace,
    ImageDataLayout, MultisampleState, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor,
    SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexState,
};

/// Live spectrum display
pub struct Spectrum {
    /// Bind group for resources that are valid forever
    static_bind_group: BindGroup,

    /// Input data texture
    input_texture: Texture,

    /// Live spectrum texture descriptor (to recreate it on window resize)
    input_texture_desc: TextureDescriptor<'static>,

    /// Bind group for resources that are recreated on window resize
    sized_bind_group: BindGroup,

    /// Size-sensitive bind group layout (to recreate bind group on resize)
    sized_bind_group_layout: BindGroupLayout,

    /// Render pipeline
    pipeline: RenderPipeline,

    /// Buffer for casting input data to half precision
    f16_input: Box<[f16]>,
}
//
impl Spectrum {
    /// Set up spectrum display
    pub fn new(
        core_context: &CoreContext,
        settings_bind_group_layout: &BindGroupLayout,
        settings_src: &'static str,
        spectrogram_texture_view: TextureView,
    ) -> Self {
        // Set up input texture sampling
        let device = core_context.device();
        let input_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Spectrum input sampler"),
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
                label: Some("Spectrum palette texture"),
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
            label: Some("Spectrum palette texture view"),
            ..Default::default()
        });

        // Set up the common bind group for things that don't need rebinding
        let static_bind_group_layout =
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
        let static_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrum static bind group"),
            layout: &static_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&input_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&palette_texture_view),
                },
            ],
        });

        // Set up a texture to hold input data
        let surface_config = core_context.surface_config();
        let input_texture_desc = TextureDescriptor {
            label: Some("Spectrum input texture"),
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
        let sized_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let mut shader_src = settings_src.to_owned();
        shader_src.push_str(include_str!("render.wgsl"));
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Spectrum rendering shaders"),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        // Set up spectrum pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spectrum pipeline layout"),
            bind_group_layouts: &[
                &settings_bind_group_layout,
                &static_bind_group_layout,
                &sized_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Set up spectrum render pipeline
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Spectrum pipeline"),
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

        // Set up size-dependent entities
        let (f16_input, input_texture, sized_bind_group) = Self::configure_sized_data(
            &device,
            &input_texture_desc,
            spectrogram_texture_view,
            &sized_bind_group_layout,
        );

        // ...and we're ready!
        Self {
            static_bind_group,
            input_texture,
            input_texture_desc,
            sized_bind_group,
            sized_bind_group_layout,
            pipeline,
            f16_input,
        }
    }

    /// Handle window resize
    pub fn handle_resize(
        &mut self,
        new_core_context: &CoreContext,
        spectrogram_texture_view: TextureView,
    ) {
        let surface_config = new_core_context.surface_config();
        self.input_texture_desc.size.width = surface_config.height as _;
        let (f16_input, input_texture, sized_bind_group) = Self::configure_sized_data(
            new_core_context.device(),
            &self.input_texture_desc,
            spectrogram_texture_view,
            &self.sized_bind_group_layout,
        );
        self.f16_input = f16_input;
        self.input_texture = input_texture;
        self.sized_bind_group = sized_bind_group;
    }

    /// Send new input to the GPU
    pub fn write_input(&mut self, queue: &Queue, input: &[f32]) {
        // Convert the new spectrum data to half precision
        for (dest, &src) in self.f16_input.iter_mut().zip(input) {
            *dest = f16::from_f32(src);
        }

        // Send the new spectrum data to the device
        queue.write_texture(
            self.input_texture.as_image_copy(),
            bytemuck::cast_slice(&self.f16_input[..]),
            ImageDataLayout::default(),
            self.input_texture_desc.size,
        );
    }

    /// Draw the live spectrum and associated spectrogram line
    ///
    /// Assumes that UI settings are bound to bind group 0
    ///
    pub fn draw_and_update_spectrogram<'a>(
        &'a self,
        render_pass: &mut RenderPass<'a>,
        spectrogram_write_idx: u32,
    ) {
        render_pass.set_bind_group(1, &self.static_bind_group, &[]);
        render_pass.set_bind_group(2, &self.sized_bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..4, spectrogram_write_idx..spectrogram_write_idx + 1);
    }

    /// (Re)configure size-dependent textures and bind groups
    fn configure_sized_data(
        device: &Device,
        input_texture_desc: &TextureDescriptor,
        spectrogram_texture_view: TextureView,
        sized_bind_group_layout: &BindGroupLayout,
    ) -> (Box<[f16]>, Texture, BindGroup) {
        // Set up half-precision spectrum data input
        let f16_input = std::iter::repeat(f16::default())
            .take(input_texture_desc.size.width as _)
            .collect();

        // Set up input texture and associated bind group
        let input_texture = device.create_texture(&input_texture_desc);
        let input_texture_view = input_texture.create_view(&TextureViewDescriptor {
            label: Some("Spectrum input texture view"),
            ..Default::default()
        });
        let sized_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrum size-sensitive bind group"),
            layout: &sized_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&input_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&spectrogram_texture_view),
                },
            ],
        });
        (f16_input, input_texture, sized_bind_group)
    }
}
