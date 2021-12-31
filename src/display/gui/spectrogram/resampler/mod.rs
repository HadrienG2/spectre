//! Spectrogram resampler

use crate::display::gui::{CoreContext, SettingsUniform};
use crevice::std140::AsStd140;
use std::num::NonZeroU32;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, BufferBinding,
    BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    Extent3d, FragmentState, FrontFace, ImageCopyBuffer, ImageDataLayout, MultisampleState,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture,
    TextureFormat, TextureViewDescriptor, VertexState,
};

/// Downscaling pipeline settings
///
/// Must be kept in sync with the downscaling shader
///
#[derive(AsStd140, Default)]
struct DownscaleSettings {
    // Last write index of the old spectrogram, minus (minimum of the width of
    // the old and new spectrograms)-1, wrapped by old spectrogram width
    old_first_write_idx: u32,

    // Stride between rows of the new spectrogram
    new_spectrogram_stride: u32,
}

/// Downscaling workgroup length
///
/// Must be ket in sync with the downscaling shader
///
const DOWNSCALE_WORKGROUP_LEN: u32 = 256;

/// Mechanism to resampler the spectrogram when the window is resized
pub struct SpectrogramResampler {
    /// Upscaling pipeline
    upscale_pipeline: RenderPipeline,

    /// Downscaling settings
    downscale_settings: SettingsUniform<DownscaleSettings>,

    /// Downscaling buffer descriptor
    downscale_buffer_desc: BufferDescriptor<'static>,

    /// Downscaling buffer_bind group descriptor
    downscale_buffer_bind_group_layout: BindGroupLayout,

    /// Downscaling pipeline
    downscale_pipeline: ComputePipeline,
}
//
impl SpectrogramResampler {
    /// Set up spectrogram resampling
    pub fn new(
        device: &Device,
        sampler_bind_group_layout: &BindGroupLayout,
        texture_bind_group_layout: &BindGroupLayout,
        spectrogram_format: TextureFormat,
    ) -> Self {
        // Load upscaling shader
        let upscale_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Spectrogram upscaling shaders"),
            source: ShaderSource::Wgsl(include_str!("upscale.wgsl").into()),
        });

        // Set up upscaling pipeline layout
        let upscale_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spectrogram upscaling pipeline layout"),
            bind_group_layouts: &[sampler_bind_group_layout, texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Set up upscaling render pipeline
        let upscale_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Spectrogram upscaling pipeline"),
            layout: Some(&upscale_pipeline_layout),
            vertex: VertexState {
                module: &upscale_shader,
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
                module: &upscale_shader,
                entry_point: "fragment",
                targets: &[ColorTargetState {
                    format: spectrogram_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        // Set up downscaling settings
        let (downscale_settings, downscale_settings_bind_group_layout) = SettingsUniform::new(
            device,
            DownscaleSettings::default(),
            ShaderStages::COMPUTE,
            "Spectrogram downscaling",
        );

        // Set up downscaling buffer descriptor & associated bind group descriptor
        let downscale_buffer_desc = BufferDescriptor {
            label: Some("Spectrogram downscaling buffer descriptor"),
            size: 0,
            usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        //
        let downscale_buffer_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Spectrogram downscaling buffer bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Load downscaling shader
        let downscale_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Spectrogram downscaling shader"),
            source: ShaderSource::Wgsl(include_str!("downscale.wgsl").into()),
        });

        // Set up downscaling pipeline layout
        let downscale_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spectrogram downscaling pipeline layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &downscale_settings_bind_group_layout,
                &downscale_buffer_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Set up downscaling pipeline
        let downscale_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Spectrogram downscaling pipeline"),
            layout: Some(&downscale_pipeline_layout),
            module: &downscale_shader,
            entry_point: "downscale",
        });

        // ...and we're ready!
        Self {
            upscale_pipeline,
            downscale_settings,
            downscale_buffer_desc,
            downscale_buffer_bind_group_layout,
            downscale_pipeline,
        }
    }

    /// Rescale a spectrogram to a different height, return the new write index
    pub fn encode_rescale(
        &mut self,
        core_context: &CoreContext,
        encoder: &mut CommandEncoder,
        sampler_bind_group: &BindGroup,
        old_texture_bind_group: &BindGroup,
        old_last_write_idx: u32,
        (old_texture_width, old_texture_height): (u32, u32),
        new_texture: &Texture,
        (new_texture_width, new_texture_height): (u32, u32),
    ) -> u32 {
        // Determine the first write of the old spectrogram we're gonna read from
        let min_texture_width = old_texture_width.min(new_texture_width);
        let min_texture_offset = min_texture_width - 1;
        let old_first_write_idx = if old_last_write_idx >= min_texture_offset {
            old_last_write_idx - min_texture_offset
        } else {
            old_texture_width - (min_texture_offset - old_last_write_idx)
        };

        // Dispatch to the appropriate rescaling pipeline
        if new_texture_height >= old_texture_height {
            self.encode_upscale(
                encoder,
                sampler_bind_group,
                old_texture_bind_group,
                old_first_write_idx,
                new_texture,
            );
        } else {
            self.encode_downscale(
                core_context,
                encoder,
                old_texture_bind_group,
                old_first_write_idx,
                new_texture,
                (new_texture_width, new_texture_height),
            );
        }

        // Return the new write index
        min_texture_offset
    }

    /// Upscale a spectrogram to a larger height
    fn encode_upscale(
        &mut self,
        encoder: &mut CommandEncoder,
        sampler_bind_group: &BindGroup,
        old_texture_bind_group: &BindGroup,
        old_first_write_idx: u32,
        new_texture: &Texture,
    ) {
        let new_texture_view = new_texture.create_view(&TextureViewDescriptor {
            label: Some("Spectrogram upscaling texture view"),
            ..Default::default()
        });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Spectrum upscaling render Pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &new_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        render_pass.set_bind_group(0, sampler_bind_group, &[]);
        render_pass.set_bind_group(1, old_texture_bind_group, &[]);
        render_pass.set_pipeline(&self.upscale_pipeline);
        render_pass.draw(0..4, old_first_write_idx..old_first_write_idx + 1);
    }

    /// Downscale a spectrogram to a smaller height
    fn encode_downscale(
        &mut self,
        core_context: &CoreContext,
        encoder: &mut CommandEncoder,
        old_texture_bind_group: &BindGroup,
        old_first_write_idx: u32,
        new_texture: &Texture,
        (new_texture_width, new_texture_height): (u32, u32),
    ) {
        // Allocate storage buffer and construct asssociated bind group
        let device = core_context.device();
        let div_round_up = |x, y| x / y + ((x % y) != 0) as u32;
        let bytes_per_texel = 2 * 2 * 2;
        let bytes_per_row = div_round_up(
            new_texture_width * bytes_per_texel,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        ) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        self.downscale_buffer_desc.size = bytes_per_row as u64 * new_texture_height as u64;
        let buffer = device.create_buffer(&self.downscale_buffer_desc);
        let buffer_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Spectrogram downscaling buffer bind group"),
            layout: &self.downscale_buffer_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // Update the downscaling settings
        self.downscale_settings.replace(DownscaleSettings {
            old_first_write_idx,
            new_spectrogram_stride: bytes_per_row / bytes_per_texel,
        });
        let settings_bind_group = self.downscale_settings.updated(core_context.queue());

        // Perform downscaling compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Downscaling compute pass"),
            });
            compute_pass.set_bind_group(0, old_texture_bind_group, &[]);
            compute_pass.set_bind_group(1, settings_bind_group, &[]);
            compute_pass.set_bind_group(2, &buffer_bind_group, &[]);
            compute_pass.set_pipeline(&self.downscale_pipeline);
            compute_pass.dispatch(
                new_texture_width,
                div_round_up(new_texture_height, DOWNSCALE_WORKGROUP_LEN),
                1,
            );
        }

        // Copy downscaled data to target texture
        debug_assert_eq!(bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: &buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(bytes_per_row),
                    rows_per_image: None,
                },
            },
            new_texture.as_image_copy(),
            Extent3d {
                width: new_texture_width,
                height: new_texture_height,
                depth_or_array_layers: 1,
            },
        );
    }
}
