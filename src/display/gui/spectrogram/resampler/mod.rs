//! Spectrogram resampler

use wgpu::{
    BindGroup, BindGroupLayout, BlendState, ColorTargetState, ColorWrites, CommandEncoder, Device,
    FragmentState, FrontFace, MultisampleState, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, TextureFormat, TextureView, VertexState,
};

/// Mechanism to resampler the spectrogram when the window is resized
pub struct SpectrogramResampler {
    /// Upscaling pipeline
    upscale_pipeline: RenderPipeline,
    // TODO: Add downscaling bind groups and pipeline
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

        // TODO: Set up downscaling shader

        Self { upscale_pipeline }
    }

    /// Rescale a spectrogram to a different height, return the new write index
    pub fn encode_rescale(
        &mut self,
        encoder: &mut CommandEncoder,
        sampler_bind_group: &BindGroup,
        old_texture_bind_group: &BindGroup,
        old_last_write_idx: u32,
        (old_texture_width, old_texture_height): (u32, u32),
        new_texture_view: &TextureView,
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
                new_texture_view,
            );
        } else {
            // TODO: Downscaling pipeline goes here
            unimplemented!()
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
        new_texture_view: &TextureView,
    ) {
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
}
