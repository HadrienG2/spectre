//! Management of shader settings that can be tuned via the user interface

use crevice::std140::{AsStd140, Std140};
use std::num::NonZeroU64;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferUsages, Device, Queue, ShaderStages,
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
struct Uniform {
    /// Horizontal fraction of the window that is occupied by the live spectrum
    spectrum_width: f32,

    /// Range of amplitudes that we can display in dB
    amp_scale: f32,
}

/// Shader settings management
pub struct Settings {
    /// UI settings
    uniform: Uniform,

    /// Buffer for holding UI settings on the device
    buffer: Buffer,

    /// Bind group for settings
    bind_group: BindGroup,

    /// Truth that settings have changed since the last upload
    updated: bool,
}
//
impl Settings {
    /// Set up GPU settings handling, provide the associated bind group layout
    /// for client shader setup
    pub fn new(device: &Device, amp_scale: f32) -> (Self, BindGroupLayout) {
        // Set up UI settings storage
        let uniform = Uniform {
            spectrum_width: DEFAULT_SPECTRUM_WIDTH,
            amp_scale,
        };

        // Set up associated buffer
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Settings uniform"),
            contents: uniform.as_std140().as_bytes(),
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
        });

        // Set up associated bind group
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Settings bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniform>() as u64),
                },
                count: None,
            }],
        });
        //
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Settings bind group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // We're ready
        (
            Self {
                uniform,
                buffer,
                bind_group,
                updated: false,
            },
            bind_group_layout,
        )
    }

    /// Update settings if needed, get the associated bind group
    pub fn updated(&mut self, queue: &Queue) -> &BindGroup {
        if self.updated {
            queue.write_buffer(&self.buffer, 0, self.uniform.as_std140().as_bytes());
            self.updated = false;
        }
        &self.bind_group
    }
}