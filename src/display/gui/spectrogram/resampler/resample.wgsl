struct VertexOutput {
    // Beware that position is given in [-1, 1] world coordinates
    // by the vertex shader, but translated into absolute screen
    // coordinates in pixels upon fragment shader invocation.
    [[ builtin(position) ]] abs_pos: vec4<f32>;

    // Relative [0, 1] vertical screen coordinate
    [[ location(0) ]] rel_y: f32;

    // Last spectrogram write index
    [[ location(1), interpolate(flat) ]] last_write_idx: u32;
};

[[ stage(vertex) ]]
fn upscale_vertex(
    [[ builtin(vertex_index) ]] vertex_idx: u32,
    [[ builtin(instance_index) ]] last_write_idx: u32,
) -> VertexOutput {
    // Emit a quad that covers the full screen
    let rel_x = f32(vertex_idx % 2u);
    let rel_y = f32(vertex_idx / 2u);
    let x = 2.0 * rel_x - 1.0;
    let y = 2.0 * rel_y - 1.0;
    return VertexOutput(
        vec4<f32>(x, y, 0.5, 1.0),
        rel_y,
        last_write_idx
    );
}

// Spectrogram sampler with wrapping x axis
[[ group(0), binding(0) ]]
var old_spectrogram_sampler: sampler;

// Spectrogram texture
[[ group(1), binding(0) ]]
var old_spectrogram_texture: texture_2d<f32>;

[[ stage(fragment) ]]
fn upscale_fragment(in: VertexOutput) -> [[ location(0) ]] vec4<f32> {
    // Probe old spectrogram at a position matching new spectrogram location...
    let old_spectrogram_width = f32(textureDimensions(old_spectrogram_texture).x);
    let shifted_x = f32(in.last_write_idx) - in.abs_pos.x;
    let rel_x = shifted_x / old_spectrogram_width;
    let old_spectrogram_color = textureSample(
        spectrogram_texture,
        spectrogram_sampler,
        vec2<f32>(rel_x, in.rel_y)
    );

    // ...but make sure newly created pixels are black
    if (in.abs_pos_x < old_spectrogram_width) {
        return old_spectrogram_color;
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
}

// TODO: Add downsampling compute shader
