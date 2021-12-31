// Must be kept in sync with other shaders & main program
struct SettingsUniform {
    // Horizontal fraction of the window that is occupied by the live spectrum
    // TODO: Allow adjusting this with mouse controls (with <-> mouse cursor UX)
    spectrum_width: f32;

    // Range of amplitudes that we can display
    amp_scale: f32;
};
//
[[ group(0), binding(0) ]]
var<uniform> settings: SettingsUniform;

struct VertexOutput {
    // Beware that position is given in [-1, 1] world coordinates
    // by the vertex shader, but translated into absolute screen
    // coordinates in pixels upon fragment shader invocation.
    [[ builtin(position) ]] abs_pos: vec4<f32>;

    // Relative vertical coordinate
    [[ location(0) ]] rel_y: f32;

    // Last spectrogram write index
    [[ location(1), interpolate(flat) ]] last_write_idx: u32;
};

[[ stage(vertex) ]]
fn vertex(
    [[ builtin(vertex_index) ]] vertex_idx: u32,
    [[ builtin(instance_index) ]] last_write_idx: u32,
) -> VertexOutput {
    // Emit a quad that covers the full screen height and a
    // uniform-configurable subset of the screen width.
    let rel_x = f32(vertex_idx % 2u);
    let rel_y = f32(vertex_idx / 2u);
    let x = -1.0 + 2.0 * settings.spectrum_width * (1.0 - rel_x) + 2.0 * rel_x;
    let y = 2.0 * rel_y - 1.0;
    return VertexOutput(
        vec4<f32>(x, y, 0.5, 1.0),
        rel_y,
        last_write_idx
    );
}

// Spectrogram sampler with wrapping x axis
[[ group(1), binding(0) ]]
var spectrogram_sampler: sampler;

// Spectrogram texture
[[ group(2), binding(0) ]]
var spectrogram_texture: texture_2d<f32>;

[[ stage(fragment) ]]
fn fragment(in: VertexOutput) -> [[ location(0) ]] vec4<f32> {
    // Get window width and live spectrum region width
    let total_width = f32(textureDimensions(spectrogram_texture).x);
    let spectrum_width = settings.spectrum_width * total_width;

    // Compute horizontal distance from end of live spectrum to current point
    let corrected_x = in.abs_pos.x - spectrum_width;

    // Load data from spectrogram texture
    let shifted_x = f32(in.last_write_idx) - corrected_x;
    let rel_x = shifted_x / total_width;
    let spectrogram_color = textureSample(
        spectrogram_texture,
        spectrogram_sampler,
        vec2<f32>(rel_x, in.rel_y)
    );

    // Render the first column white to separate live spectrum vs spectrogram
    if (corrected_x < 1.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    } else {
        return spectrogram_color;
    }
}
