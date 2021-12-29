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

    // Relative horizontal position within the quad
    [[ location(0) ]] rel_x: f32;

    // Spectrogram write index (same for every vertex)
    [[ location(1), interpolate(flat) ]] spectrogram_write_idx: u32;
};

[[ stage(vertex) ]]
fn vertex(
    [[ builtin(vertex_index) ]] vertex_idx: u32,
    [[ builtin(instance_index) ]] spectrogram_write_idx: u32,
) -> VertexOutput {
    // Emit a quad that covers the full screen height and a
    // uniform-configurable subset of the screen width.
    let rel_x = f32(vertex_idx % 2u);
    let rel_y = f32(vertex_idx / 2u);
    let x = -1.0 + 2.0 * settings.spectrum_width * rel_x;
    let y = 2.0 * rel_y - 1.0;
    return VertexOutput(
        vec4<f32>(x, y, 0.5, 1.0),
        rel_x,
        spectrogram_write_idx
    );
}

// Sampler for spectra and spectrograms
[[ group(1), binding(0) ]]
var spectrum_sampler: sampler;

// Spectrum color palette
[[ group(1), binding(1) ]]
var palette_texture: texture_1d<f32>;

// Live spectrum texture
[[ group(2), binding(0) ]]
var spectrum_texture: texture_1d<f32>;

// Spectrogram texture
[[ group(2), binding(1) ]]
var spectrogram_texture: texture_storage_2d<rgba16float, write>;

[[ stage(fragment) ]]
fn fragment(in: VertexOutput) -> [[ location(0) ]] vec4<f32> {
    // Compute useful quantities from interpolated vertex output
    let rel_amp = -in.rel_x;

    // Find spectrum amplitude at current vertical position
    let spectrum_len = f32(textureDimensions(spectrum_texture));
    let spectrum_abs_pos = spectrum_len - in.abs_pos.y - 1.0;
    let spectrum_rel_pos = spectrum_abs_pos / (spectrum_len - 1.0);
    let spectrum_amp = textureSample(spectrum_texture, spectrum_sampler, spectrum_rel_pos).x;
    let spectrum_color = textureSample(
        palette_texture,
        spectrum_sampler,
        1.0 + spectrum_amp/settings.amp_scale
    );

    // Record the spectrum color in the spectrogram image
    if (in.abs_pos.x < 1.0) {
        textureStore(
            spectrogram_texture,
            vec2<i32>(i32(in.spectrogram_write_idx), i32(spectrum_abs_pos)),
            spectrum_color
        );
    }

    // Only draw if current pixel is below scaled vertical amplitude
    if (rel_amp * settings.amp_scale > spectrum_amp) {
        discard;
    }

    // Display the live spectrum using our color palette for each line
    return spectrum_color;
}
