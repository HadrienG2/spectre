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
    // to the vertex shader, but translated into absolute screen
    // coordinates in pixels for the fragment shader.
    [[builtin(position)]] abs_pos: vec4<f32>;

    // Relative horizontal position within the quad
    [[location(0)]] rel_x: f32;
};

[[stage(vertex)]]
fn vertex(
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {
    // Emit a quad that covers the full screen height and a
    // uniform-configurable subset of the screen width.
    let rel_x = f32(vertex_index % 2u);
    let rel_y = f32(vertex_index / 2u);
    let x = -1.0 + 2.0 * settings.spectrum_width * rel_x;
    let y = 2.0 * rel_y - 1.0;
    return VertexOutput(
        vec4<f32>(x, y, 0.5, 1.0),
        rel_x,
    );
}

// Sampler for spectra and spectrograms
[[ group(0), binding(1) ]]
var spectrum_sampler: sampler;

// Live spectrum texture
[[ group(1), binding(0) ]]
var spectrum_texture: texture_1d<f32>;

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // Compute useful quantities from interpolated vertex output
    let pixel_pos = in.abs_pos;
    let rel_amp = -in.rel_x;

    // Find spectrum amplitude at current vertical position
    // TODO: To do multiple spectra with a blur effect, pass instance
    //       number from vertex and use to look up a texture array or 2D texture
    let spectrum_len = f32(textureDimensions(spectrum_texture));
    let spectrum_pos = (spectrum_len - in.abs_pos.y) / spectrum_len;
    let spectrum_amp = textureSample(spectrum_texture, spectrum_sampler, spectrum_pos).x;

    // Only draw if current pixel is below scaled vertical amplitude
    if (rel_amp * settings.amp_scale > spectrum_amp) {
        discard;
    }

    // TODO: To prepare a spectrogram, the first fragment shader instance can
    //       just write the shaded data into a storage image that will be
    //       subsequently read by the spectrogram shader. But we need the
    //       instance number to be just right for this to work.

    // Live spectrum pixels are yellow
    // TODO: Try using the palette color to see how it looks like
    return vec4<f32>(1.0, 1.0, 0.0, 1.0);
}