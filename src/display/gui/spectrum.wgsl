struct VertexOutput {
    // Beware that position is given in [-1, 1] world coordinates
    // by the vertex shader, but translated into absolute screen
    // coordinates for the fragment shader.
    [[builtin(position)]] clip_position: vec4<f32>;

    // Relative horizontal position within the quad
    [[location(0)]] rel_x: f32;
};

// TODO: Make this a uniform
// TODO: Allow adjusting this with mouse controls (with <-> mouse cursor UX)
let spectrum_width = 0.3;

[[stage(vertex)]]
fn vertex(
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {
    // Emit a quad that covers the full screen height and a
    // uniform-configurable subset of the screen width.
    let rel_x = f32(vertex_index % 2u);
    let x = -1.0 + 2.0 * spectrum_width * rel_x;
    let y = 2.0 * f32(vertex_index / 2u) - 1.0;
    let clip_position = vec4<f32>(x, y, 0.5, 1.0);
    return VertexOutput(
        clip_position,
        rel_x
    );
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // Compute useful quantities from interpolated vertex output
    let pixel_pos = in.clip_position;
    let rel_amp = 1.0 - in.rel_x;

    // TODO: Replace this sine with a 1D amplitude texture
    //       Be sure to lookup at textureDimensions(t) - pixel_pos.y
    //       so that the highest frequencies are on top.
    // TODO: To do multiple spectra with a blur effect, pass instance
    //       number from vertex and use it to look up a texture array
    // TODO: To prepare a spectrogram, the first fragment shader instance can
    //       just write the shaded data into a storage image that will be
    //       subsequently read by the spectrogram shader. But we need the
    //       instance number to be just right for this to work.
    let pi = acos(-1.0);
    if (rel_amp > 0.3 * sin(2.0 * pi * pixel_pos.y / 100.0) + 0.5) {
        discard;
    }

    // TODO: Pick color from a palette texture
    return vec4<f32>(
        rel_amp,
        1.0 - rel_amp,
        0.0,
        1.0
    );
}