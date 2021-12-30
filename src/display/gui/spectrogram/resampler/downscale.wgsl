// Old spectrogram sampler with wrapping x axis
[[ group(0), binding(0) ]]
var old_spectrogram_sampler: sampler;

struct SettingsUniform {
    // Last write index of the old spectrogram, minus (minimum of the width of
    // the old and new spectrograms)-1, wrapped by old spectrogram width
    old_first_write_idx: u32;

    // Width of the new spectrogram
    new_spectrogram_width: u32,
};
//
[[ group(0), binding(1) ]]
var<uniform> settings: SettingsUniform;

// Old spectrogram texture
[[ group(1), binding(0) ]]
var old_spectrogram_texture: texture_2d<f32>;

// Minimal workgroup size requirement from WebGPU downlevel defaults
let workgroup_len: u32 = 256;

// rgba16 pixel emulated using u32 atomics
struct AtomicRgba16 {
    // First integer corresponds to R and G channels, second to B and A channels
    rg_ba: array<atomic<u32>, 2>;
};

// New spectrogram buffer, must be in rgba16float format
[[ group(1), binding(1) ]]
var<storage, read_write> new_spectrogram_buffer: array<AtomicRgba16>;

// Workgroup-local new spectrogram accumulator, can be in any rgba16 format and
// for our purpose rgba16uint will be most convenient.
var<workgroup> new_spectrogram_accumulator: array<AtomicRgba16, workgroup_len+1>;

// Invoked with a dispatch domain whose width corresponds to the new spectrogram
// width (to generate all rows of output texels) and whose height corresponds to
// the old spectrogram height (so that each workitem processes one input texel),
// rounded up to the nearest multiple of the workgroup length
//
// Last write index of the new spectrogram will be (minimum of the width of the
// old and new spectrograms)-1.
//
[[ stage(compute), workgroup_size(1u, workgroup_len) ]]
fn downscale(
    [[ builtin(local_invocation_index) ]] local_idx: u32,
    [[ builtin(global_invocation_id) ]] global_id: vec3<u32>,
    [[ builtin(workgroup_id) ]] group_id: vec3<u32>,
) {
    // Determine and load input texel associated with this invocation
    let input_color = load_input(global_id.xy);

    // Determine corresponding position in output image
    let out_x = global_id.x;
    let out_y_start = first_output_y(global_id.y);

    // Determine first and last output texels the workgroup maps into
    let group_range = group_y_range(group_id.y);

    // Deduce target range of vertical output positions,
    // relative to the start of the workgroup.
    let local_y_start = out_y_start - f32(group_range[0]);
    let out_height = rel_texel_height();
    let local_y_end = rel_y_start + out_height;

    // Do these fall into the same texel or two consecutive texels?
    let first_idx = u32(local_y_start);
    let second_idx = u32(local_y_end);
    if (first_idx == second_idx) {
        // If it's the same texel, integrate everything into that texel
        add_contribution(input_color, out_height, first_idx);
    } else {
        // Otherwise, integrate partial contributions to each output texel
        let first_height = f32(second_idx) - local_y_start;
        add_contribution(input_color, first_height, first_idx);
        let second_height = out_height - first_height;
        add_contribution(input_color, second_height, second_idx);
    }

    // Wait for the end of workgroup-local integration
    workgroupBarrier();

    // We are now done with collective operations and can discard workitems that
    // do not map into a local accumulator or global input pixel.
    let num_out_texels = group_range[1] - group_range[0] + 1;
    if ((local_idx >= num_out_texels) && (global_id.y >= textureDimensions(old_spectrogram).y)) {
        return;
    }

    // Merge workgroup-local accumulators into the output buffer
    let out_y = group_range[0] + local_idx;
    let out_pos = vec2<u32>(out_x, out_y);
    let acc = load_accumulator(local_idx);
    if ((local_idx == 0) || (local_idx == out_texels-1)) {
        // The first and last texels may alias with the previous and next
        // workgroup respectively and should be merged through atomic sums
        merge_accumulator(acc, out_pos);
    } else {
        // The middle output texels are specific to our workgroup, no other
        // workgroup will contribute to them, so an atomic store is enough
        store_accumulator(acc, out_pos);
    }

    // Handle edge case where there is one more accumulator than workitems
    // This can happen when there are almost as many output texels as inputs,
    // but inputs map into slightly shifted texels in the output
    if ((local_idx == 0) && (out_texels == workgroup_len+1)) {
        let out_pos = vec2<u32>(out_x, group_range[1]);
        let acc = load_accumulator(workgroup_len);
        merge_accumulator(acc, out_pos);
    }
}

// Load input texel, clamp it to [0, 1], return a transparent texel if out of bounds
fn load_input(pos: vec2<u32>) -> vec4<f32> {
    // Determine input texel location
    let old_spectrogram_dims = textureDimensions(old_spectrogram_texture);
    let abs_x = settings.old_first_write_idx + pos.x;
    let rel_x = f32(abs_x) / f32(old_spectrogram_dims.x);
    let rel_y = f32(pos.y) / f32(old_spectrogram_dims.y);

    // Load texel with wraparound, but make sure new texels are transparent
    let old_spectrogram_color = textureSample(
        old_spectrogram_texture,
        old_spectrogram_sampler,
        vec2<f32>(rel_x, rel_y)
    );
    if (all(pos < old_spectrogram_dims)) {
        return clamp(old_spectrogram_color, vec4<f32>(0.0), vec4<f32>(1.0));
    }
        return vec4<f32>(0.0);
    }
}

// Determine of the height of the new spectrogram
fn new_spectrogram_height() -> u32 {
    let new_spectrogram_len = arrayLength(&new_spectrogram_buffer);
    return new_spectrogram_len / settings.new_spectrogram_width;
}

// Determine how many fractional output texels each input texel maps into
// Since we are downscaling, this number is guaranteed to be <= 1.0
fn rel_texel_height() -> f32 {
    let old_spectrogram_height = textureDimensions(old_spectrogram_texture).y;
    return f32(new_spectrogram_height()) / f32(old_spectrogram_height);
}

// Determine the first vertical position which the input texel maps into
fn first_output_y(global_y: u32) -> OutputPosition {
    return f32(global_y) * rel_texel_height();
}

// Determine the first and last vertical texel which the workgroup maps into
fn group_y_range(group_y: u32) -> array<u32, 2> {
    let global_y_start = group_y * workgroup_len;
    let global_y_end = global_y_start + workgroup_len;
    return array<u32, 2>(
        u32(floor(first_output_y(global_y_start))),
        u32(ceil(first_output_y(global_y_end)))
    );
}

// Split a vec4<f32> into two consecutive vec2<f32>
fn split_vec4(in: vec4<f32>) -> array<vec2<f32>, 2> {
    return array<vec2<f32>, 2>(in.rg, in.ba);
}

// Integrate a contribution into a texel of the local spectrogram accumulator
fn add_contribution(color: vec4<f32>, weight: f32, local_idx: u32) {
    let contribution = weight * color;
    let sub_contribs = split_vec4(contribution);
    for (var i: u32 = 0; i < 2; i++) {
        // Using an u32 add to emulate two u16 adds works as long as the
        // individual adds do not overflow. Which shouldn't happen, since
        // spectrogram color channels were clamped to the [0, 1] range and
        // the sum of contribution weights should be equal to 1.
        let sub_contrib_u32 = pack2x16unorm(sub_contribs[i]);
        atomicAdd(&new_spectrogram_accumulator[local_idx].rg_ba[i], sub_contrib_u32);
    }
}

// Load accumulated contribution from a texel of the local spectrogram accumulator
// Must be separated from add_contribution calls by a workgroupBarrier
fn load_accumulator(local_idx: u32) -> vec4<f32> {
    var sub_results: array<vec2<f32>, 2>;
    for (var i: u32 = 0; i < 2; i++) {
        let sub_result_u32 = atomicLoad(&new_spectrogram_accumulator[local_idx].rg_ba[i]);
        sub_results[i] = unpack2x16unorm(sub_result_u32);
    }
    return vec4<f32>(sub_results[0], sub_results[1]);
}

// Compute output storage buffer index associated with a certain output position
fn output_idx(out_pos: vec2<u32>) -> u32 {
    return out_pos.y * settings.new_spectrogram_width + out_pos.x;
}

// Merge accumulated contribution into a texel of the output storage buffer
fn merge_accumulator(acc: vec4<f32>, out_pos: vec2<u32>) {
    let out_idx = output_idx(out_pos);
    let sub_accs = split_vec4(acc);
    for (var i: u32 = 0; i < 2; i++) {
        // Simulate a 2xf16 atomic add using u32 compare-and-swap
        var old_u32: u32 = atomicLoad(&new_spectrogram_buffer[out_idx].rg_ba[i]);
        var exchanged: bool = false;
        while (!exchanged) {
            let old = unpack2x16float(old_u32);
            let new_u32 = pack2x16float(old + sub_accs[i]);
            let result = atomicCompareExchangeWeak(
                &new_spectrogram_buffer[out_idx].rg_ba[i],
                old_u32,
                new_u32
            );
            old_u32 = result.old_value;
            exchanged = result.exchanged;
        }
    }
}

// Store accumulated contribution into a texel of the output storage buffer
fn store_accumulator(acc: vec4<f32>, out_pos: vec2<u32>) {
    let out_idx = output_idx(out_pos);
    let sub_accs = split_vec4(acc);
    for (var i: u32 = 0; i < 2; i++) {
        atomicStore(
            &new_spectrogram_buffer[out_idx].rg_ba[i],
            pack2x16float(sub_accs[i]),
        );
    }
}
