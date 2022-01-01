// Must be kept in sync with the main program
struct SettingsUniform {
    // Last write index of the old spectrogram, minus (minimum of the width of
    // the old and new spectrograms)-1, wrapped by old spectrogram width
    old_first_write_idx: u32;

    // Stride between rows of the new spectrogram
    new_spectrogram_stride: u32;
};
//
[[ group(0), binding(0) ]]
var<uniform> settings: SettingsUniform;

// Old spectrogram texture
[[ group(1), binding(0) ]]
var old_spectrogram_texture: texture_2d<f32>;

// rgba16 pixel emulated using u32 atomics
struct AtomicRgba16 {
    // First integer corresponds to R and G channels, second to B and A channels
    rg_ba: [[ stride(4) ]] array<atomic<u32>, 2>;
};

// New spectrogram buffer, must be in rgba16float format
struct AtomicRgba16Array {
    texels: [[ stride(8) ]] array<AtomicRgba16>;
};
//
[[ group(2), binding(0) ]]
var<storage, read_write> new_spectrogram_buffer: AtomicRgba16Array;

// Minimal workgroup size requirement from WebGPU downlevel defaults
//
// Must be kept in sync with mod.rs since WebGPU does not allow setting
// specialization constants yet.
//
// FIXME: Sync up workgroup length in workgroup_size attribute below
let workgroup_len: u32 = 256u;
let workgroup_len_p1: u32 = 257u;

// Workgroup-local new spectrogram accumulator, can be in any rgba format and
// for our purpose rgba16unorm will be most convenient.
var<workgroup> new_spectrogram_accumulator: array<AtomicRgba16, workgroup_len_p1>;

// Load input texel, clamp it to [0, 1], return a transparent texel if out of bounds
fn load_input(pos: vec2<u32>) -> vec4<f32> {
    // Determine input texel location
    let old_spectrogram_dims = vec2<u32>(textureDimensions(old_spectrogram_texture).xy);
    let abs_x = (settings.old_first_write_idx + pos.x) % old_spectrogram_dims.x;

    // Load texel with wraparound, but make sure new texels are transparent
    let old_spectrogram_color = textureLoad(
        old_spectrogram_texture,
        vec2<i32>(i32(abs_x), i32(pos.y)),
        0
    );
    if (all(pos < old_spectrogram_dims)) {
        return old_spectrogram_color;
    } else {
        return vec4<f32>(0.0);
    }
}

// Determine of the height of the new spectrogram
fn new_spectrogram_height() -> u32 {
    let new_spectrogram_len = arrayLength(&new_spectrogram_buffer.texels);
    return new_spectrogram_len / settings.new_spectrogram_stride;
}

// Determine how many fractional output texels each input texel maps into
// Since we are downscaling, this number is guaranteed to be <= 1.0
fn rel_texel_height() -> f32 {
    let old_spectrogram_height = textureDimensions(old_spectrogram_texture).y;
    return f32(new_spectrogram_height()) / f32(old_spectrogram_height);
}

// Determine the first vertical position which the input texel maps into
fn first_output_y(global_y: u32) -> f32 {
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

// Initialize the local spectrogram accumulator
fn init_contribution(local_idx: u32) {
    for (var i: i32 = 0; i < 2; i = i+1) {
        atomicStore(&new_spectrogram_accumulator[local_idx].rg_ba[i], 0u);
    }
}

// Integrate a contribution into a texel of the local spectrogram accumulator
fn add_contribution(color: vec4<f32>, weight: f32, local_idx: u32) {
    let contribution = weight * color;
    var sub_contribs: array<vec2<f32>, 2> = split_vec4(contribution);
    let sub_contribs_ptr = &sub_contribs;
    for (var i: i32 = 0; i < 2; i = i+1) {
        // Using an u32 add to emulate two u16 adds works as long as the
        // individual adds do not overflow. Which shouldn't happen, since
        // spectrogram color channels were clamped to the [0, 1] range and
        // the sum of contribution weights should be equal to 1.
        let sub_contrib_u32 = pack2x16unorm((*sub_contribs_ptr)[i]);
        atomicAdd(&new_spectrogram_accumulator[local_idx].rg_ba[i], sub_contrib_u32);
    }
}

// Load accumulated contribution from a texel of the local spectrogram accumulator
// Must be separated from add_contribution calls by a workgroupBarrier
fn load_accumulator(local_idx: u32) -> vec4<f32> {
    var sub_results: array<vec2<f32>, 2>;
    for (var i: i32 = 0; i < 2; i = i+1) {
        let sub_result_u32 = atomicLoad(&new_spectrogram_accumulator[local_idx].rg_ba[i]);
        sub_results[i] = unpack2x16unorm(sub_result_u32);
    }
    return vec4<f32>(sub_results[0], sub_results[1]);
}

// Compute output storage buffer index associated with a certain output position
fn output_idx(out_pos: vec2<u32>) -> u32 {
    return out_pos.y * settings.new_spectrogram_stride + out_pos.x;
}

// Merge accumulated contribution into a texel of the output storage buffer
fn merge_accumulator(acc: vec4<f32>, out_pos: vec2<u32>) {
    let out_idx = output_idx(out_pos);
    var sub_accs: array<vec2<f32>, 2> = split_vec4(acc);
    let sub_accs_ptr = &sub_accs;
    for (var i: i32 = 0; i < 2; i = i+1) {
        // Simulate a 2xf16 atomic add using u32 swap
        // FIXME: atomicCompareExchangeWeak is better, but not implemented,
        //        see https://github.com/gfx-rs/naga/issues/1413
        let target = &new_spectrogram_buffer.texels[out_idx].rg_ba[i];
        var sub_acc: vec2<f32> = (*sub_accs_ptr)[i];
        var expected_old_u32: u32 = atomicLoad(target);
        var expected_old: vec2<f32> = unpack2x16float(expected_old_u32);
        loop {
            // Compute new accumulator value
            let new_u32 = pack2x16float(expected_old + sub_accs[i]);

            // Try to inject our new value
            let actual_old_u32 = atomicExchange(target, new_u32);
            if (actual_old_u32 == expected_old_u32) { break; }

            // We accidentally undid another work-item's merging, must undo that
            let actual_old = unpack2x16float(expected_old_u32);
            sub_acc = actual_old - expected_old;
            expected_old_u32 = actual_old_u32;
            expected_old = actual_old;
        }
    }
}

// Store accumulated contribution into a texel of the output storage buffer
fn store_accumulator(acc: vec4<f32>, out_pos: vec2<u32>) {
    let out_idx = output_idx(out_pos);
    var sub_accs: array<vec2<f32>, 2> = split_vec4(acc);
    let sub_accs_ptr = &sub_accs;
    for (var i: i32 = 0; i < 2; i = i+1) {
        atomicStore(
            &new_spectrogram_buffer.texels[out_idx].rg_ba[i],
            pack2x16float((*sub_accs_ptr)[i]),
        );
    }
}

// Invoked with a dispatch domain whose width corresponds to the new spectrogram
// width (to generate all rows of output texels) and whose height corresponds to
// the old spectrogram height (so that each workitem processes one input texel),
// rounded up to the nearest multiple of the workgroup length
//
// Last write index of the new spectrogram will be (minimum of the width of the
// old and new spectrograms)-1.
//
// FIXME: WGSL spec says workgroup_len should be accepted as a parameter to the
//        workgroup_size attribute, but it currently isn't accepted by wgpu.
[[ stage(compute), workgroup_size(1u, 256u) ]]
fn downscale(
    [[ builtin(local_invocation_index) ]] local_idx: u32,
    [[ builtin(global_invocation_id) ]] global_id: vec3<u32>,
    [[ builtin(workgroup_id) ]] group_id: vec3<u32>,
) {
    // Initialize this workgroup's contribution
    init_contribution(local_idx);
    if (local_idx == 0u) {
        init_contribution(workgroup_len);
    }

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
    let local_y_end = local_y_start + out_height;

    // Wait for accumulator initialization to be finished
    workgroupBarrier();

    // Do start and end fall into the same texel or two consecutive texels?
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
    let num_out_texels = group_range[1] - group_range[0] + 1u;
    let old_height = u32(textureDimensions(old_spectrogram_texture).y);
    if ((local_idx >= num_out_texels) && (global_id.y >= old_height)) {
        return;
    }

    // Merge workgroup-local accumulators into the output buffer
    let out_y = group_range[0] + local_idx;
    let out_pos = vec2<u32>(out_x, out_y);
    let acc = load_accumulator(local_idx);
    if ((local_idx == 0u) || (local_idx == (num_out_texels - 1u))) {
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
    if ((local_idx == 0u) && (num_out_texels == (workgroup_len + 1u))) {
        let out_pos = vec2<u32>(out_x, group_range[1]);
        let acc = load_accumulator(workgroup_len);
        merge_accumulator(acc, out_pos);
    }
}
