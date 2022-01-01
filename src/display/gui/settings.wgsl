// Must be kept in sync with the main program
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
