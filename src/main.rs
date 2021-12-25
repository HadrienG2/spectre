mod audio;
mod display;
mod fourier;
pub mod math;
mod resample;

use crate::{
    audio::AudioSetup, display::FrameResult, fourier::SteadyQTransform, resample::FourierResampler,
};
use log::{debug, error};
use rt_history::Overrun;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use structopt::StructOpt;

/// Default Result type used throughout this app whenever bubbling errors up
/// seems to be the only sensible option.
pub use anyhow::Result;

// Command-line parameters
#[derive(Debug, StructOpt)]
struct CliOpts {
    /// Minimum displayed frequency in Hz
    #[structopt(long, default_value = "20.0")]
    min_freq: f32,

    /// Maximum displayed frequency in Hz
    #[structopt(long, default_value = "20000.0")]
    max_freq: f32,

    /// Minimal frequency resolution in Hz
    ///
    /// This is the minimal FFT bin spacing at 20Hz. Actual frequency resolution
    /// will be a bit more, depending on the choice of window function.
    ///
    #[structopt(long, default_value = "1.0")]
    freq_res: f32,

    /// Minimal time resolution in ms
    ///
    /// This is the time resolution provided by the FFT at 20kHz. It cannot be
    /// set indefinitely small, at some point the limit of the constant Q
    /// transform's ability to accomodate both frequency and time resolution
    /// constraints will be reached.
    ///
    #[structopt(long, default_value = "7.0")]
    time_res: f32,

    /// Use a linear frequency scale
    ///
    /// By default, spectre will use a log scale as it better matches human
    /// audition, but linear scales may be better for specific applications.
    ///
    #[structopt(long)]
    lin_freqs: bool,

    /// Window function to be applied
    ///
    /// "rectangular" has minimal central peak width (1 bin), but maximal
    /// leakage (first sidelobe at -15dB, down to -40dB when 40 bins away).
    ///
    /// "triangular" has a central peak width of 1.3 bins, first sidelobes at
    /// -30dB, down to -70dB when 40 bins away.
    ///
    /// "hann" has a central peak width of 1.5 bins, first sidelobes at -30dB,
    /// down to -105dB when 40 bins away.
    ///
    /// "blackman" has a central peak width of 1.7 bins, first sidelobes at
    /// -60dB, down to -115dB when 40 bins away.
    ///
    /// "nuttall" has a central peak width of 2.0 bins, first sidelobes at
    /// -95dB, down to -130dB when 40 bins away.
    ///
    #[structopt(long, default_value = "hann")]
    window: String,

    /// Amplitude range in dBFS
    ///
    /// Signal amplitudes lower than this amount below 0dBFS will not be
    /// rendered, typically you will want to set this at your "noise floor".
    ///
    #[structopt(long, default_value = "96")]
    amp_range: f32,
}

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Decode and validate CLI arguments
    let mut opts = CliOpts::from_args();
    debug!("Got CLI options {:?}", opts);
    if !opts.min_freq.is_finite() || opts.min_freq < 0.0 {
        panic!("Please specify a sensible minimum frequency");
    }
    if !opts.max_freq.is_finite() || opts.max_freq <= opts.min_freq {
        panic!("Please specify a sensible maximum frequency");
    }
    if !opts.freq_res.is_finite() || opts.freq_res <= 0.0 {
        panic!("Please specify a sensible frequency resolution");
    }
    if !opts.time_res.is_finite() || opts.time_res <= 0.0 {
        panic!("Please specify a sensible time resolution");
    }
    if !opts.amp_range.is_finite() {
        panic!("Please specify a sensible amplitude scale");
    }
    opts.amp_range = opts.amp_range.abs();

    // Set up the audio stack
    let audio = AudioSetup::new()?;
    let sample_rate = audio.sample_rate();
    if opts.max_freq > (sample_rate / 2) as f32 {
        panic!("Requested max frequency can't be probed at current sampling rate");
    }

    // Set up the Fourier transform
    let mut fourier =
        SteadyQTransform::new(opts.freq_res, opts.time_res, sample_rate, &opts.window);

    // Start recording audio, keeping enough history that the audio thread can
    // write two full periods before triggering an FFT input readout overrun.
    let history_len = if audio.buffer_size() <= fourier.input().len() / 2 {
        2 * fourier.input().len()
    } else {
        4 * audio.buffer_size()
    };
    let mut recording = audio.start_recording(history_len)?;

    // Initialize the GUI display
    #[cfg(feature = "cli")]
    let spectrum_display = crate::display::CliDisplay::new(opts.amp_range)?;
    #[cfg(all(feature = "gui", not(feature = "cli")))]
    let spectrum_display = crate::display::GuiDisplay::new(opts.amp_range)?;

    // Prepare to resample the Fourier transform for display purposes
    let fourier_len = fourier.output_len();
    let setup_resampler = move |display_width| {
        FourierResampler::new(
            fourier_len,
            sample_rate,
            display_width,
            opts.min_freq,
            opts.max_freq,
            !opts.lin_freqs,
        )
    };
    let mut resampler = setup_resampler(spectrum_display.width());

    // Handle user shutdown requests (Ctrl+C)
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_2 = shutdown.clone();
    ctrlc::set_handler(move || shutdown_2.store(true, Ordering::Relaxed))?;

    // Start computing some FFTs
    let mut last_clock = 0;
    spectrum_display.run_event_loop(move |display, frame_input| {
        // Check if the user has requested shutdown via Ctrl+C
        if shutdown.load(Ordering::Relaxed) {
            return Ok(FrameResult::Stop);
        }

        // Check if the display width has changed, recreate resampler if need be
        if let Some(new_display_width) = frame_input.new_display_width {
            resampler = setup_resampler(new_display_width);
        }

        // Read latest audio history, handle xruns and audio thread errors
        let mut underrun = false;
        let mut overrun = None;
        last_clock = match recording.read_history(fourier.input()) {
            // Successfully read latest FFT history with a certain timestamp
            Ok(Ok(clock)) => {
                if clock == last_clock {
                    underrun = true;
                }
                clock
            }

            // Some history was overwritten by the audio thread (overrun)
            Ok(Err(Overrun {
                clock,
                excess_entries,
            })) => {
                overrun = Some(excess_entries);
                clock
            }

            // The audio threads have crashed, report their errors and die
            mut audio_error @ Err(_) => {
                let terminal_reset_result = display.reset_terminal();
                while let Err(error) = audio_error {
                    error!("Audio thread error: {:?}", error);
                    audio_error = recording.read_history(fourier.input());
                }
                error!("Audio thread exited due to errors, time to die...");
                return terminal_reset_result.map(|()| FrameResult::Stop);
            }
        };

        // Display the result of the data acquisition
        match (underrun, overrun) {
            // Everything went fine
            (false, None) => {
                // Compute the Fourier transform
                let fft_amps = fourier.compute();

                // Resample it to the desired number of output bins
                let output_bins = resampler.resample(fft_amps);

                // Display the resampled FFT bins
                display.render(output_bins)?;
            }

            // Buffer underrun (no new data)
            (true, _) => { /* FIXME: display.report_underrun()?; */ }

            // Buffer overrun (audio thread overwrote buffer while we were reading)
            (false, Some(excess_samples)) => { /* FIXME: display.report_overrun(excess_samples)?; */
            }
        }

        // All good and ready for the next frame
        return Ok(FrameResult::Continue);
    })
}
