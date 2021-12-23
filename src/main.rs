mod audio;
mod fourier;
mod resample;
pub mod simd;

use crate::{audio::AudioSetup, fourier::FourierTransform, resample::FourierResampler};
use log::{debug, error, warn};
use rt_history::Overrun;
use std::time::{Duration, Instant};
use structopt::StructOpt;

/// Default Result type used throughout this app whenever bubbling errors up
/// seems to be the only sensible option.
pub use anyhow::Result;

// Command-line parameters
#[derive(Debug, StructOpt)]
struct CliOpts {
    /// Minimum monitored frequency in Hz
    #[structopt(long, default_value = "20.0")]
    min_freq: f32,

    /// Maximum monitored frequency in Hz
    #[structopt(long, default_value = "20000.0")]
    max_freq: f32,

    /// Minimal frequency resolution in Hz
    ///
    /// This is the minimal FFT bin spacing at 20Hz, actual frequency resolution
    /// will be a bit more, depending on the choice of window function.
    ///
    #[structopt(long, default_value = "1.0")]
    freq_res: f32,

    /// Use a linear frequency scale
    ///
    /// By default, spectre will use a log scale as it better matches human
    /// audition, but linear scales may be better for specific applications.
    ///
    #[structopt(long)]
    lin_freqs: bool,

    /// Amplitude scale in dBFS
    ///
    /// Signal amplitudes lower than this amount below 0dBFS will not be
    /// rendered, typically you will want to set this at your "noise floor".
    ///
    #[structopt(long, default_value = "96")]
    amp_scale: f32,
}

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Decode CLI arguments
    let opts = CliOpts::from_args();
    debug!("Got CLI options {:?}", opts);
    if !opts.min_freq.is_finite() || opts.min_freq < 0.0 {
        panic!("Please enter a sensible minimum frequency");
    }
    if !opts.max_freq.is_finite() || opts.max_freq <= opts.min_freq {
        panic!("Please enter a sensible maximum frequency");
    }
    if !opts.freq_res.is_finite() || opts.freq_res <= 0.0 {
        panic!("Please enter a sensible frequency resolution");
    }
    if !opts.amp_scale.is_finite() {
        panic!("Please enter a sensible amplitude scale");
    }
    let amplitude_scale = opts.amp_scale.abs();

    // Set up the audio stack
    let audio = AudioSetup::new()?;
    let sample_rate = audio.sample_rate();
    if opts.max_freq > (sample_rate / 2) as f32 {
        panic!("Requested max frequency can't be probed at current sampling rate");
    }

    // Set up the Fourier transform
    let mut fourier = FourierTransform::new(opts.freq_res, sample_rate);

    // Start recording audio, keeping enough history that the audio thread can
    // write two full periods before triggering an FFT input readout overrun.
    let history_len = if audio.buffer_size() <= fourier.input().len() / 2 {
        2 * fourier.input().len()
    } else {
        4 * audio.buffer_size()
    };
    let mut recording = audio.start_recording(history_len)?;

    // Prepare to resample the Fourier transform for display purposes
    // FIXME: Replace all these hardcoded test parameters with real parameters
    //        that can be tuned at runtime.
    const NUM_OUTPUT_BINS: usize = 200; // FIXME: Should be provided by terminal display
    let mut resampler = FourierResampler::new(
        fourier.output_len(),
        sample_rate,
        NUM_OUTPUT_BINS,
        opts.min_freq,
        opts.max_freq,
        !opts.lin_freqs,
    );

    // Start computing some FFTs
    let mut last_clock = 0;
    let mut last_refresh = Instant::now();
    loop {
        // Simulate vertical synchronization
        // FIXME: Extract to console backend
        const REFRESH_PERIOD: Duration = Duration::from_millis(100);
        std::thread::sleep(REFRESH_PERIOD.saturating_sub(last_refresh.elapsed()));
        last_refresh = Instant::now();

        // Read latest audio history, handle xruns and audio thread errors
        // TODO: Report audio errors visually in the final display
        last_clock = match recording.read_history(fourier.input()) {
            // Successfully read latest FFT history with a certain timestamp
            Ok(Ok(clock)) => {
                if clock == last_clock {
                    warn!("Buffer underrun! (No new data from audio thread)");
                }
                clock
            }

            // Some history was overwritten by the audio thread (overrun)
            Ok(Err(Overrun {
                clock,
                excess_entries,
            })) => {
                error!(
                    "Buffer overrun! (Audio thread overwrote {} samples)",
                    excess_entries
                );
                clock
            }

            // The audio threads have crashed, report their errors and die
            mut audio_error @ Err(_) => {
                while let Err(error) = audio_error {
                    error!("Audio thread error: {:?}", error);
                    audio_error = recording.read_history(fourier.input());
                }
                error!("Audio thread exited due to errors, time to die...");
                std::process::exit(1);
            }
        };

        // Compute the Fourier transform
        let fft_amps = fourier.compute();

        // Resample it to the desired number of output bins
        let output_bins = resampler.resample(fft_amps);

        // Display the resampled FFT bins
        // FIXME: Move to a real GUI display, but maybe keep this one around
        //        just for fun and as an exercise in abstraction.
        const SPARKLINE: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        print!("FFT: ");
        for &bin in output_bins {
            let spark = if bin < -amplitude_scale {
                SPARKLINE[0]
            } else if bin > 0.0 {
                SPARKLINE.last().unwrap().clone()
            } else {
                let normalized = bin / amplitude_scale + 1.0;
                let idx = (normalized * (SPARKLINE.len() - 2) as f32) as usize + 1;
                SPARKLINE[idx]
            };
            print!("{}", spark);
        }
        println!();
    }
}
