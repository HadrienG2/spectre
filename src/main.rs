mod audio;
mod fourier;
mod resample;
pub mod simd;

use crate::{audio::AudioSetup, fourier::FourierTransform, resample::FourierResampler};
use log::{debug, error, warn};
use rt_history::Overrun;
use structopt::StructOpt;

/// Default Result type used throughout this app whenever bubbling errors up
/// seems to be the only sensible option.
pub use anyhow::Result;

// TODO: Use CLI options here instead of consts, for anything that we may
//       want to change dynamically.
#[derive(Debug, StructOpt)]
struct CliOpts {
    /// Minimum monitored frequency in Hz
    #[structopt(long, default_value = "20.0")]
    min_frequency: f32,

    /// Maximum monitored frequency in Hz
    #[structopt(long, default_value = "20000.0")]
    max_frequency: f32,

    /// Minimal frequency resolution in Hz
    #[structopt(long, default_value = "1.0")]
    frequency_resolution: f32,

    /// Use a linear frequency scale
    ///
    /// By default, spectre will use a log scale as it better matches human
    /// audition, but linear scales may be better for specific applications.
    ///
    #[structopt(long)]
    linear_scale: bool,

    /// Amplitude scale in dBFS
    ///
    /// Signal amplitudes lower than this amount below 0dBFS will not be
    /// rendered, typically you will want to set this at your "noise floor".
    ///
    #[structopt(long, default_value = "96")]
    amplitude_scale: f32,
}

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Decode CLI arguments
    let opts = CliOpts::from_args();
    debug!("Got CLI options {:?}", opts);
    if !opts.min_frequency.is_finite() || opts.min_frequency < 0.0 {
        panic!("Please enter a sensible minimum frequency");
    }
    if !opts.max_frequency.is_finite() || opts.max_frequency <= opts.min_frequency {
        panic!("Please enter a sensible maximum frequency");
    }
    if !opts.frequency_resolution.is_finite() || opts.frequency_resolution <= 0.0 {
        panic!("Please enter a sensible frequency resolution");
    }
    if !opts.amplitude_scale.is_finite() {
        panic!("Please enter a sensible amplitude scale");
    }
    let amplitude_scale = opts.amplitude_scale.abs();

    // Set up the audio stack
    let audio = AudioSetup::new()?;
    let sample_rate = audio.sample_rate();
    if opts.max_frequency > (sample_rate / 2) as f32 {
        panic!("Requested max frequency can't be probed at current sampling rate");
    }

    // Set up the Fourier transform
    let mut fourier = FourierTransform::new(opts.frequency_resolution, sample_rate);

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
        opts.min_frequency,
        opts.max_frequency,
        !opts.linear_scale,
    );

    // Start computing some FFTs
    let mut last_clock = 0;
    loop {
        // FIXME: Simulate vertical synchronization
        std::thread::sleep(
            // TODO: Should be provided by terminal display
            std::time::Duration::from_millis(100), /* std::time::Duration::from_millis(7) */
        );

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
        //        just for fun
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
