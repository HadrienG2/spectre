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
    /// Minimal frequency resolution in Hz
    #[structopt(long, default_value = "1.0")]
    frequency_resolution: f32,

    /// Use a linear scale (the default log scale better matches human audition)
    #[structopt(long)]
    linear_scale: bool,
}

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Decode CLI arguments
    let opts = CliOpts::from_args();
    debug!("Got CLI options {:?}", opts);

    // Set up the audio stack
    let audio = AudioSetup::new()?;
    let sample_rate = audio.sample_rate();

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
    const NUM_OUTPUT_BINS: usize = 10;
    let min_freq = 20.0;
    let max_freq = 20_000.0;
    let mut resampler = FourierResampler::new(
        fourier.output_len(),
        sample_rate,
        NUM_OUTPUT_BINS,
        min_freq,
        max_freq,
        !opts.linear_scale,
    );

    // Start computing some FFTs
    let mut last_clock = 0;
    for _ in 0..1000 {
        // FIXME: Simulate vertical synchronization
        std::thread::sleep(
            std::time::Duration::from_secs(1), /* std::time::Duration::from_millis(7) */
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
        // FIXME: Move to a real GUI display
        println!("FFT bins: {:?}", output_bins);
    }

    Ok(())
}
