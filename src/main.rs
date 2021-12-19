mod audio;
mod fourier;
pub mod simd;

use crate::{audio::AudioSetup, fourier::FourierTransform};
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
}

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();

    // Decode CLI arguments
    let opt = CliOpts::from_args();
    debug!("Got CLI options {:?}", opt);

    // Set up the audio stack
    let audio = AudioSetup::new()?;

    // Set up the Fourier transform
    let mut fourier = FourierTransform::new(opt.frequency_resolution, audio.sample_rate());

    // Start recording audio, keeping enough history that the audio thread can
    // write two full periods before triggering an FFT input readout overrun.
    let history_len = if audio.buffer_size() <= fourier.input().len() / 2 {
        2 * fourier.input().len()
    } else {
        4 * audio.buffer_size()
    };
    let mut recording = audio.start_recording(history_len)?;

    // Start computing some FFTs
    let mut last_clock = 0;
    for _ in 0..1000 {
        // FIXME: Simulate vertical synchronization
        std::thread::sleep(std::time::Duration::from_millis(7));

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
        let _fft_amps = fourier.compute();

        // TODO: Display the FFT
        // println!("Current FFT is {:?}", fft_amps);
    }

    Ok(())
}
