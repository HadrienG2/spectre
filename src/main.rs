mod audio;

use self::audio::AudioRecording;
use log::{debug, error, info, warn};
use realfft::RealFftPlanner;
use rt_history::Overrun;
use structopt::StructOpt;

/// Default Result type used throughout this app whenever bubbling errors up
/// seems to be the only sensible option.
pub use anyhow::Result;

/// Remove DC offsets before computing Fourier transform
const REMOVE_DC: bool = true;

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

    // Set up JACK client and port
    let (jack_client, status) =
        jack::Client::new(env!("CARGO_PKG_NAME"), jack::ClientOptions::NO_START_SERVER)?;
    debug!("Got jack client with status: {:?}", status);

    // Translate the desired frequency resolution into an FFT length
    //
    // Given 2xN input data point, a real-fft produces N+1 frequency bins
    // ranging from 0 frequency to sampling_rate/2. So bins spacing df is
    // sampling_rate/(2*N) Hz.
    //
    // By inverting this relation, we get that the smallest N needed to achieve
    // a bin spacing smaller than df is Nmin = sampling_rate / (2 * df). We turn
    // back that Nmin to a number of points 2xNmin, and we round that to the
    // next power of two.
    //
    let sample_rate = jack_client.sample_rate();
    let fft_len = 2_usize.pow(
        (sample_rate as f32 / opt.frequency_resolution)
            .log2()
            .ceil() as _,
    );
    info!(
        "At a sampling rate of {} Hz, achieving the requested frequency resolution of {} Hz requires a {}-points FFT",
        jack_client.sample_rate(),
        opt.frequency_resolution,
        fft_len
    );

    // Start recording audio after making sure that the audio thread can write
    // two periods before triggering an overrun (which should always be true,
    // since we're computing long FFTs).
    assert!((jack_client.buffer_size() as usize) <= fft_len / 2);
    let mut recording = AudioRecording::start(jack_client, 2 * fft_len)?;

    // Prepare for the FFT computation
    let mut fft_planner = RealFftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(fft_len);
    let mut fft_input = fft.make_input_vec().into_boxed_slice();
    let mut fft_output = fft.make_output_vec().into_boxed_slice();
    let mut fft_scratch = fft.make_scratch_vec().into_boxed_slice();
    let mut fft_amps = vec![0.0; fft_output.len()].into_boxed_slice();

    // Start computing some FFTs
    let mut last_clock = 0;
    for _ in 0..1000 {
        // DEBUG: Simulate vertical synchronization
        std::thread::sleep(std::time::Duration::from_millis(7));

        // Read latest FFT history, handle xruns and audio thread errors
        last_clock = match recording.read_history(&mut fft_input[..]) {
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
                    audio_error = recording.read_history(&mut fft_input[..]);
                }
                error!("Audio thread exited due to errors, time to die...");
                std::process::exit(1);
            }
        };

        // Remove DC offset
        if REMOVE_DC {
            let average = fft_input.iter().sum::<f32>() / fft_len as f32;
            fft_input.iter_mut().for_each(|elem| *elem -= average);
        }

        // Compute FFT
        fft.process_with_scratch(
            &mut fft_input[..],
            &mut fft_output[..],
            &mut fft_scratch[..],
        )
        .expect("Failed to compute FFT");

        // Normalize amplitudes, convert to dBm, and send the result out
        let norm_sqr = 1.0 / fft_input.len() as f32;
        for (coeff, amp) in fft_output.iter().zip(fft_amps.iter_mut()) {
            *amp = 10.0 * (coeff.norm_sqr() * norm_sqr).log10();
        }

        // TODO: Display the FFT
        // println!("Current FFT is {:?}", fft_amps);
    }

    Ok(())
}
