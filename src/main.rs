mod audio;
mod fourier;
mod resample;
pub mod simd;

use crate::{audio::AudioSetup, fourier::FourierTransform, resample::FourierResampler};
use crossterm::{cursor, terminal, QueueableCommand};
use log::{debug, error};
use rt_history::Overrun;
use std::{
    io::Write,
    time::{Duration, Instant},
};
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
    let mut fourier = FourierTransform::new(opts.freq_res, sample_rate, &opts.window);

    // Start recording audio, keeping enough history that the audio thread can
    // write two full periods before triggering an FFT input readout overrun.
    let history_len = if audio.buffer_size() <= fourier.input().len() / 2 {
        2 * fourier.input().len()
    } else {
        4 * audio.buffer_size()
    };
    let mut recording = audio.start_recording(history_len)?;

    // Initialize the terminal display
    let (term_width, term_height) = terminal::size().unwrap_or((80, 25));
    let (term_width, term_height): (usize, usize) = (term_width.into(), term_height.into());
    let spectrum_height = term_height - 1;
    let mut stdout = std::io::stdout();
    stdout.queue(cursor::Hide)?;
    stdout.queue(terminal::EnterAlternateScreen)?;
    stdout.queue(terminal::DisableLineWrap)?;
    let restore_terminal = || {
        let mut stdout = std::io::stdout();
        stdout.queue(cursor::Show).unwrap();
        stdout.queue(terminal::LeaveAlternateScreen).unwrap();
        stdout.queue(terminal::EnableLineWrap).unwrap();
        stdout.flush().unwrap();
    };
    ctrlc::set_handler(move || {
        restore_terminal();
        std::process::exit(0);
    })?;
    const SPARKLINE: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut spectrum = String::with_capacity(
        term_width * term_height * SPARKLINE.iter().map(|c| c.len_utf8()).max().unwrap_or(1),
    );
    let row_height = amplitude_scale / spectrum_height as f32;

    // Prepare to resample the Fourier transform for display purposes
    let mut resampler = FourierResampler::new(
        fourier.output_len(),
        sample_rate,
        term_width,
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
                restore_terminal();
                while let Err(error) = audio_error {
                    error!("Audio thread error: {:?}", error);
                    audio_error = recording.read_history(fourier.input());
                }
                error!("Audio thread exited due to errors, time to die...");
                std::process::exit(1);
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
                // FIXME: Move to a real GUI display, but maybe keep this one around
                //        just for fun and as an exercise in abstraction.
                spectrum.clear();
                for row in 0..spectrum_height {
                    let max_val = -(row as f32) * row_height;
                    let min_val = -(row as f32 + 1.0) * row_height;
                    for &bin in output_bins {
                        let spark = if bin < min_val {
                            SPARKLINE[0]
                        } else if bin > max_val {
                            SPARKLINE.last().unwrap().clone()
                        } else {
                            let normalized = (bin - min_val) / row_height;
                            let idx = (normalized * (SPARKLINE.len() - 2) as f32) as usize + 1;
                            SPARKLINE[idx]
                        };
                        spectrum.push(spark);
                    }
                    spectrum.push('\n');
                }
                stdout.queue(cursor::MoveTo(0, 0))?;
                print!("{}", spectrum);
                stdout.queue(terminal::Clear(terminal::ClearType::CurrentLine))?;
                stdout.flush()?;
            }

            // Buffer underrun (no new data)
            (true, _) => {
                stdout.queue(cursor::MoveTo(0, spectrum_height as _))?;
                stdout.queue(terminal::Clear(terminal::ClearType::CurrentLine))?;
                print!("No new audio data since last refresh!");
                stdout.flush()?;
            }

            // Buffer overrun (audio thread overwrote buffer while we were reading)
            (false, Some(excess_samples)) => {
                stdout.queue(cursor::MoveTo(0, spectrum_height as _))?;
                stdout.queue(terminal::Clear(terminal::ClearType::CurrentLine))?;
                print!(
                    "Audio thread overwrote {} samples during readout!",
                    excess_samples,
                );
                stdout.flush()?;
            }
        }
    }
}
