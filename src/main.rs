use anyhow::Result;
use jack::{AudioIn, Control, Frames, NotificationHandler, Port, ProcessHandler, ProcessScope};
use log::{debug, error, info, warn};
use realfft::RealFftPlanner;
use rt_history::{Overrun, RTHistory};
use std::panic::{catch_unwind, AssertUnwindSafe, UnwindSafe};
use structopt::StructOpt;

/// Remove DC offsets before computing Fourier transform
const REMOVE_DC: bool = true;

fn handle_panics(f: impl UnwindSafe + FnOnce() -> Control) -> Control {
    match catch_unwind(f) {
        Ok(c) => c,
        Err(_e) => {
            error!("Panic occurred in JACK thread, aborting...");
            Control::Quit
        }
    }
}

// TODO: Use CLI options here instead of consts, for anything that we may
//       want to change dynamically.
#[derive(Debug, StructOpt)]
struct Opt {
    /// Minimal frequency resolution in Hz
    #[structopt(long, default_value = "1.0")]
    frequency_resolution: f32,
}

struct NotificationState {
    /// Last supported sample rate
    ///
    /// We don't support sample rate changes yet, even though JACK theoretically
    /// does, because that requires FFT width changes, which requires FFT buffer
    /// reallocations and thus tricky lock-free algorithms in a RT environment.
    ///
    sample_rate: Frames,
}

impl NotificationHandler for NotificationState {
    fn sample_rate(&mut self, _: &jack::Client, srate: Frames) -> Control {
        handle_panics(|| {
            if self.sample_rate != srate {
                // FIXME: Instead of bombing, rerun bits of initialization that depends
                //        on the sample rate, like FFT buffer allocation.
                //        Should only be implemented once the code is rather mature and
                //        we know well what must be done here.
                eprintln!("Sample rate changes are not supported yet!");
                Control::Quit
            } else {
                Control::Continue
            }
        })
    }
}

struct ProcessState {
    /// Last observed buffer size
    ///
    /// We don't support buffer size changes yet, but if the code doesn't change
    /// too much, we may be able to.
    ///
    buffer_size: Frames,

    /// Port which input data is coming from
    input_port: Port<AudioIn>,

    /// Output location to which audio frames are sent
    output_hist: rt_history::Input<f32>,
}

impl ProcessHandler for ProcessState {
    fn process(&mut self, _: &jack::Client, process_scope: &ProcessScope) -> Control {
        // AssertUnwindSafe seems reasonable here because JACK will not call us
        // back if Control::Quit is returned and the state is not accessible
        // after the thread has exited.
        handle_panics(AssertUnwindSafe(|| {
            // Forward new audio data from JACK into our history ring buffer
            self.output_hist
                .write(self.input_port.as_slice(process_scope));
            Control::Continue
        }))
    }

    fn buffer_size(&mut self, _: &jack::Client, size: Frames) -> Control {
        if self.buffer_size != size {
            // FIXME: Instead of bombing, rerun bits of initialization that depend
            //        on the buffer size, like latency sanity checks.
            //        Should only be implemented once the code is rather mature and
            //        we know well what must be done here.
            eprintln!("Buffer size changes are not supported yet!");
            Control::Quit
        } else {
            Control::Continue
        }
    }
}

fn main() -> Result<()> {
    // Set up logging
    env_logger::init();
    jack::set_error_callback(|msg| error!("JACK said: {}", msg));
    jack::set_info_callback(|msg| info!("JACK said: {}", msg));

    // Decode CLI arguments
    let opt = Opt::from_args();
    debug!("Got CLI options {:?}", opt);

    // Set up JACK client and port
    let (jack_client, status) =
        jack::Client::new(env!("CARGO_PKG_NAME"), jack::ClientOptions::NO_START_SERVER)?;
    debug!("Got jack client with status: {:?}", status);
    let input_port = jack_client.register_port("input", AudioIn)?;

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
    let sample_rate = jack_client.sample_rate() as Frames;
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

    // Set up a communication channel with the audio thread
    let (hist_input, hist_output) = RTHistory::new(3 * fft_len).split();

    // Start recording audio
    let notification_handler = NotificationState { sample_rate };
    let process_handler = ProcessState {
        buffer_size: jack_client.buffer_size() as _,
        input_port,
        output_hist: hist_input,
    };
    let _jack_client = jack_client.activate_async(notification_handler, process_handler)?;

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

        // Fetch some FFT history, check for overruns and underruns
        last_clock = match hist_output.read(&mut fft_input[..]) {
            Ok(clock) => {
                if clock == last_clock {
                    warn!("Buffer underrun! (No new data from audio thread)");
                }
                clock
            }
            Err(Overrun {
                clock,
                excess_entries,
            }) => {
                error!(
                    "Buffer overrun! (Audio thread overwrote {} samples)",
                    excess_entries
                );
                clock
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
        // eprintln!("Current FFT is {:?}", fft_amps);
    }

    Ok(())
}
