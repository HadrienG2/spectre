use anyhow::Result;
use jack::{AudioIn, Control, Frames, NotificationHandler, Port, ProcessHandler, ProcessScope};
use log::{debug, error, info, warn};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use realfft::RealFftPlanner;
use rt_history::{Overrun, RTHistory};
use std::{
    panic::{catch_unwind, AssertUnwindSafe, UnwindSafe},
    sync::{
        atomic::{self, AtomicUsize, Ordering},
        Arc,
    },
};
use structopt::StructOpt;

// Stuff that we may wish to configure at compile time, but the user is very
// unlikely to ever want to change at runtime.

/// Remove DC offsets before computing Fourier transform
const REMOVE_DC: bool = true;

/// Fatal errors that can occur within the audio thread
#[derive(Debug, FromPrimitive)]
pub enum AudioError {
    /// An audio callback has panicked
    CallbackPanicked = 0,

    /// The sample rate has changed (and we aren't ready to handle that)
    SampleRateChanged,

    /// The history buffer must be reallocated (and we aren't ready to do so)
    MustReallocateHistory,
}

/// Setup audio thread error notification mechanism
fn setup_error_channel() -> (ErrorInput, ErrorOutput) {
    let flag = Arc::new(AtomicUsize::new(0));
    (ErrorInput(flag.clone()), ErrorOutput(flag))
}

/// Mechanism to notify the main thread of audio thread errors
#[derive(Clone)]
struct ErrorInput(Arc<AtomicUsize>);
//
impl ErrorInput {
    /// Notify the main thread that an audio thread error has occured
    fn notify_error(&self, what: AudioError) {
        // Set the new error flag
        self.0.fetch_or(1 << (what as u32), Ordering::Relaxed);

        // Error must be notified before we touch any other shared state
        atomic::fence(Ordering::Release);
    }

    /// Run a JACK callback, catch panics and report them to the main thread
    /// while avoiding unwind-through-C undefined behavior.
    fn handle_panics(&self, f: impl UnwindSafe + FnOnce() -> Control) -> Control {
        match catch_unwind(f) {
            Ok(c) => c,
            Err(_e) => {
                self.notify_error(AudioError::CallbackPanicked);
                Control::Quit
            }
        }
    }
}

/// Mechanism to receive audio thread errors in the main thread
pub struct ErrorOutput(Arc<AtomicUsize>);
//
impl ErrorOutput {
    /// Look for the next audio thread error, if any
    pub fn next_error(&mut self) -> Option<AudioError> {
        // Query the current error flags
        // Must be Acquire because we want to make sure that error readout
        // occurs before any other shared state is touched.
        let flags = self.0.load(Ordering::Acquire);

        // Early exit if no error occured
        if flags == 0 {
            return None;
        }

        // Find the first error number and unset the associated flag so that we
        // return the next audio thread error on the next call to this method.
        let first_error = flags.trailing_zeros();
        self.0.fetch_and(!(1 << first_error), Ordering::Relaxed);

        // Convert the error number back to an AudioError
        let error = AudioError::from_u32(first_error);
        assert!(
            error.is_some(),
            "Encountered unknown audio thread error code {}",
            first_error
        );
        error
    }
}

// TODO: Use CLI options here instead of consts, for anything that we may
//       want to change dynamically.
#[derive(Debug, StructOpt)]
struct CliOpts {
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

    /// Audio thread error notification mechanism
    error_input: ErrorInput,
}

impl NotificationHandler for NotificationState {
    fn sample_rate(&mut self, _: &jack::Client, srate: Frames) -> Control {
        self.error_input.handle_panics(|| {
            if self.sample_rate != srate {
                // FIXME: Instead of bombing, rerun bits of initialization that depends
                //        on the sample rate, like FFT buffer allocation.
                //        Should only be implemented once the code is rather mature and
                //        we know well what must be done here.
                self.error_input.notify_error(AudioError::SampleRateChanged);
                Control::Quit
            } else {
                Control::Continue
            }
        })
    }
}

struct ProcessState {
    /// Port which input data is coming from
    input_port: Port<AudioIn>,

    /// Output location to which audio frames are sent
    output_hist: rt_history::Input<f32>,

    /// Audio thread error notification mechanism
    error_input: ErrorInput,
}

impl ProcessHandler for ProcessState {
    fn process(&mut self, _: &jack::Client, process_scope: &ProcessScope) -> Control {
        // AssertUnwindSafe seems reasonable here because JACK will not call us
        // back if Control::Quit is returned and the state is not accessible
        // after the thread has exited.
        self.error_input.handle_panics(AssertUnwindSafe(|| {
            // Forward new audio data from JACK into our history ring buffer
            self.output_hist
                .write(self.input_port.as_slice(process_scope));
            Control::Continue
        }))
    }

    fn buffer_size(&mut self, _: &jack::Client, size: Frames) -> Control {
        // FIXME: Implement support for reallocating self.output_hist storage,
        //        this should be easy-ish to do since the buffer_size callback
        //        is allowed to do RT-unsafe things like allocating memory and
        //        the main thread has no RT-safety requirements.
        self.error_input.handle_panics(AssertUnwindSafe(|| {
            if size as usize > self.output_hist.capacity() {
                self.error_input
                    .notify_error(AudioError::MustReallocateHistory);
                Control::Quit
            } else {
                if size as usize > self.output_hist.capacity() / 4 {
                    // Can emit a warning since this callback does not need to be RT-safe
                    warn!("Should reallocate ring buffer, overruns are likely to occur!");
                }
                Control::Continue
            }
        }))
    }
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

    // Set up a communication channel with the audio thread, ensuring that the
    // audio thread can write two periods before triggering an overrun.
    assert!((jack_client.buffer_size() as usize) <= fft_len / 2);
    let (hist_input, hist_output) = RTHistory::new(2 * fft_len).split();

    // Start recording audio
    let (error_input, mut audio_errors) = setup_error_channel();
    let notification_handler = NotificationState {
        sample_rate,
        error_input: error_input.clone(),
    };
    let process_handler = ProcessState {
        input_port,
        output_hist: hist_input,
        error_input,
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

        // Handle audio thread errors, if any
        let mut error_opt = audio_errors.next_error();
        if error_opt.is_some() {
            while let Some(error) = error_opt.take() {
                error!("Audio thread reported an error: {:?}", error);
                error_opt = audio_errors.next_error();
            }
            error!("Audio thread exited due to errors, time to die...");
            std::process::exit(1);
        }

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
        // println!("Current FFT is {:?}", fft_amps);
    }

    Ok(())
}
