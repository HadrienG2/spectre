use jack::{
    AsyncClient, AudioIn, Control, Frames, NotificationHandler, Port, ProcessHandler, ProcessScope,
};
use log::warn;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use rt_history::{Overrun, RTHistory};
use std::{
    panic::{catch_unwind, AssertUnwindSafe, UnwindSafe},
    sync::{
        atomic::{self, AtomicUsize, Ordering},
        Arc,
    },
};

/// Fatal errors that can occur within the audio threads
#[derive(Debug, FromPrimitive)]
pub enum AudioError {
    /// An audio callback has panicked
    CallbackPanicked = 0,

    /// The sample rate has changed (and we aren't ready to handle that)
    SampleRateChanged,

    /// The history buffer must be reallocated (and we aren't ready to do so)
    MustReallocateHistory,
}

/// Handle to an audio recording pipeline
pub struct AudioRecording {
    /// Underlying connection to the JACK audio server
    _jack_client: AsyncClient<NotificationState, ProcessState>,

    /// Mechanism to query errors from the audio threads
    error_output: ErrorOutput,

    /// Mechanism to read the latest audio history from the audio threads
    hist_output: rt_history::Output<f32>,
}
//
impl AudioRecording {
    /// Start recording audio into a history buffer of user-specified length
    pub fn start(jack_client: jack::Client, history_len: usize) -> crate::Result<Self> {
        // Allocate history buffer
        let (hist_input, hist_output) = RTHistory::new(history_len).split();

        // Setup audio input port
        let input_port = jack_client.register_port("input", AudioIn)?;

        // Prepare to handle audio thread errors
        let (error_input, error_output) = setup_error_channel();

        // Start recording audio
        let notification_handler = NotificationState {
            sample_rate: jack_client.sample_rate() as Frames,
            error_input: error_input.clone(),
        };
        let process_handler = ProcessState {
            input_port,
            output_hist: hist_input,
            error_input,
        };
        let _jack_client = jack_client.activate_async(notification_handler, process_handler)?;

        // Give the caller a handle onto the audio recording process
        Ok(AudioRecording {
            _jack_client,
            error_output,
            hist_output,
        })
    }

    /// Read latest audio history after checking for audio thread errors
    pub fn read_history(
        &mut self,
        target: &mut [f32],
    ) -> Result<Result<rt_history::Clock, Overrun>, AudioError> {
        if let Some(error) = self.error_output.next_error() {
            Err(error)
        } else {
            Ok(self.hist_output.read(target))
        }
    }
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
