mod errors;

use self::errors::{ErrorInput, ErrorOutput};
use jack::{
    AsyncClient, AudioIn, Client, Control, Frames, NotificationHandler, Port, ProcessHandler,
    ProcessScope,
};
use rt_history::{Overrun, RTHistory};
use std::panic::AssertUnwindSafe;

// Expose audio thread errors so the main thread can process them
pub use errors::AudioError;

/// Handle to a prepared audio setup that is not recording data yet
pub struct AudioSetup(Client);
//
impl AudioSetup {
    /// Set up the audio stack
    pub fn new() -> crate::Result<Self> {
        // Set up a JACK client
        let (jack_client, status) =
            jack::Client::new(env!("CARGO_PKG_NAME"), jack::ClientOptions::NO_START_SERVER)?;
        log::debug!("Got jack client with status: {:?}", status);
        Ok(Self(jack_client))
    }

    /// Query audio sampling rate
    pub fn sample_rate(&self) -> usize {
        self.0.sample_rate()
    }

    /// Granularity at which history data will be written by the audio thread
    pub fn buffer_size(&self) -> usize {
        self.0.buffer_size() as usize
    }

    /// Start recording audio data into a history buffer of a certain length
    pub fn start_recording(self, history_len: usize) -> crate::Result<AudioRecording> {
        // Allocate history buffer
        let (hist_input, hist_output) = RTHistory::new(history_len).split();

        // Setup audio input port
        let jack_client = self.0;
        let input_port = jack_client.register_port("input", AudioIn)?;

        // Prepare to handle audio thread errors
        let (error_input, error_output) = errors::setup_error_channel();

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
}

/// Handle to an active audio recording pipeline
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
        // after the thread has exited, except for output_hist but that can't
        // be too badly corrupted by a panic.
        self.error_input.handle_panics(AssertUnwindSafe(|| {
            // Forward new audio data from JACK into our history ring buffer
            self.output_hist
                .write(self.input_port.as_slice(process_scope));
            Control::Continue
        }))
    }

    // By special exemption, this callback is allowed to do allocation-heavy
    // stuff like emitting logs, and we're going to leverage that
    fn buffer_size(&mut self, _: &jack::Client, size: Frames) -> Control {
        // AssertUnwindSafe seems reasonable for the same reason as above.
        self.error_input.handle_panics(AssertUnwindSafe(|| {
            // FIXME: Implement support for reallocating self.output_hist storage,
            //        this should be easy-ish to do since the buffer_size callback
            //        is allowed to do RT-unsafe things like allocating memory and
            //        the main thread has no RT-safety requirements.
            use log::{error, info, warn};
            if size as usize > self.output_hist.capacity() {
                error!(
                    "New JACK buffer size {} is above history capacity {}. \
                     Must reallocate history buffer!",
                    size,
                    self.output_hist.capacity()
                );
                self.error_input
                    .notify_error(AudioError::MustReallocateHistory);
                Control::Quit
            } else {
                if size as usize > self.output_hist.capacity() / 4 {
                    warn!(
                        "New JACK buffer size {} is more than 1/4 of history capacity {}. \
                         Overruns are likely to occur. Should reallocate history buffer!",
                        size,
                        self.output_hist.capacity()
                    );
                } else {
                    info!("Switching to new supported JACK buffer size {}", size);
                }
                Control::Continue
            }
        }))
    }
}
