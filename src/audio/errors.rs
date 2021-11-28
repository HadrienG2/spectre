use jack::Control;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use std::{
    panic::{catch_unwind, UnwindSafe},
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

/// Setup audio thread error notification mechanism
pub fn setup_error_channel() -> (ErrorInput, ErrorOutput) {
    let flag = Arc::new(AtomicUsize::new(0));
    (ErrorInput(flag.clone()), ErrorOutput(flag))
}

/// Mechanism to notify the main thread of audio thread errors
#[derive(Clone)]
pub struct ErrorInput(Arc<AtomicUsize>);
//
impl ErrorInput {
    /// Notify the main thread that an audio thread error has occured
    pub fn notify_error(&self, what: AudioError) {
        // Set the new error flag
        self.0.fetch_or(1 << (what as u32), Ordering::Relaxed);

        // Error must be notified before we touch any other shared state
        atomic::fence(Ordering::Release);
    }

    /// Run a JACK callback, catch panics and report them to the main thread
    /// while avoiding unwind-through-C undefined behavior.
    pub fn handle_panics(&self, f: impl UnwindSafe + FnOnce() -> Control) -> Control {
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
