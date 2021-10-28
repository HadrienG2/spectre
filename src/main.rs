use anyhow::Result;
use jack::{AudioIn, Control, Frames, NotificationHandler, Port, ProcessHandler, ProcessScope};
use log::{debug, error, info, warn};
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use structopt::StructOpt;
use triple_buffer::TripleBuffer;

/// Remove DC offsets before computing Fourier transform
const REMOVE_DC: bool = true;

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

    /// Last observed samples from input_port, in a ring buffer layout.
    last_points: Box<[f32]>,

    /// Index at which we're currently writing in the last_points ring buffer
    last_point_idx: usize,

    /// FFT implementation
    fft: Arc<dyn RealToComplex<f32>>,

    /// FFT input buffer, will be overwritten
    fft_input: Box<[f32]>,

    /// FFT output buffer, will hold final coefficients
    fft_output: Box<[Complex<f32>]>,

    /// FFT scratch space, will be overwritten
    fft_scratch: Box<[Complex<f32>]>,

    /// Triple buffer to send FFT amplitudes to the main thread
    fft_amps_in: triple_buffer::Input<Box<[f32]>>,

    /// Number of audio periods to go through before computing an FFT
    fft_period: usize,

    /// Number of audio periods that we went through so far
    curr_period: usize,

    /// Truth that the display thread is ready to accept input
    display_ready: Arc<AtomicBool>,
}

impl ProcessHandler for ProcessState {
    fn process(&mut self, _: &jack::Client, process_scope: &ProcessScope) -> Control {
        // Collect new audio data from JACK into our FFT input ring buffer
        let new_data = self.input_port.as_slice(process_scope);
        let (new_points, old_points) = self.last_points.split_at_mut(self.last_point_idx);
        for (src, target) in new_data
            .iter()
            .zip(old_points.iter_mut().chain(new_points.iter_mut()))
        {
            *target = *src;
        }
        self.last_point_idx = (self.last_point_idx + new_data.len()) % self.last_points.len();

        // Compute a new FFT if it's time to do so
        self.curr_period += 1;
        if self.curr_period >= self.fft_period {
            // Linearize recent signal history
            let (new_points, old_points) = self.last_points.split_at_mut(self.last_point_idx);
            let (old_target, new_target) = self.fft_input.split_at_mut(old_points.len());
            old_target.copy_from_slice(old_points);
            new_target.copy_from_slice(new_points);

            // Remove DC offset
            if REMOVE_DC {
                let average = self.fft_input.iter().sum::<f32>() / self.fft_input.len() as f32;
                self.fft_input.iter_mut().for_each(|elem| *elem -= average);
            }

            // Compute FFT
            let result = self.fft.process_with_scratch(
                &mut self.fft_input[..],
                &mut self.fft_output[..],
                &mut self.fft_scratch[..],
            );
            if let Err(err) = result {
                error!("Failed to compute FFT: {}", err);
                return Control::Quit;
            }

            // Normalize amplitudes, convert to dBm, and send the result out
            let amps = self.fft_amps_in.input_buffer();
            let norm_sqr = 1.0 / self.fft_input.len() as f32;
            for (coeff, amp) in self.fft_output.iter().zip(amps.iter_mut()) {
                *amp = 10.0 * (coeff.norm_sqr() * norm_sqr).log10();
            }
            let overwrite = self.fft_amps_in.publish();

            // The triple buffer tells us if we did overwrite an unread FFT, and
            // we use that to probe the downstream readout rate and adjust our
            // own FFT computation rate accordingly.
            if overwrite && self.display_ready.load(Ordering::Relaxed) {
                // FIXME: Make sure this only triggers after client started
                //        receiving data, via an atomic.
                self.fft_period += 1;
            } else {
                self.fft_period = self.fft_period.saturating_sub(1);
            }
            self.curr_period = 0;
        }
        Control::Continue
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

    // Prepare for the FFT computation
    let mut fft_planner = RealFftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(fft_len);
    let fft_input = vec![0.; fft_len].into_boxed_slice();
    let fft_output = fft.make_output_vec().into_boxed_slice();
    let fft_scratch = fft.make_scratch_vec().into_boxed_slice();

    // Prepare triple buffer for sending FFTs to the main thread
    let fft_amps = TripleBuffer::new(vec![0.; fft_output.len()].into_boxed_slice());
    let (fft_amps_in, mut fft_amps_out) = fft_amps.split();

    // Start processing audio (FIXME: should be done once downstream chain is ready.
    let display_ready = Arc::new(AtomicBool::new(false));
    let notification_handler = NotificationState { sample_rate };
    let process_handler = ProcessState {
        buffer_size: jack_client.buffer_size() as _,
        input_port,
        last_points: fft_input.clone(),
        last_point_idx: 0,
        fft,
        fft_input,
        fft_output,
        fft_scratch,
        fft_amps_in,
        fft_period: 1,
        curr_period: 0,
        display_ready: display_ready.clone(),
    };
    let _jack_client = jack_client.activate_async(notification_handler, process_handler)?;

    // HACK: Sleep a bit to see if audio thread is stable
    display_ready.store(true, Ordering::Relaxed);
    for _ in 0..1000 {
        std::thread::sleep(std::time::Duration::from_millis(17));
        if !fft_amps_out.update() {
            warn!("Display thread got a stale FFT. Audio thread is overloaded or buffer size is too high.");
        }
        // eprintln!("Current FFT is {:?}", fft_amps_out.output_buffer());
    }

    Ok(())
}
