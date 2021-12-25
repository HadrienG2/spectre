//! In-terminal spectrum display

use crate::{display::FrameResult, Result};
use crossterm::{cursor, terminal, QueueableCommand};
use std::{
    io::Write,
    time::{Duration, Instant},
};

/// Useful Unicode chars for in-terminal graphs
const SPARKLINE: [&'static str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// In-terminal spectrum display
pub struct CliDisplay {
    /// Terminal width
    width: u16,

    /// Terminal height
    height: u16,

    /// Terminal char height in dBFS
    char_amp_scale: f32,

    /// Spectrum display buffer
    spectrum: String,

    /// Last display timestamp
    last_display: Instant,
}
//
impl CliDisplay {
    /// Set up the terminal display
    pub fn new(amp_scale: f32) -> Result<Self> {
        assert!(amp_scale > 0.0);
        let (width, height) = terminal::size().unwrap_or((80, 25));
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        stdout.queue(cursor::Hide)?;
        stdout.queue(terminal::EnterAlternateScreen)?;
        stdout.queue(terminal::DisableLineWrap)?;
        stdout.flush()?;
        let spectrum = String::with_capacity(
            width as usize
                * height as usize
                * SPARKLINE
                    .iter()
                    .map(|c| c.len())
                    .max()
                    .expect("There has to be sparkline chars"),
        );
        Ok(Self {
            width,
            height,
            char_amp_scale: amp_scale / (height - 1) as f32,
            spectrum,
            last_display: Instant::now(),
        })
    }

    /// Report terminal width in chars
    pub fn width(&self) -> usize {
        self.width.into()
    }

    /// Start the event loop, run a user-provided callback on every frame
    ///
    /// This function will call `reset()` at the end, so no other method of the
    /// CliDisplay should be called after it has returned.
    ///
    pub fn run_event_loop(
        &mut self,
        mut frame_callback: impl FnMut(&mut Self) -> Result<FrameResult>,
    ) -> Result<()> {
        let result = loop {
            match frame_callback(self) {
                Ok(FrameResult::Continue) => {}
                Ok(FrameResult::Stop) => break Ok(()),
                Err(e) => break Err(e),
            }
            self.wait_for_frame();
        };
        self.reset_terminal()?;
        result
    }

    /// Display a spectrum
    pub fn render(&mut self, data: &[f32]) -> Result<()> {
        // Validate input
        assert_eq!(data.len(), self.width as usize);

        // Cache some useful quantities
        let char_amp_norm = 1. / self.char_amp_scale;

        // Prepare spectrum display
        self.spectrum.clear();
        for row in 0..self.spectrum_height() {
            let max_val = -(row as f32) * self.char_amp_scale;
            let min_val = -(row as f32 + 1.0) * self.char_amp_scale;
            for &bin in data {
                let spark = if bin < min_val {
                    SPARKLINE[0]
                } else if bin >= max_val {
                    SPARKLINE
                        .last()
                        .expect("There has to be sparkline chars")
                        .clone()
                } else {
                    let normalized = (bin - min_val) * char_amp_norm;
                    let idx = (normalized * (SPARKLINE.len() - 2) as f32) as usize + 1;
                    SPARKLINE[idx]
                };
                self.spectrum.push_str(spark);
            }
            self.spectrum.push('\n');
        }

        // Display the rendered spectrum and clear status line
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        stdout.queue(cursor::MoveTo(0, 0))?;
        write!(stdout, "{}", self.spectrum)?;
        stdout.queue(terminal::Clear(terminal::ClearType::CurrentLine))?;
        stdout.flush()?;

        // We're done
        Ok(())
    }

    /// Report a buffer underrun (audio thread provided no new data)
    pub fn report_underrun(&mut self) -> Result<()> {
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        stdout.queue(cursor::MoveTo(0, self.spectrum_height()))?;
        stdout.queue(terminal::Clear(terminal::ClearType::CurrentLine))?;
        write!(stdout, "No new audio data since last refresh!")?;
        stdout.flush()?;
        Ok(())
    }

    /// Report a buffer overrun (audio thread overwrote some data we were reading)
    pub fn report_overrun(&mut self, excess_samples: usize) -> Result<()> {
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        stdout.queue(cursor::MoveTo(0, self.spectrum_height()))?;
        stdout.queue(terminal::Clear(terminal::ClearType::CurrentLine))?;
        write!(
            stdout,
            "Audio thread overwrote {} samples during readout!",
            excess_samples,
        )?;
        stdout.flush()?;
        Ok(())
    }

    /// Restore the terminal to its initial state
    ///
    /// It is safe to call this function multiple times, but no other function
    /// should be called after it, or terminal corruption will occur.
    ///
    pub fn reset_terminal(&mut self) -> Result<()> {
        let mut stdout = std::io::stdout();
        stdout.queue(cursor::Show)?;
        stdout.queue(terminal::LeaveAlternateScreen)?;
        stdout.queue(terminal::EnableLineWrap)?;
        stdout.flush()?;
        Ok(())
    }

    /// Report spectrum height in chars
    fn spectrum_height(&self) -> u16 {
        self.height - 1
    }

    /// Wait for the previous submitted spectrum to be displayed
    fn wait_for_frame(&mut self) {
        // CLI APIs don't do VSync, but we assume a max display rate of 144Hz
        const MIN_REFRESH_PERIOD: Duration = Duration::from_millis(7);
        let now = Instant::now();
        let next_frame = self.last_display + MIN_REFRESH_PERIOD;
        if now < next_frame {
            std::thread::sleep(next_frame - now)
        }
        self.last_display = Instant::now();
    }
}
//
impl Drop for CliDisplay {
    fn drop(&mut self) {
        self.reset_terminal().expect("Failed to reset the terminal");
    }
}
