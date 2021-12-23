//! In-terminal spectrum display

use crate::Result;
use crossterm::{cursor, terminal, QueueableCommand};
use std::{
    io::Write,
    time::{Duration, Instant},
};

/// Useful Unicode chars for in-terminal graphs
const SPARKLINE: [&'static str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// Let's cap refreshes to 144Hz for now.
const MIN_REFRESH_PERIOD: Duration = Duration::from_millis(7);

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
            width as usize * height as usize * SPARKLINE.iter().map(|c| c.len()).max().unwrap_or(1),
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

    /// Report spectrum height in chars
    fn spectrum_height(&self) -> u16 {
        self.height - 1
    }

    /// Display a spectrum
    pub fn display(&mut self, data: &[f32]) -> Result<()> {
        // Validate input
        assert_eq!(data.len(), self.width as usize);

        // Apply display rate limiting
        std::thread::sleep(MIN_REFRESH_PERIOD.saturating_sub(self.last_display.elapsed()));
        self.last_display = Instant::now();

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
                    SPARKLINE.last().unwrap().clone()
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
}
//
impl Drop for CliDisplay {
    fn drop(&mut self) {
        let mut stdout = std::io::stdout();
        stdout.queue(cursor::Show).unwrap();
        stdout.queue(terminal::LeaveAlternateScreen).unwrap();
        stdout.queue(terminal::EnableLineWrap).unwrap();
        stdout.flush().unwrap();
    }
}
