//! WebGPU-based spectrum display
// FIXME: This module is getting very long and should be split into smaller entities

mod core;
mod settings;
mod spectrogram;
mod spectrum;

use self::{
    core::HighLevelEvent, settings::SettingsUniform, spectrogram::Spectrogram, spectrum::Spectrum,
};
use crate::{
    display::{FrameInput, FrameResult},
    Result,
};
use crevice::std140::AsStd140;
use wgpu::{ShaderStages, SurfaceError, TextureViewDescriptor};
use winit::event_loop::ControlFlow;

/// Re-export core context type for child modules
pub(self) use self::core::CoreContext;

/// Custom winit event type
type CustomEvent = ();
type EventLoop = winit::event_loop::EventLoop<CustomEvent>;
type Event<'a> = winit::event::Event<'a, CustomEvent>;

/// Default fraction of the window used by the live spectrum
const DEFAULT_SPECTRUM_WIDTH: f32 = 0.25;

/// Uniform for passing UI settings to rendering shaders
///
/// Must be kept in sync with the rendering shaders
///
#[derive(AsStd140)]
struct Settings {
    /// Horizontal fraction of the window that is occupied by the live spectrum
    spectrum_width: f32,

    /// Range of amplitudes that we can display in dB
    amp_scale: f32,
}

/// GPU-accelerated spectrum display
pub struct GuiDisplay {
    /// Event loop
    event_loop: Option<EventLoop>,

    /// Core context
    core_context: CoreContext,

    /// UI settings
    settings: SettingsUniform<Settings>,

    /// Spectrogram renderer
    spectrogram: Spectrogram,

    /// Spectrum renderer
    spectrum: Spectrum,
}
//
impl GuiDisplay {
    /// Set up the GPU display
    pub fn new(amp_scale: f32, spectrogram_refresh_rate: f32) -> Result<Self> {
        assert!(amp_scale > 0.0);

        // Set up the event loop
        let event_loop = EventLoop::new();

        // Set up the core context
        let core_context = CoreContext::new(&event_loop)?;

        // Set up GPU UI settings
        let device = core_context.device();
        let (settings, settings_bind_group_layout) = SettingsUniform::new(
            device,
            Settings {
                spectrum_width: DEFAULT_SPECTRUM_WIDTH,
                amp_scale,
            },
            ShaderStages::VERTEX_FRAGMENT,
        );

        // Set up spectrogram
        let (spectrogram, spectrogram_texture_view) = Spectrogram::new(
            &core_context,
            &settings_bind_group_layout,
            spectrogram_refresh_rate,
        );

        // Set up live spectrum
        let spectrum = Spectrum::new(
            &core_context,
            &settings_bind_group_layout,
            spectrogram_texture_view,
        );

        // ...and we're ready!
        Ok(Self {
            event_loop: Some(event_loop),
            core_context,
            settings,
            spectrogram,
            spectrum,
        })
    }

    /// Report desired spectrum length in bins
    pub fn spectrum_len(&self) -> usize {
        self.core_context.surface_config().height as _
    }

    /// Start the event loop, run a user-provided callback on every frame
    pub fn run_event_loop(
        mut self,
        mut frame_callback: impl FnMut(&mut Self, FrameInput) -> Result<FrameResult> + 'static,
    ) -> ! {
        // Display the first frame
        let first_result = frame_callback(
            &mut self,
            FrameInput {
                new_spectrum_len: None,
            },
        )
        .expect("Failed to render first frame");
        if first_result == FrameResult::Stop {
            std::mem::drop(frame_callback);
            std::process::exit(0);
        }
        self.core_context.show_window();

        // Start the event loop
        let mut frame_callback = Some(frame_callback);
        let mut resized = false;
        self.event_loop
            .take()
            .expect("Event loop should be present")
            .run(move |event, _target, control_flow| {
                // Perform basic event handling, extract higher-level ops
                match self.core_context.handle_event(event, control_flow) {
                    // Window has been resized, DPI may have changed as well
                    Some(HighLevelEvent::Resized { scale_factor_ratio }) => {
                        resized = true;
                        if let Some(scale_factor_ratio) = scale_factor_ratio {
                            self.spectrogram
                                .handle_scale_factor_change(scale_factor_ratio);
                        }
                    }

                    // It is time to draw a new frame
                    Some(HighLevelEvent::Redraw) => {
                        let mut frame_input = FrameInput {
                            new_spectrum_len: None,
                        };
                        if resized {
                            frame_input.new_spectrum_len =
                                Some(self.core_context.surface_config().height as _);
                            self.handle_resize();
                            resized = false;
                        }
                        match frame_callback.as_mut().expect("Callback should be present")(
                            &mut self,
                            frame_input,
                        ) {
                            Ok(FrameResult::Continue) => {}
                            Ok(FrameResult::Stop) => *control_flow = ControlFlow::Exit,
                            Err(e) => panic!("Frame processing failed: {}", e),
                        }
                    }

                    // The event loop will be destroyed after this call, drop
                    // the things that need dropping for correctness
                    Some(HighLevelEvent::Exit) => std::mem::drop(frame_callback.take()),

                    // TODO: Provide some mouse controls: adjust spectrum width
                    //       via click-drag around separator, spectrum and
                    //       spectrogram zoom-in via click drag + zoom out via
                    //       right click, zoom specific to time, frequency or
                    //       magnitude scales once we have scale bars for those.

                    // This event need not concern us
                    None => {}
                }
            })
    }

    /// Display a spectrum
    pub fn render(&mut self, data: &[f32]) -> Result<()> {
        // Try to access the next window texture
        let window_texture = match self.core_context.current_surface_texture() {
            // Succeeded
            Ok(texture) => texture,

            // Some errors will just resolve themselves, perhaps with some help
            Err(SurfaceError::Timeout | SurfaceError::Outdated) => return Ok(()),
            Err(SurfaceError::Lost) => {
                self.core_context.recreate_surface();
                return Ok(());
            }

            // Other errors are presumed to be serious ones and kill the app*
            Err(e @ SurfaceError::OutOfMemory) => Err(e)?,
        };

        // Acquire a texture view (needed for render passes)
        let window_view = window_texture.texture.create_view(&TextureViewDescriptor {
            label: Some("Window surface texture view"),
            ..Default::default()
        });

        // Prepare to render on the screen texture
        let mut encoder =
            self.core_context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Spectrum render encoder"),
                });

        // Send new spectrum data to the device
        let queue = self.core_context.queue();
        self.spectrum.write_input(&queue, data);

        // Move spectrogram forward if enough time elapsed
        let spectrogram_write_idx = self.spectrogram.write_idx();

        // Update the settings
        let settings_bind_group = self.settings.updated(queue);

        // Display the spectrum and spectrogram
        {
            // Set up a render pass with a black clear color
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Spectrum render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &window_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            // Draw the live spectrum and produce a new spectrogram line
            render_pass.set_bind_group(0, settings_bind_group, &[]);
            self.spectrum
                .draw_and_update_spectrogram(&mut render_pass, spectrogram_write_idx);
        }
        {
            // Spectrogram can't be in above render pass because its spectrogram
            // texture reads would race with the spectrogram texture writes
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Spectrogram render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &window_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            // Draw the spectrogram
            render_pass.set_bind_group(0, settings_bind_group, &[]);
            self.spectrogram.draw(&mut render_pass);
        }

        // Submit our render command
        queue.submit(Some(encoder.finish()));

        // Make sure the output gets displayed on the screen
        window_texture.present();
        Ok(())
    }

    /// Restore the terminal to its initial state
    pub fn reset_terminal(&mut self) -> Result<()> {
        // The GUI backend does not alter the terminal state, so this is easy
        Ok(())
    }

    /// Reallocate structures that depend on the window size after a resize
    fn handle_resize(&mut self) {
        // Reallocate window surface
        self.core_context.recreate_surface();

        // Resize spectrogram texture
        let mut encoder =
            self.core_context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Spectrum rescaling encoder"),
                });
        let spectrogram_texture_view = self
            .spectrogram
            .handle_resize(&self.core_context, &mut encoder);

        // Resize live spectrum texture
        self.spectrum
            .handle_resize(&self.core_context, spectrogram_texture_view);

        // Submit rescaling commands
        self.core_context.queue().submit(Some(encoder.finish()));
    }
}
