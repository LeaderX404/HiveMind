//! The chat window (wave #49): message list, model picker
//! and input box - all drawn immediately by egui. No HTML,
//! no CSS, no browser; when the engine is missing the
//! replies are tagged mock, never faked as real.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Duration;

use eframe::egui;

use crate::engine::{self, BridgeHandle, ChatMessage, EngineEvent, Speaker};

pub struct HiveMindApp {
    messages: Vec<ChatMessage>,
    draft: String,
    models: Vec<String>,
    selected_model: usize,
    status: String,
    engine_ready: bool,
    busy: bool,
    rx: Receiver<EngineEvent>,
    tx: Sender<EngineEvent>,
    bridge: Option<BridgeHandle>,
}

impl HiveMindApp {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        engine::discover(std::sync::mpsc::Sender::clone(&tx));
        Self {
            messages: vec![ChatMessage {
                speaker: Speaker::Assistant,
                text: "HiveMind native UI online. Discovering the engine..."
                    .to_string(),
                mock: false,
            }],
            draft: String::new(),
            models: Vec::new(),
            selected_model: 0,
            status: "discovering the engine...".to_string(),
            engine_ready: false,
            busy: false,
            rx,
            tx,
            bridge: None,
        }
    }

    /// Drain everything the background threads reported.
    fn pump(&mut self, ctx: &egui::Context) {
        loop {
            match self.rx.try_recv() {
                Ok(EngineEvent::Status(status)) => self.status = status,
                Ok(EngineEvent::Ready(ready)) => self.engine_ready = ready,
                Ok(EngineEvent::Models(models)) => {
                    self.models = models;
                    self.selected_model = 0;
                }
                Ok(EngineEvent::BridgeUp(handle)) => self.bridge = Some(handle),
                Ok(EngineEvent::Reply { text, mock, error }) => {
                    self.busy = false;
                    if error {
                        self.status = text;
                    } else {
                        self.messages.push(ChatMessage {
                            speaker: Speaker::Assistant,
                            text,
                            mock,
                        });
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        // Keep the chat alive without burning the CPU:
        // workers push through the channel, we poll calmly.
        ctx.request_repaint_after(Duration::from_millis(120));
    }

    fn send(&mut self) {
        let prompt = self.draft.trim().to_string();
        if prompt.is_empty() || self.busy {
            return;
        }
        self.draft.clear();
        self.messages.push(ChatMessage {
            speaker: Speaker::User,
            text: prompt.clone(),
            mock: false,
        });
        self.busy = true;
        self.status = "generating...".to_string();
        let model = if self.engine_ready {
            self.models
                .get(self.selected_model)
                .cloned()
                .unwrap_or_else(|| "mock-echo".to_string())
        } else {
            "offline-mock".to_string()
        };
        let tx = std::sync::mpsc::Sender::clone(&self.tx);
        engine::spawn_chat(
            tx,
            engine::DEFAULT_PORT,
            self.engine_ready,
            prompt,
            model,
        );
    }

    fn send_enabled(&self) -> bool {
        !self.busy && !self.models.is_empty()
    }
}

impl eframe::App for HiveMindApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump(ctx);

        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("HiveMind");
                ui.separator();
                ui.label(self.status.as_str());
            });
        });

        egui::SidePanel::left("model_panel").show(ctx, |ui| {
            ui.heading("Model");
            if self.models.is_empty() {
                ui.label("(discovering...)");
            }
            for (index, name) in self.models.iter().enumerate() {
                if ui
                    .selectable_label(index == self.selected_model, name.as_str())
                    .clicked()
                {
                    self.selected_model = index;
                }
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.small("wave #49 - native egui UI");
                ui.small("no browser, no HTML, no CSS");
            });
        });

        egui::TopBottomPanel::bottom("composer").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 90.0).max(60.0);
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.draft)
                        .hint_text("Type a message...")
                        .desired_width(width),
                );
                if response.lost_focus()
                    && ctx.input(|i| i.key_pressed(egui::Key::Enter))
                {
                    self.send();
                }
                if ui
                    .add_enabled(self.send_enabled(), egui::Button::new("Send"))
                    .clicked()
                {
                    self.send();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for message in &self.messages {
                        let color = match message.speaker {
                            Speaker::User => egui::Color32::from_rgb(120, 170, 255),
                            Speaker::Assistant => {
                                egui::Color32::from_rgb(190, 235, 175)
                            }
                        };
                        let mut rich =
                            egui::RichText::new(message.text.as_str()).color(color);
                        if message.mock {
                            rich = rich.weak();
                        }
                        ui.label(rich);
                        ui.add_space(4.0);
                    }
                });
        });
    }
}
