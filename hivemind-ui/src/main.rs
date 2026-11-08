//! HiveMind native UI (wave #49): a real desktop window,
//! no browser anywhere. The exe is self-contained: egui
//! draws everything directly, and the Python engine is
//! reached through the local bridge (hivemind_bridge.py)
//! over HTTP. Without python the window still works in a
//! clearly-tagged offline mock mode - honesty over illusion.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod engine;

use std::process::exit;

use eframe::egui;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--selftest") {
        if engine::selftest() {
            println!("selftest: ok");
        } else {
            eprintln!("selftest: FAILED");
            exit(1);
        }
        return;
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 640.0])
            .with_min_inner_size([520.0, 360.0]),
        ..Default::default()
    };
    let result = eframe::run_native(
        "HiveMind",
        options,
        Box::new(|_cc| Ok(Box::new(app::HiveMindApp::new()))),
    );
    if let Err(err) = result {
        eprintln!("ui error: {}", err);
        exit(1);
    }
}
