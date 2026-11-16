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
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            Ok(Box::new(app::HiveMindApp::new()))
        }),
    );
    if let Err(err) = result {
        eprintln!("ui error: {}", err);
        exit(1);
    }
}

/// Wave #51: Arabic must render as letters, not boxes. The
/// Noto Naskh Arabic font is fetched at build time from a
/// commit-pinned URL (see assets/fetch-font.ps1) and embedded
/// into the exe here. It is APPENDED to every family, so
/// Latin text keeps the default fonts and Arabic falls back
/// to Noto.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let arabic = egui::FontData::from_static(include_bytes!(
        "../assets/NotoNaskhArabic-Regular.ttf"
    ));
    fonts
        .font_data
        .insert("noto_naskh_arabic".to_owned(), arabic.into());
    for family in [
        egui::FontFamily::Proportional,
        egui::FontFamily::Monospace,
    ] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("noto_naskh_arabic".to_owned());
    }
    ctx.set_fonts(fonts);
}
