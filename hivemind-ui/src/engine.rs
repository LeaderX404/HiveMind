//! The engine bridge (wave #49, port protocol wave #50):
//! discover or spawn the python bridge process and speak
//! JSON over localhost HTTP. One blocking worker thread per
//! request - no async runtime keeps the exe under the 20 MB
//! gate. Every mock reply says it is mock. No fixed port is
//! trusted: the bridge publishes pid+port to port.txt and
//! the UI follows that file.

use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 7860;
const PORT_FILE: &str = "port.txt";
const DISCOVER_TRIES: u32 = 80;
const DISCOVER_SLEEP_MS: u64 = 250;

/// Who said a chat line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speaker {
    User,
    Assistant,
}

/// One rendered chat message.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub speaker: Speaker,
    pub text: String,
    pub mock: bool,
}

/// Events background workers send to the UI thread.
pub enum EngineEvent {
    Status(String),
    Port(u16),
    Ready(bool),
    Models(Vec<String>),
    BridgeUp(BridgeHandle),
    Reply { text: String, mock: bool, error: bool },
}

/// Owns a spawned bridge process; dropping it kills the
/// child so no python ghost survives the app.
pub struct BridgeHandle {
    child: Option<Child>,
}

impl BridgeHandle {
    pub fn spawn() -> Option<BridgeHandle> {
        let script = bridge_script()?;
        let pythons: &[&str] = if cfg!(windows) {
            &["python", "py"]
        } else {
            &["python3", "python"]
        };
        for program in pythons {
            let attempt = Command::new(program)
                .arg(&script)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();
            if let Ok(child) = attempt {
                return Some(BridgeHandle {
                    child: Some(child),
                });
            }
        }
        None
    }

    pub fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for BridgeHandle {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Where hivemind_bridge.py lives: beside the exe (the
/// installed layout), else the working directory (repo).
fn bridge_script() -> Option<std::path::PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("hivemind_bridge.py");
            if beside.is_file() {
                return Some(beside);
            }
        }
    }
    let local = std::path::PathBuf::from("hivemind_bridge.py");
    if local.is_file() {
        return Some(local);
    }
    None
}

/// Where port.txt lives: the same search order as
/// bridge_script() - beside the exe, else the working
/// directory. The bridge writes the file beside itself,
/// which is that same folder in both layouts.
fn port_file_path() -> Option<std::path::PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(PORT_FILE);
            if beside.is_file() {
                return Some(beside);
            }
        }
    }
    let local = std::path::PathBuf::from(PORT_FILE);
    if local.is_file() {
        return Some(local);
    }
    None
}

/// Parse the pid=N / port=M pairs the bridge publishes.
/// Anything malformed means: treat the file as absent.
fn parse_port_file(text: &str) -> Option<(u32, u16)> {
    let mut pid: Option<u32> = None;
    let mut port: Option<u16> = None;
    for line in text.lines() {
        let mut parts = line.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        if let Some(value) = parts.next().map(str::trim) {
            if key == "pid" {
                pid = value.parse::<u32>().ok();
            } else if key == "port" {
                port = value.parse::<u16>().ok();
            }
        }
    }
    Some((pid?, port?))
}

/// The published pid+port pair, if a parseable port file
/// exists right now.
fn read_port_file() -> Option<(u32, u16)> {
    let path = port_file_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    parse_port_file(&text)
}

fn agent(timeout_secs: u64) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
}

fn url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{}{}", port, path)
}

/// Fetch the model list from the bridge; None when the
/// bridge is not answering.
pub fn fetch_models(port: u16) -> Option<Vec<String>> {
    let response = agent(3).get(&url(port, "/api/models")).call().ok()?;
    let value: serde_json::Value = response.into_json().ok()?;
    let array = value.get("models")?.as_array()?;
    let names: Vec<String> = array
        .iter()
        .filter_map(|item| item.as_str().map(String::from))
        .collect();
    if names.is_empty() {
        None
    } else {
        Some(names)
    }
}

/// One chat round-trip. Returns (text, mock, error).
pub fn chat(port: u16, prompt: &str, model: &str) -> (String, bool, bool) {
    let body = serde_json::json!({
        "prompt": prompt,
        "model": model,
    });
    let request = agent(600).post(&url(port, "/api/chat"));
    match request.send_json(body) {
        Ok(response) => match response.into_json::<serde_json::Value>() {
            Ok(value) => {
                if let Some(err) = value.get("error").and_then(|e| e.as_str()) {
                    return (format!("engine error: {}", err), false, true);
                }
                let text = value
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let mock = value.get("mode").and_then(|m| m.as_str()) == Some("mock");
                let error = text.is_empty();
                (text, mock, error)
            }
            Err(err) => (format!("bad response: {}", err), false, true),
        },
        Err(err) => (format!("bridge unreachable: {}", err), false, true),
    }
}

/// The built-in offline mock (no python at all): clearly
/// tagged, never pretending to be a real model.
pub fn mock_reply(prompt: &str, model: &str) -> String {
    format!(
        "[mock:{}] offline mode - no bridge behind this window. echoing: {}",
        model, prompt
    )
}

/// Background discovery: reuse a bridge that is already
/// listening on the default port, else spawn one and follow
/// the port.txt protocol, else fall back to offline mock.
pub fn discover(tx: Sender<EngineEvent>) {
    std::thread::spawn(move || {
        // Compat: a bridge may already be up - started by
        // hand, or an older build without a port file.
        if let Some(models) = fetch_models(DEFAULT_PORT) {
            let _ = tx.send(EngineEvent::Status(
                "engine: bridge detected".to_string(),
            ));
            let _ = tx.send(EngineEvent::Port(DEFAULT_PORT));
            let _ = tx.send(EngineEvent::Ready(true));
            let _ = tx.send(EngineEvent::Models(models));
            return;
        }
        // A stale port.txt from a dead run is identified by
        // its pid: only a pid we have NOT seen before counts
        // as the bridge we just spawned.
        let baseline = read_port_file();
        let mut handle = match BridgeHandle::spawn() {
            Some(handle) => handle,
            None => {
                let _ = tx.send(EngineEvent::Status(
                    "offline: no python - mock mode".to_string(),
                ));
                let _ = tx.send(EngineEvent::Ready(false));
                let _ = tx.send(EngineEvent::Models(vec![
                    "offline-mock".to_string(),
                ]));
                return;
            }
        };
        for _ in 0..DISCOVER_TRIES {
            std::thread::sleep(Duration::from_millis(DISCOVER_SLEEP_MS));
            if let Some((pid, port)) = read_port_file() {
                let fresh = match baseline {
                    Some((old_pid, _)) => pid != old_pid,
                    None => true,
                };
                if !fresh {
                    continue;
                }
                if let Some(models) = fetch_models(port) {
                    let _ = tx.send(EngineEvent::Status(
                        "engine: bridge up (python)".to_string(),
                    ));
                    let _ = tx.send(EngineEvent::Port(port));
                    let _ = tx.send(EngineEvent::Ready(true));
                    let _ = tx.send(EngineEvent::Models(models));
                    let _ = tx.send(EngineEvent::BridgeUp(handle));
                    return;
                }
            }
        }
        handle.kill();
        let _ = tx.send(EngineEvent::Status(
            "offline: bridge did not answer - mock mode".to_string(),
        ));
        let _ = tx.send(EngineEvent::Ready(false));
        let _ = tx.send(EngineEvent::Models(vec![
            "offline-mock".to_string(),
        ]));
    });
}

/// Worker for one chat request; the reply lands in the UI
/// through the channel.
pub fn spawn_chat(
    tx: Sender<EngineEvent>,
    port: u16,
    engine_ready: bool,
    prompt: String,
    model: String,
) {
    std::thread::spawn(move || {
        let (text, mock, error) = if engine_ready {
            chat(port, &prompt, &model)
        } else {
            (mock_reply(&prompt, &model), true, false)
        };
        let _ = tx.send(EngineEvent::Reply { text, mock, error });
    });
}

/// Headless selftest (CI): the mock must work, and when a
/// python is present the full port-file + HTTP path must
/// answer.
pub fn selftest() -> bool {
    let mock = mock_reply("hello", "selftest");
    if mock.is_empty() {
        eprintln!("selftest: mock generator produced nothing");
        return false;
    }
    println!("selftest: mock generation ok");
    let baseline = read_port_file();
    let mut handle = match BridgeHandle::spawn() {
        Some(handle) => handle,
        None => {
            println!("selftest: no python - offline mock path only");
            return true;
        }
    };
    for _ in 0..DISCOVER_TRIES {
        if let Some((pid, port)) = read_port_file() {
            let fresh = match baseline {
                Some((old_pid, _)) => pid != old_pid,
                None => true,
            };
            if !fresh {
                continue;
            }
            if let Some(models) = fetch_models(port) {
                let model = models
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "mock-echo".to_string());
                let (text, _mock, error) = chat(port, "self test", &model);
                handle.kill();
                if !error && !text.is_empty() {
                    println!(
                        "selftest: bridge chat ok on port {}: {}",
                        port, text
                    );
                    return true;
                }
                eprintln!("selftest: bridge chat failed: {}", text);
                return false;
            }
        }
        std::thread::sleep(Duration::from_millis(DISCOVER_SLEEP_MS));
    }
    eprintln!("selftest: bridge did not come up");
    handle.kill();
    false
}
