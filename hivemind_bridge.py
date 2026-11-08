"""HiveMind engine bridge (wave #49).

The native Rust UI (hivemind-ui/) talks to the Python engine
through this stdlib-only local HTTP bridge: GET /api/models
lists local models, POST /api/chat runs generation. Heavy
imports (torch, transformers, hivemind itself) happen
lazily, ONLY when a real model is requested, so the bridge
starts on any machine - and the UI keeps working in a
clearly-tagged mock mode when the engine is absent (wave
#49 gate: the exe must generate text even when it is fake;
mocks always say they are mocks).

The real-engine path mirrors the retired Streamlit UI
(legacy/streamlit/app.py): scan ~/.hivemind/models, build
the paged engine from FileBackedPager, generate greedy.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS_DIR = Path.home() / ".hivemind" / "models"
DEFAULT_PORT = 7860
CHAT_MAX_NEW_TOKENS = 64

_engines = {}
_engines_lock = threading.Lock()


def _core_dirs():
    """Candidate core locations (wave #47 layout): core beside
    the bridge (repo layout) or beside the app dir (install
    layout)."""
    here = Path(__file__).resolve().parent
    return [here / "core", here.parent / "core"]


def _ensure_core_on_path():
    for core in _core_dirs():
        if (core / "torch").is_dir():
            if str(core) not in sys.path:
                sys.path.insert(0, str(core))
            return True
    return False


def scan_models():
    """Every subfolder of ~/.hivemind/models that looks like a
    model - the same rule the retired Streamlit app used."""
    models = []
    if MODELS_DIR.exists():
        for entry in sorted(MODELS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            has_config = (entry / "config.json").exists()
            has_weights = any(entry.glob("*.safetensors"))
            if has_config or has_weights:
                models.append(entry.name)
    return models


def _load_engine(model_name):
    """Build the paged engine for one local model. Single-file
    Safetensors only - same limits as the retired chat."""
    model_dir = MODELS_DIR / model_name
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("no safetensors weights in the model dir")
    if len(weights) > 1:
        raise RuntimeError("multi-shard weights are not wired in yet")
    from transformers import AutoTokenizer
    from hivemind.core.file_pager import FileBackedPager
    from hivemind.core.transformer_engine import (
        PagedTransformerEngine,
        TransformerConfig,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    config = TransformerConfig.from_hf_config(
        str(model_dir / "config.json"))
    pager = FileBackedPager(str(weights[0]), device="cpu")
    return PagedTransformerEngine(pager, tokenizer, config,
                                  device="cpu")


def _get_engine(model_name):
    with _engines_lock:
        if model_name not in _engines:
            _engines[model_name] = _load_engine(model_name)
        return _engines[model_name]


def mock_reply(prompt, model):
    return ("[mock:%s] no engine behind this bridge yet - "
            "echoing: %s" % (model, prompt))


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok", "mode": "bridge"})
        elif self.path == "/api/models":
            models = scan_models() or ["mock-echo"]
            self._send_json(200, {"models": models})
        else:
            self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json(404, {"error": "unknown endpoint"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            self._send_json(400, {"error": "invalid json"})
            return
        prompt = str(request.get("prompt", "")).strip()
        model = str(request.get("model", "mock-echo"))
        max_new = int(request.get("max_new_tokens", CHAT_MAX_NEW_TOKENS))
        if not prompt:
            self._send_json(400, {"error": "empty prompt"})
            return
        if model == "mock-echo" or model.startswith("offline-"):
            self._send_json(200, {"text": mock_reply(prompt, model),
                                  "mode": "mock"})
            return
        try:
            _ensure_core_on_path()
            engine = _get_engine(model)
            text = engine.generate(prompt, max_new_tokens=max_new,
                                   temperature=0)
            self._send_json(200, {"text": text, "mode": "engine"})
        except Exception as exc:
            # Log the full details locally; the client gets a
            # generic message (exception text can leak paths).
            print("bridge: engine failure for %s: %s"
                  % (model, exc), file=sys.stderr, flush=True)
            self._send_json(
                200,
                {"text": mock_reply(prompt, model)
                 + " (engine unavailable - see the bridge console)",
                 "mode": "mock"})

    def log_message(self, *args):
        pass  # quiet by design: the UI shows the status


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind engine bridge (wave #49)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("bridge: listening on %s:%d" % (args.host, args.port),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
