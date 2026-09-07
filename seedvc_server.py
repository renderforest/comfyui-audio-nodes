"""Persistent seed-vc server: loads the models ONCE and converts on request,
so each job pays ~5s inference instead of ~15s model-load + inference.

Runs in the seed-vc venv. POST /convert with JSON
{source, target, output_dir, diffusion_steps, f0_condition, auto_f0_adjust}
-> {"output": "<wav path>"}. GET /health -> "ok".
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import seed_vc.inference as inf

_LOCK = threading.Lock()
_CACHE = {}  # f0_condition -> loaded models tuple

# Base args used only for model loading (source/target set per request).
def _base_args(f0_condition):
    return SimpleNamespace(
        f0_condition=f0_condition, auto_f0_adjust=True, semi_tone_shift=0,
        checkpoint=None, config=None, fp16=True,
        diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7,
        source="", target="", output="",
    )

# Wrap load_models so main() reuses the cached models instead of reloading.
_orig_load = inf.load_models
def _cached_load(args):
    key = bool(args.f0_condition)
    if key not in _CACHE:
        _CACHE[key] = _orig_load(args)
    return _CACHE[key]
inf.load_models = _cached_load

# Warm the default (f0_condition False) model at startup.
_cached_load(_base_args(False))
print("seed-vc models warm", flush=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/convert":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        args = _base_args(bool(req.get("f0_condition", False)))
        args.auto_f0_adjust = bool(req.get("auto_f0_adjust", True))
        args.diffusion_steps = int(req.get("diffusion_steps", 25))
        args.source = req["source"]
        args.target = req["target"]
        args.output = req["output_dir"]
        os.makedirs(args.output, exist_ok=True)
        before = set(os.listdir(args.output))
        try:
            with _LOCK:  # single GPU — serialize
                inf.main(args)
        except Exception as error:  # noqa: BLE001
            return self._send(500, {"error": str(error)[:400]})
        produced = [f for f in os.listdir(args.output) if f.endswith(".wav") and f not in before]
        if not produced:
            return self._send(500, {"error": "no output produced"})
        self._send(200, {"output": os.path.join(args.output, sorted(produced)[-1])})

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    port = int(os.environ.get("SEEDVC_PORT", "8199"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
