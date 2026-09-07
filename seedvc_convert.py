"""Zero-shot voice conversion (seed-vc) — re-voices the SOURCE audio in the
REFERENCE speaker's timbre, frame by frame, preserving all content including
singing (melody kept). No transcription, so unlike a transcribe->re-speak
pipeline it never drops sung/unrecognized sections.

seed-vc has heavy, conflicting deps (numpy<2, pinned huggingface_hub), so it
lives in its own venv and is invoked as a subprocess — its GPU models load and
free per call, isolated from ComfyUI's environment.
"""

import glob
import json
import os
import subprocess
import tempfile
import urllib.request
import uuid

import numpy as np
import torch

# Isolated seed-vc environment (see comfyui-nodes/requirements-seedvc.txt).
_SEEDVC_PYTHON = os.environ.get("SEEDVC_PYTHON", "/workspace/seedvc-venv/bin/python")
# Persistent warm seed-vc server (models loaded once) — avoids the ~15s
# per-call model load of the subprocess path. Falls back to the subprocess
# when the server isn't reachable.
_SEEDVC_SERVER = os.environ.get("SEEDVC_SERVER", "http://127.0.0.1:8199")
_FFMPEG = "/usr/bin/ffmpeg"
_SR = 44100
# a conversion runs ~0.13x realtime; give generous headroom for long sources
_TIMEOUT_S = int(os.environ.get("SEEDVC_TIMEOUT_S", "1800"))


def _convert_via_server(src_path, ref_path, out_dir, diffusion_steps, f0_condition, auto_f0_adjust):
    """Ask the warm seed-vc server to convert; returns the output path or None
    if the server is unreachable (caller then falls back to the subprocess)."""
    payload = json.dumps({
        "source": src_path, "target": ref_path, "output_dir": out_dir,
        "diffusion_steps": int(diffusion_steps),
        "f0_condition": bool(f0_condition), "auto_f0_adjust": bool(auto_f0_adjust),
    }).encode()
    try:
        req = urllib.request.Request(
            f"{_SEEDVC_SERVER}/convert", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as response:
            return json.load(response)["output"]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None


def _write_wav(waveform, sample_rate, path):
    channels = waveform.shape[0]
    raw = waveform.t().contiguous().numpy().astype(np.float32).tobytes()
    subprocess.run(
        [_FFMPEG, "-v", "quiet", "-y", "-f", "f32le", "-ar", str(sample_rate),
         "-ac", str(channels), "-i", "pipe:0", path],
        input=raw, check=True,
    )


def _read_wav(path):
    raw = subprocess.run(
        [_FFMPEG, "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(_SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(bytearray(raw), dtype=np.float32).copy()).unsqueeze(0)


class SeedVCConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("AUDIO", {"tooltip": "The audio to re-voice (content / vocals)."}),
                "reference": ("AUDIO", {"tooltip": "The target voice to convert into."}),
                # 15 steps runs ~2x faster than 25 with no measurable identity
                # loss (speaker-embedding cosine held at ~0.86).
                "diffusion_steps": ("INT", {"default": 15, "min": 1, "max": 100}),
                # The base model (f0_condition False) matches the target TIMBRE
                # markedly better (speaker-embedding cosine ~0.85 vs ~0.79 for
                # the f0 singing model) while still covering the full duration,
                # so it is the default. Enable f0_condition only when strict
                # pitch/melody preservation matters more than voice identity.
                "f0_condition": ("BOOLEAN", {"default": False}),
                "auto_f0_adjust": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = "audio"
    DESCRIPTION = "Zero-shot voice conversion (seed-vc): re-voice source in the reference timbre, melody preserved."

    def convert(self, source, reference, diffusion_steps, f0_condition, auto_f0_adjust):
        tmp = tempfile.gettempdir()
        tag = uuid.uuid4().hex
        src_path = os.path.join(tmp, f"svc-src-{tag}.wav")
        ref_path = os.path.join(tmp, f"svc-ref-{tag}.wav")
        out_dir = os.path.join(tmp, f"svc-out-{tag}")
        os.makedirs(out_dir, exist_ok=True)

        try:
            _write_wav(source["waveform"][0].float(), int(source["sample_rate"]), src_path)
            _write_wav(reference["waveform"][0].float(), int(reference["sample_rate"]), ref_path)

            # Warm server first (fast); subprocess fallback if it is down.
            output_path = _convert_via_server(src_path, ref_path, out_dir, diffusion_steps, f0_condition, auto_f0_adjust)
            if not output_path:
                subprocess.run(
                    [_SEEDVC_PYTHON, "-m", "seed_vc.inference",
                     "--source", src_path, "--target", ref_path, "--output", out_dir,
                     "--diffusion-steps", str(int(diffusion_steps)),
                     "--f0-condition", "True" if f0_condition else "False",
                     "--auto-f0-adjust", "True" if auto_f0_adjust else "False",
                     "--fp16", "True"],
                    check=True, capture_output=True, timeout=_TIMEOUT_S,
                )
                produced = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
                if not produced:
                    raise RuntimeError("seed-vc produced no output")
                output_path = produced[0]
            converted = _read_wav(output_path)
        except subprocess.CalledProcessError as error:
            tail = (error.stderr or b"").decode("utf-8", "ignore")[-500:]
            raise RuntimeError(f"seed-vc failed: {tail}")
        finally:
            for path in [src_path, ref_path]:
                try:
                    os.remove(path)
                except OSError:
                    pass
            for path in glob.glob(os.path.join(out_dir, "*")):
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(out_dir)
            except OSError:
                pass

        return ({"waveform": converted.unsqueeze(0), "sample_rate": _SR},)


NODE_CLASS_MAPPINGS = {
    "SeedVCConvert": SeedVCConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVCConvert": "Voice Conversion (seed-vc)",
}
