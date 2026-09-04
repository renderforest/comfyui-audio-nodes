"""Split audio into vocal and accompaniment stems (Mel-Band Roformer).

Outputs both the voice and the backing music, so a workflow can transcribe /
re-voice the vocals and later mix the new voice back over the accompaniment.
With pick_best_seconds > 0 the VOCALS output is cut to the loudest vocal
window of that length (the best clip to clone a voice from); the
accompaniment output always keeps the full length.

Mel-Band Roformer (via the `audio-separator` UVR engine) replaced HDemucs
because HDemucs leaves a substantial ghost of the original voice in the
instrumental (~9 dB below the mix even on clean speech), which the mix-back
step then played under the new voice as noise. Roformer's instrumental is
near-silent on clean speech (~-80 dB) and clean on music, so the residue
disappears and no separate music-detection step is needed.
"""

import os
import subprocess
import tempfile
import uuid

import numpy as np
import torch

# UVR Mel-Band Roformer vocals model — auto-downloaded on first load into the
# audio-separator model dir.
_MODEL_FILENAME = os.environ.get(
    "ROFORMER_MODEL", "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
)
_SEPARATOR = None
_FFMPEG = "/usr/bin/ffmpeg"
_SR = 44100


def _get_separator():
    global _SEPARATOR
    if _SEPARATOR is None:
        from audio_separator.separator import Separator

        separator = Separator(output_dir=tempfile.gettempdir(), output_format="WAV")
        separator.load_model(model_filename=_MODEL_FILENAME)
        _SEPARATOR = separator
    return _SEPARATOR


def _write_wav(waveform, sample_rate, path):
    """Write [C, T] float tensor to a wav via ffmpeg (no torch audio backend needed)."""
    channels = waveform.shape[0]
    raw = waveform.t().contiguous().numpy().astype(np.float32).tobytes()
    subprocess.run(
        [_FFMPEG, "-v", "quiet", "-y", "-f", "f32le", "-ar", str(sample_rate),
         "-ac", str(channels), "-i", "pipe:0", path],
        input=raw, check=True,
    )


def _read_wav(path):
    """Read a wav to a [1, T] mono float tensor at _SR."""
    raw = subprocess.run(
        [_FFMPEG, "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(_SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(bytearray(raw), dtype=np.float32).copy()).unsqueeze(0)


def _frame_energy(mono, frame):
    n = mono.shape[-1] // frame * frame
    return (mono[:n].reshape(-1, frame) ** 2).mean(dim=1)


def _best_window(vocals_mono, sample_rate, seconds):
    """Start index of the loudest vocal window (the cleanest clip to clone from)."""
    win = int(seconds * sample_rate)
    if win >= vocals_mono.shape[-1]:
        return 0
    frame = int(0.05 * sample_rate)
    k = max(1, win // frame)
    ve = torch.nn.functional.avg_pool1d(_frame_energy(vocals_mono, frame)[None, None], k, stride=1).squeeze()
    return int(ve.argmax()) * frame


class AudioExtractVocals:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio that may contain background music."}),
                "pick_best_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.5,
                    "tooltip": "0 keeps the full length; >0 cuts the VOCALS output to the loudest vocal window of this length (use ~15 for a cloning reference).",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "accompaniment")
    FUNCTION = "extract"
    CATEGORY = "audio"
    DESCRIPTION = "Split into vocal and backing-music stems (Mel-Band Roformer)."

    def extract(self, audio, pick_best_seconds):
        wav = audio["waveform"][0].float()  # [C, T]
        sample_rate = int(audio["sample_rate"])

        tmp_dir = tempfile.gettempdir()
        tag = uuid.uuid4().hex
        in_path = os.path.join(tmp_dir, f"sep-in-{tag}.wav")
        produced = []

        try:
            _write_wav(wav, sample_rate, in_path)

            separator = _get_separator()
            # returns the two stem filenames (relative to the separator's out dir)
            outputs = separator.separate(in_path)
            produced = [os.path.join(tmp_dir, name) for name in outputs]

            def _find(kind):
                match = next((p for p in produced if kind in os.path.basename(p).lower()), None)
                if not match:
                    raise RuntimeError(f"Roformer produced no {kind} stem: {outputs}")
                return match

            vocals = _read_wav(_find("vocal"))
            accompaniment = _read_wav(_find("instrument"))
        finally:
            for path in [in_path, *produced]:
                try:
                    os.remove(path)
                except OSError:
                    pass

        if pick_best_seconds > 0:
            start = _best_window(vocals[0], _SR, pick_best_seconds)
            vocals = vocals[:, start:start + int(pick_best_seconds * _SR)]

        return (
            {"waveform": vocals.unsqueeze(0), "sample_rate": _SR},
            {"waveform": accompaniment.unsqueeze(0), "sample_rate": _SR},
        )


NODE_CLASS_MAPPINGS = {
    "AudioExtractVocals": AudioExtractVocals,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioExtractVocals": "Extract Vocals (voice / music stems)",
}
