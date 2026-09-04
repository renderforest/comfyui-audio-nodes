"""Overlay a (cloned) voice onto a music/accompaniment track with timing control.

- stretch_to_seconds > 0: time-stretch the voice (ffmpeg atempo, pitch kept)
  so it spans exactly that many seconds — match it to the original speech span.
- offset_seconds: where the voice starts in the mix — match it to the original
  speech start so timing lines up with the source recording.
The output keeps the music track's length (extended if the voice overruns).
"""

import subprocess

import torch
import torchaudio

FFMPEG = "/usr/bin/ffmpeg"


def _atempo_chain(factor):
    """Split a tempo factor into ffmpeg-safe [0.5, 2.0] stages."""
    stages = []
    while factor > 2.0:
        stages.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        stages.append(0.5)
        factor /= 0.5
    stages.append(factor)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def _stretch(wav, sr, factor):
    """Time-stretch [C, T] by `factor` (>1 = faster/shorter), pitch preserved."""
    channels = wav.shape[0]
    raw = wav.T.contiguous().numpy().astype("float32").tobytes()
    out = subprocess.run(
        [FFMPEG, "-v", "quiet",
         "-f", "f32le", "-ar", str(sr), "-ac", str(channels), "-i", "-",
         "-af", _atempo_chain(factor),
         "-f", "f32le", "-"],
        input=raw, capture_output=True, check=True,
    ).stdout
    res = torch.frombuffer(bytearray(out), dtype=torch.float32).reshape(-1, channels)
    return res.T.contiguous()


class AudioOverlayVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "music": ("AUDIO", {"tooltip": "Backing track / accompaniment."}),
                "voice": ("AUDIO", {"tooltip": "Voice to lay over the music."}),
                "offset_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Where the voice starts in the mix (link the transcriber's speech_start to keep original timing).",
                }),
                "stretch_to_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "0 keeps the voice's natural pace; >0 time-stretches it to span exactly this long (link the transcriber's speech_span).",
                }),
                "voice_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 24.0, "step": 0.5,
                    "tooltip": "Gain applied to the voice before mixing.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mix",)
    FUNCTION = "overlay"
    CATEGORY = "audio"
    DESCRIPTION = "Mix a voice over music at a given start time, optionally stretched to a target span."

    def overlay(self, music, voice, offset_seconds, stretch_to_seconds, voice_gain_db):
        sr = int(music["sample_rate"])
        mus = music["waveform"][0].float()  # [C, T]
        vox = voice["waveform"][0].float()

        # No-music gate: when the accompaniment is effectively silent (content
        # had no real music — see AudioExtractVocals bypass), return the voice
        # untouched. This makes one workflow serve both cases: plain cloned
        # speech for music-less audio, timed mix-back when music exists.
        music_db = 10 * torch.log10((mus ** 2).mean() + 1e-12)
        if music_db < -60.0:
            import logging
            logging.getLogger(__name__).info(
                f"[OverlayVoice] music input is silent ({music_db:.1f} dB) — passing voice through untouched"
            )
            return (voice,)

        if int(voice["sample_rate"]) != sr:
            vox = torchaudio.functional.resample(vox, int(voice["sample_rate"]), sr)

        # match channel counts
        if vox.shape[0] < mus.shape[0]:
            vox = vox.repeat(mus.shape[0] // vox.shape[0], 1)[: mus.shape[0]]
        elif vox.shape[0] > mus.shape[0]:
            vox = vox.mean(dim=0, keepdim=True).repeat(mus.shape[0], 1)

        if stretch_to_seconds > 0:
            factor = (vox.shape[-1] / sr) / stretch_to_seconds
            # bound the correction: stretching beyond ~±30% makes speech
            # sound smeared/metallic — prefer slight misalignment over noise
            factor = min(max(factor, 0.75), 1.5)
            if abs(factor - 1.0) > 0.01:
                vox = _stretch(vox, sr, factor)

        if voice_gain_db:
            vox = vox * (10 ** (voice_gain_db / 20))

        offset = int(offset_seconds * sr)
        total = max(mus.shape[-1], offset + vox.shape[-1])
        mix = torch.zeros(mus.shape[0], total)
        mix[:, : mus.shape[-1]] = mus
        mix[:, offset: offset + vox.shape[-1]] += vox
        peak = mix.abs().max()
        if peak > 1.0:
            mix = mix / peak

        return ({"waveform": mix.unsqueeze(0), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "AudioOverlayVoice": AudioOverlayVoice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioOverlayVoice": "Overlay Voice On Music (timed)",
}
