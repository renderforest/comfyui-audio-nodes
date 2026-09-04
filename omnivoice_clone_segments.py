"""Segment-wise voice cloning: re-voice a long recording, word-timing aligned.

Takes the word timestamps from OmniVoiceWhisperTranscribe, groups words into
phrases, clones each phrase with OmniVoice, time-fits every phrase into its
original slot (pitch-preserving), and returns one full-length voice track.
Made for whole songs / long voiceovers, where one-shot TTS cannot follow the
original phrasing.
"""

import json
import logging
import subprocess

import torch

FFMPEG = "/usr/bin/ffmpeg"
logger = logging.getLogger(__name__)


def _atempo_chain(factor):
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
    """Time-stretch mono [1, T] by factor (>1 = shorter), pitch preserved."""
    raw = wav[0].contiguous().numpy().astype("float32").tobytes()
    out = subprocess.run(
        [FFMPEG, "-v", "quiet", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-af", _atempo_chain(factor), "-f", "f32le", "-"],
        input=raw, capture_output=True, check=True,
    ).stdout
    return torch.frombuffer(bytearray(out), dtype=torch.float32).unsqueeze(0)


def _group_words(words, max_gap, max_words):
    segs, cur = [], None
    for w in words:
        word = str(w.get("word", "")).strip()
        start, end = w.get("start"), w.get("end")
        if not word:
            continue
        if start is None or end is None:
            if cur:
                cur["words"].append(word)
            continue
        if cur and (start - cur["end"] > max_gap or len(cur["words"]) >= max_words):
            segs.append(cur)
            cur = None
        if cur is None:
            cur = {"start": float(start), "end": float(end), "words": [word]}
        else:
            cur["words"].append(word)
            cur["end"] = max(cur["end"], float(end))
    if cur:
        segs.append(cur)
    return [
        {"text": " ".join(s["words"]), "start": s["start"], "end": s["end"]}
        for s in segs if " ".join(s["words"]).strip()
    ]


class OmniVoiceCloneSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "words_json": ("STRING", {"forceInput": True, "tooltip": "Word timestamps from OmniVoice Whisper Transcribe."}),
                "ref_audio": ("AUDIO", {"tooltip": "Voice sample to clone."}),
                "steps": ("INT", {"default": 32, "min": 1, "max": 128}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "max_gap": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "A pause longer than this (seconds) starts a new phrase.",
                }),
                # max_words and seed are STRING-typed on purpose: graphs imported
                # with shifted widget values send things like "" or "randomize"
                # here, and the validator hard-fails INT conversion before any
                # custom validation can run. Strings always validate; generate()
                # parses them and falls back to sane values.
                "max_words": ("STRING", {"default": "24", "tooltip": "Longest phrase, in words (4-100)."}),
                "seed": ("STRING", {"default": "7", "tooltip": "Number for reproducible output, or 'randomize' for a new seed each run."}),
            },
            "optional": {
                "ref_text": ("STRING", {"default": "", "tooltip": "Transcript of the voice sample; leave empty to auto-transcribe once."}),
                "whisper_model": ("WHISPER_ASR",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("voice_track",)
    FUNCTION = "generate"
    CATEGORY = "OmniVoice"
    DESCRIPTION = "Clone a voice phrase by phrase, keeping every phrase at its original word timing."

    @classmethod
    def VALIDATE_INPUTS(cls, steps=None, guidance_scale=None, max_gap=None, max_words=None, seed=None):
        # Skip built-in min/max rejection for these — generate() clamps them
        # instead, so a graph imported with shifted widget values still runs.
        return True

    def generate(self, words_json, ref_audio, steps, guidance_scale, seed, max_gap, max_words, ref_text="", whisper_model=None):
        import random

        steps = min(max(int(steps), 8), 128)
        guidance_scale = min(max(float(guidance_scale), 0.1), 10.0)
        max_gap = min(max(float(max_gap), 0.1), 5.0)
        try:
            max_words = min(max(int(float(max_words)), 4), 100)
        except (TypeError, ValueError):
            max_words = 24
        try:
            seed = int(float(seed))
        except (TypeError, ValueError):
            seed = random.randint(1, 2**31)  # e.g. "randomize" or empty
        import nodes as comfy_nodes
        import comfy.model_management as mm

        clone_cls = comfy_nodes.NODE_CLASS_MAPPINGS["OmniVoiceVoiceCloneTTS"]
        clone_fn = getattr(clone_cls(), clone_cls.FUNCTION)

        segs = _group_words(json.loads(words_json), max_gap, max_words)
        if not segs:
            raise ValueError("words_json contains no timed words to speak.")
        logger.info(f"[CloneSegments] {len(segs)} phrases to generate")

        sr, pieces = None, []
        for i, seg in enumerate(segs):
            mm.throw_exception_if_processing_interrupted()
            out = clone_fn(
                model="OmniVoice (auto download)", text=seg["text"],
                ref_audio=ref_audio, ref_text=ref_text,
                steps=steps, guidance_scale=guidance_scale, t_shift=0.1,
                speed=1.0, duration=0.0, device="auto", dtype="auto",
                attention="auto", seed=seed + i, position_temperature=5.0,
                class_temperature=0.0, layer_penalty_factor=5.0, denoise=True,
                preprocess_prompt=True, postprocess_output=True,
                keep_model_loaded=True, instruct="", whisper_model=whisper_model,
            )[0]
            # after the first phrase we know the reference transcript is cached
            wav = out["waveform"][0].float().mean(dim=0, keepdim=True)
            sr = int(out["sample_rate"])

            span = max(0.2, seg["end"] - seg["start"])
            gen_dur = wav.shape[-1] / sr
            factor = min(max(gen_dur / span, 0.5), 2.5)
            if abs(gen_dur / span - 1.0) > 0.03:
                wav = _stretch(wav, sr, factor)
            target = int(span * sr)
            if wav.shape[-1] > target:
                wav = wav[:, :target].clone()
                fade = torch.linspace(1, 0, min(int(0.03 * sr), wav.shape[-1]))
                wav[:, -fade.shape[0]:] *= fade
            pieces.append((seg["start"], wav))
            logger.info(f"[CloneSegments] {i + 1}/{len(segs)} span={span:.2f}s gen={gen_dur:.2f}s '{seg['text'][:50]}'")

        total = int((max(s["end"] for s in segs) + 0.5) * sr)
        track = torch.zeros(1, total)
        for start, wav in pieces:
            offset = int(start * sr)
            end = min(offset + wav.shape[-1], total)
            track[:, offset:end] += wav[:, : end - offset]

        return ({"waveform": track.unsqueeze(0), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "OmniVoiceCloneSegments": OmniVoiceCloneSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceCloneSegments": "OmniVoice Clone Segments (timing-aligned)",
}
