"""Standalone speech-to-text node reusing the OmniVoice pack's Whisper model.

Connect OmniVoiceWhisperLoader's output and any AUDIO. Returns:
- text: the plain transcription
- words_json: per-word timestamps from the source audio, as JSON
  [{"word": "...", "start": 1.23, "end": 1.57}, ...]
- speech_start: when the first word begins (seconds)
- speech_span: first word start -> last word end (seconds)
No extra models or packages needed.
"""

import json

import numpy as np


class OmniVoiceWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to transcribe."}),
                "whisper_model": ("WHISPER_ASR", {"tooltip": "From OmniVoice Whisper Loader."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("text", "words_json", "speech_start", "speech_span")
    FUNCTION = "transcribe"
    CATEGORY = "OmniVoice"
    DESCRIPTION = "Transcribe audio to text + word timestamps with the OmniVoice Whisper ASR model."

    def transcribe(self, audio, whisper_model):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        wav = waveform[0] if waveform.dim() == 3 else waveform
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        audio_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)

        pipe = whisper_model["pipeline"]
        # The OmniVoice pack offloads the cached Whisper model to CPU after
        # each TTS run; the pipeline still preprocesses inputs on its original
        # device, so the weights must be moved back before inference.
        try:
            if next(pipe.model.parameters()).device != pipe.device:
                pipe.model.to(pipe.device)
        except StopIteration:
            pass

        # Transcribe in bounded windows: Whisper's single-pass word-timestamp
        # alignment holds cross-attention for the whole clip and OOMs on long
        # audio (a 4-minute song exceeded a 16 GB container). 55s windows with
        # a 2s look-ahead keep memory flat; each window owns the words that
        # START inside its own 55s span, so the overlap never double-counts.
        WINDOW_S, LOOKAHEAD_S = 55.0, 2.0
        duration = audio_np.shape[-1] / sample_rate
        words = []
        t = 0.0
        while t < duration:
            seg = audio_np[int(t * sample_rate): int((t + WINDOW_S + LOOKAHEAD_S) * sample_rate)]
            if seg.shape[-1] < int(0.2 * sample_rate):
                break
            result = pipe({"array": seg, "sampling_rate": sample_rate}, return_timestamps="word")
            seg_dur = seg.shape[-1] / sample_rate
            for chunk in result.get("chunks", []):
                start, end = chunk.get("timestamp") or (None, None)
                if start is not None and float(start) >= WINDOW_S and t + WINDOW_S < duration:
                    continue  # owned by the next window
                words.append({
                    "word": str(chunk.get("text", "")).strip(),
                    "start": round(t + float(start), 3) if start is not None else None,
                    # whisper leaves the last end open sometimes; fall back to window end
                    "end": round(t + float(end if end is not None else seg_dur), 3),
                })
            t += WINDOW_S
        text = " ".join(w["word"] for w in words if w["word"]).strip()

        # Whisper stretches words across non-speech (intro/outro music, leading
        # silence): "When" spanning 2.6-13.3s. No real word exceeds ~1.5s, so
        # clamp: interior words keep their END (anchored to the next word),
        # the final word keeps its START (anchored to the previous word).
        MAX_WORD_S = 1.5
        for i, w in enumerate(words):
            if w["start"] is None or w["end"] is None:
                continue
            if w["end"] - w["start"] > MAX_WORD_S:
                if i == len(words) - 1:
                    w["end"] = round(w["start"] + MAX_WORD_S, 3)
                else:
                    w["start"] = round(w["end"] - MAX_WORD_S, 3)

        # Correct the first word's start with the waveform's own energy onset
        # (never past its end).
        onset = self._energy_onset(audio_np, sample_rate)
        if words and words[0]["start"] is not None and abs(words[0]["start"] - onset) > 0.3:
            words[0]["start"] = round(min(max(words[0]["start"], onset), words[0]["end"]), 3)

        starts = [w["start"] for w in words if w["start"] is not None]
        ends = [w["end"] for w in words if w["end"] is not None]
        speech_start = min(starts) if starts else 0.0
        speech_span = (max(ends) - speech_start) if ends else duration

        return (text, json.dumps(words, ensure_ascii=False), float(speech_start), float(speech_span))

    @staticmethod
    def _energy_onset(audio_np, sample_rate, min_hold_s=0.1):
        """First time (s) the signal stays near its own loud level for min_hold_s.

        The threshold adapts to the clip (20 dB under its 95th-percentile frame
        level), so quiet separation bleed doesn't trigger a false onset.
        """
        frame = max(1, int(0.02 * sample_rate))
        n = audio_np.shape[-1] // frame * frame
        if n == 0:
            return 0.0
        rms = np.sqrt((audio_np[:n].reshape(-1, frame) ** 2).mean(axis=1)) + 1e-10
        db = 20 * np.log10(rms)
        threshold_db = max(float(np.percentile(db, 95)) - 20.0, -45.0)
        loud = db > threshold_db
        hold = max(1, int(min_hold_s / 0.02))
        run = 0
        for i, is_loud in enumerate(loud):
            run = run + 1 if is_loud else 0
            if run >= hold:
                return (i - hold + 1) * 0.02
        return 0.0


NODE_CLASS_MAPPINGS = {
    "OmniVoiceWhisperTranscribe": OmniVoiceWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceWhisperTranscribe": "OmniVoice Whisper Transcribe (audio → text + words)",
}
