"""Split audio into vocal and accompaniment stems (HDemucs via torchaudio).

Outputs both the voice and the backing music, so a workflow can transcribe /
re-voice the vocals and later mix the new voice back over the accompaniment.
With pick_best_seconds > 0 the VOCALS output is cut to the loudest vocal
window of that length (the best clip to clone a voice from); the
accompaniment output always keeps the full length.
"""

import torch
import torchaudio
from torchaudio.transforms import Fade

_BUNDLE = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _BUNDLE.get_model().eval()
    return _MODEL


def _separate_sources(model, mix, sample_rate, segment=10.0, overlap=1.0):
    """Chunked separation with linear crossfades (torchaudio tutorial scheme)."""
    device = mix.device
    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    overlap_frames = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)
    start, end = 0, chunk_len
    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:start + out.shape[-1]] += out
        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def _frame_energy(mono, frame):
    n = mono.shape[-1] // frame * frame
    return (mono[:n].reshape(-1, frame) ** 2).mean(dim=1)


def _best_window(vocals_mono, accomp_mono, sample_rate, seconds):
    """Start index of the loudest vocal window.

    (A/B tests showed picking by vocal loudness gives the cleanest clone;
    scoring by music-bleed ratio selected breathier vocals and made the
    generated voice hissier.)
    """
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
    DESCRIPTION = "Split into vocal and backing-music stems (HDemucs)."

    def extract(self, audio, pick_best_seconds):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sr = _BUNDLE.sample_rate  # 44100

        wav = audio["waveform"][0].float()  # [C, T]
        if int(audio["sample_rate"]) != sr:
            wav = torchaudio.functional.resample(wav, int(audio["sample_rate"]), sr)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        model = _get_model().to(device)
        try:
            wav = wav.to(device)
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            sources = _separate_sources(model, ((wav - mean) / std)[None], sr)[0]
            vocal_idx = model.sources.index("vocals")
            vocals = sources[vocal_idx] * std + mean
            accompaniment = (sources.sum(dim=0) - sources[vocal_idx]) * std + mean
        finally:
            model.to("cpu")
            if device == "cuda":
                torch.cuda.empty_cache()

        vocals, accompaniment = vocals.cpu(), accompaniment.cpu()

        # No real music? Bypass: keep the ORIGINAL audio untouched (separation
        # artifacts add noise to clean speech) and output true silence as the
        # accompaniment.
        voc_db = 10 * torch.log10((vocals ** 2).mean() + 1e-12)
        acc_db = 10 * torch.log10((accompaniment ** 2).mean() + 1e-12)
        if acc_db < -55.0 or acc_db - voc_db < -25.0:
            import logging
            logging.getLogger(__name__).info(
                f"[ExtractVocals] no significant music (accomp {acc_db:.1f} dB, vocals {voc_db:.1f} dB) — bypassing separation"
            )
            vocals = wav.cpu()
            accompaniment = torch.zeros_like(vocals)

        if pick_best_seconds > 0:
            start = _best_window(vocals.mean(0), accompaniment.mean(0), sr, pick_best_seconds)
            vocals = vocals[:, start:start + int(pick_best_seconds * sr)]

        return (
            {"waveform": vocals.unsqueeze(0), "sample_rate": sr},
            {"waveform": accompaniment.unsqueeze(0), "sample_rate": sr},
        )


NODE_CLASS_MAPPINGS = {
    "AudioExtractVocals": AudioExtractVocals,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioExtractVocals": "Extract Vocals (voice / music stems)",
}
