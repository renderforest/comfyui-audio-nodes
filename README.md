# comfyui-audio-nodes

ComfyUI custom nodes for AI audio, used by the Renderforest `text-to-speech-api`
change-voice / speech-to-speech pipeline. Drop this repo into ComfyUI's
`custom_nodes/` on a GPU worker.

## Nodes

| Node | Class | Purpose |
|------|-------|---------|
| Extract Vocals | `AudioExtractVocals` | Split audio into vocal + accompaniment stems (Mel-Band Roformer, via `audio-separator`). Near-silent instrumental on clean speech, so no ghost-voice noise on mix-back. |
| Voice Conversion (seed-vc) | `SeedVCConvert` | Zero-shot voice conversion — re-voice source audio in a reference speaker's timbre, frame by frame. Handles singing / full songs end-to-end (no transcription). Calls a warm server for speed. |
| `seedvc_server.py` | — | Persistent seed-vc server: loads models once and converts on request (~5x faster than per-call model load). Run as a long-lived service. |
| Overlay Voice | `AudioOverlayVoice` | Mix a converted voice over a backing-music bed (offset / stretch / gain). |
| Whisper Transcribe | `OmniVoiceWhisperTranscribe` | Transcribe vocals with word timings (legacy transcribe-and-respeak path). |
| Clone Segments | `OmniVoiceCloneSegments` | Re-speak transcribed segments in a cloned voice (legacy path). |

The current speech-to-speech workflow uses `AudioExtractVocals` →
`SeedVCConvert` → `AudioOverlayVoice`. The OmniVoice nodes are the earlier
transcribe-and-respeak approach, kept for reference.

## Install

Clone into `custom_nodes/` and restart ComfyUI:

```bash
cd /workspace/comfyui/ComfyUI/custom_nodes
git clone git@github.com:renderforest/comfyui-audio-nodes.git
```

### Dependencies

The ComfyUI venv needs the `AudioExtractVocals` deps:

```bash
pip install -r comfyui-audio-nodes/requirements.txt   # audio-separator[gpu], audioread
```

`SeedVCConvert` runs seed-vc in its **own** venv (its deps conflict with
ComfyUI's). See `requirements-seedvc.txt` for the full setup, the BigVGAN
one-line patch, and how to run the persistent warm server
(`seedvc_server.py`, listens on `127.0.0.1:8199`).

## Environment

| Var | Default | Used by |
|-----|---------|---------|
| `SEEDVC_PYTHON` | `/workspace/seedvc-venv/bin/python` | `SeedVCConvert` subprocess fallback |
| `SEEDVC_SERVER` | `http://127.0.0.1:8199` | `SeedVCConvert` warm-server path |
| `SEEDVC_PORT` | `8199` | `seedvc_server.py` |
| `ROFORMER_MODEL` | `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` | `AudioExtractVocals` |
