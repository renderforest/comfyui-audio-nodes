"""Renderforest audio nodes — ComfyUI custom node pack.

ComfyUI loads a directory under ``custom_nodes/`` as a Python package, i.e.
through this file only; without it a ``git clone`` of the repo registers
nothing. Each node module also stays importable on its own, so copying the
loose ``.py`` files straight into ``custom_nodes/`` keeps working.

``seedvc_server.py`` is intentionally not imported: it is the standalone warm
seed-vc server (imports ``seed_vc`` at top level), launched separately.
"""

from . import (
    audio_extract_vocals,
    audio_overlay_voice,
    omnivoice_clone_segments,
    omnivoice_whisper_transcribe,
    seedvc_convert,
)

_NODE_MODULES = (
    audio_extract_vocals,
    audio_overlay_voice,
    omnivoice_clone_segments,
    omnivoice_whisper_transcribe,
    seedvc_convert,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module in _NODE_MODULES:
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
