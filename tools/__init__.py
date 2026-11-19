"""Tools available to the local chat application.

Submodules are imported explicitly (``from tools.calculator import calculate``).
This package deliberately imports nothing eagerly: the calculator worker
subprocess imports ``tools.calculator``, and pulling in ``tools.rag`` or
``tools.tts`` here would load torch, sentence-transformers and kokoro in that
subprocess too, adding ~10 s of startup to every calculator call.
"""
