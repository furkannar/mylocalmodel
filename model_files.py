"""Find which model files still need downloading, and warn when disk space is short.

Nothing here downloads anything; the loaders in engine.py and tools/ do that, in
parallel, after this report is printed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import ChatConfig

GIB = 1024 ** 3
_EMBEDDING_MODEL_BYTES = 100 * 1024 ** 2  # all-MiniLM-L6-v2 weights + tokenizer, approx.
_TTS_BYTES = {"kokoro-v1.0.onnx": 326 * 1024 ** 2, "voices-v1.0.bin": 28 * 1024 ** 2}
_SAFETY_MARGIN = 1.10  # ask for 10% more free space than the downloads need


@dataclass
class PendingDownload:
    label: str
    size_bytes: Optional[int]  # None when unknown; 0 when streamed and not stored on disk
    target_dir: Path


def _hub_cache_dir() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def _is_cached(repo_id: str, filename: str, repo_type: str = "model") -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo_id, filename, repo_type=repo_type), str)


def _remote_size(repo_id: str, filename: str) -> Optional[int]:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        return get_hf_file_metadata(hf_hub_url(repo_id, filename)).size
    except Exception:
        return None  # offline or unknown; the download itself will report the problem


def find_pending_downloads(config: ChatConfig, include_tts: bool = False) -> List[PendingDownload]:
    pending: List[PendingDownload] = []
    hub_cache = _hub_cache_dir()

    if not config.local_model_path and not _is_cached(config.model_name, config.gguf_file):
        pending.append(PendingDownload(
            f"LLM {config.model_name}/{config.gguf_file}",
            _remote_size(config.model_name, config.gguf_file),
            hub_cache,
        ))

    if config.enable_rag:
        from tools.rag import INDEX_FILENAME
        rag = config.rag
        if not _is_cached(rag.embedding_model_name, "config.json"):
            pending.append(PendingDownload(f"Embedding model {rag.embedding_model_name}", _EMBEDDING_MODEL_BYTES, hub_cache))
        if not (rag.cache_dir / INDEX_FILENAME).exists():
            # Streamed: only the needed Parquet row groups are fetched, nothing is stored.
            pending.append(PendingDownload(
                f"RAG dataset {rag.dataset_name}, first {rag.max_rows} rows (to build the index)",
                0,
                hub_cache,
            ))

    if include_tts:
        from tools.tts import missing_model_files
        for name, path in missing_model_files():
            pending.append(PendingDownload(f"TTS {name}", _TTS_BYTES.get(name), path.parent))

    return pending


def _existing_parent(path: Path) -> Path:
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def report_pending_downloads(pending: List[PendingDownload]) -> None:
    """Print what will be downloaded and warn when a drive looks too full for it."""
    if not pending:
        print("All model files are already available locally.")
        return

    print(f"Downloading {len(pending)} missing item(s) in parallel:")
    for item in pending:
        if item.size_bytes == 0:
            size = "streamed, not stored on disk"
        elif item.size_bytes:
            size = f"~{item.size_bytes / GIB:.2f} GB"
        else:
            size = "size unknown"
        print(f"  - {item.label} ({size})")

    needed_per_drive: dict = {}
    for item in pending:
        anchor = _existing_parent(item.target_dir)
        drive = anchor.anchor or str(anchor)
        entry = needed_per_drive.setdefault(drive, [anchor, 0])
        entry[1] += item.size_bytes or 0
    for drive, (anchor, needed) in needed_per_drive.items():
        free = shutil.disk_usage(anchor).free
        if needed and free < needed * _SAFETY_MARGIN:
            print(
                f"WARNING: {drive} has {free / GIB:.2f} GB free but the downloads need about "
                f"{needed / GIB:.2f} GB. Free up space or set HF_HOME to another drive."
            )
