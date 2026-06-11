from __future__ import annotations

import os
from functools import lru_cache


def snapshot_download(*args, **kwargs):
    from huggingface_hub import snapshot_download as _snapshot_download

    return _snapshot_download(*args, **kwargs)


@lru_cache(maxsize=None)
def cached_snapshot_path(model_id: str) -> str | None:
    if os.path.exists(model_id):
        return model_id

    try:
        return snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception:
        return None


def cached_from_pretrained(loader, model_id: str, **kwargs):
    local_path = cached_snapshot_path(model_id)
    if local_path is not None:
        try:
            return loader.from_pretrained(local_path, **kwargs)
        except Exception:
            pass

    return loader.from_pretrained(model_id, **kwargs)
