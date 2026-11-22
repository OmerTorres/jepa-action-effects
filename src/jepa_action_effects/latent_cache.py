"""Content-addressed, manifest-checked latent cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(_canonical(list(contiguous.shape)))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class LatentCache:
    """Store float32 arrays atomically with a content and provenance manifest."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def key(kind: str, inputs: dict[str, np.ndarray], fingerprint: dict[str, Any]) -> str:
        if kind not in {"real", "prediction"}:
            raise ValueError("Cache kind must be real or prediction")
        payload = {
            "kind": kind,
            "inputs": {
                name: array_digest(np.asarray(value)) for name, value in sorted(inputs.items())
            },
            "fingerprint": fingerprint,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def write(
        self,
        kind: str,
        key: str,
        embeddings: np.ndarray,
        fingerprint: dict[str, Any],
        identifiers: dict[str, Any] | None = None,
    ) -> Path:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim < 2 or values.shape[-1] != 192 or not np.isfinite(values).all():
            raise ValueError("Cached embeddings must be finite float32 arrays ending in width 192")
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key}.npz"
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "key": key,
            "embedding_shape": list(values.shape),
            "embedding_dtype": "float32",
            "embedding_sha256": array_digest(values),
            "fingerprint": fingerprint,
            "identifiers": identifiers or {},
        }
        handle, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".npz", dir=directory)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary, embeddings=values, manifest=np.array(json.dumps(manifest))
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def read(self, kind: str, key: str, fingerprint: dict[str, Any]) -> tuple[np.ndarray, dict]:
        path = self.root / kind / f"{key}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"embeddings", "manifest"}:
                    raise ValueError("Cache entry is incomplete")
                values = archive["embeddings"]
                manifest = json.loads(str(archive["manifest"].item()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid cache entry {path}") from error
        if manifest.get("kind") != kind or manifest.get("key") != key:
            raise ValueError("Cache manifest identity mismatch")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Cache fingerprint mismatch")
        if values.dtype != np.float32 or list(values.shape) != manifest.get("embedding_shape"):
            raise ValueError("Cache embedding type or shape mismatch")
        if not np.isfinite(values).all() or array_digest(values) != manifest.get(
            "embedding_sha256"
        ):
            raise ValueError("Cache embedding hash or finiteness check failed")
        return values, manifest
