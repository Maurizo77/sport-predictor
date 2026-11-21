"""Small persistent cache for models trained from unchanged local CSV data."""

import hashlib
import os
import tempfile

import joblib


BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, ".cache")


def _source_signature(source_path):
    stat = os.stat(source_path)
    digest = hashlib.sha256()
    with open(source_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return stat.st_size, stat.st_mtime_ns, digest.hexdigest()


def load_model_cache(name, source_path, version):
    """Return a cached payload when the input file and cache version still match."""
    cache_path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(cache_path):
        return None

    try:
        artifact = joblib.load(cache_path)
        if (artifact.get("version") == version
                and artifact.get("source_signature") == _source_signature(source_path)):
            return artifact.get("payload")
    except Exception:
        # A partial or incompatible cache should be rebuilt from the source data.
        pass
    return None


def save_model_cache(name, source_path, version, payload):
    """Atomically store a trained-model payload so interrupted writes are harmless."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, name)
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".tmp", delete=False) as temp:
        temporary_path = temp.name
    try:
        joblib.dump(
            {
                "version": version,
                "source_signature": _source_signature(source_path),
                "payload": payload,
            },
            temporary_path,
        )
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
