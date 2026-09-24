"""Hash-checked loading of the frozen ambient clocked classifier."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from . import clocked

MODEL_SHA256 = "d0f0d8170229da869332ae0754e05e488602d13f2fac6cd2ef6ae80a974ad551"
MODEL_REVISION = "ambient-multipitch-v2"
MODEL_SKLEARN_VERSION = "1.6.1"
MODEL_FILENAME = f"ambient-multipitch-v2-{MODEL_SHA256}.joblib"
MODEL_URL = ("https://github.com/kenspiredcode/B4-BL/releases/download/"
             f"model-ambient-multipitch-v2/{MODEL_FILENAME}")


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    return (Path(root) if root else Path.home() / ".cache") / "b4bl" / "models"


def model_path() -> Path:
    return cache_dir() / MODEL_FILENAME


def verify_model(path=None) -> str:
    """Verify bytes before joblib can deserialize them; return the SHA256."""
    path = Path(path) if path is not None else model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"model missing: {path}. Run 'b4bl models fetch', or "
            "'b4bl models install PATH' with a trusted local copy.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != MODEL_SHA256:
        raise ValueError(f"model SHA256 mismatch: expected {MODEL_SHA256}, got {actual}")
    return actual


def load_model(path=None):
    """Load only the exact frozen model. Never pass untrusted joblib files here."""
    path = Path(path) if path is not None else model_path()
    verify_model(path)
    try:
        installed_sklearn = version("scikit-learn")
    except PackageNotFoundError as exc:
        raise RuntimeError("decoding requires: pip install 'b4bl[decode]'") from exc
    if installed_sklearn != MODEL_SKLEARN_VERSION:
        raise RuntimeError(
            f"frozen model requires scikit-learn {MODEL_SKLEARN_VERSION}; "
            f"found {installed_sklearn}. Install 'b4bl[decode]' in a clean environment.")
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("decoding requires: pip install 'b4bl[decode]'") from exc
    model = joblib.load(path)
    if (not isinstance(model, dict) or model.get("profile") != clocked.PROFILE or
            not hasattr(model.get("model"), "predict_proba")):
        raise ValueError("verified model has incompatible clocked profile or classifier")
    model["_b4bl_sha256"] = MODEL_SHA256
    model["_b4bl_revision"] = MODEL_REVISION
    return model


def install_model(source) -> Path:
    """Copy a locally supplied trusted model into the cache after hash verification."""
    source = Path(source)
    verify_model(source)
    destination = model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        verify_model(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fetch_model() -> Path:
    """Fetch the pinned official release asset, verifying it before installation."""
    destination = model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(MODEL_URL, timeout=30) as response:
                shutil.copyfileobj(response, stream)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"could not fetch model from {MODEL_URL}: {exc}") from exc
    try:
        verify_model(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
