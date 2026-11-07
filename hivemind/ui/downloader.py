"""Resumable model downloads (issue #40, wave #38).

download_model(repo_id, on_progress) pulls every wanted file
from the Hugging Face hub with hf_hub_download - one call per
file, resume built in - and reports live progress as
(downloaded_mb, total_mb, speed_mbps).

Integrity: after the last byte, every file is verified against
what the hub publishes - LFS files by sha256 (the lfs oid),
plain files by their git blob sha1. A mismatch raises: a
corrupted download must never pretend to be a model.
"""
import hashlib
import threading
import time
from pathlib import Path

MODELS_DIR = Path.home() / ".hivemind" / "models"
WANTED_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".jinja")
_HASH_CHUNK = 1024 * 1024
_PLAIN_VERIFY_LIMIT = 64 * 1024 * 1024


def model_local_dir(repo_id: str, models_dir: Path = MODELS_DIR) -> Path:
    """Where a repo lands locally: models_dir/<repo name>."""
    return Path(models_dir) / repo_id.split("/")[-1]


def _dir_bytes(path: Path) -> int:
    """Bytes already on disk under path, including partial files."""
    total = 0
    for f in Path(path).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1_of(path: Path) -> str:
    data = Path(path).read_bytes()
    h = hashlib.sha1()
    h.update(b"blob " + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def download_model(repo_id: str, on_progress=None,
                   models_dir: Path = MODELS_DIR) -> Path:
    """Download repo_id resumably, verify hashes, return the dir.

    on_progress(downloaded_mb, total_mb, speed_mbps) fires from
    a monitor thread while files stream in. Raises RuntimeError
    when the hub publishes no Safetensors weights or a hash
    check fails.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    siblings = list(info.siblings or [])
    wanted = [s for s in siblings
              if s.rfilename.endswith(WANTED_SUFFIXES)
              and not s.rfilename.startswith("original/")]
    if not any(s.rfilename.endswith(".safetensors") for s in wanted):
        raise RuntimeError(
            "no Safetensors weights in %s - HiveMind reads "
            "Safetensors only (GGUF: not implemented yet)" % repo_id)
    total_bytes = sum((s.size or 0) for s in wanted)
    if total_bytes <= 0:
        raise RuntimeError("the hub reports no file sizes for %s"
                           % repo_id)
    target = model_local_dir(repo_id, models_dir)
    target.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    if on_progress is not None:
        def _monitor():
            last_bytes = _dir_bytes(target)
            last_at = time.monotonic()
            while not stop.is_set():
                time.sleep(0.25)
                now_bytes = _dir_bytes(target)
                now_at = time.monotonic()
                speed = ((now_bytes - last_bytes)
                         / max(now_at - last_at, 1e-6))
                on_progress(now_bytes / 1e6, total_bytes / 1e6,
                            speed / 1e6)
                last_bytes = now_bytes
                last_at = now_at
        threading.Thread(target=_monitor, daemon=True).start()

    try:
        for s in wanted:
            hf_hub_download(repo_id=repo_id, filename=s.rfilename,
                            local_dir=str(target))
    finally:
        stop.set()
    if on_progress is not None:
        on_progress(total_bytes / 1e6, total_bytes / 1e6, 0.0)

    bad = []
    for s in wanted:
        local = target / s.rfilename
        lfs = getattr(s, "lfs", None)
        oid = getattr(lfs, "oid", None) if lfs else None
        if oid:
            if _sha256_of(local) != oid:
                bad.append(s.rfilename)
        elif s.blob_id:
            if (local.stat().st_size <= _PLAIN_VERIFY_LIMIT
                    and _git_blob_sha1_of(local) != s.blob_id):
                bad.append(s.rfilename)
    if bad:
        raise RuntimeError(
            "hash verification failed for: %s - delete the "
            "model folder and download again" % ", ".join(bad))
    return target
