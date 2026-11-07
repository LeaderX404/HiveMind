"""Hardware profile detection for TPS prediction (issue #31, phase 1).

Detects the machine's available RAM, disk read speed, CPU core
count and operating system, and caches the profile to
~/.hivemind/profile.json. The cache is re-measured when it is
older than PROFILE_TTL_DAYS days or when force_refresh=True.

Rules honored here:
- No absolute paths: everything lives under Path.home() / ".hivemind".
- No OS assumption: platform.system() decides, O_DIRECT is tried
  only when the OS provides it.
- The disk probe file is created with tempfile.mkstemp() INSIDE
  ~/.hivemind so the measurement hits the same disk that will
  hold the models, not some system temp directory.
- The probe is GRADED by model size (probe_size_for_model): a
  100 MiB probe measures the NVMe cache layer; a multi-GB model
  streams through the sustained layer, so bigger models earn
  bigger probes (see MEASUREMENT_POLICY.md).
- A real load is journaled by record_actual_load() into
  "measured_bw": a measurement outranks every estimate.
"""
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

PROFILE_TTL_DAYS = 7
DEFAULT_PROBE_MIB = 100  # default probe size; graded per model size
WRITE_CHUNK_BYTES = 64 * 1024 * 1024  # caps the RAM the probe needs
READ_CHUNK_BYTES = 4 * 1024 * 1024   # O_DIRECT-safe, page-aligned
BUFFERED_READ_FACTOR = 0.7  # buffered reads are page-cache warm; declared correction


def _hivemind_dir() -> Path:
    """Return ~/.hivemind, creating it when missing."""
    root = Path.home() / ".hivemind"
    root.mkdir(parents=True, exist_ok=True)
    return root


def probe_size_for_model(model_size_gb: float) -> int:
    """Probe size in MiB, graded by model size (MEASUREMENT_POLICY).

    A 100 MiB probe measures the disk's cache layer, not the
    sustained rate a multi-GB model actually streams at. The probe
    grows with the model so the measurement reaches deeper layers:
    under 2 GB -> 100 MiB, 2-10 GB -> 500 MiB, 10-50 GB -> 2 GiB,
    above 50 GB -> 5 GiB.
    """
    if model_size_gb <= 0:
        raise ValueError("model_size_gb must be positive")
    if model_size_gb < 2:
        return 100
    if model_size_gb < 10:
        return 500
    if model_size_gb <= 50:
        return 2048
    return 5120


def detect_ram_gb() -> float:
    """Available RAM in GB (decimal): psutil.virtual_memory().available / 1e9."""
    import psutil  # lazy: run.sh / run.bat install it before the UI starts
    return psutil.virtual_memory().available / 1e9


def detect_disk_speed_gbs(probe_size_mib: int = DEFAULT_PROBE_MIB) -> float:
    """Disk read speed in GB/s, measured with a probe of a chosen size.

    Writes probe_size_mib MiB of random data to a tempfile inside
    ~/.hivemind, fsyncs it, then reads it back with O_DIRECT when
    the OS provides it. If O_DIRECT is unavailable (Windows) or its
    read fails (alignment, driver quirks), a buffered read is used
    and the result is multiplied by BUFFERED_READ_FACTOR (0.7): a
    buffered read right after a write is page-cache warm and
    overstates the disk. That correction is declared here, not
    hidden. The probe writes and reads in fixed-size chunks, so a
    5 GiB probe never needs 5 GiB of RAM.
    """
    if probe_size_mib <= 0:
        raise ValueError("probe_size_mib must be positive")
    probe_bytes = probe_size_mib * 1024 * 1024
    fd, tmp_name = tempfile.mkstemp(prefix="disk_probe_", dir=str(_hivemind_dir()))
    tmp = Path(tmp_name)
    try:
        chunk = os.urandom(min(probe_bytes, WRITE_CHUNK_BYTES))
        try:
            with os.fdopen(fd, "wb") as f:
                written = 0
                while written < probe_bytes:
                    remaining = probe_bytes - written
                    piece = chunk if remaining >= len(chunk) else chunk[:remaining]
                    wrote = f.write(piece)
                    if wrote <= 0:
                        raise OSError("disk probe write made no progress")
                    written += wrote
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise RuntimeError(
                "disk probe failed to write " + str(probe_size_mib) +
                " MiB - not enough free disk space?"
            ) from exc
        factor = 1.0
        elapsed = 0.0
        direct_ok = False
        direct_flag = getattr(os, "O_DIRECT", None)
        rfd = None
        if direct_flag is not None:
            try:
                rfd = os.open(str(tmp), os.O_RDONLY | direct_flag)
            except OSError:
                rfd = None
        if rfd is not None:
            try:
                start = time.perf_counter()
                total = 0
                while total < probe_bytes:
                    data = os.read(rfd, READ_CHUNK_BYTES)
                    if not data:
                        break
                    total += len(data)
                elapsed = time.perf_counter() - start
                direct_ok = (total == probe_bytes)
            except OSError:
                direct_ok = False  # opened, but the aligned read failed
            finally:
                os.close(rfd)
        if not direct_ok:
            factor = BUFFERED_READ_FACTOR
            with tmp.open("rb") as f:
                start = time.perf_counter()
                total = 0
                while True:
                    data = f.read(READ_CHUNK_BYTES)
                    if not data:
                        break
                    total += len(data)
                elapsed = time.perf_counter() - start
            if total != probe_bytes:
                raise RuntimeError("disk probe read back the wrong number of bytes")
        if elapsed <= 0:
            raise RuntimeError("disk probe measured a non-positive duration")
        return (probe_bytes / 1e9) / elapsed * factor
    finally:
        tmp.unlink(missing_ok=True)


def detect_cpu_cores() -> int:
    """Logical CPU core count (psutil first, os.cpu_count as fallback)."""
    cores = None
    try:
        import psutil
        cores = psutil.cpu_count(logical=True)
    except ImportError:
        cores = None
    if not cores:
        cores = os.cpu_count()
    if not cores:
        raise RuntimeError("cannot detect CPU core count on this machine")
    return int(cores)


def detect_os() -> str:
    """Operating system name: platform.system() ("Linux", "Darwin", "Windows")."""
    return platform.system()


def record_actual_load(model_name: str, size_gb: float, load_seconds: float) -> None:
    """Journal an actual load's bandwidth into the profile.

    Phase 2 calls this right after the first real load of a
    model: the effective streaming bandwidth (size_gb /
    load_seconds) becomes the model's measured entry under
    "measured_bw" in ~/.hivemind/profile.json, keyed by model
    name, with its own timestamp. From then on a measurement
    outranks every estimate for that model.
    """
    if not model_name:
        raise ValueError("model_name must not be empty")
    if size_gb <= 0:
        raise ValueError("size_gb must be positive")
    if load_seconds <= 0:
        raise ValueError("load_seconds must be positive")
    cache = _hivemind_dir() / "profile.json"
    profile = {}
    if cache.exists():
        try:
            with cache.open("r", encoding="utf-8") as f:
                profile = json.load(f)
        except (OSError, ValueError):
            profile = {}  # corrupt cache: keep the new measurement, drop the rest
    measured = profile.get("measured_bw")
    if not isinstance(measured, dict):
        measured = {}
    measured[model_name] = {
        "gbs": size_gb / load_seconds,
        "measured_at": datetime.now().isoformat(),
    }
    profile["measured_bw"] = measured
    with cache.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")


def get_profile(force_refresh: bool = False) -> dict:
    """Return the hardware profile, cached for PROFILE_TTL_DAYS days.

    Reads ~/.hivemind/profile.json when it exists and is younger
    than 7 days; otherwise re-measures every component and writes
    the cache back. force_refresh=True ignores the cache. A
    corrupt or unreadable cache is re-measured, never guessed.
    """
    cache = _hivemind_dir() / "profile.json"
    if not force_refresh and cache.exists():
        try:
            with cache.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            measured = datetime.fromisoformat(cached["measured_at"])
            if datetime.now() - measured < timedelta(days=PROFILE_TTL_DAYS):
                return cached
        except (KeyError, TypeError, ValueError, OSError):
            pass  # corrupt or stale cache: fall through and re-measure
    profile = {
        "ram_gb": detect_ram_gb(),
        "disk_gbs": detect_disk_speed_gbs(),
        "disk_probe_mib": DEFAULT_PROBE_MIB,
        "cores": detect_cpu_cores(),
        "os": detect_os(),
        "measured_at": datetime.now().isoformat(),
    }
    with cache.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")
    return profile
