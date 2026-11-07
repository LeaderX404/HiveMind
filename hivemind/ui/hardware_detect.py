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
"""
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

PROFILE_TTL_DAYS = 7
DISK_PROBE_BYTES = 100 * 1024 * 1024  # 100 MiB probe
BUFFERED_READ_FACTOR = 0.7  # buffered reads are page-cache warm; declared correction


def _hivemind_dir() -> Path:
    """Return ~/.hivemind, creating it when missing."""
    root = Path.home() / ".hivemind"
    root.mkdir(parents=True, exist_ok=True)
    return root


def detect_ram_gb() -> float:
    """Available RAM in GB (decimal): psutil.virtual_memory().available / 1e9."""
    import psutil  # lazy: run.sh / run.bat install it before the UI starts
    return psutil.virtual_memory().available / 1e9


def detect_disk_speed_gbs() -> float:
    """Disk read speed in GB/s, measured with a 100 MiB probe.

    Writes 100 MiB of random data to a tempfile inside ~/.hivemind,
    fsyncs it, then reads it back with O_DIRECT when the OS provides
    it. If O_DIRECT is unavailable (Windows) or its read fails
    (alignment, driver quirks), a buffered read is used and the
    result is multiplied by BUFFERED_READ_FACTOR (0.7): a buffered
    read right after a write is page-cache warm and overstates the
    disk. That correction is declared here, not hidden.
    """
    probe = os.urandom(DISK_PROBE_BYTES)
    fd, tmp_name = tempfile.mkstemp(prefix="disk_probe_", dir=str(_hivemind_dir()))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(probe)
            f.flush()
            os.fsync(f.fileno())
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
                data = os.read(rfd, DISK_PROBE_BYTES)
                elapsed = time.perf_counter() - start
                direct_ok = (len(data) == DISK_PROBE_BYTES)
            except OSError:
                direct_ok = False  # opened, but the aligned read failed
            finally:
                os.close(rfd)
        if not direct_ok:
            factor = BUFFERED_READ_FACTOR
            with tmp.open("rb") as f:
                start = time.perf_counter()
                data = f.read()
                elapsed = time.perf_counter() - start
            if len(data) != DISK_PROBE_BYTES:
                raise RuntimeError("disk probe read back the wrong number of bytes")
        if elapsed <= 0:
            raise RuntimeError("disk probe measured a non-positive duration")
        return (DISK_PROBE_BYTES / 1e9) / elapsed * factor
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
        "cores": detect_cpu_cores(),
        "os": detect_os(),
        "measured_at": datetime.now().isoformat(),
    }
    with cache.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")
    return profile
