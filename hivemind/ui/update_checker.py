"""Version check against the public GitHub releases page.

The app knows one thing about the internet: whether a newer
HiveMind release exists. Model downloads are untouched - this
module never goes near them. Every failure (no network, slow
network, malformed reply, version text we cannot compare)
returns None: a silent app beats a guessing app.

Results are cached for 24 hours in ~/.hivemind/update_check.json
so the UI does not wait on the network on every rerun.
"""
import json
import os
import time
import urllib.request

APP_VERSION = "0.1.1"
RELEASES_API = ("https://api.github.com/repos/LeaderX404/HiveMind/"
                "releases/latest")
RELEASES_PAGE = ("https://github.com/LeaderX404/HiveMind/"
                 "releases/latest")
CACHE_PATH = os.path.expanduser("~/.hivemind/update_check.json")
CACHE_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 5


def _version_tuple(text):
    """'0.1.2' -> (0, 1, 2); None when the text is not plain
    dot-separated integers (prerelease suffixes do not guess)."""
    parts = text.strip().lstrip("v").split(".")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def _fetch_latest_version():
    """Ask the GitHub API for the latest release tag, or None."""
    request = urllib.request.Request(
        RELEASES_API, headers={"User-Agent": "HiveMind-update-check"})
    try:
        with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    tag = data.get("tag_name") if isinstance(data, dict) else None
    return tag if isinstance(tag, str) else None


def _read_cache():
    """The cached check as a dict, or None when stale/absent."""
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.loads(f.read())
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict):
        return None
    if time.time() - cached.get("checked_at", 0) > CACHE_TTL_SECONDS:
        return None
    latest = cached.get("latest_version")
    if latest is not None and not isinstance(latest, str):
        return None
    return cached


def _write_cache(latest_version):
    """Best-effort cache write; failures are not the app's problem."""
    payload = {"checked_at": time.time(), "latest_version": latest_version}
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
    except OSError:
        pass


def check_for_update():
    """{'version': ..., 'url': ...} when a strictly newer public
    release exists, otherwise None. Never raises, never guesses."""
    cached = _read_cache()
    if cached is not None:
        latest = cached.get("latest_version")
    else:
        latest = _fetch_latest_version()
        _write_cache(latest)
    if latest is None:
        return None
    ours = _version_tuple(APP_VERSION)
    theirs = _version_tuple(latest)
    if ours is None or theirs is None or ours >= theirs:
        return None
    return {"version": latest.lstrip("v"), "url": RELEASES_PAGE}
