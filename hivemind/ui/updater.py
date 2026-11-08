"""Delta updates and the one-time core bootstrap (wave #47).

Laws of this module:
- No unverified code runs: every downloaded archive is checked
  against its SHA-256 before extraction, and every patched
  file is checked again before it replaces anything on disk.
- No guessing: any doubt - bad hash, missing key, dead
  network - leaves the installed build exactly as it was.
- Small is the point: only files whose hash changed in the
  manifest are fetched, so a normal release patches kilobytes
  instead of the whole bundle.
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile

from hivemind.ui.update_checker import (APP_VERSION,
                                        REQUEST_TIMEOUT_SECONDS,
                                        _version_tuple)

MANIFEST_URL = ("https://github.com/LeaderX404/HiveMind/"
                "releases/latest/download/update_manifest.json")
STAGE_ROOT = os.path.expanduser(os.path.join("~", ".hivemind", "staging"))
MAX_APPLY_ATTEMPTS = 3
_CHUNK = 64 * 1024
_PROGRESS_EVERY = 64 * 1024 * 1024


def _sha256_file(path):
    """SHA-256 of a file, hex lowercase."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, dest_path):
    """Stream url to dest_path with progress notes. False on
    any failure - the partial file is removed."""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "HiveMind-update-check"})
        with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS) as resp, \
                open(dest_path, "wb") as out:
            total = 0
            next_note = _PROGRESS_EVERY
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                if total >= next_note:
                    print("updater: %s %.0f MB so far"
                          % (os.path.basename(dest_path),
                             total // (1024 * 1024)))
                    next_note += _PROGRESS_EVERY
        return True
    except OSError as exc:
        print("updater: download failed: %s" % exc)
        try:
            os.remove(dest_path)
        except OSError:
            pass
        return False


def fetch_manifest():
    """The latest public update manifest, or None on any failure."""
    try:
        request = urllib.request.Request(
            MANIFEST_URL, headers={"User-Agent": "HiveMind-update-check"})
        with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            manifest = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if not isinstance(manifest.get("version"), str):
        return None
    return manifest


def ensure_core(core_dir):
    """Download + verify + extract the heavy core once. True when
    the core is ready to use."""
    if os.path.isdir(os.path.join(core_dir, "torch")):
        return True
    manifest = fetch_manifest()
    if manifest is None:
        print("updater: no manifest - cannot fetch the core")
        return False
    core = manifest.get("core")
    if not isinstance(core, dict):
        print("updater: manifest has no core entry")
        return False
    url = core.get("url")
    sha = core.get("sha256")
    if not isinstance(url, str) or not isinstance(sha, str):
        print("updater: core entry incomplete")
        return False
    parent = os.path.dirname(os.path.abspath(core_dir))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "HiveMind-Core.zip.part")
    print("updater: downloading the core bundle")
    if not _download(url, zip_path):
        return False
    print("updater: verifying the core archive")
    if _sha256_file(zip_path) != sha.lower():
        print("updater: core sha256 mismatch - discarding")
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return False
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(core_dir)
    try:
        os.remove(zip_path)
    except OSError:
        pass
    if not os.path.isdir(os.path.join(core_dir, "torch")):
        print("updater: core archive did not contain torch/")
        return False
    print("updater: core installed")
    return True


def _pending_path(stage_dir):
    return os.path.join(stage_dir, "pending.json")


def _unsafe_path(path):
    """True for absolute paths or anything that climbs out."""
    if os.path.isabs(path):
        return True
    parts = path.replace("\\", "/").split("/")
    return ".." in parts


def download_patch(manifest, stage_dir):
    """Fetch the patch archive, verify it, then extract and
    per-file verify its contents. False on any doubt."""
    patch = manifest.get("patch")
    files = manifest.get("files")
    if not isinstance(patch, dict) or not isinstance(files, list) \
            or not files:
        print("updater: manifest has no patch entry")
        return False
    url = patch.get("url")
    sha = patch.get("sha256")
    if not isinstance(url, str) or not isinstance(sha, str):
        print("updater: patch entry incomplete")
        return False
    for entry in files:
        if not isinstance(entry, dict) \
                or not isinstance(entry.get("path"), str) \
                or not isinstance(entry.get("sha256"), str):
            print("updater: malformed file entry")
            return False
        if _unsafe_path(entry["path"]):
            print("updater: refusing unsafe path %s" % entry["path"])
            return False
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir, ignore_errors=True)
    os.makedirs(stage_dir, exist_ok=True)
    zip_path = os.path.join(stage_dir, "patch.zip")
    if not _download(url, zip_path):
        return False
    print("updater: verifying the patch archive")
    if _sha256_file(zip_path) != sha.lower():
        print("updater: patch sha256 mismatch - discarding")
        shutil.rmtree(stage_dir, ignore_errors=True)
        return False
    extract_dir = os.path.join(stage_dir, "patch")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    try:
        os.remove(zip_path)
    except OSError:
        pass
    for entry in files:
        local = os.path.join(extract_dir,
                             *entry["path"].replace("\\", "/").split("/"))
        if not os.path.isfile(local) \
                or _sha256_file(local) != entry["sha256"].lower():
            print("updater: patched file failed verification: %s"
                  % entry["path"])
            shutil.rmtree(stage_dir, ignore_errors=True)
            return False
    pending = {"version": manifest["version"], "files": files,
               "attempts": 0}
    with open(_pending_path(stage_dir), "w", encoding="utf-8") as f:
        f.write(json.dumps(pending))
    print("updater: patch staged and verified - %d files"
          % len(files))
    return True


def apply_patch(stage_dir, app_root):
    """Move verified files into place. True when a relaunch is
    needed - the executable was replaced, or locked files
    remain staged for the next start."""
    pending_path = _pending_path(stage_dir)
    if not os.path.isfile(pending_path):
        return False
    with open(pending_path, "r", encoding="utf-8") as f:
        pending = json.loads(f.read())
    extract_dir = os.path.join(stage_dir, "patch")
    pending["attempts"] = pending.get("attempts", 0) + 1
    remaining = []
    exe_replaced = False
    for entry in pending.get("files", []):
        path = entry["path"]
        src = os.path.join(extract_dir, *path.replace("\\", "/").split("/"))
        dst = os.path.join(app_root, *path.replace("\\", "/").split("/"))
        if not os.path.isfile(src):
            print("updater: staged file vanished: %s" % path)
            remaining.append(entry)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if os.path.abspath(dst) == os.path.abspath(sys.executable):
                old = dst + ".old"
                if os.path.exists(old):
                    try:
                        os.remove(old)
                    except OSError:
                        pass
                os.rename(dst, old)
                shutil.copy2(src, dst)
                exe_replaced = True
                print("updater: replaced the executable")
            else:
                os.replace(src, dst)
                print("updater: replaced %s" % path)
        except OSError as exc:
            print("updater: %s locked - stays staged: %s"
                  % (path, exc))
            remaining.append(entry)
    if remaining and pending["attempts"] >= MAX_APPLY_ATTEMPTS:
        print("updater: giving up on %d locked files after %d "
              "attempts" % (len(remaining), pending["attempts"]))
        remaining = []
    if remaining:
        pending["files"] = remaining
        with open(pending_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(pending))
        return True
    shutil.rmtree(stage_dir, ignore_errors=True)
    return exe_replaced


def check_and_apply(app_root):
    """One update cycle. True means: relaunch the process now."""
    # Finish an interrupted update first - converges over a few
    # starts; locked files are given up on, never looped over.
    if os.path.isdir(STAGE_ROOT):
        for name in sorted(os.listdir(STAGE_ROOT)):
            candidate = os.path.join(STAGE_ROOT, name)
            if os.path.isfile(_pending_path(candidate)):
                print("updater: finishing a previous update")
                return apply_patch(candidate, app_root)
    # Is there anything newer?
    manifest = fetch_manifest()
    if manifest is None:
        return False
    theirs = _version_tuple(manifest["version"])
    ours = _version_tuple(APP_VERSION)
    if theirs is None or ours is None or theirs <= ours:
        return False
    print("updater: v%s is available" % manifest["version"])
    stage_dir = os.path.join(STAGE_ROOT, "v" + manifest["version"])
    if not download_patch(manifest, stage_dir):
        return False
    return apply_patch(stage_dir, app_root)
