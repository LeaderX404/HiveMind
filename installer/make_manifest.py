"""Build update_manifest.json and the UI delta patch (wave #47).

Inputs: the freshly built onedir app (dist/HiveMind-UI), the
core bundle zip, the release version, and optionally the
previous release's UI zip. Outputs, in --out-dir:
- update_manifest.json: version, core url+sha256, patch
  url+sha256, and the per-file sha256 list.
- HiveMind-UI-patch.zip: only the files that changed, or all
  of them when no previous bundle exists - the first wave of a
  new architecture is a full patch by definition.

Every path written into the manifest is forward-slashed and
relative to the app root; the updater re-verifies every hash
before anything on disk is replaced.
"""
import argparse
import hashlib
import io
import json
import os
import zipfile

BASE_URL = ("https://github.com/LeaderX404/HiveMind/releases/"
            "download/v%s")


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_zip(zip_path):
    """Hash every file inside a zip - entries may or may not
    carry a top folder."""
    entries = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            entries[name] = _hash_bytes(zf.read(name))
    return entries


def _prev_paths(entries):
    """Map previous-zip entry names to app-relative paths by
    stripping a single top folder if one is present."""
    tops = set()
    for name in entries:
        parts = name.split("/")
        if len(parts) > 1:
            tops.add(parts[0])
    has_top = len(tops) == 1 and all(
        name.split("/")[0] in tops for name in entries)
    mapping = {}
    for name in entries:
        parts = name.split("/")
        rel = "/".join(parts[1:]) if has_top else name
        mapping[rel] = entries[name]
    return mapping


def _collect_app_files(ui_dir):
    """Map every file in the freshly built app directory to its
    app-relative path and sha256."""
    files = {}
    for root, _dirs, names in os.walk(ui_dir):
        for name in names:
            absolute = os.path.join(root, name)
            rel = os.path.relpath(absolute, ui_dir)
            rel = rel.replace(os.sep, "/")
            files[rel] = (absolute, _hash_file(absolute))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Build the wave #47 update manifest and patch")
    parser.add_argument("--ui-dir", required=True,
                        help="the built app directory - dist/HiveMind-UI")
    parser.add_argument("--core-zip", required=True,
                        help="the core bundle zip")
    parser.add_argument("--version", required=True,
                        help="release version without the v prefix")
    parser.add_argument("--prev-ui-zip", default=None,
                        help="the previous release UI zip - optional")
    parser.add_argument("--out-dir", default=".",
                        help="where to write the outputs")
    args = parser.parse_args()

    version = args.version.strip().lstrip("v")
    base = BASE_URL % version
    app_files = _collect_app_files(args.ui_dir)
    if not app_files:
        raise SystemExit("no files found under %s" % args.ui_dir)

    changed = dict(app_files)
    if args.prev_ui_zip and os.path.isfile(args.prev_ui_zip):
        prev = _prev_paths(_hash_zip(args.prev_ui_zip))
        changed = {}
        for rel, entry in sorted(app_files.items()):
            if prev.get(rel) != entry[1]:
                changed[rel] = entry
        print("manifest: %d of %d files changed"
              % (len(changed), len(app_files)))
    else:
        print("manifest: no previous bundle - full patch, "
              "%d files" % len(changed))

    patch_name = "HiveMind-UI-patch.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, entry in sorted(changed.items()):
            zf.write(entry[0], rel)
    patch_data = buffer.getvalue()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    patch_path = os.path.join(out_dir, patch_name)
    with open(patch_path, "wb") as f:
        f.write(patch_data)

    files_list = [{"path": rel, "sha256": app_files[rel][1]}
                  for rel in sorted(app_files)]
    manifest = {
        "version": version,
        "core": {
            "url": "%s/HiveMind-Core-windows-x64.zip" % base,
            "sha256": _hash_file(args.core_zip),
        },
        "patch": {
            "url": "%s/%s" % (base, patch_name),
            "sha256": _hash_bytes(patch_data),
        },
        "files": files_list,
    }
    manifest_path = os.path.join(out_dir, "update_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2))
    print("manifest: wrote %s - version %s, patch %d files, "
          "%.1f MB" % (manifest_path, version, len(changed),
                       len(patch_data) / (1024 * 1024)))


if __name__ == "__main__":
    main()
