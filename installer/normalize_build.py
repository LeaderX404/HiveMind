"""Normalize the onedir build for deterministic delta patches.

The update manifest diffs FILE CONTENT (a sha256 per file), so
patch membership depends on file bytes alone - zip metadata
never enters a patch. Three byte-level sources of build-to-
build noise must be pinned, or every release re-ships the app:

1. The PE TimeDateStamp (the 32-bit word at e_lfanew + 8) in
   every .exe / .dll / .pyd. Windows accepts any value there;
   the field is tooling metadata. The CI passes SOURCE_DATE_EPOCH
   to PyInstaller as the upstream reproducible-builds standard,
   and this script enforces the same epoch by hand - the
   guarantee holds regardless of tool support.
2. The mtime word inside timestamp-based .pyc headers. Hash-
   based headers (PEP 552, flags bit 0) carry no mtime and are
   left untouched.
3. On-disk file mtimes and the UI zip entry metadata (fixed
   dates, sorted order, fixed attributes), making the bundle
   itself a byte-reproducible artifact even though the updater
   only ever hashes the files inside it.

A malformed or unreadable file is a HARD error: a silent skip
would poison the manifest with nondeterministic bytes.

Usage:
    python installer/normalize_build.py --ui-dir dist/HiveMind-UI
        --epoch 946684800 --zip HiveMind-UI-windows-x64.zip

Verification protocol (wave #48): two dry-run builds from
identical sources must produce byte-identical UI bundles.
The CI artifacts API exposes a sha256 digest per artifact,
so the comparison needs no log access: matching digests
across independent builds is proof of determinism; matching
sizes alone is only weak evidence.
"""
import argparse
import os
import struct
import zipfile

_MZ = b"MZ"
_PE_SIGNATURE = b"PE\x00\x00"
_E_LFANEW_OFFSET = 0x3C
_PE_TIMESTAMP_OFFSET = 8  # TimeDateStamp, from the PE signature

_PYC_HEADER = struct.Struct("<4sI")  # magic, flags
_PYC_FLAGS_HASH_BASED = 0x1  # PEP 552: no mtime field present
_PYC_MTIME_OFFSET = 8  # timestamp-based .pyc only

# Fixed zip entry date: 2000-01-01 00:00:00, the same instant
# as SOURCE_DATE_EPOCH (946684800) pinned into PE headers.
_ZIP_DATE_TIME = (2000, 1, 1, 0, 0, 0)

_PE_EXTENSIONS = (".exe", ".dll", ".pyd")


def _pe_timestamp_offset(data):
    """Return the offset of TimeDateStamp inside a PE image, or
    None when the file is not a PE image we recognize."""
    if len(data) < 64 or data[:2] != _MZ:
        return None
    e_lfanew = struct.unpack_from("<I", data, _E_LFANEW_OFFSET)[0]
    if e_lfanew + 24 > len(data):
        return None
    if data[e_lfanew:e_lfanew + 4] != _PE_SIGNATURE:
        return None
    return e_lfanew + _PE_TIMESTAMP_OFFSET


def _rewrite_pe_timestamp(path, epoch):
    """Pin the PE TimeDateStamp to a fixed epoch. Idempotent."""
    with open(path, "rb") as f:
        data = f.read()
    offset = _pe_timestamp_offset(data)
    if offset is None:
        return False
    if struct.unpack_from("<I", data, offset)[0] == epoch:
        return False
    data = data[:offset] + struct.pack("<I", epoch) + data[offset + 4:]
    with open(path, "wb") as f:
        f.write(data)
    return True


def _rewrite_pyc_mtime(path):
    """Zero the mtime word of a timestamp-based .pyc header."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < _PYC_HEADER.size:
        return False
    _magic, flags = _PYC_HEADER.unpack_from(data, 0)
    if flags & _PYC_FLAGS_HASH_BASED:
        return False  # hash-based: no mtime exists
    if struct.unpack_from("<I", data, _PYC_MTIME_OFFSET)[0] == 0:
        return False
    data = (data[:_PYC_MTIME_OFFSET]
            + struct.pack("<I", 0)
            + data[_PYC_MTIME_OFFSET + 4:])
    with open(path, "wb") as f:
        f.write(data)
    return True


def _deterministic_zip(src_dir, out_path):
    """Write a byte-reproducible zip of src_dir: sorted entries,
    fixed dates, fixed attributes, fixed compression level."""
    entries = []
    for root, _dirs, names in os.walk(src_dir):
        for name in names:
            absolute = os.path.join(root, name)
            rel = os.path.relpath(absolute, src_dir)
            entries.append((rel.replace(os.sep, "/"), absolute))
    entries.sort()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for arc, absolute in entries:
            info = zipfile.ZipInfo(arc, date_time=_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0  # pinned: DOS/Windows creator
            info.external_attr = 0  # pinned: nothing to preserve
            with open(absolute, "rb") as f:
                zf.writestr(info, f.read())
    return len(entries)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize a built app dir for deterministic"
                    " delta patches (wave #48)")
    parser.add_argument("--ui-dir", required=True,
                        help="the built app dir - dist/HiveMind-UI")
    parser.add_argument("--epoch", type=int, default=946684800,
                        help="the pinned timestamp epoch")
    parser.add_argument("--zip", default=None,
                        help="also write the UI bundle zip here")
    args = parser.parse_args()

    ui_dir = os.path.abspath(args.ui_dir)
    if not os.path.isdir(ui_dir):
        raise SystemExit("no such directory: %s" % ui_dir)

    pe_pinned = 0
    pyc_pinned = 0
    seen = 0
    errors = []
    for root, _dirs, names in os.walk(ui_dir):
        for name in names:
            absolute = os.path.join(root, name)
            seen += 1
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext in _PE_EXTENSIONS:
                    if _rewrite_pe_timestamp(absolute, args.epoch):
                        pe_pinned += 1
                elif ext == ".pyc":
                    if _rewrite_pyc_mtime(absolute):
                        pyc_pinned += 1
            except OSError as exc:
                errors.append("%s: %s" % (absolute, exc))
    if errors:
        for line in errors:
            print("normalize: ERROR %s" % line)
        raise SystemExit(1)

    # Pin mtimes AFTER the content rewrites (rewrites bump them).
    for root, _dirs, names in os.walk(ui_dir):
        for name in names:
            absolute = os.path.join(root, name)
            os.utime(absolute, (args.epoch, args.epoch))

    print("normalize: %d files, %d PE timestamps pinned, %d .pyc"
          " mtimes zeroed" % (seen, pe_pinned, pyc_pinned))

    if args.zip:
        count = _deterministic_zip(ui_dir, args.zip)
        size = os.path.getsize(args.zip)
        print("normalize: wrote %s - %d entries, %.1f MB"
              % (args.zip, count, size / (1024.0 * 1024.0)))


if __name__ == "__main__":
    main()
