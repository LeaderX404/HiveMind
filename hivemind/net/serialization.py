"""
Safe tensor wire format for the gRPC inference protocol.

The old wire format used torch.save/torch.load, i.e. pickle, on
payloads that come from the network. Unpickling attacker-controlled
data is remote code execution, and weights_only=True is not a hard
guarantee: CVE-2025-32434 (fixed in torch 2.6.0) and
PYSEC-2026-2286 (fixed in torch 2.10.0) both bypass it.

This module replaces pickle with an explicit, non-executable
container:

    [ 4B magic "HMT1" ][ 4B header length ][ JSON header ][ raw bytes ]

The JSON header carries the tensor shape and a dtype taken from a
strict allowlist. Deserialization only reads bytes and builds a
tensor; nothing in the payload is ever executed. Legacy
torch.save/pickle payloads are rejected by the magic check.
"""
import json
import struct
import torch

MAGIC = b"HMT1"
# Little-endian: 4-byte magic followed by 4-byte header length.
_HEADER = struct.Struct("<4sI")

# Strict allowlist: dtypes that may cross the wire. Anything else is
# rejected during both serialization and deserialization.
_ALLOWED_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

# Byte width per allowed dtype. Kept as an explicit table (not
# torch.dtype.element_size, which only exists on tensor instances)
# because the deserializer validates the payload byte length
# against it: the table itself is part of the security boundary.
_DTYPE_ITEM_SIZES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _dtype_name(tensor: torch.Tensor) -> str:
    """Return the bare dtype name, e.g. 'float32' for torch.float32."""
    return str(tensor.dtype).replace("torch.", "")


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Pack a tensor into the safe container format (no pickle)."""
    tensor = tensor.detach().cpu().contiguous()
    name = _dtype_name(tensor)
    if name not in _ALLOWED_DTYPES:
        raise ValueError(f"dtype {name} is not allowed on the wire")
    header = json.dumps(
        {"shape": list(tensor.shape), "dtype": name}
    ).encode("utf-8")
    if tensor.numel() == 0:
        raw = b""
    else:
        # Reinterpret the buffer as uint8 to copy raw element bytes
        # without any pickling. Works for every allowed dtype.
        raw = tensor.view(torch.uint8).numpy().tobytes()
    return _HEADER.pack(MAGIC, len(header)) + header + raw


def deserialize_tensor(payload: bytes) -> torch.Tensor:
    """Rebuild a tensor from the safe container format (no pickle)."""
    if len(payload) < _HEADER.size:
        raise ValueError("payload rejected: too short to be a tensor container")
    magic, header_len = _HEADER.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(
            "payload rejected: not a HiveMind tensor container; "
            "legacy torch.save/pickle payloads are no longer accepted"
        )
    header_start = _HEADER.size
    header_end = header_start + header_len
    if header_end > len(payload):
        raise ValueError("payload rejected: truncated header")
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"payload rejected: malformed header: {e}")
    
    shape = [int(dim) for dim in header.get("shape", [])]
    dtype_name = header.get("dtype")
    if dtype_name not in _ALLOWED_DTYPES:
        raise ValueError(f"payload rejected: dtype {dtype_name!r} is not allowed")
    dtype = _ALLOWED_DTYPES[dtype_name]
    if dtype_name not in _DTYPE_ITEM_SIZES:
        raise ValueError(f"payload rejected: dtype {dtype_name!r} has no size entry")
    
    raw = payload[header_end:]
    numel = 1
    for dim in shape:
        numel *= dim
    # numel can go negative with attacker-crafted negative dims;
    # a negative product can never equal a non-negative byte count.
    if numel * _DTYPE_ITEM_SIZES[dtype_name] != len(raw):
        raise ValueError(
            "payload rejected: byte length does not match shape/dtype"
        )
    
    # bytearray() gives frombuffer a writable buffer; clone() makes
    # the tensor own its memory instead of aliasing the payload.
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
    if shape:
        tensor = tensor.reshape(shape)
    return tensor.clone()
