from __future__ import annotations

import binascii
import logging
import os
import struct
from multiprocessing import shared_memory as _shm
from typing import Any

from vllm_omni.config.yaml_util import to_dict as _omega_to_dict
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def set_stage_devices(
    stage_id: int,
    devices: str | int | None,
) -> str | None:
    """Configure per-stage device visibility and current device (CUDA or NPU).

    This function sets environment variables that control which devices are visible
    to the process. It must be called BEFORE worker initialization so that workers
    see the correct devices.


    NOTE: This will set the control variable for the appropriate platform.
        - CUDA: CUDA_VISIBLE_DEVICES
        - NPU: ASCEND_RT_VISIBLE_DEVICES

    Args:
        stage_id: Stage identifier for logging
        devices: Devices specified as either:
            - None / "cpu"; uses the default visibility.
            - An int or a str composed of one or more ints separated by commas,
              which correspond to logical indices. If the control env var is
              set, e.g., CUDA_VISIBLE_DEVICES, we will map the logical indices
              to physical, e.g.,
                    devices: [0,1,2,3]
                    CUDA_VISIBLE_DEVICES -> [1, 3, 4, 5, 6]
            will leverage [1, 3, 4, 5]

    Returns:
        The list of physical devices that were set for the given stage
        or None if we have no passed devices / are using cpu.
    """
    env_var = current_omni_platform.device_control_env_var
    vis = os.environ.get(env_var)

    if devices in (None, "cpu"):
        logger.debug("[Stage-%s] Using default device visibility (devices=%s)", stage_id, devices)
        return None

    elif isinstance(devices, (int, str)):
        device_list = _parse_device_list(devices)
        if vis is not None:
            visible_device_list = _parse_device_list(vis)
            device_list = _map_device_list(stage_id, device_list, visible_device_list)
        device_str = ",".join(device_list)
        current_omni_platform.set_device_control_env_var(device_str)
        return device_str

    raise TypeError(f"Expected str or int device IDs for stage initialization, got type {type(devices)}")


def _parse_device_list(devices: str | int) -> list[str]:
    """Given an int or a str representing one or more comma separated
    non-negative IDs, coerce it to a list of strs.

    Args:
        devices: devices to be converted to a list of strs.
    """
    if isinstance(devices, int):
        if devices < 0:
            raise ValueError("Device IDs must be non-negative integers!")
        return [str(devices)]
    # Devices will usually be ints, but not always
    # so we don't explicitly validate that here.
    return [t.strip() for t in devices.split(",") if t.strip() != ""]


def _map_device_list(stage_id: int, device_list: list[str], visible_device_list: list[str]) -> list[str]:
    """Maps logical to physical devices if we have enough visible devices available.

    Args:
        stage_id: The stage ID currently configuring devices.
        device_list: List of (logical) devices to be used, which are strings
            holding non-negative nums counting from 0, 1, ..., n devices needed.
        visible_device_list: List of physical devices available.
    """
    num_visible = len(visible_device_list)
    num_logical = len(device_list)
    if num_visible < num_logical:
        raise ValueError(f"Stage {stage_id} requires {num_logical} devices, but only {num_visible} devices are visible")

    # Ensure that the logical IDs are actually in range to avoid index errors;
    # If the check above passes and those below fail, the logical devices are wrong,
    # i.e., not actually 0, 1, ..., n
    if not all(device.isdigit() for device in device_list):
        raise ValueError("Logical devices must be non-negative integers")

    logical_ids = [int(device) for device in device_list]
    if max(logical_ids) >= num_visible:
        raise ValueError(
            f"Stage {stage_id} has logical IDs {device_list}, one or more of which exceed the number of visible devices"
        )
    return [visible_device_list[idx] for idx in logical_ids]


def serialize_obj(obj: Any) -> bytes:
    """Serialize a Python object to bytes using centralized serializer (defaults to cloudpickle)."""
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    return OmniSerializer.serialize(obj)


_SHM_MAGIC = b"OMNIIPC1"
_SHM_HEADER_STRUCT = struct.Struct(">8sQI")
_SHM_HEADER_SIZE = _SHM_HEADER_STRUCT.size


def _pack_shm_payload(payload: bytes) -> bytes:
    crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    header = _SHM_HEADER_STRUCT.pack(_SHM_MAGIC, len(payload), crc32)
    return header + payload


def _unpack_shm_payload(raw: bytes) -> bytes:
    if len(raw) < _SHM_HEADER_SIZE:
        raise ValueError(
            f"Shared memory payload too small for header: got={len(raw)} bytes, need>={_SHM_HEADER_SIZE}"
        )
    magic, payload_size, expected_crc32 = _SHM_HEADER_STRUCT.unpack(raw[:_SHM_HEADER_SIZE])
    if magic != _SHM_MAGIC:
        raise ValueError(
            "Shared memory payload header mismatch: "
            f"expected_magic={_SHM_MAGIC!r}, got_magic={magic!r}, raw_size={len(raw)}"
        )
    payload = raw[_SHM_HEADER_SIZE:]
    if len(payload) != payload_size:
        raise ValueError(
            "Shared memory payload length mismatch: "
            f"header_payload_size={payload_size}, actual_payload_size={len(payload)}"
        )
    actual_crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    if actual_crc32 != expected_crc32:
        raise ValueError(
            "Shared memory payload crc32 mismatch: "
            f"expected={expected_crc32:#010x}, actual={actual_crc32:#010x}, payload_size={payload_size}"
        )
    return payload


def shm_write_bytes(payload: bytes, name: str | None = None) -> dict[str, Any]:
    """Write bytes into SharedMemory and return meta dict {name,size}.

    Caller should close the segment; the receiver should unlink.
    """
    packed_payload = _pack_shm_payload(payload)
    try:
        shm = _shm.SharedMemory(create=True, size=len(packed_payload), name=name)
    except FileExistsError:
        if name:
            # If name is specified and exists, unlink it and try again
            try:
                existing = _shm.SharedMemory(name=name)
                existing.unlink()
            except Exception:
                pass
            shm = _shm.SharedMemory(create=True, size=len(packed_payload), name=name)
        else:
            raise

    mv = memoryview(shm.buf)
    mv[: len(packed_payload)] = packed_payload
    del mv
    meta = {
        "name": shm.name,
        "size": len(packed_payload),
        "payload_size": len(payload),
        "header_size": _SHM_HEADER_SIZE,
    }
    try:
        shm.close()
    except Exception as e:
        logger.debug("Failed to close shared memory: %s", e)
    return meta


def shm_read_bytes(meta: dict[str, Any]) -> bytes:
    """Read bytes from SharedMemory by meta {name,size} and cleanup."""
    shm = _shm.SharedMemory(name=meta["name"])  # type: ignore[index]
    mv = memoryview(shm.buf)
    data = bytes(mv[: meta["size"]])
    del mv
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
    return _unpack_shm_payload(data)


def maybe_load_from_ipc_with_metrics(
    container: dict[str, Any], obj_key: str, shm_key: str
) -> tuple[Any, dict[str, float]]:
    """Load object and return (object, metrics) with RX bytes and decode time.

    Metrics keys:
      - rx_transfer_bytes: int
      - rx_decode_time_ms: float
    """
    import time as _time  # local import to avoid overhead at module import

    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    t0 = _time.time()
    if shm_key in container:
        meta = container[shm_key]  # type: ignore[index]
        payload = shm_read_bytes(meta)
        obj = OmniSerializer.deserialize(payload)
        try:
            rx_bytes = int(meta.get("size", len(payload)))  # type: ignore[call-arg]
        except Exception:
            rx_bytes = len(payload)
    else:
        obj = container[obj_key]
        try:
            rx_bytes = len(serialize_obj(obj))
        except Exception:
            rx_bytes = 0
    t1 = _time.time()
    rx_decode_ms = (t1 - t0) * 1000.0
    return obj, {
        "rx_transfer_bytes": int(rx_bytes),
        "rx_decode_time_ms": float(rx_decode_ms),
    }


# Convert OmegaConf/objects to plain dicts
def _to_dict(x: Any) -> dict[str, Any]:
    try:
        if isinstance(x, dict):
            return dict(x)
        return _omega_to_dict(x)
    except Exception:
        try:
            return dict(x)
        except Exception:
            return {}
