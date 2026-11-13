"""Small, dependency-light helpers for ScanNet++ rendered-depth evaluation."""

import struct
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_PIXEL_TYPES = {1: np.dtype("<f2"), 2: np.dtype("<f4")}
_LINES_PER_CHUNK = {0: 1, 2: 1, 3: 16}


def _exr_header(data):
    if data[:4] != b"\x76\x2f\x31\x01":
        raise ValueError("Not an OpenEXR file")
    if struct.unpack_from("<I", data, 4)[0] & 0x1A00:
        raise ValueError("Only single-part scanline OpenEXR files are supported")
    attributes, position = {}, 8
    while data[position] != 0:
        end = data.index(b"\0", position)
        name = data[position:end].decode()
        position = data.index(b"\0", end + 1) + 1
        size = struct.unpack_from("<i", data, position)[0]
        attributes[name] = data[position + 4:position + 4 + size]
        position += 4 + size
    return attributes, position + 1


def _exr_channels(chlist):
    channels, position = [], 0
    while chlist[position] != 0:
        end = chlist.index(b"\0", position)
        name = chlist[position:end].decode()
        pixel_type, _, x_sampling, y_sampling = struct.unpack_from("<i4s2i", chlist, end + 1)
        if (x_sampling, y_sampling) != (1, 1):
            raise ValueError("Subsampled OpenEXR channels are not supported")
        channels.append((name, pixel_type))
        position = end + 17
    return channels


def _exr_unzip(chunk, expected_size):
    if len(chunk) == expected_size:
        return chunk
    values = np.frombuffer(zlib.decompress(chunk), np.uint8)
    values = np.cumsum(values.astype(np.int64) - 128, dtype=np.int64)
    values = ((values + 128) & 0xFF).astype(np.uint8)
    output = np.empty_like(values)
    half = (len(values) + 1) // 2
    output[0::2], output[1::2] = values[:half], values[half:]
    return output.tobytes()


def read_exr_depth(path):
    """Read the single half/float channel in a ScanNet++ depth EXR."""
    data = Path(path).read_bytes()
    attributes, position = _exr_header(data)
    channels = _exr_channels(attributes["channels"])
    if len(channels) != 1 or channels[0][1] not in _PIXEL_TYPES:
        raise ValueError(f"Expected one half or float channel, got {channels}: {path}")
    dtype = _PIXEL_TYPES[channels[0][1]]
    compression = attributes["compression"][0]
    if compression not in _LINES_PER_CHUNK:
        raise ValueError(f"Unsupported OpenEXR compression {compression}: {path}")
    x_min, y_min, x_max, y_max = struct.unpack("<4i", attributes["dataWindow"])
    width, height = x_max - x_min + 1, y_max - y_min + 1
    lines = _LINES_PER_CHUNK[compression]
    offsets = struct.unpack_from(f"<{-(-height // lines)}Q", data, position)
    image = np.empty((height, width), dtype)
    for offset in offsets:
        y, size = struct.unpack_from("<ii", data, offset)
        row = y - y_min
        count = min(lines, height - row)
        expected = count * width * dtype.itemsize
        chunk = data[offset + 8:offset + 8 + size]
        raw = chunk if compression == 0 else _exr_unzip(chunk, expected)
        image[row:row + count] = np.frombuffer(raw, dtype, count * width).reshape(count, width)
    return image.astype(np.float32)


def lift_depth_array_to_world(depth, valid, intrinsics, camera_to_world, *, stride, max_depth):
    """Lift camera-z depth samples into world coordinates."""
    if stride < 1:
        raise ValueError("point stride must be positive")
    rows = np.arange(0, depth.shape[0], stride)
    columns = np.arange(0, depth.shape[1], stride)
    grid_x, grid_y = np.meshgrid(columns, rows)
    sampled = depth[np.ix_(rows, columns)]
    keep = valid[np.ix_(rows, columns)] & np.isfinite(sampled)
    keep &= (sampled > 0) & (sampled <= max_depth)
    z = sampled[keep]
    x = (grid_x[keep] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (grid_y[keep] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    camera_points = np.column_stack((x, y, z))
    points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    return points.astype(np.float32), grid_y[keep], grid_x[keep]


def cloud_scores(points, gt_points, tolerances):
    """Symmetric nearest-neighbour accuracy, completeness and F1."""
    tolerances = np.asarray(tolerances, dtype=np.float64)
    if tolerances.ndim != 1 or not len(tolerances) or np.any(tolerances <= 0):
        raise ValueError("tolerances must be positive")
    if not len(points) or not len(gt_points):
        zeros = [0.0] * len(tolerances)
        return {"accuracy": zeros, "completeness": zeros, "f1_score": zeros}
    limit = tolerances.max()
    to_gt = cKDTree(gt_points).query(points, distance_upper_bound=limit, workers=-1)[0]
    to_points = cKDTree(points).query(gt_points, distance_upper_bound=limit, workers=-1)[0]
    accuracy = (to_gt[:, None] <= tolerances).mean(0)
    completeness = (to_points[:, None] <= tolerances).mean(0)
    denominator = np.where(accuracy + completeness > 0, accuracy + completeness, 1.0)
    f1 = 2 * accuracy * completeness / denominator
    return {"accuracy": accuracy.tolist(), "completeness": completeness.tolist(),
            "f1_score": f1.tolist()}
