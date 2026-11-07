"""
pcio.py -- point-cloud readers and writers that need no Open3D.

read_points(path) understands
    .txt .xyz .pts .csv .asc   ASCII columns (x y z [r g b | intensity ...]); delimiter and header sniffed
    .ply                       plyfile (vertex x/y/z, optional red/green/blue)
    .pcd                       ASCII and binary PCD (binary_compressed falls back to Open3D if installed)
    .las .laz                  laspy (.laz needs the lazrs backend: pip install "laspy[lazrs]")
    .npy .npz                  arrays with >= 3 columns, or an npz holding "xyz"
and returns {"xyz": (N,3) float64, "rgb": (N,3) uint8 or None, "n_cols": int, "format": str}.
"""
import io
import os
import re

import numpy as np

_NUM = re.compile(r"^[\s,;\t]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


# ----------------------------------------------------------------------------- ASCII
def _sniff_ascii(path, n_lines=60):
    with open(path, "r", errors="replace") as f:
        lines = [f.readline() for _ in range(n_lines)]
    lines = [l for l in lines if l.strip()]
    skip = 0
    for l in lines:                       # header lines: anything not starting with a number
        if _NUM.match(l):
            break
        skip += 1
    body = lines[skip:]
    if body and len(body[0].split()) == 1 and "," not in body[0] and len(body) > 1 and len(body[1].split()) >= 3:
        skip += 1                         # .pts files start with the point count
        body = body[1:]
    sample = body[0] if body else ""
    if "," in sample:
        delim = ","
    elif ";" in sample:
        delim = ";"
    else:
        delim = None                      # any whitespace
    ncol = len([c for c in (sample.split(delim) if delim else sample.split()) if c.strip()])
    return skip, delim, ncol


def read_ascii(path):
    skip, delim, ncol = _sniff_ascii(path)
    if ncol < 3:
        raise ValueError(f"{path}: fewer than 3 numeric columns")
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=r"\s+" if delim is None else delim, header=None, skiprows=skip,
                         usecols=list(range(ncol)), engine="c" if delim else "python", dtype=np.float64,
                         on_bad_lines="skip")
        arr = df.to_numpy()
    except Exception:
        arr = np.loadtxt(path, delimiter=delim, skiprows=skip, usecols=range(ncol), ndmin=2)
    arr = arr[np.isfinite(arr[:, :3]).all(1)]
    rgb = None
    if ncol >= 6:
        c = arr[:, 3:6]
        if c.size and c.min() >= 0 and c.max() <= 255 and np.allclose(c, np.round(c)):
            rgb = c.astype(np.uint8)
    return {"xyz": arr[:, :3].astype(np.float64), "rgb": rgb, "n_cols": ncol, "format": "ascii"}


# ----------------------------------------------------------------------------- PLY
def read_ply(path):
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = None
    for r, g, b in (("red", "green", "blue"), ("r", "g", "b")):
        if r in names:
            rgb = np.stack([v[r], v[g], v[b]], 1)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            break
    return {"xyz": xyz, "rgb": rgb, "n_cols": len(names), "format": "ply"}


def read_ply_mesh(path):
    """(vertices (V,3) float64, faces (F,3) int64) of a triangle mesh PLY."""
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"].data
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    faces = np.zeros((0, 3), np.int64)
    if "face" in ply:
        fd = ply["face"].data
        key = "vertex_indices" if "vertex_indices" in fd.dtype.names else fd.dtype.names[0]
        tri = [f for f in fd[key] if len(f) >= 3]
        if tri:
            faces = np.array([f[:3] for f in tri], dtype=np.int64)
    return verts, faces


def write_ply(path, xyz, rgb=None, scalars=None):
    """Binary PLY with optional uint8 colour and float32 scalar fields."""
    from plyfile import PlyData, PlyElement
    n = len(xyz)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    scalars = scalars or {}
    for k, val in scalars.items():
        fields.append((k, "i4" if np.issubdtype(np.asarray(val).dtype, np.integer) else "f4"))
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for k, val in scalars.items():
        arr[k] = val
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


# ----------------------------------------------------------------------------- PCD
_PCD_TYPES = {("F", 4): np.float32, ("F", 8): np.float64, ("U", 1): np.uint8, ("U", 2): np.uint16,
              ("U", 4): np.uint32, ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32}


def read_pcd(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            k, *vals = line.split()
            header[k.upper()] = vals
            if k.upper() == "DATA":
                break
        data_kind = header["DATA"][0].lower()
        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        npts = int(header["POINTS"][0])
        if data_kind == "ascii":
            arr = np.loadtxt(io.TextIOWrapper(f, encoding="ascii"), ndmin=2)
            cols = {}
            j = 0
            for name, c in zip(fields, counts):
                cols[name] = arr[:, j:j + c]
                j += c
        elif data_kind == "binary":
            dt = np.dtype([(n if c == 1 else n, _PCD_TYPES[(t, s)], (c,) if c > 1 else ())
                           for n, s, t, c in zip(fields, sizes, types, counts)])
            rec = np.frombuffer(f.read(npts * dt.itemsize), dtype=dt, count=npts)
            cols = {n: rec[n].reshape(npts, -1) for n in fields}
        else:
            try:
                import open3d as o3d
            except ImportError as e:
                raise ValueError(f"{path}: binary_compressed PCD needs open3d (pip install open3d)") from e
            pc = o3d.io.read_point_cloud(path)
            xyz = np.asarray(pc.points, dtype=np.float64)
            rgb = (np.asarray(pc.colors) * 255).astype(np.uint8) if pc.has_colors() else None
            return {"xyz": xyz, "rgb": rgb, "n_cols": len(fields), "format": "pcd-compressed"}
    xyz = np.concatenate([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    rgb = None
    if "rgb" in cols or "rgba" in cols:
        packed = cols.get("rgb", cols.get("rgba")).astype(np.float32).ravel().view(np.uint32)
        rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], 1).astype(np.uint8)
    return {"xyz": xyz, "rgb": rgb, "n_cols": len(fields), "format": f"pcd-{data_kind}"}


# ----------------------------------------------------------------------------- LAS
def read_las(path):
    import laspy
    las = laspy.read(path)
    xyz = np.stack([las.x, las.y, las.z], 1).astype(np.float64)
    rgb = None
    if all(hasattr(las, c) for c in ("red", "green", "blue")):
        c = np.stack([las.red, las.green, las.blue], 1).astype(np.float64)
        if c.max() > 0:
            rgb = (c / (65535.0 if c.max() > 255 else 255.0) * 255).astype(np.uint8)
    return {"xyz": xyz, "rgb": rgb, "n_cols": len(las.point_format.dimension_names), "format": "las"}


# ----------------------------------------------------------------------------- dispatch
def read_points(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".xyz", ".pts", ".csv", ".asc"):
        return read_ascii(path)
    if ext == ".ply":
        return read_ply(path)
    if ext == ".pcd":
        return read_pcd(path)
    if ext in (".las", ".laz"):
        return read_las(path)
    if ext == ".npy":
        a = np.load(path)
        return {"xyz": a[:, :3].astype(np.float64), "rgb": None, "n_cols": a.shape[1], "format": "npy"}
    if ext == ".npz":
        z = np.load(path)
        a = z["xyz"] if "xyz" in z else z[list(z.keys())[0]]
        return {"xyz": a[:, :3].astype(np.float64), "rgb": None, "n_cols": a.shape[1], "format": "npz"}
    raise ValueError(f"unsupported point file: {path}")


def read_gaussians_ply(path):
    """3DGS / 2DGS point_cloud.ply -> (centres (N,3), opacity in [0,1] (N,))."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64))) if "opacity" in v.dtype.names else np.ones(len(xyz))
    return xyz, op


def ply_vertex_count(path):
    """Vertex count from the PLY header only (fast, for Gaussian counts)."""
    with open(path, "rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None
