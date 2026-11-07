"""
colmap_io.py -- minimal, dependency-free readers for COLMAP sparse models.

Reads   cameras.bin / images.bin / points3D.bin   (or the .txt equivalents)
exactly as documented in the COLMAP file format:
https://colmap.github.io/format.html

Only the fields needed here are kept (no 2D observations). Also: camera centres and viewing
directions from images.bin, and a cameras.bin writer for rescaled PINHOLE cameras.
"""
import os
import struct
from collections import namedtuple

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "num_points2d", "num_points3d"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "track_length"])

# model_id -> (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
_NAME_TO_NPARAMS = {v[0]: v[1] for v in CAMERA_MODELS.values()}


def _read(fid, fmt):
    return struct.unpack("<" + fmt, fid.read(struct.calcsize("<" + fmt)))


# ----------------------------------------------------------------------------- cameras
def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, model_id, w, h = _read(f, "iiQQ")
            name, npar = CAMERA_MODELS[model_id]
            params = _read(f, "d" * npar)
            cams[cid] = Camera(cid, name, w, h, list(params))
    return cams


def read_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            e = line.split()
            cid, model, w, h = int(e[0]), e[1], int(e[2]), int(e[3])
            cams[cid] = Camera(cid, model, w, h, [float(x) for x in e[4:]])
    return cams


# ----------------------------------------------------------------------------- images
def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid = _read(f, "i")[0]
            qvec = _read(f, "dddd")
            tvec = _read(f, "ddd")
            cam_id = _read(f, "i")[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            (npts,) = _read(f, "Q")
            # x, y (double), point3D_id (int64)  -> 24 bytes per observation
            data = f.read(24 * npts)
            ids = struct.unpack("<" + "ddq" * npts, data)[2::3] if npts else ()
            n3d = sum(1 for p in ids if p != -1)
            imgs[iid] = Image(iid, qvec, tvec, cam_id, name.decode("utf-8"), npts, n3d)
    return imgs


def read_images_txt(path):
    """Two lines per image; the second (observations) may be empty for images without points."""
    imgs = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        e = lines[i].split()
        iid = int(e[0])
        qvec = tuple(float(x) for x in e[1:5])
        tvec = tuple(float(x) for x in e[5:8])
        cam_id, name = int(e[8]), e[9]
        obs = lines[i + 1].split() if i + 1 < len(lines) else []
        npts = len(obs) // 3
        n3d = sum(1 for k in range(2, len(obs), 3) if obs[k] != "-1")
        imgs[iid] = Image(iid, qvec, tvec, cam_id, name, npts, n3d)
        i += 2
    return imgs


# ----------------------------------------------------------------------------- points
def read_points3d_bin(path):
    pts = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            pid = _read(f, "Q")[0]
            xyz = _read(f, "ddd")
            rgb = _read(f, "BBB")
            err = _read(f, "d")[0]
            (tl,) = _read(f, "Q")
            f.read(8 * tl)  # (image_id int32, point2D_idx int32) per track element
            pts[pid] = Point3D(pid, xyz, rgb, err, tl)
    return pts


def read_points3d_txt(path):
    pts = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            e = line.split()
            pid = int(e[0])
            xyz = tuple(float(x) for x in e[1:4])
            rgb = tuple(int(x) for x in e[4:7])
            err = float(e[7])
            tl = (len(e) - 8) // 2
            pts[pid] = Point3D(pid, xyz, rgb, err, tl)
    return pts


# ----------------------------------------------------------------------------- model
def read_model(sparse_dir):
    """Return (cameras, images, points3D) from <sparse_dir> holding .bin or .txt files."""
    def pick(base):
        for ext in (".bin", ".txt"):
            p = os.path.join(sparse_dir, base + ext)
            if os.path.exists(p):
                return p, ext
        raise FileNotFoundError(f"{base}.bin/.txt not found in {sparse_dir}")

    pc, ec = pick("cameras")
    pi, ei = pick("images")
    pp, ep = pick("points3D")
    cams = read_cameras_bin(pc) if ec == ".bin" else read_cameras_txt(pc)
    imgs = read_images_bin(pi) if ei == ".bin" else read_images_txt(pi)
    pts = read_points3d_bin(pp) if ep == ".bin" else read_points3d_txt(pp)
    return cams, imgs, pts


def llff_split(names, llffhold=8):
    """
    Reproduce the official 3DGS --eval split (scene/dataset_readers.py):
    names sorted, index % llffhold == 0 -> test, else train.
    """
    names = sorted(names)
    test = [n for i, n in enumerate(names) if i % llffhold == 0]
    train = [n for i, n in enumerate(names) if i % llffhold != 0]
    return train, test


# ----------------------------------------------------------------------------- extras (final project)
def write_cameras_bin(cams, path):
    """Write cameras.bin (any model; params as stored in the Camera tuples)."""
    name_to_id = {v[0]: k for k, v in CAMERA_MODELS.items()}
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid in sorted(cams):
            c = cams[cid]
            f.write(struct.pack("<iiQQ", c.id, name_to_id[c.model], int(c.width), int(c.height)))
            f.write(struct.pack("<" + "d" * len(c.params), *c.params))


def qvec_to_rotmat(q):
    w, x, y, z = q
    return [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]]


def camera_poses(images):
    """{name: (centre (3,), viewing direction (3,))} in world coordinates (COLMAP: x_cam = R x + t)."""
    import numpy as np
    out = {}
    for im in images.values():
        R = np.array(qvec_to_rotmat(im.qvec))
        t = np.array(im.tvec)
        out[im.name] = (-R.T @ t, R.T @ np.array([0.0, 0.0, 1.0]))
    return out
