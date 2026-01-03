#!/usr/bin/env python3
"""
bulk_cr2_wb_export.py

Bulk-process Canon CR2 RAW files:
- Apply a global white balance from Temperature (K) and Tint
  * Tint can be provided in RawTherapee-like form (--tint-rt, multiplier, 1.0 neutral)
  * or in an offset form (--tint-ev, 0 neutral, negative greener, positive magenta)
- Apply exposure compensation in EV stops (--exposure-ev), mapped to rawpy/LibRaw exp_shift (linear scale)
- Export as sRGB JPEG with 4:4:4 (no chroma subsampling) at chosen quality

Install:
  pip install rawpy pillow numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rawpy
from PIL import Image


# ---- White point / WB helpers ----

def cct_to_xy(cct_kelvin: float) -> Tuple[float, float]:
    """Approximate CCT (Kelvin) -> CIE 1931 chromaticity (x, y)."""
    T = float(cct_kelvin)
    T = max(1667.0, min(25000.0, T))

    if T <= 4000.0:
        x = (-0.2661239e9 / (T**3)
             - 0.2343580e6 / (T**2)
             + 0.8776956e3 / T
             + 0.179910)
    else:
        x = (-3.0258469e9 / (T**3)
             + 2.1070379e6 / (T**2)
             + 0.2226347e3 / T
             + 0.240390)

    if T <= 2222.0:
        y = (-1.1063814 * (x**3)
             - 1.34811020 * (x**2)
             + 2.18555832 * x
             - 0.20219683)
    elif T <= 4000.0:
        y = (-0.9549476 * (x**3)
             - 1.37418593 * (x**2)
             + 2.09137015 * x
             - 0.16748867)
    else:
        y = (3.0817580 * (x**3)
             - 5.87338670 * (x**2)
             + 3.75112997 * x
             - 0.37001483)

    return float(x), float(y)


def xy_to_XYZ(x: float, y: float, Y: float = 1.0) -> np.ndarray:
    if y <= 0.0:
        raise ValueError("Invalid y chromaticity (must be > 0).")
    X = (x * Y) / y
    Z = ((1.0 - x - y) * Y) / y
    return np.array([X, Y, Z], dtype=np.float64)


def tint_ev_to_rt_multiplier(tint_ev: float) -> float:
    """
    Convert offset-style tint into a RawTherapee-like multiplier.
      tint_ev = 0 -> 1.0
      tint_ev < 0 -> greener -> multiplier > 1.0
      tint_ev > 0 -> magenta -> multiplier < 1.0
    """
    return float(2.0 ** (-float(tint_ev) / 100.0))


def compute_user_wb_from_temp_and_tint(
    cam_matrix: np.ndarray,
    temp_k: float,
    tint_rt: float,
    *,
    reference_xy: Tuple[float, float] = (0.3127, 0.3290),  # D65
) -> Tuple[float, float, float, float]:
    """
    Convert (temp_k, tint_rt) into rawpy's user_wb multipliers (R, G1, B, G2).

    Handles camera->XYZ matrices of shapes:
      - (4,3)  (LibRaw cam_xyz style; 4 channels -> XYZ)
      - (3,4)  (rawpy color_matrix style; XYZ = M @ cam4)
      - (3,3)  (RGB -> XYZ; will duplicate green for G2)

    tint_rt is RawTherapee-style: 1.0 neutral, >1 greener, <1 more magenta.
    """
    mat = np.asarray(cam_matrix, dtype=np.float64)

    if tint_rt <= 0.0 or not np.isfinite(tint_rt):
        raise ValueError(f"tint_rt must be a finite positive number, got {tint_rt}")

    # Build a 3xN matrix M such that XYZ = M @ camN
    if mat.shape == (4, 3):
        # LibRaw "cam_xyz": rows are channels, cols are XYZ -> transpose to 3x4
        M = mat.T  # 3x4
        n_ch = 4
    elif mat.shape == (3, 4):
        M = mat  # 3x4
        n_ch = 4
    elif mat.shape == (3, 3):
        M = mat  # 3x3
        n_ch = 3
    else:
        raise ValueError(f"Unsupported camera matrix shape: {mat.shape} (expected 4x3, 3x4, or 3x3)")

    # Desired illuminant (from temp) and reference white (D65)
    x_i, y_i = cct_to_xy(temp_k)
    XYZ_illum = xy_to_XYZ(x_i, y_i, Y=1.0)

    x_r, y_r = reference_xy
    XYZ_ref = xy_to_XYZ(x_r, y_r, Y=1.0)

    # Map XYZ -> camera channels
    if n_ch == 3:
        xyz2cam = np.linalg.inv(M)          # 3x3
        cam_illum = xyz2cam @ XYZ_illum     # 3
        cam_ref = xyz2cam @ XYZ_ref         # 3
        # Expand to 4 WB coefficients: (R, G1, B, G2)
        cam_illum = np.array([cam_illum[0], cam_illum[1], cam_illum[2], cam_illum[1]], dtype=np.float64)
        cam_ref   = np.array([cam_ref[0],   cam_ref[1],   cam_ref[2],   cam_ref[1]],   dtype=np.float64)
    else:
        # 3x4: use least-squares inverse (pseudoinverse)
        xyz2cam = np.linalg.pinv(M)         # 4x3
        cam_illum = xyz2cam @ XYZ_illum     # 4
        cam_ref = xyz2cam @ XYZ_ref         # 4

        # If LibRaw provided a "dummy" 4th channel (e.g., zeros), copy green.
        eps0 = 1e-10
        if abs(cam_illum[3]) < eps0 and abs(cam_ref[3]) < eps0:
            cam_illum[3] = cam_illum[1]
            cam_ref[3] = cam_ref[1]

    # Compute per-channel gains to map illuminant white -> reference white
    eps = 1e-12
    denom = np.where(np.abs(cam_illum) < eps, eps, cam_illum)
    gains = cam_ref / denom
    gains = np.clip(gains, 1e-6, 1e6)

    # Normalize so average green gain is 1.0 (scale-invariant anyway)
    g_mean = 0.5 * (gains[1] + gains[3])
    if not np.isfinite(g_mean) or g_mean <= 0:
        g_mean = 1.0
    gains = gains / g_mean

    # Apply RawTherapee-style tint as scaling on both greens
    gains[1] *= float(tint_rt)
    gains[3] *= float(tint_rt)

    return (float(gains[0]), float(gains[1]), float(gains[2]), float(gains[3]))


# ---- Exposure helpers ----

def ev_to_exp_shift(ev: float) -> float:
    """
    Convert exposure compensation in EV (stops) to LibRaw/rawpy exp_shift (linear).
    exp_shift = 2^EV
    """
    return float(2.0 ** float(ev))


def clamp_exp_shift(exp_shift: float) -> float:
    """
    LibRaw/rawpy document exp_shift usable range as 0.25 .. 8.0.
    """
    return float(max(0.25, min(8.0, exp_shift)))


# ---- File processing ----

def iter_cr2_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    exts = {".cr2"}
    if recursive:
        yield from (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    else:
        yield from (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def select_demosaic(name: str) -> rawpy.DemosaicAlgorithm:
    n = name.strip().upper()
    mapping = {
        "AHD": rawpy.DemosaicAlgorithm.AHD,
        "AMAZE": rawpy.DemosaicAlgorithm.AMAZE,
        "DCB": rawpy.DemosaicAlgorithm.DCB,
        "DHT": rawpy.DemosaicAlgorithm.DHT,
        "LINEAR": rawpy.DemosaicAlgorithm.LINEAR,
        "PPG": rawpy.DemosaicAlgorithm.PPG,
        "VNG": rawpy.DemosaicAlgorithm.VNG,
    }
    if n not in mapping:
        raise ValueError(f"Unsupported demosaic '{name}'. Choose one of: {', '.join(mapping.keys())}")
    return mapping[n]


def atomic_save_jpeg(img: Image.Image, out_path: Path, *, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".jpg", dir=str(out_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        img.save(
            str(tmp_path),
            format="JPEG",
            quality=int(quality),
            subsampling=0,   # 4:4:4
            optimize=True,
        )
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists() and tmp_path != out_path:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def process_one(
    in_path: Path,
    out_path: Path,
    *,
    temp_k: float,
    tint_rt: float,
    exposure_ev: float,
    quality: int,
    demosaic: rawpy.DemosaicAlgorithm,
) -> None:
    exp_shift = ev_to_exp_shift(exposure_ev)
    exp_shift_clamped = clamp_exp_shift(exp_shift)

    with rawpy.imread(str(in_path)) as raw:
        cam2xyz = np.array(raw.rgb_xyz_matrix, dtype=np.float64)

        user_wb = compute_user_wb_from_temp_and_tint(
            cam_matrix=cam2xyz,
            temp_k=temp_k,
            tint_rt=tint_rt,
        )

        rgb8 = raw.postprocess(
            user_wb=user_wb,
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,  # keep deterministic; exposure handled via exp_shift
            exp_shift=exp_shift_clamped,             # linear exposure multiplier
            exp_preserve_highlights=0.0,             # keep simple; add CLI if you want
            output_color=rawpy.ColorSpace.sRGB,
            demosaic_algorithm=demosaic,
            output_bps=8,
        )

    img = Image.fromarray(rgb8, mode="RGB")
    atomic_save_jpeg(img, out_path, quality=quality)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk convert Canon CR2 to sRGB JPEG (4:4:4) with global Temperature(K)+Tint and exposure EV."
    )
    ap.add_argument("--input", "-i", required=True, type=Path, help="Input directory containing .CR2 files.")
    ap.add_argument("--output", "-o", required=True, type=Path, help="Output directory for .jpg files.")
    ap.add_argument("--temp", required=True, type=float, help="White balance temperature in Kelvin, e.g. 5500.")
    ap.add_argument("--quality", "-q", default=92, type=int,
                    help="JPEG quality. Practical range is typically 1..95.")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories.")
    ap.add_argument("--demosaic", default="AHD",
                    help="Demosaic: AHD, AMAZE, DCB, DHT, LINEAR, PPG, VNG.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output JPEGs.")

    ap.add_argument(
        "--exposure-ev",
        type=float,
        default=0.0,
        help="Exposure compensation in EV (stops). Example: +1.0 doubles exposure, -1.0 halves it."
    )

    tint_grp = ap.add_mutually_exclusive_group(required=True)
    tint_grp.add_argument(
        "--tint-rt",
        type=float,
        help="RawTherapee-style tint multiplier: 1.0 neutral; >1 greener; <1 more magenta."
    )
    tint_grp.add_argument(
        "--tint-ev",
        type=float,
        help="Offset-style tint: 0 neutral; negative greener; positive magenta. Mapped to tint-rt via 2^(-t/100)."
    )

    args = ap.parse_args()

    if not args.input.exists() or not args.input.is_dir():
        print(f"ERROR: input is not a directory: {args.input}", file=sys.stderr)
        return 2

    quality = max(1, min(95, int(args.quality)))
    demosaic = select_demosaic(args.demosaic)

    tint_rt = float(args.tint_rt) if args.tint_rt is not None else tint_ev_to_rt_multiplier(float(args.tint_ev))

    # Warn if exp_shift clamps (keeps behavior explicit)
    exp_shift = ev_to_exp_shift(float(args.exposure_ev))
    exp_shift_clamped = clamp_exp_shift(exp_shift)
    if abs(exp_shift - exp_shift_clamped) > 1e-12:
        print(
            f"WARNING: --exposure-ev {args.exposure_ev} maps to exp_shift {exp_shift:.4g}, "
            f"clamped to {exp_shift_clamped:.4g} (LibRaw/rawpy documented range 0.25..8.0).",
            file=sys.stderr,
        )

    files = sorted(iter_cr2_files(args.input, args.recursive))
    if not files:
        print("No .CR2 files found.", file=sys.stderr)
        return 1

    total = len(files)
    for idx, in_path in enumerate(files, start=1):
        rel = in_path.relative_to(args.input) if args.recursive else Path(in_path.name)
        out_path = (args.output / rel).with_suffix(".jpg")

        if out_path.exists() and not args.overwrite:
            print(f"[{idx}/{total}] SKIP exists: {out_path}")
            continue

        print(f"[{idx}/{total}] {in_path} -> {out_path}")
        try:
            process_one(
                in_path=in_path,
                out_path=out_path,
                temp_k=float(args.temp),
                tint_rt=tint_rt,
                exposure_ev=float(args.exposure_ev),
                quality=quality,
                demosaic=demosaic,
            )
        except Exception as e:
            print(f"ERROR processing {in_path}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
