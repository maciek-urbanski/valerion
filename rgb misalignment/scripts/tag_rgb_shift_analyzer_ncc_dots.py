#!/usr/bin/env python3
"""
Tag-based RGB shift analyzer (multi-image). Uses bounded NCC on a dot-texture region.

Pattern/layout assumed (must match generator):
- Pattern size: 3840x2160
- Grid: 16x9 cells (CELL=240)
- Plate: 224x224 at (PAD_B=8) inside each cell
- Tag pad and marker: TAG=120 centered inside plate (PAD_W=52)
- Dot texture exists on the plate OUTSIDE the tag pad and is used for shift estimation.

Key behavior:
- Detect ArUco markers on grayscale (optionally inverted).
- For each visible marker:
    - Use its corners to warp the *PLATE* region into a canonical 224x224 patch.
    - Compute bounded NCC shifts (±max_shift pattern px) for R->G and B->G using:
        * gradient magnitude
        * a mask that excludes the tag pad region (only dots + plate area used)

Outputs:
- vectors_overlay.png (one arrow per tag; R->G red, B->G blue)
- shifts.npz (per-image and median arrays + errors)
- tag_summary.csv, per_image_report.csv
"""

import argparse
import csv
import glob
import math
import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import cv2


# -------------------------
# Layout constants
# -------------------------
PAT_W, PAT_H = 3840, 2160
CELL, PLATE, TAG = 240, 224, 120
PAD_B = (CELL - PLATE) // 2      # 8
PAD_W = (PLATE - TAG) // 2       # 52
NX, NY = PAT_W // CELL, PAT_H // CELL  # 16, 9

ARUCO_DICT = cv2.aruco.DICT_6X6_250


# -------------------------
# ArUco detection helpers
# -------------------------
def make_aruco_detector(dict_id: int, detect_inverted: bool = False):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dict_id)

    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    if hasattr(params, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if detect_inverted and hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        return detector, dictionary, params
    return None, dictionary, params


def detect_markers(gray_u8: np.ndarray, dict_id: int, detect_inverted: bool = False):
    aruco = cv2.aruco
    detector, dictionary, params = make_aruco_detector(dict_id, detect_inverted=detect_inverted)

    if detector is not None:
        corners_list, ids, rejected = detector.detectMarkers(gray_u8)
    else:
        corners_list, ids, rejected = aruco.detectMarkers(gray_u8, dictionary, parameters=params)

    if ids is None:
        return [], np.array([], dtype=np.int32), rejected
    return corners_list, ids.astype(np.int32).ravel(), rejected


def detect_markers_robust(gray_u8: np.ndarray, dict_id: int, detect_inverted: bool,
                          max_dim: int = 1600, use_clahe: bool = True):
    """
    Robust ArUco detection for very high-res close-up images.
    - Downscales to max_dim for detection stability + speed
    - Uses larger adaptive threshold windows appropriate for large markers
    - Optionally applies CLAHE to stabilize local contrast
    Returns: corners_list, ids (ravel int32), rejected_list
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dict_id)

    # Downscale for detection
    h, w = gray_u8.shape[:2]
    s = 1.0
    if max(h, w) > max_dim:
        s = max_dim / float(max(h, w))
        gray = cv2.resize(gray_u8, (int(round(w * s)), int(round(h * s))),
                          interpolation=cv2.INTER_AREA)
    else:
        gray = gray_u8

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Mild blur can help adaptive threshold on projector captures
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Params
    params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()

    # Close-up scale: widen adaptive threshold windows (must be odd)
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 31
        params.adaptiveThreshWinSizeMax = 401
        params.adaptiveThreshWinSizeStep = 20
    if hasattr(params, "adaptiveThreshConstant"):
        params.adaptiveThreshConstant = 7

    # Contour / quad acceptance
    if hasattr(params, "polygonalApproxAccuracyRate"):
        params.polygonalApproxAccuracyRate = 0.05
    if hasattr(params, "minCornerDistanceRate"):
        params.minCornerDistanceRate = 0.01
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = 0.01
    if hasattr(params, "maxMarkerPerimeterRate"):
        params.maxMarkerPerimeterRate = 4.0

    # Refinement / robustness knobs (only if available in your OpenCV build)
    if hasattr(params, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "useAruco3Detection"):
        params.useAruco3Detection = True
    if detect_inverted and hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True

    # Detect
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners_list, ids, rejected = detector.detectMarkers(gray)
    else:
        corners_list, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        return [], np.array([], dtype=np.int32), rejected

    ids = ids.astype(np.int32).ravel()

    # Scale corners back to original image coordinates if downscaled
    if s != 1.0:
        invs = 1.0 / s
        corners_list = [c.astype(np.float32) * invs for c in corners_list]
        if rejected is not None:
            rejected = [c.astype(np.float32) * invs for c in rejected]

    return corners_list, ids, rejected

def detect_markers_multiscale_closeup(
    gray_u8: np.ndarray,
    dict_id: int,
    detect_inverted: bool = False,
    max_dims=(6400, 6000, 4500, 3200, 2400, 1800),
    use_clahe: bool = True,
    median_ksize: int = 5,
):
    """
    Close-up robust ArUco detection for high-res projector captures with dot textures.
    Key behaviors:
      - medianBlur BEFORE resize (suppresses dot texture / aliasing into tag)
      - multi-scale pyramid that includes 6000 (as discovered empirically)
      - increased minMarkerPerimeterRate to avoid dot-quads
    Returns: corners_list, ids(int32 ravel), rejected_list, scale_used
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dict_id)

    # Preprocess once at full-res
    base = gray_u8
    if median_ksize and median_ksize >= 3:
        base = cv2.medianBlur(base, median_ksize)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = clahe.apply(base)

    # A mild blur helps stabilize adaptive thresholding
    base = cv2.GaussianBlur(base, (3, 3), 0)

    h0, w0 = base.shape[:2]

    best = (0, -1, None, None, None, 1.0)  # detected, rejected, corners, ids, rej, scale

    for max_dim in max_dims:
        s = 1.0
        if max(h0, w0) > max_dim:
            s = max_dim / float(max(h0, w0))
            g = cv2.resize(base, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        else:
            g = base

        # Detector parameters tuned for your scenario (large markers, dot texture)
        params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()

        # Adaptive threshold windows (odd sizes). Keep wide enough for close-ups.
        if hasattr(params, "adaptiveThreshWinSizeMin"):
            params.adaptiveThreshWinSizeMin = 31
            params.adaptiveThreshWinSizeMax = 401
            params.adaptiveThreshWinSizeStep = 20
        if hasattr(params, "adaptiveThreshConstant"):
            params.adaptiveThreshConstant = 7

        # Strongly reduce false candidates from dot texture by requiring larger perimeter
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.08  # try 0.05..0.15 if needed
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 4.0

        # Candidate polygon approximation
        if hasattr(params, "polygonalApproxAccuracyRate"):
            params.polygonalApproxAccuracyRate = 0.05

        # Decode robustness
        if hasattr(params, "perspectiveRemovePixelPerCell"):
            params.perspectiveRemovePixelPerCell = 20
        if hasattr(params, "perspectiveRemoveIgnoredMarginPerCell"):
            params.perspectiveRemoveIgnoredMarginPerCell = 0.15
        if hasattr(params, "errorCorrectionRate"):
            params.errorCorrectionRate = 1.0

        # Refinement / robustness
        if hasattr(params, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        if hasattr(params, "useAruco3Detection"):
            params.useAruco3Detection = True
        if detect_inverted and hasattr(params, "detectInvertedMarker"):
            params.detectInvertedMarker = True

        # Detect
        if hasattr(aruco, "ArucoDetector"):
            det = aruco.ArucoDetector(dictionary, params)
            corners, ids, rejected = det.detectMarkers(g)
        else:
            corners, ids, rejected = aruco.detectMarkers(g, dictionary, parameters=params)

        detected = 0 if ids is None else len(ids)
        rejcnt = 0 if rejected is None else len(rejected)

        # Prefer more detections; tie-break on more candidates found
        if (detected, rejcnt) > (best[0], best[1]):
            best = (detected, rejcnt, corners, ids, rejected, s)

    detected, rejcnt, corners, ids, rejected, s = best
    if ids is None:
        return [], np.array([], dtype=np.int32), rejected, s

    ids = ids.astype(np.int32).ravel()

    # Scale corners back to original coordinates if downscaled
    if s != 1.0:
        invs = 1.0 / s
        corners = [c.astype(np.float32) * invs for c in corners]
        if rejected is not None:
            rejected = [c.astype(np.float32) * invs for c in rejected]

    return corners, ids, rejected, s


def _robust_center_and_inliers(v: np.ndarray, k: float = 3.0):
    """
    v: (N,2) float32
    Returns (center(2,), inlier_mask(N,))
    Uses vector-median-ish center (component-wise median) + MAD on radial residuals.
    """
    if v.shape[0] == 0:
        return np.array([np.nan, np.nan], np.float32), np.zeros((0,), bool)

    c = np.nanmedian(v, axis=0).astype(np.float32)
    r = np.sqrt((v[:, 0] - c[0])**2 + (v[:, 1] - c[1])**2).astype(np.float32)

    r_med = float(np.nanmedian(r))
    mad = float(np.nanmedian(np.abs(r - r_med))) + 1e-6
    thr = r_med + k * 1.4826 * mad  # 1.4826 makes MAD ~ std for Gaussian
    inl = r <= thr
    return c, inl


def robust_weighted_mean_2d(v: np.ndarray, w: np.ndarray | None = None, k: float = 3.0):
    """
    v: (N,2), w: (N,)
    Returns dict with:
      mean(2,), median(2,), inliers(int), total(int), rms_inlier(float), mad_thr(float)
    """
    v = v.astype(np.float32)
    if w is None:
        w = np.ones((v.shape[0],), np.float32)
    else:
        w = w.astype(np.float32)

    # Basic cleanup
    valid = np.isfinite(v[:, 0]) & np.isfinite(v[:, 1]) & np.isfinite(w) & (w > 0)
    v = v[valid]
    w = w[valid]

    out = {
        "mean": np.array([np.nan, np.nan], np.float32),
        "median": np.array([np.nan, np.nan], np.float32),
        "inliers": 0,
        "total": int(valid.sum()),
        "rms_inlier": np.nan,
        "mad_thr": np.nan,
    }
    if v.shape[0] == 0:
        return out

    med = np.nanmedian(v, axis=0).astype(np.float32)
    c, inl = _robust_center_and_inliers(v, k=k)

    # compute the threshold used (for reporting)
    r = np.sqrt((v[:, 0] - c[0])**2 + (v[:, 1] - c[1])**2).astype(np.float32)
    r_med = float(np.nanmedian(r))
    mad = float(np.nanmedian(np.abs(r - r_med))) + 1e-6
    thr = r_med + k * 1.4826 * mad

    out["median"] = med
    out["inliers"] = int(inl.sum())
    out["mad_thr"] = float(thr)

    if out["inliers"] == 0:
        return out

    vv = v[inl]
    ww = w[inl]
    ww = ww / (float(np.sum(ww)) + 1e-12)

    mean = (vv * ww[:, None]).sum(axis=0).astype(np.float32)
    out["mean"] = mean

    rr = np.sqrt((vv[:, 0] - mean[0])**2 + (vv[:, 1] - mean[1])**2).astype(np.float32)
    out["rms_inlier"] = float(np.sqrt(np.mean(rr**2))) if rr.size else np.nan
    return out


def global_shift_from_tag_medians(shift_med: np.ndarray, count: np.ndarray, err_med: np.ndarray | None = None):
    """
    Uses per-tag medians (NY,NX,2) to compute a single global shift vector.
    weights: count, optionally down-weight by (1/(err+eps)) if err is available and finite.
    """
    valid = (count > 0) & np.isfinite(shift_med[..., 0]) & np.isfinite(shift_med[..., 1])
    v = shift_med[valid].reshape(-1, 2).astype(np.float32)
    w = count[valid].reshape(-1).astype(np.float32)

    if err_med is not None:
        e = err_med[valid].reshape(-1).astype(np.float32)
        good_e = np.isfinite(e) & (e > 0)
        # down-weight noisy tags; keep bounded so a single tiny error can't dominate
        if np.any(good_e):
            inv = np.zeros_like(e, np.float32)
            inv[good_e] = 1.0 / (e[good_e] + 1e-3)
            inv = np.clip(inv, 0.0, 50.0)
            w = w * np.where(good_e, inv, 1.0)

    return robust_weighted_mean_2d(v, w=w, k=3.0)


def global_shift_from_per_image(shift_per_image: np.ndarray):
    """
    shift_per_image: (Nimg, NY, NX, 2)
    Computes robust mean per image (across observed tags), then median across images.
    Returns dict with:
      per_image_means (Nimg,2) with NaNs for images with insufficient tags
      global_median (2,)
      global_mean_of_means (2,)
      n_images_used
    """
    nimg = shift_per_image.shape[0]
    per_means = np.full((nimg, 2), np.nan, np.float32)

    for k in range(nimg):
        a = shift_per_image[k].reshape(-1, 2)
        valid = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
        v = a[valid].astype(np.float32)
        if v.shape[0] < 3:
            continue
        per_means[k] = robust_weighted_mean_2d(v, w=None, k=3.0)["mean"]

    valid_img = np.isfinite(per_means[:, 0]) & np.isfinite(per_means[:, 1])
    used = int(valid_img.sum())
    if used == 0:
        return {
            "per_image_means": per_means,
            "global_median": np.array([np.nan, np.nan], np.float32),
            "global_mean_of_means": np.array([np.nan, np.nan], np.float32),
            "n_images_used": 0,
        }

    global_median = np.nanmedian(per_means[valid_img], axis=0).astype(np.float32)
    global_mean = np.nanmean(per_means[valid_img], axis=0).astype(np.float32)
    return {
        "per_image_means": per_means,
        "global_median": global_median,
        "global_mean_of_means": global_mean,
        "n_images_used": used,
    }


# -------------------------
# Geometry mapping
# -------------------------
def id_to_ij(mid: int) -> Tuple[int, int]:
    return int(mid % NX), int(mid // NX)


def ij_to_id(i: int, j: int) -> int:
    return int(j * NX + i)


def marker_corners_in_pattern(i: int, j: int) -> np.ndarray:
    x0 = i * CELL + PAD_B + PAD_W
    y0 = j * CELL + PAD_B + PAD_W
    return np.array([[x0, y0],
                     [x0 + TAG, y0],
                     [x0 + TAG, y0 + TAG],
                     [x0, y0 + TAG]], dtype=np.float32)


def plate_topleft_in_pattern(i: int, j: int) -> Tuple[float, float]:
    return (i * CELL + PAD_B, j * CELL + PAD_B)


def plate_center_in_pattern(i: int, j: int) -> Tuple[float, float]:
    x0, y0 = plate_topleft_in_pattern(i, j)
    return (x0 + PLATE / 2.0, y0 + PLATE / 2.0)


# -------------------------
# Background for overlay
# -------------------------
def render_background(dict_id: int) -> np.ndarray:
    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(dict_id)
    img = np.zeros((PAT_H, PAT_W), np.uint8)

    def gen_marker(mid: int, size: int) -> np.ndarray:
        if hasattr(aruco, "generateImageMarker"):
            return aruco.generateImageMarker(d, int(mid), int(size))
        m = np.zeros((size, size), np.uint8)
        aruco.drawMarker(d, int(mid), int(size), m, 1)
        return m

    for j in range(NY):
        for i in range(NX):
            mid = ij_to_id(i, j)
            x_cell, y_cell = i * CELL, j * CELL
            x0, y0 = x_cell + PAD_B, y_cell + PAD_B
            img[y0:y0 + PLATE, x0:x0 + PLATE] = 32  # dark plate preview
            xt, yt = x0 + PAD_W, y0 + PAD_W
            img[yt:yt + TAG, xt:xt + TAG] = 220  # bright pad
            img[yt:yt + TAG, xt:xt + TAG] = gen_marker(mid, TAG)

    return img


# -------------------------
# Homography (image -> pattern) using all markers
# -------------------------
def estimate_homography_img_to_pattern(corners_list, ids, ransac_thresh: float = 3.0):
    img_pts, pat_pts = [], []
    used = 0
    for corners, mid in zip(corners_list, ids):
        i, j = id_to_ij(int(mid))
        if not (0 <= i < NX and 0 <= j < NY):
            continue
        img_pts.append(corners.reshape(4, 2).astype(np.float32))
        pat_pts.append(marker_corners_in_pattern(i, j))
        used += 1

    info = {"markers_used": used}
    if used == 0:
        H = np.full((3, 3), np.nan, dtype=np.float64)
        info["status"] = "no_markers"
        return H, info

    img_pts = np.concatenate(img_pts, axis=0).reshape(-1, 1, 2)
    pat_pts = np.concatenate(pat_pts, axis=0).reshape(-1, 1, 2)

    if used == 1:
        H = cv2.getPerspectiveTransform(img_pts.reshape(4, 2), pat_pts.reshape(4, 2)).astype(np.float64)
        H /= H[2, 2]
        info["status"] = "ok_direct_1marker"
        info["inliers"] = 4
        return H, info

    H, mask = cv2.findHomography(img_pts, pat_pts, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        H = np.full((3, 3), np.nan, dtype=np.float64)
        info["status"] = "findHomography_failed"
        return H, info

    inliers = int(mask.ravel().sum())
    info["status"] = "ok_ransac"
    info["inliers"] = inliers
    info["inlier_ratio"] = float(inliers) / float(mask.size)
    H = H.astype(np.float64)
    H /= H[2, 2]
    return H, info


# -------------------------
# Bounded NCC shift estimation on plate excluding tag pad
# -------------------------
def plate_mask_excluding_tag() -> np.ndarray:
    m = np.ones((PLATE, PLATE), np.float32)
    m[PAD_W:PAD_W + TAG, PAD_W:PAD_W + TAG] = 0.0
    return m


def weighted_zncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    sw = float(np.sum(w))
    if sw < 1:
        return -1.0
    ma = float(np.sum(a * w) / sw)
    mb = float(np.sum(b * w) / sw)
    a0 = (a - ma) * w
    b0 = (b - mb) * w
    den = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if den <= 1e-12:
        return -1.0
    return float(np.sum(a0 * b0) / den)


def shift_score_map(ref: np.ndarray, mov: np.ndarray, w: np.ndarray, D: int) -> np.ndarray:
    Hh, Ww = ref.shape
    scores = np.full((2 * D + 1, 2 * D + 1), -1.0, dtype=np.float32)

    for dy in range(-D, D + 1):
        for dx in range(-D, D + 1):
            x0 = max(0, dx); x1 = min(Ww, Ww + dx)
            y0 = max(0, dy); y1 = min(Hh, Hh + dy)

            rx0, rx1 = x0, x1
            ry0, ry1 = y0, y1
            mx0, mx1 = x0 - dx, x1 - dx
            my0, my1 = y0 - dy, y1 - dy

            ww = w[ry0:ry1, rx0:rx1]
            if np.sum(ww) < 64:
                continue

            scores[dy + D, dx + D] = weighted_zncc(ref[ry0:ry1, rx0:rx1], mov[my0:my1, mx0:mx1], ww)

    return scores


def subpixel_parabola_1d(v_m1: float, v0: float, v_p1: float) -> float:
    denom = (v_m1 - 2 * v0 + v_p1)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (v_m1 - v_p1) / denom


def best_shift_bounded_ncc(ref_u8: np.ndarray, mov_u8: np.ndarray, mask: np.ndarray, max_shift: float):
    D = int(math.ceil(max_shift))

    # Gradient magnitude improves robustness
    ref = cv2.GaussianBlur(ref_u8.astype(np.float32), (0, 0), 1.0)
    mov = cv2.GaussianBlur(mov_u8.astype(np.float32), (0, 0), 1.0)

    ref_g = cv2.magnitude(cv2.Sobel(ref, cv2.CV_32F, 1, 0, 3), cv2.Sobel(ref, cv2.CV_32F, 0, 1, 3))
    mov_g = cv2.magnitude(cv2.Sobel(mov, cv2.CV_32F, 1, 0, 3), cv2.Sobel(mov, cv2.CV_32F, 0, 1, 3))

    scores = shift_score_map(ref_g, mov_g, mask, D)
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    best = float(scores[iy, ix])
    dx_i = int(ix - D)
    dy_i = int(iy - D)

    dx_f, dy_f = float(dx_i), float(dy_i)
    if 0 < ix < scores.shape[1] - 1:
        dx_f += subpixel_parabola_1d(float(scores[iy, ix - 1]), float(scores[iy, ix]), float(scores[iy, ix + 1]))
    if 0 < iy < scores.shape[0] - 1:
        dy_f += subpixel_parabola_1d(float(scores[iy - 1, ix]), float(scores[iy, ix]), float(scores[iy + 1, ix]))

    if abs(dx_f) > max_shift or abs(dy_f) > max_shift:
        return float("nan"), float("nan"), best
    return dx_f, dy_f, best


# -------------------------
# Per-image per-tag: warp PLATE using per-marker homography
# -------------------------
def tag_shifts_for_image(bgr: np.ndarray, corners_list, ids, max_shift_pat_px: float):
    shift_rg = np.full((NY, NX, 2), np.nan, dtype=np.float32)
    shift_bg = np.full((NY, NX, 2), np.nan, dtype=np.float32)
    score_rg = np.full((NY, NX), np.nan, dtype=np.float32)
    score_bg = np.full((NY, NX), np.nan, dtype=np.float32)

    mask = plate_mask_excluding_tag()

    for corners, mid in zip(corners_list, ids):
        i, j = id_to_ij(int(mid))
        if not (0 <= i < NX and 0 <= j < NY):
            continue

        src = corners.reshape(4, 2).astype(np.float32)
        dst_marker_pat = marker_corners_in_pattern(i, j)

        # Exact per-marker image->pattern homography
        H_img2pat = cv2.getPerspectiveTransform(src, dst_marker_pat)

        # Shift pattern coords into plate-local coords
        px0, py0 = plate_topleft_in_pattern(i, j)
        T = np.array([[1, 0, -px0],
                      [0, 1, -py0],
                      [0, 0, 1]], dtype=np.float32)

        H_img2plate = T @ H_img2pat

        plate = cv2.warpPerspective(
            bgr, H_img2plate, (PLATE, PLATE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        b = plate[:, :, 0]
        g = plate[:, :, 1]
        r = plate[:, :, 2]

        dx_rg, dy_rg, sc_rg = best_shift_bounded_ncc(g, r, mask, max_shift_pat_px)
        dx_bg, dy_bg, sc_bg = best_shift_bounded_ncc(g, b, mask, max_shift_pat_px)

        if math.isfinite(dx_rg) and math.isfinite(dy_rg):
            shift_rg[j, i, :] = (dx_rg, dy_rg)
            score_rg[j, i] = sc_rg
        if math.isfinite(dx_bg) and math.isfinite(dy_bg):
            shift_bg[j, i, :] = (dx_bg, dy_bg)
            score_bg[j, i] = sc_bg

    return shift_rg, shift_bg, score_rg, score_bg


# -------------------------
# Reduce + errors (NaN if <2 samples)
# -------------------------
def reduce_median_and_error(per_image: np.ndarray):
    valid = np.isfinite(per_image[..., 0]) & np.isfinite(per_image[..., 1])
    count = np.sum(valid, axis=0).astype(np.int32)

    with np.errstate(all="ignore"):
        med = np.nanmedian(per_image, axis=0).astype(np.float32)
        diff = per_image - med[None, ...]
        mag = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
        mag = np.where(valid, mag, np.nan)
        err_med = np.nanmedian(mag, axis=0).astype(np.float32)
        err_rms = np.sqrt(np.nanmean(mag ** 2, axis=0)).astype(np.float32)

    err_med[count < 2] = np.nan
    err_rms[count < 2] = np.nan
    return med, err_med, err_rms, count


# -------------------------
# Overlay
# -------------------------
def draw_vectors_overlay(shift_rg_med, shift_bg_med, count_rg, count_bg, out_path,
                        arrow_scale: float = 20.0, downscale: float = 0.5, dict_id: int = ARUCO_DICT):
    bg = render_background(dict_id)
    vis = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    if downscale != 1.0:
        vis = cv2.resize(vis, (int(PAT_W * downscale), int(PAT_H * downscale)), interpolation=cv2.INTER_AREA)

    def pt(x, y): return (int(round(x * downscale)), int(round(y * downscale)))

    for j in range(NY):
        for i in range(NX):
            cx, cy = plate_center_in_pattern(i, j)

            if count_rg[j, i] > 0:
                dx, dy = shift_rg_med[j, i, 0], shift_rg_med[j, i, 1]
                if np.isfinite(dx) and np.isfinite(dy):
                    cv2.arrowedLine(vis, pt(cx, cy), pt(cx + dx * arrow_scale, cy + dy * arrow_scale),
                                    (0, 0, 255), 1, tipLength=0.25)
            if count_bg[j, i] > 0:
                dx, dy = shift_bg_med[j, i, 0], shift_bg_med[j, i, 1]
                if np.isfinite(dx) and np.isfinite(dy):
                    cv2.arrowedLine(vis, pt(cx, cy), pt(cx + dx * arrow_scale, cy + dy * arrow_scale),
                                    (255, 0, 0), 1, tipLength=0.25)

    cv2.imwrite(out_path, vis)


# -------------------------
# CSV writers
# -------------------------
def write_csvs(out_dir: str, paths: List[str], per_image_info: List[Dict],
               shift_rg_per_image, shift_bg_per_image,
               shift_rg_med, shift_bg_med,
               err_rg_med, err_bg_med,
               count_rg, count_bg):

    report_path = os.path.join(out_dir, "per_image_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "status", "markers_used", "inliers", "inlier_ratio"])
        for info in per_image_info:
            w.writerow([
                os.path.basename(info.get("path", "")),
                info.get("status", ""),
                info.get("markers_used", ""),
                info.get("inliers", ""),
                info.get("inlier_ratio", ""),
            ])

    tag_path = os.path.join(out_dir, "tag_summary.csv")
    nimg = shift_rg_per_image.shape[0]
    with open(tag_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "tag_id", "i", "j",
            "count_RtoG", "RtoG_dx_med", "RtoG_dy_med", "RtoG_err_med",
            "count_BtoG", "BtoG_dx_med", "BtoG_dy_med", "BtoG_err_med",
            "RtoG_per_image(dx,dy)", "BtoG_per_image(dx,dy)"
        ])

        for j in range(NY):
            for i in range(NX):
                mid = ij_to_id(i, j)
                rg_list, bg_list = [], []
                for k in range(nimg):
                    dx, dy = shift_rg_per_image[k, j, i, 0], shift_rg_per_image[k, j, i, 1]
                    if np.isfinite(dx) and np.isfinite(dy):
                        rg_list.append(f"{os.path.basename(paths[k])}:{dx:+.3f},{dy:+.3f}")
                    dx, dy = shift_bg_per_image[k, j, i, 0], shift_bg_per_image[k, j, i, 1]
                    if np.isfinite(dx) and np.isfinite(dy):
                        bg_list.append(f"{os.path.basename(paths[k])}:{dx:+.3f},{dy:+.3f}")

                w.writerow([
                    mid, i, j,
                    int(count_rg[j, i]),
                    float(shift_rg_med[j, i, 0]) if np.isfinite(shift_rg_med[j, i, 0]) else "",
                    float(shift_rg_med[j, i, 1]) if np.isfinite(shift_rg_med[j, i, 1]) else "",
                    float(err_rg_med[j, i]) if np.isfinite(err_rg_med[j, i]) else "",
                    int(count_bg[j, i]),
                    float(shift_bg_med[j, i, 0]) if np.isfinite(shift_bg_med[j, i, 0]) else "",
                    float(shift_bg_med[j, i, 1]) if np.isfinite(shift_bg_med[j, i, 1]) else "",
                    float(err_bg_med[j, i]) if np.isfinite(err_bg_med[j, i]) else "",
                    ";".join(rg_list),
                    ";".join(bg_list),
                ])


# -------------------------
# Main
# -------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help='Glob for input images, e.g. "captures/*.jpg"')
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--dict", type=int, default=ARUCO_DICT, help="cv2.aruco dictionary id")
    ap.add_argument("--detect-inverted", action="store_true", help="Enable inverted-marker detection")
    ap.add_argument("--max-shift", type=float, default=2.0, help="Max expected shift in PATTERN pixels (bound)")
    ap.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold (pixels)")
    ap.add_argument("--arrow-scale", type=float, default=20.0, help="Arrow scale for overlay")
    ap.add_argument("--viz-downscale", type=float, default=0.5, help="Downscale overlay image")
    args = ap.parse_args()

    ensure_dir(args.out)
    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise RuntimeError(f"No inputs matched: {args.inputs}")

    nimg = len(paths)
    shift_rg_per_image = np.full((nimg, NY, NX, 2), np.nan, dtype=np.float32)
    shift_bg_per_image = np.full((nimg, NY, NX, 2), np.nan, dtype=np.float32)
    score_rg_per_image = np.full((nimg, NY, NX), np.nan, dtype=np.float32)
    score_bg_per_image = np.full((nimg, NY, NX), np.nan, dtype=np.float32)
    homographies = np.full((nimg, 3, 3), np.nan, dtype=np.float64)

    per_image_info = []

    for k, p in enumerate(paths):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            per_image_info.append({"path": p, "status": "read_failed"})
            print(f"[{k+1}/{nimg}] {os.path.basename(p)} read_failed")
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        #corners_list, ids, _ = detect_markers(gray, args.dict, detect_inverted=args.detect_inverted)

        #corners_list, ids, rejected = detect_markers_robust( gray, args.dict, detect_inverted=args.detect_inverted, max_dim=8000, use_clahe=True)
        #print(f"... markers={len(ids)} rejected={(0 if rejected is None else len(rejected))} ...")

        corners_list, ids, rejected, scale_used = detect_markers_multiscale_closeup(
            gray,
            args.dict,
            detect_inverted=args.detect_inverted,
            max_dims=(6400, 6000, 4500, 3200, 2400, 1800),
            use_clahe=True,
            median_ksize=5,
        )
        print(f"... markers={len(ids)} rejected={(0 if rejected is None else len(rejected))} scale={scale_used:.3f}")


        H, info = estimate_homography_img_to_pattern(corners_list, ids, ransac_thresh=args.ransac_thresh)
        homographies[k] = H
        info["path"] = p
        per_image_info.append(info)

        print(f"[{k+1}/{nimg}] {os.path.basename(p)}  markers={info.get('markers_used',0)}  "
              f"H={info.get('status','')}  inliers={info.get('inliers','')}")

        if info.get("status", "").startswith("ok"):
            s_rg, s_bg, sc_rg, sc_bg = tag_shifts_for_image(bgr, corners_list, ids, max_shift_pat_px=args.max_shift)
            shift_rg_per_image[k] = s_rg
            shift_bg_per_image[k] = s_bg
            score_rg_per_image[k] = sc_rg
            score_bg_per_image[k] = sc_bg

    shift_rg_med, err_rg_med, err_rg_rms, count_rg = reduce_median_and_error(shift_rg_per_image)
    shift_bg_med, err_bg_med, err_bg_rms, count_bg = reduce_median_and_error(shift_bg_per_image)

    # -------------------------
    # Global (whole-image) average shifts (noise-reduced)
    # -------------------------
    glob_rg = global_shift_from_tag_medians(shift_rg_med, count_rg, err_med=err_rg_med)
    glob_bg = global_shift_from_tag_medians(shift_bg_med, count_bg, err_med=err_bg_med)

    rg = glob_rg["mean"]
    bg = glob_bg["mean"]

    def _fmt(v):
        if not (np.isfinite(v[0]) and np.isfinite(v[1])):
            return "dx=NaN dy=NaN |d|=NaN"
        mag = float(np.sqrt(v[0]*v[0] + v[1]*v[1]))
        return f"dx={v[0]:+.4f} dy={v[1]:+.4f} |d|={mag:.4f}"

    print("\nWhole-image global shifts (robust, from per-tag medians):")
    print(f"  R->G: {_fmt(rg)}   (inliers={glob_rg['inliers']}/{glob_rg['total']}, rms_inlier={glob_rg['rms_inlier']:.4f})")
    print(f"  B->G: {_fmt(bg)}   (inliers={glob_bg['inliers']}/{glob_bg['total']}, rms_inlier={glob_bg['rms_inlier']:.4f})")

    # Also compute a per-image robust mean (then aggregate across images)
    pi_rg = global_shift_from_per_image(shift_rg_per_image)
    pi_bg = global_shift_from_per_image(shift_bg_per_image)

    print("\nWhole-image global shifts (robust, via per-image means):")
    print(f"  R->G: median_over_images {_fmt(pi_rg['global_median'])}   (images_used={pi_rg['n_images_used']})")
    print(f"  B->G: median_over_images {_fmt(pi_bg['global_median'])}   (images_used={pi_bg['n_images_used']})")

    # Overlay
    overlay_path = os.path.join(args.out, "vectors_overlay.png")
    draw_vectors_overlay(shift_rg_med, shift_bg_med, count_rg, count_bg,
                        overlay_path, arrow_scale=args.arrow_scale,
                        downscale=args.viz_downscale, dict_id=args.dict)
    vis = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
    if vis is not None:
        txt1 = f"Global R->G: {rg[0]:+.3f}, {rg[1]:+.3f} px"
        txt2 = f"Global B->G: {bg[0]:+.3f}, {bg[1]:+.3f} px"
        cv2.putText(vis, txt1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(vis, txt2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
        cv2.imwrite(overlay_path, vis)

    # NPZ
    npz_path = os.path.join(args.out, "shifts.npz")
    np.savez_compressed(
        npz_path,
        shift_R_to_G_per_image=shift_rg_per_image,
        shift_B_to_G_per_image=shift_bg_per_image,
        shift_G_to_R_per_image=-shift_rg_per_image,
        shift_G_to_B_per_image=-shift_bg_per_image,
        score_R_to_G_per_image=score_rg_per_image,
        score_B_to_G_per_image=score_bg_per_image,
        shift_R_to_G_median=shift_rg_med,
        shift_B_to_G_median=shift_bg_med,
        shift_G_to_R_median=-shift_rg_med,
        shift_G_to_B_median=-shift_bg_med,
        err_R_to_G_median=err_rg_med,
        err_B_to_G_median=err_bg_med,
        err_R_to_G_rms=err_rg_rms,
        err_B_to_G_rms=err_bg_rms,
        count_R_to_G=count_rg,
        count_B_to_G=count_bg,
        homographies=homographies,
        image_paths=np.array(paths, dtype=object),
        layout=np.array([PAT_W, PAT_H, NX, NY, CELL, PLATE, TAG, PAD_B, PAD_W], dtype=np.int32),
        meaning=np.array(["dx,dy are shifts to apply to (R or B) to align to G; units=pattern pixels"], dtype=object),
        global_R_to_G_from_tags=rg,
        global_B_to_G_from_tags=bg,
        global_R_to_G_from_images=pi_rg["global_median"],
        global_B_to_G_from_images=pi_bg["global_median"],
        per_image_mean_R_to_G=pi_rg["per_image_means"],
        per_image_mean_B_to_G=pi_bg["per_image_means"],
    )

    # CSVs
    write_csvs(args.out, paths, per_image_info,
               shift_rg_per_image, shift_bg_per_image,
               shift_rg_med, shift_bg_med,
               err_rg_med, err_bg_med,
               count_rg, count_bg)

    print("\nOutputs:")
    print(f"  Overlay:   {overlay_path}")
    print(f"  NPZ:       {npz_path}")
    print(f"  Tag CSV:   {os.path.join(args.out, 'tag_summary.csv')}")
    print(f"  Img CSV:   {os.path.join(args.out, 'per_image_report.csv')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
