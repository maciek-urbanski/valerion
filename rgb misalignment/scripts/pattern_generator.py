#!/usr/bin/env python3
"""
Generate a 3840x2160 ArUco grid test pattern:
- CELL=240, PLATE=224, TAG=120 (16x9 tags)
- Black background
- Dark-gray plate per cell
- Bright dot texture on plate (excluding tag area)
- ArUco marker centered on a bright pad (same size as TAG)

Requires: pip install opencv-contrib-python pillow numpy
"""

import argparse
import numpy as np
import cv2
from PIL import Image


def gen_pattern(
    w=3840, h=2160,
    cell=240, plate=224, tag=120,
    bg=0, plate_val=32, dot_val=220, pad_val=220,
    dots_per_tag=220, dot_r=3,
    dict_id=cv2.aruco.DICT_6X6_250,
    seed_base=10_000
):
    pad_b = (cell - plate) // 2
    pad_w = (plate - tag) // 2
    nx, ny = w // cell, h // cell

    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(dict_id)

    def marker(mid: int, size: int) -> np.ndarray:
        if hasattr(aruco, "generateImageMarker"):
            return aruco.generateImageMarker(d, int(mid), int(size))
        m = np.zeros((size, size), np.uint8)
        aruco.drawMarker(d, int(mid), int(size), m, 1)
        return m

    img = np.full((h, w), bg, np.uint8)

    for j in range(ny):
        for i in range(nx):
            mid = j * nx + i
            x0, y0 = i * cell + pad_b, j * cell + pad_b
            img[y0:y0 + plate, x0:x0 + plate] = plate_val

            xt, yt = x0 + pad_w, y0 + pad_w
            img[yt:yt + tag, xt:xt + tag] = pad_val

            rng = np.random.default_rng(seed_base + mid)
            for _ in range(dots_per_tag):
                x = int(rng.integers(x0 + dot_r, x0 + plate - dot_r))
                y = int(rng.integers(y0 + dot_r, y0 + plate - dot_r))
                if (xt - 2 * dot_r <= x <= xt + tag + 2 * dot_r) and (yt - 2 * dot_r <= y <= yt + tag + 2 * dot_r):
                    continue
                cv2.circle(img, (x, y), dot_r, dot_val, -1)

            img[yt:yt + tag, xt:xt + tag] = marker(mid, tag)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="aruco_grid_darkplate_brightdots_tagpad.png")
    ap.add_argument("--dots", type=int, default=220, help="dots per tag")
    ap.add_argument("--dot-r", type=int, default=3, help="dot radius (px)")
    ap.add_argument("--plate", type=int, default=32, help="plate intensity (0-255)")
    ap.add_argument("--dot", type=int, default=220, help="dot intensity (0-255)")
    ap.add_argument("--pad", type=int, default=220, help="tag pad intensity (0-255)")
    args = ap.parse_args()

    img = gen_pattern(dots_per_tag=args.dots, dot_r=args.dot_r,
                      plate_val=args.plate, dot_val=args.dot, pad_val=args.pad)
    Image.fromarray(img, "L").convert("RGB").save(args.out, optimize=True)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
