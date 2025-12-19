"""
Image to Pencil Sketch (OpenCV)

Pipeline:
1) Read RGB(BGR) image
2) Convert to grayscale
3) Invert grayscale
4) Blur inverted image (Gaussian blur)
5) Dodge blend: gray / (255 - blurred)  -> pencil-like effect
6) Post-processing (contrast, threshold, edges)

Usage:
    python sketch.py --input ./input.jpg --output ./sketch.png --ksize 21 --sigma 0
"""

from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(path: str) -> np.ndarray:
    """Read image from disk in BGR format (OpenCV default)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from: {path}")
    return img


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def invert(gray: np.ndarray) -> np.ndarray:
    """Invert grayscale image."""
    return 255 - gray


def gaussian_blur(img: np.ndarray, ksize: int = 21, sigma: float = 0.0) -> np.ndarray:
    """
    Apply Gaussian blur.
    - ksize must be odd and >= 3
    - sigma=0 lets OpenCV choose based on ksize
    """
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def dodge_blend(gray: np.ndarray, blurred_inverted: np.ndarray) -> np.ndarray:
    """
    Dodge blend (pencil sketch effect):
        sketch = gray * 255 / (255 - blurred_inverted)
    Use cv2.divide for stability.
    """
    # ConvertScaleAbs isn't needed if we keep uint8 carefully.
    denom = 255 - blurred_inverted
    # Avoid divide-by-zero
    denom = np.where(denom == 0, 1, denom).astype(np.uint8)

    sketch = cv2.divide(gray, denom, scale=255)
    return sketch


def postprocess(sketch: np.ndarray, sharpen: bool = False, threshold: int | None = None) -> np.ndarray:
    """
    post-processing:
    - sharpen: simple unsharp masking-like sharpening
    - threshold: binary threshold to make it more "line art"
    """
    out = sketch.copy()

    if sharpen:
        # Simple sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, ddepth=-1, kernel=kernel)

    if threshold is not None:
        threshold = int(threshold)
        threshold = max(0, min(255, threshold))
        _, out = cv2.threshold(out, threshold, 255, cv2.THRESH_BINARY)

    return out


def pencil_sketch_from_bgr(
    bgr: np.ndarray,
    ksize: int = 21,
    sigma: float = 0.0,
    sharpen: bool = False,
    threshold: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run full pipeline and return intermediates for reporting/visualization.
    """
    gray = to_grayscale(bgr)
    inv = invert(gray)
    blur = gaussian_blur(inv, ksize=ksize, sigma=sigma)
    sketch = dodge_blend(gray, blur)
    sketch_pp = postprocess(sketch, sharpen=sharpen, threshold=threshold)

    return {
        "bgr": bgr,
        "gray": gray,
        "inverted": inv,
        "blurred_inverted": blur,
        "sketch_raw": sketch,
        "sketch_final": sketch_pp,
    }


def save_image(path: str, img: np.ndarray) -> None:
    """Save image to disk. Creates parent folder if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), img)
    if not ok:
        raise IOError(f"Failed to write image to: {path}")


def show_images(results: dict[str, np.ndarray], max_width: int = 1200) -> None:
    """
    Display intermediate results in windows.
    Press any key to close.
    """
    for name, img in results.items():
        vis = img
        # Resize large images for display
        h, w = vis.shape[:2]
        if w > max_width:
            scale = max_width / w
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imshow(name, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an image to a pencil sketch using OpenCV.")
    parser.add_argument("--input", "-i", required=True, help="Input image path (color/RGB image).")
    parser.add_argument("--output", "-o", default="sketch.png", help="Output image path.")
    parser.add_argument("--ksize", type=int, default=21, help="Gaussian blur kernel size (odd number).")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian blur sigma (0 lets OpenCV choose).")
    parser.add_argument("--sharpen", action="store_true", help="Apply optional sharpening.")
    parser.add_argument("--threshold", type=int, default=None, help="Optional threshold (0-255) for line-art style.")
    parser.add_argument("--show", action="store_true", help="Show intermediate results in windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bgr = read_image_bgr(args.input)
    results = pencil_sketch_from_bgr(
        bgr=bgr,
        ksize=args.ksize,
        sigma=args.sigma,
        sharpen=args.sharpen,
        threshold=args.threshold,
    )

    save_image(args.output, results["sketch_final"])
    print(f"Saved sketch to: {args.output}")

    if args.show:
        show_images(results)


if __name__ == "__main__":
    main()
