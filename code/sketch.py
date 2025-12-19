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
    denom = 255 - blurred_inverted
    denom = np.where(denom == 0, 1, denom).astype(np.uint8)
    sketch = cv2.divide(gray, denom, scale=255)
    return sketch


def reinforce_edges(gray: np.ndarray, sketch: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return sketch
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_not(edges)
    enhanced = cv2.multiply(sketch, edges, scale=1 / 255.0)
    s = float(np.clip(strength, 0.0, 1.0))
    return cv2.addWeighted(sketch, 1.0 - s, enhanced, s, 0.0)


def reinforce_texture(gray: np.ndarray, sketch: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    if strength <= 0:
        return sketch

    g = gray.astype(np.float32)
    base = cv2.GaussianBlur(g, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    detail = cv2.absdiff(g, base)
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    detail = cv2.GaussianBlur(detail, (0, 0), sigmaX=0.6, sigmaY=0.6)

    mask = 255 - detail
    textured = cv2.multiply(sketch, mask, scale=1 / 255.0)

    s = float(np.clip(strength, 0.0, 1.0))
    return cv2.addWeighted(sketch, 1.0 - s, textured, s, 0.0)


def postprocess(
    sketch: np.ndarray,
    sharpen: bool = False,
    threshold: int | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    darken: int = 0,
) -> np.ndarray:
    """
    post-processing:
    - contrast: linear contrast (alpha). >1 makes strokes darker.
    - gamma: gamma correction. <1 darkens mid-tones (more depth).
    - darken: subtract brightness to deepen strokes (0~80 is typical).
    """
    out = sketch.copy()

    if contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=float(contrast), beta=0)

    if gamma != 1.0:
        g = float(gamma)
        if g <= 0:
            g = 1.0
        inv_gamma = 1.0 / g
        table = (np.linspace(0, 1, 256) ** inv_gamma) * 255
        table = np.clip(table, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, table)

    if darken > 0:
        out = cv2.subtract(out, int(darken))

    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, ddepth=-1, kernel=kernel)

    if threshold is not None:
        threshold = int(np.clip(threshold, 0, 255))
        _, out = cv2.threshold(out, threshold, 255, cv2.THRESH_BINARY)

    return out


def pencil_sketch_from_bgr(
    bgr: np.ndarray,
    ksize: int = 21,
    sigma: float = 0.0,
    sharpen: bool = False,
    threshold: int | None = None,
    contrast: float = 1.0,
    gamma: float = 1.0,
    darken: int = 0,
    edge_strength: float = 0.0,
    texture_strength: float = 0.35,
    detail_sigma: float = 2.0,
) -> dict[str, np.ndarray]:
    gray = to_grayscale(bgr)
    inv = invert(gray)
    blur = gaussian_blur(inv, ksize=ksize, sigma=sigma)
    sketch = dodge_blend(gray, blur)
    sketch = reinforce_edges(gray, sketch, edge_strength)
    sketch = reinforce_texture(gray, sketch, sigma=detail_sigma, strength=texture_strength)

    sketch_pp = postprocess(
        sketch,
        sharpen=sharpen,
        threshold=threshold,
        contrast=contrast,
        gamma=gamma,
        darken=darken,
    )

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
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--darken", type=int, default=0)
    parser.add_argument("--edge_strength", type=float, default=0.0)
    parser.add_argument("--texture_strength", type=float, default=0.35)
    parser.add_argument("--detail_sigma", type=float, default=2.0)
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
        contrast=args.contrast,
        gamma=args.gamma,
        darken=args.darken,
        edge_strength=args.edge_strength,
        texture_strength=args.texture_strength,
        detail_sigma=args.detail_sigma,
    )

    save_image(args.output, results["sketch_final"])
    print(f"Saved sketch to: {args.output}")

    if args.show:
        show_images(results)


if __name__ == "__main__":
    main()
