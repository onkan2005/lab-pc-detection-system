"""Image preprocessing built from the course's earlier lessons, applied to every camera frame
before YOLO26 and face recognition.

Notation (Lesson: Definitions and Notations)
  A digital image is a function f(x, y) sampled on an M x N grid of pixels. Each pixel holds an
  intensity r in [0, L-1]; for 8-bit images L = 2^8 = 256, so r is 0 (black) .. 255 (white).

Data formats (Lesson: Image Data Formats)
  Frames arrive as JPEG bytes and are decoded by OpenCV into an M x N x 3 array of uint8 in BGR
  order. Brightness work is done on one intensity channel: either grayscale
      Y = 0.299 R + 0.587 G + 0.114 B
  or the Y (luma) channel of YCrCb, so colors (Cr, Cb) are left untouched.

Lesson 1 - Brightness and contrast
  brightness = mean intensity, contrast = standard deviation of intensity.
  Linear point operation:  g(x, y) = alpha * f(x, y) + beta   (clipped to 0..255)
Lesson 2 - Digital negative and normalization
  negative:       s = (L - 1) - r
  normalization:  s = (r - r_min) / (r_max - r_min) * (L - 1)   (min-max contrast stretch)
Lesson 3 - Histogram equalization
  s_k = (L - 1) * sum_{j<=k} p(r_j)   where p(r_j) = n_j / (M N)   (the CDF of the histogram)
Beyond the course - CLAHE
  Contrast Limited Adaptive Histogram Equalization: equalizes small tiles separately with a clip
  limit, so a dark face next to a bright window is still lifted without blowing out the window.
"""
import cv2
import numpy as np

import config

L = 256   # number of intensity levels for 8-bit images


# ---------- measurements ----------
def to_gray(img):
    """BGR (uint8, M x N x 3) -> grayscale intensity image (uint8, M x N)."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def stats(img):
    """Lesson 1: brightness (mean) and contrast (standard deviation) of the intensities."""
    g = to_gray(img)
    return {"brightness": round(float(g.mean()), 1), "contrast": round(float(g.std()), 1),
            "min": int(g.min()), "max": int(g.max())}


def histogram(img, bins=L):
    """Number of pixels n_k at each intensity level k (used by equalization and shown in the UI)."""
    return np.bincount(to_gray(img).ravel(), minlength=L).reshape(bins, -1).sum(axis=1).tolist()


# ---------- point operations (Lessons 1-2) ----------
def brightness_contrast(img, alpha=1.0, beta=0.0):
    """Lesson 1: g = alpha * f + beta, clipped to the valid range [0, L-1]."""
    return np.clip(img.astype(np.float32) * alpha + beta, 0, L - 1).astype(np.uint8)


def negative(img):
    """Lesson 2: digital negative s = (L - 1) - r."""
    return (L - 1) - img


def normalize(img):
    """Lesson 2: min-max normalization; stretches [r_min, r_max] to the full [0, L-1] range."""
    r_min, r_max = int(img.min()), int(img.max())
    if r_max == r_min:
        return img.copy()
    return ((img.astype(np.float32) - r_min) * (L - 1) / (r_max - r_min)).astype(np.uint8)


# ---------- histogram equalization (Lesson 3) ----------
def equalize_gray(gray):
    """Lesson 3, written out: s_k = (L-1) * CDF(r_k). (Same result as cv2.equalizeHist.)"""
    hist = np.bincount(gray.ravel(), minlength=L)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    if cdf[-1] == cdf_min:
        return gray.copy()
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * (L - 1)).astype(np.uint8)
    return lut[gray]


def on_luma(img, fn):
    """Apply an intensity operation to the Y (brightness) channel only, keeping the colors."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = fn(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def clahe(gray):
    """Beyond the course: adaptive (tile-based) equalization with a contrast limit."""
    return _clahe.apply(gray)


METHODS = {
    "normalize": ("Min-max normalization (Lesson 2)", lambda img: on_luma(img, normalize)),
    "equalize": ("Histogram equalization (Lesson 3)", lambda img: on_luma(img, equalize_gray)),
    "clahe": ("CLAHE, adaptive equalization (beyond the course)", lambda img: on_luma(img, clahe)),
    "brightness": ("Brightness/contrast g = a*f + b (Lesson 1)",
                   lambda img: brightness_contrast(img, 110 / max(1.0, stats(img)["brightness"]), 0)),
}


def detail(img):
    """Contrast after a light blur (removes sensor noise). A covered lens has almost none."""
    return float(cv2.GaussianBlur(to_gray(img), (5, 5), 0).std())


# ---------- the automatic step used on every live frame ----------
def enhance(frame, method=None):
    """Measures the frame (Lesson 1) and enhances it only when it is dark or flat.
    Returns (image for detection, report).

    - dark or low contrast -> config.ENHANCE_METHOD (brightness/contrast g = a*f + b by default, see config.py)
    - over-exposed / glare -> flagged with the digital negative (Lesson 2): a blinded camera's
      negative is almost black, i.e. mean((L-1) - r) is close to 0
    """
    method = method or config.ENHANCE_METHOD
    before = stats(frame)
    steps, out = [], frame
    if before["brightness"] < config.DARK_MEAN or before["contrast"] < config.LOW_CONTRAST_STD:
        name, fn = METHODS[method]
        out = fn(frame)
        why = "dark" if before["brightness"] < config.DARK_MEAN else "low contrast"
        steps.append(f"{name}: frame was {why}")
    negative_mean = (L - 1) - before["brightness"]     # mean of the digital negative
    glare = negative_mean < (L - 1) - config.GLARE_MEAN
    return out, {"before": before, "after": stats(out) if steps else before, "steps": steps,
                 "glare": glare, "detail": round(detail(frame), 1),
                 "format": f"{frame.shape[1]}x{frame.shape[0]}, {frame.dtype}, BGR, L={L} levels"}
