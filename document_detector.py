"""ID card isolation using adaptive thresholding + contour detection."""
from __future__ import annotations

import cv2
import numpy as np

import config


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    rect[1] = pts[np.argmin(d)]    # top-right
    rect[3] = pts[np.argmax(d)]    # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    dst = np.array(
        [
            [0, 0],
            [config.WARP_WIDTH - 1, 0],
            [config.WARP_WIDTH - 1, config.WARP_HEIGHT - 1],
            [0, config.WARP_HEIGHT - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (config.WARP_WIDTH, config.WARP_HEIGHT))


def _find_card_contour(gray: np.ndarray, min_area: float):
    """Try adaptive threshold first, then Canny edges, to find a 4-corner contour."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    candidates = []

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    candidates.append(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3))

    edges = cv2.Canny(blur, 50, 150)
    candidates.append(cv2.dilate(edges, kernel, iterations=2))

    for mask in candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx
            # Fallback for slightly rounded corners: use the min-area rectangle
            box = cv2.boxPoints(cv2.minAreaRect(c))
            if cv2.contourArea(c) / max(cv2.contourArea(box.astype("float32")), 1) > 0.85:
                return box.reshape(4, 1, 2).astype("int32")
    return None


def detect_document(image_bgr: np.ndarray):
    """Isolate the ID card.

    Returns (rectified_card, annotated_original, found_flag).
    If no card outline is found, the full image is returned and found_flag is False.
    """
    h, w = image_bgr.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale < 1 else image_bgr.copy()
    ratio = 1.0 / scale if scale < 1 else 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    min_area = config.MIN_DOC_AREA_RATIO * small.shape[0] * small.shape[1]
    contour = _find_card_contour(gray, min_area)

    annotated = image_bgr.copy()
    if contour is None:
        return cv2.resize(image_bgr, (config.WARP_WIDTH, config.WARP_HEIGHT)), annotated, False

    pts = contour.reshape(4, 2).astype("float32") * ratio
    cv2.polylines(annotated, [pts.astype("int32")], True, (0, 255, 0), max(2, int(w / 300)))
    warped = _four_point_transform(image_bgr, pts)

    # Portrait-shaped capture of a landscape card -> rotate so text is horizontal
    ordered = _order_points(pts)
    width = np.linalg.norm(ordered[1] - ordered[0])
    height = np.linalg.norm(ordered[3] - ordered[0])
    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        warped = cv2.resize(warped, (config.WARP_WIDTH, config.WARP_HEIGHT))
    return warped, annotated, True
