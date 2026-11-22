"""Face detection (Haar cascades) + FaceNet512 embeddings + cosine similarity."""
from __future__ import annotations

import cv2
import numpy as np

import config

_CASCADES = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _CASCADES:
        _CASCADES[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _CASCADES[name]


def detect_face(image_bgr: np.ndarray, margin: float = 0.25):
    """Detect the largest face with Haar cascades.

    Tries several cascades and an upscaled pass (ID-card photos are tiny).
    Returns (face_crop_bgr, (x, y, w, h)) or (None, None).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    attempts = []
    for upscale in (1.0, 1.5, 2.0):
        g = cv2.resize(gray, None, fx=upscale, fy=upscale) if upscale != 1.0 else gray
        for name in (
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
        ):
            for min_neighbors in (5, 3):
                faces = _cascade(name).detectMultiScale(
                    g, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(40, 40)
                )
                if len(faces):
                    attempts = [(tuple(int(v / upscale) for v in f)) for f in faces]
                    break
            if attempts:
                break
        if attempts:
            break

    if not attempts:
        return None, None

    x, y, w, h = max(attempts, key=lambda f: f[2] * f[3])
    mx, my = int(w * margin), int(h * margin)
    H, W = image_bgr.shape[:2]
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, W), min(y + h + my, H)
    return image_bgr[y0:y1, x0:x1], (x, y, w, h)


def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """512-d FaceNet embedding of an already-cropped face (DeepFace)."""
    from deepface import DeepFace  # imported lazily: heavy TensorFlow import

    result = DeepFace.represent(
        img_path=face_bgr,                # numpy arrays are treated as BGR
        model_name=config.MODEL_NAME,
        detector_backend="skip",          # we already cropped with Haar
        enforce_detection=False,
    )
    return np.asarray(result[0]["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compare_faces(emb_a: np.ndarray, emb_b: np.ndarray, distance_threshold: float):
    """Return (is_match, cosine_similarity, cosine_distance)."""
    sim = cosine_similarity(emb_a, emb_b)
    dist = 1.0 - sim
    return dist <= distance_threshold, sim, dist
