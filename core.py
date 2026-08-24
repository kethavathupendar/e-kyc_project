"""
core.py
--------
All computer-vision / OCR logic for the eKYC app, in one place:
  1. ID card extraction from an uploaded photo   (was preprocess.py)
  2. Face detection + face-match verification    (was face_verification.py)
  3. OCR text extraction                         (was ocr_engine.py)
  4. Parsing OCR text into structured fields      (was postprocess.py)

Fixes vs. the originals:
  - No dependency on a missing utils.py / config.yaml — paths and
    thresholds are plain constants at the top of this file.
  - extract_id_card no longer crashes when no contour is found
    (`largest_contour.any()` on None used to raise AttributeError).
  - Uses OpenCV's bundled Haar cascade instead of a HAARCASCADE_PATH
    that was never actually provided anywhere in the original project.
  - PAN parsing uses a regex for the ID/DOB instead of assuming fixed
    line positions, so it degrades gracefully instead of throwing
    IndexError when OCR misses a line.
  - Windows-only path strings ("data\\...") replaced with os.path.join.
  - Single logging setup shared by the whole app (call `setup_logging()`
    once from app.py).
"""

import os
import re
import logging
from datetime import datetime

import cv2
import numpy as np
import easyocr
from deepface import DeepFace

# --------------------------------------------------------------------- #
# Config / constants
# --------------------------------------------------------------------- #
INTERMEDIATE_DIR = os.path.join("data", "intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

CONTOUR_FILE = os.path.join(INTERMEDIATE_DIR, "id_card_contour.jpg")
UPLOADED_FACE_FILE = os.path.join(INTERMEDIATE_DIR, "uploaded_face.jpg")
EXTRACTED_FACE_FILE = os.path.join(INTERMEDIATE_DIR, "extracted_face.jpg")

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

OCR_CONFIDENCE_THRESHOLD = 0.3
FACE_MODEL_NAME = "Facenet512"
FACE_DETECTOR_BACKEND = "opencv"

_ocr_reader = None  # lazy-loaded singleton, EasyOCR is slow to init


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "ekyc_logs.log"),
        level=logging.INFO,
        format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
        filemode="a",
    )


logger = logging.getLogger("ekyc")


# --------------------------------------------------------------------- #
# 1. Image I/O + ID card extraction
# --------------------------------------------------------------------- #
def read_uploaded_image(uploaded_file):
    """Read a Streamlit UploadedFile into an OpenCV BGR image."""
    try:
        image_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — file may not be a valid image.")
        return img
    except Exception as e:
        logger.error(f"Error reading uploaded image: {e}")
        return None


def extract_id_card(img):
    """
    Locate the largest rectangular contour in `img` (assumed to be the ID
    card) and return the cropped card image. Returns None if nothing
    usable is found, instead of crashing.
    """
    if img is None:
        logger.error("extract_id_card received None image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No contours found in ID card image.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 1000:  # too small to be a real card
        logger.warning("Largest contour is too small to be an ID card.")
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y + h, x:x + w]

    cv2.imwrite(CONTOUR_FILE, cropped)
    logger.info(f"ID card extracted, bbox=({x},{y},{w},{h}), saved to {CONTOUR_FILE}")
    return cropped


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    logger.info(f"Image saved: {path}")
    return path


# --------------------------------------------------------------------- #
# 2. Face detection + verification
# --------------------------------------------------------------------- #
def detect_and_extract_face(img, save_path=EXTRACTED_FACE_FILE):
    """Detect the largest face in `img` (BGR) and save a crop of it."""
    if img is None:
        logger.error("detect_and_extract_face received None image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        logger.warning("No face detected in image.")
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # pad the crop by 50% so DeepFace has context around the face
    pad_w, pad_h = int(w * 0.25), int(h * 0.25)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(img.shape[1], x + w + pad_w)
    y1 = min(img.shape[0], y + h + pad_h)

    face_crop = img[y0:y1, x0:x1]
    save_image(face_crop, save_path)
    return save_path


def verify_faces(id_face_path: str, selfie_face_path: str) -> bool:
    """Return True if the two face images belong to the same person."""
    if not (id_face_path and os.path.exists(id_face_path)):
        logger.error(f"Face image not found: {id_face_path}")
        return False
    if not (selfie_face_path and os.path.exists(selfie_face_path)):
        logger.error(f"Face image not found: {selfie_face_path}")
        return False

    try:
        result = DeepFace.verify(
            img1_path=id_face_path,
            img2_path=selfie_face_path,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=True,
        )
        logger.info(f"Face verification result: {result['verified']} (distance={result.get('distance')})")
        return bool(result["verified"])
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        return False


def get_face_embedding(image_path: str):
    if not (image_path and os.path.exists(image_path)):
        logger.error(f"Face image not found: {image_path}")
        return None
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=True,
        )
        return embedding[0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding extraction error: {e}")
        return None


# --------------------------------------------------------------------- #
# 3. OCR
# --------------------------------------------------------------------- #
def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Loading EasyOCR reader (first call only)...")
        _ocr_reader = easyocr.Reader(["en"])
    return _ocr_reader


def extract_text(image, confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD) -> str:
    """
    Run OCR on `image` (path or ndarray) and return recognized text
    fragments joined with '|', keeping only fragments above the
    confidence threshold.
    """
    reader = _get_ocr_reader()
    try:
        results = reader.readtext(image)
        fragments = [text for (_, text, conf) in results if conf > confidence_threshold]
        joined = "|" + "|".join(fragments) + "|"
        logger.info(f"OCR extracted {len(fragments)} fragments above threshold.")
        return joined
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


# --------------------------------------------------------------------- #
# 4. Parse OCR text -> structured PAN card fields
# --------------------------------------------------------------------- #
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
DATE_REGEX = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def extract_information(ocr_text: str) -> dict:
    """
    Parse the raw '|'-delimited OCR string from a PAN card into a
    structured dict. Falls back to regex matching for the PAN number and
    DOB (robust to OCR line-order noise), and takes the two long
    uppercase-letter fragments after "GOVT OF INDIA" as name / father's
    name when present.
    """
    info = {"ID": "", "Name": "", "Father's Name": "", "DOB": "", "ID Type": "PAN"}

    words = [w.strip() for w in ocr_text.replace(".", "").split("|") if len(w.strip()) > 1]

    # PAN number: regex match anywhere is far more robust than a fixed index
    for w in words:
        if PAN_REGEX.fullmatch(w.replace(" ", "")):
            info["ID"] = w.replace(" ", "")
            break

    # DOB: first date-shaped fragment
    for w in words:
        if DATE_REGEX.fullmatch(w):
            try:
                info["DOB"] = datetime.strptime(w, "%d/%m/%Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
            break

    # Name / father's name: the two alphabetic fragments right after
    # "GOVT OF INDIA", matching the layout of an Indian PAN card.
    try:
        idx = words.index("GOVT OF INDIA")
        candidates = [w for w in words[idx + 1:] if re.fullmatch(r"[A-Za-z ]{3,}", w)]
        if len(candidates) >= 1:
            info["Name"] = candidates[0]
        if len(candidates) >= 2:
            info["Father's Name"] = candidates[1]
    except ValueError:
        logger.warning("'GOVT OF INDIA' marker not found in OCR text; name fields left blank.")

    missing = [k for k, v in info.items() if not v and k != "ID Type"]
    if missing:
        logger.warning(f"Could not extract fields: {missing}")

    return info
