"""Central configuration. Override any value with an environment variable."""
import os

# ---------- MySQL ----------
DB_HOST = os.getenv("EKYC_DB_HOST", "localhost")
DB_PORT = int(os.getenv("EKYC_DB_PORT", "3306"))
DB_USER = os.getenv("EKYC_DB_USER", "root")
DB_PASSWORD = os.getenv("EKYC_DB_PASSWORD", "password")
DB_NAME = os.getenv("EKYC_DB_NAME", "ekyc_db")

# ---------- Face matching ----------
MODEL_NAME = "Facenet512"          # 512-dimensional embeddings
# DeepFace's published cosine-distance threshold for Facenet512 is 0.30.
# Selfie-vs-ID-photo is a harder comparison (small, printed, compressed photo),
# so the UI lets you tune this. Lower = stricter.
DEFAULT_COSINE_DISTANCE_THRESHOLD = 0.30

# ---------- Document detection ----------
MIN_DOC_AREA_RATIO = 0.20          # card must cover at least 20% of the image
WARP_WIDTH = 1000                  # standard width of the rectified card
WARP_HEIGHT = 630                  # ~ID-1 card ratio (85.6 x 53.98 mm)

# ---------- OCR ----------
OCR_LANGS = ["en"]
