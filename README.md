# eKYC Verification System

Streamlit app that verifies identity from a PAN card + live selfie.

## Pipeline
1. **Document detection** (`document_detector.py`) - Gaussian blur, adaptive thresholding, morphological close, contour detection, 4-point perspective warp (Canny fallback).
2. **Face detection** (`face_utils.py`) - Haar cascades (default + alt2, multi-scale retries) on the card and the selfie.
3. **Face matching** (`face_utils.py`) - DeepFace `Facenet512` -> 512-d embeddings -> cosine similarity/distance.
4. **OCR** (`ocr_extractor.py`) - EasyOCR run on 3 preprocessed variants (raw, CLAHE, adaptive threshold); regex + position-aware character correction for PAN (`AAAAA9999A`), DOB validation, layout-based Name/Father's name extraction.
5. **Storage** (`db.py`) - Verified records saved to MySQL (duplicate PAN blocked).

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# MySQL must be running. Set credentials (database + table are auto-created):
export EKYC_DB_HOST=localhost
export EKYC_DB_USER=root
export EKYC_DB_PASSWORD=yourpassword
export EKYC_DB_NAME=ekyc_db

streamlit run app.py
```
Windows PowerShell: `$env:EKYC_DB_PASSWORD="yourpassword"`

First run downloads the FaceNet512 weights and EasyOCR models (needs internet once).

## Usage
Upload a PAN card photo, take a selfie, click **Run verification**, correct any OCR mistakes in the editable fields, then **Save verified record**. A record can only be saved if the PAN format is valid, DOB is valid, and the faces match.

## Tuning
- Sidebar slider changes the cosine-distance threshold (default 0.30, DeepFace's value for Facenet512). ID-card photos are small/printed, so 0.35-0.45 is often needed in practice.
- Use a flat, well-lit, high-resolution card photo on a contrasting background for best contour detection and OCR.

## Files
| File | Role |
|---|---|
| `app.py` | Streamlit workflow |
| `document_detector.py` | Card isolation |
| `face_utils.py` | Haar detection, FaceNet512, cosine similarity |
| `ocr_extractor.py` | EasyOCR + regex parsing |
| `db.py` | MySQL layer |
| `config.py` | Settings / env vars |
