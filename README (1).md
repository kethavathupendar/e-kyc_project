# eKYC Verification — Consolidated

Went from **6 files** (`preprocess.py`, `postprocess.py`, `ocr_engine.py`,
`face_verification.py`, `mysqldb_operations.py`, `app.py`) + a missing
`utils.py`/`config.yaml` down to **3 files**:

| File | Replaces | Responsibility |
|---|---|---|
| `app.py` | old `app.py` | Streamlit UI + pipeline orchestration |
| `core.py` | `preprocess.py`, `ocr_engine.py`, `postprocess.py`, `face_verification.py` | ID card cropping, face detection/verification, OCR, text parsing |
| `database.py` | `mysqldb_operations.py` | MySQL connection + CRUD |

## Setup

```bash
pip install -r requirements.txt
```

Set your DB credentials as environment variables (don't hardcode them —
that was the original bug):

```bash
export EKYC_DB_HOST=localhost
export EKYC_DB_USER=root
export EKYC_DB_PASSWORD=yourpassword
export EKYC_DB_NAME=ekyc
export EKYC_DB_PORT=3306
```

Then run:

```bash
streamlit run app.py
```

The `users` table is created automatically on first DB connection — no
manual `CREATE TABLE` step needed anymore.

## What was actually broken in the original code

1. **`extract_id_card` crashed** — it called `largest_contour.any()`
   without checking if `largest_contour` was `None` first, which raised
   `AttributeError` whenever no contour was found.
2. **DB connection at import time** — `mysqldb_operations.py` opened a
   MySQL connection with a hardcoded password the moment the module was
   imported, so the app crashed on startup if MySQL wasn't reachable, and
   the password was committed to source.
3. **`HAARCASCADE_PATH` pointed nowhere** — `face_verification.py` read
   it from `config.yaml`, but no working cascade file was ever shipped
   with the project. Now uses OpenCV's built-in cascade
   (`cv2.data.haarcascades`), so no extra file is needed.
4. **Two DB connections at once** — `app.py` created a
   `st.connection('mysql', type='sql')` that was never actually used,
   while a separate raw `mysql.connector` connection (from
   `mysqldb_operations.py`) did the real work. Removed the dead one.
5. **Fragile OCR parsing** — `postprocess.py` assumed the PAN number and
   name always sat at fixed line indices in the OCR output, so a single
   missed OCR line threw `IndexError`. Replaced with regex matching for
   the PAN number (`[A-Z]{5}[0-9]{4}[A-Z]`) and DOB, which degrades
   gracefully instead of crashing.
6. **Windows-only paths** — `"data\\02_intermediate_data"` broke on
   Mac/Linux. Replaced with `os.path.join`.
7. **Missing `utils.py`** — `preprocess.py` and `face_verification.py`
   imported `read_yaml`/`file_exists` from a `utils.py` that wasn't
   included, so the project couldn't even run. Removed the YAML config
   layer entirely in favor of plain constants/env vars in `core.py` and
   `database.py`.

## Pipeline flow

```
Upload ID card + selfie
        │
        ▼
extract_id_card()        (core.py)
        │
        ▼
detect_and_extract_face()  (core.py)  ──► compare with selfie via verify_faces()
        │                                          │
        ▼                                     fail → stop, show error
extract_text() + extract_information() (core.py)
        │
        ▼
is_duplicate() → insert_record()  (database.py)
```
