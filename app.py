"""eKYC verification app.

Run:  streamlit run app.py
Flow: upload PAN card -> detect card -> detect ID face -> OCR PAN/Name/DOB
      -> take selfie -> FaceNet512 + cosine similarity -> save to MySQL.
"""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

import config
import db
import document_detector as dd
import face_utils as fu
import ocr_extractor as ocr

st.set_page_config(page_title="eKYC Verification", page_icon="🪪", layout="wide")


def decode(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Max cosine distance for a match",
        0.10, 0.60, config.DEFAULT_COSINE_DISTANCE_THRESHOLD, 0.01,
        help="Lower = stricter. DeepFace's default for Facenet512 is 0.30.",
    )
    ok, msg = db.check_connection()
    if ok:
        st.success("MySQL connected")
    else:
        st.error("MySQL not reachable")
        st.caption(msg)
        st.caption("Set EKYC_DB_HOST / EKYC_DB_USER / EKYC_DB_PASSWORD env vars.")
    db_ok = ok

st.title("🪪 eKYC Identity Verification")
tab_verify, tab_records = st.tabs(["Verify identity", "Verified records"])

# ---------------------------------------------------------------- verify tab
with tab_verify:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. PAN card")
        id_file = st.file_uploader("Upload PAN card image", type=["jpg", "jpeg", "png"])
    with col_b:
        st.subheader("2. Live selfie")
        mode = st.radio("Selfie source", ["Camera", "Upload"], horizontal=True)
        selfie_file = (
            st.camera_input("Take a selfie")
            if mode == "Camera"
            else st.file_uploader("Upload selfie", type=["jpg", "jpeg", "png"], key="selfie_up")
        )

    if st.button("Run verification", type="primary", disabled=not (id_file and selfie_file)):
        id_img, selfie_img = decode(id_file), decode(selfie_file)
        result: dict = {"errors": []}

        with st.status("Verifying...", expanded=True) as status:
            st.write("Detecting ID card (adaptive threshold + contours)...")
            card, annotated, found = dd.detect_document(id_img)
            result.update(card=card, annotated=annotated, card_found=found)

            st.write("Detecting faces (Haar cascades)...")
            id_face, id_box = fu.detect_face(card)
            if id_face is None:  # fall back to the raw upload
                id_face, id_box = fu.detect_face(id_img)
            selfie_face, selfie_box = fu.detect_face(selfie_img)
            result.update(id_face=id_face, selfie_face=selfie_face)
            if id_face is None:
                result["errors"].append("No face found on the ID card.")
            if selfie_face is None:
                result["errors"].append("No face found in the selfie.")

            if id_face is not None and selfie_face is not None:
                st.write("Computing FaceNet512 embeddings...")
                e1, e2 = fu.get_embedding(id_face), fu.get_embedding(selfie_face)
                result["embedding"] = e2
                result["emb_dim"] = len(e1)

            st.write("Reading text (EasyOCR)...")
            result["fields"] = ocr.extract_pan_details(card)
            status.update(label="Processing complete", state="complete")

        if "embedding" in result:
            match, sim, dist = fu.compare_faces(e1, e2, threshold)
            result.update(match=match, sim=sim, dist=dist)
        st.session_state["result"] = result

    res = st.session_state.get("result")
    if res:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.image(rgb(res["annotated"]), caption=f"Card outline ({'found' if res['card_found'] else 'not found - using full image'})")
        c2.image(rgb(res["card"]), caption="Rectified card")
        with c3:
            if res.get("id_face") is not None:
                st.image(rgb(res["id_face"]), caption="Face on ID", width=140)
            if res.get("selfie_face") is not None:
                st.image(rgb(res["selfie_face"]), caption="Selfie face", width=140)

        for err in res["errors"]:
            st.error(err)

        if "sim" in res:
            m1, m2, m3 = st.columns(3)
            m1.metric("Cosine similarity", f"{res['sim']:.3f}")
            m2.metric("Cosine distance", f"{res['dist']:.3f}", help=f"match if <= {threshold}")
            m3.metric("Face match", "✅ MATCH" if res["match"] else "❌ NO MATCH")

        st.subheader("Extracted details (edit if OCR misread anything)")
        f = res["fields"]
        e1c, e2c, e3c, e4c = st.columns(4)
        pan = e1c.text_input("PAN", f["pan"] or "").strip().upper()
        name = e2c.text_input("Name", f["name"] or "").strip().upper()
        father = e3c.text_input("Father's name", f["father_name"] or "").strip().upper()
        dob = e4c.text_input("DOB (DD/MM/YYYY)", f["dob"] or "").strip()
        with st.expander("Raw OCR text"):
            st.text(f["raw_text"] or "(nothing read)")

        # ---- validation
        problems = []
        if not ocr.is_valid_pan(pan):
            problems.append("PAN must look like ABCDE1234F.")
        if not name:
            problems.append("Name is missing.")
        try:
            from datetime import datetime
            datetime.strptime(dob, "%d/%m/%Y")
        except ValueError:
            problems.append("DOB must be a valid DD/MM/YYYY date.")
        if not res.get("match"):
            problems.append("Selfie does not match the ID photo.")

        if problems:
            st.warning("Cannot mark as verified:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            st.success("All checks passed.")

        if st.button("Save verified record to MySQL", disabled=bool(problems) or not db_ok):
            try:
                if db.pan_exists(pan):
                    st.error(f"PAN {pan} is already registered.")
                else:
                    rid = db.save_record(pan, name, father, dob, res["sim"], res["dist"], res["embedding"])
                    st.success(f"Saved. Record id = {rid}")
                    st.balloons()
            except Exception as e:  # noqa: BLE001
                st.error(f"Database error: {e}")

# ---------------------------------------------------------------- records tab
with tab_records:
    if db_ok:
        rows = db.list_records()
        st.dataframe(rows, use_container_width=True) if rows else st.info("No records yet.")
    else:
        st.info("Connect MySQL to see saved records.")
