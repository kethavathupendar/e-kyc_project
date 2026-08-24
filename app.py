"""
app.py
-------
Streamlit front-end and orchestration for the eKYC pipeline:

  1. User uploads a PAN card photo + a selfie.
  2. core.extract_id_card()        -> crop the card out of the photo
  3. core.detect_and_extract_face() -> find the face on the card
  4. core.verify_faces()           -> compare card face vs. selfie
  5. core.extract_text() + extract_information() -> OCR the card
  6. database.is_duplicate() / insert_record()   -> store the result

Fixes vs. the original app.py:
  - Removed the unused, conflicting `st.connection('mysql', ...)` — the
    app used two different DB connection mechanisms at once and only
    one was ever actually used.
  - Every pipeline step is checked for None/failure before the next step
    runs, with a clear st.error() instead of an unhandled exception
    crashing the whole app.
  - DOB formatting is handled once, consistently, in core.py.
"""

import logging
import streamlit as st

import core
import database

core.setup_logging()
logger = logging.getLogger("ekyc")


# --------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------- #
def configure_page():
    st.set_page_config(page_title="eKYC Verification", layout="wide")
    st.markdown(
        """
        <style>
            .block-container { max-width: 1100px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section() -> str:
    st.sidebar.title("ID Card Type")
    return st.sidebar.selectbox("Select ID type", ["PAN"])


# --------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------- #
def run_ekyc_pipeline(id_card_file, selfie_file):
    id_card_img = core.read_uploaded_image(id_card_file)
    if id_card_img is None:
        st.error("Could not read the ID card image. Try a different file.")
        return

    selfie_img = core.read_uploaded_image(selfie_file)
    if selfie_img is None:
        st.error("Could not read the selfie image. Try a different file.")
        return

    with st.spinner("Extracting ID card..."):
        card_crop = core.extract_id_card(id_card_img)
    if card_crop is None:
        st.error("Couldn't detect an ID card in the uploaded photo. Try a clearer, well-lit shot.")
        return

    with st.spinner("Locating face on ID card..."):
        card_face_path = core.detect_and_extract_face(card_crop, core.EXTRACTED_FACE_FILE)
    if card_face_path is None:
        st.error("Couldn't detect a face on the ID card.")
        return

    with st.spinner("Locating face in selfie..."):
        selfie_path = core.detect_and_extract_face(selfie_img, core.UPLOADED_FACE_FILE)
    if selfie_path is None:
        st.error("Couldn't detect a face in the selfie. Try a clearer, front-facing photo.")
        return

    with st.spinner("Verifying faces..."):
        verified = core.verify_faces(card_face_path, selfie_path)

    col1, col2 = st.columns(2)
    with col1:
        st.image(card_face_path, caption="Face on ID card", channels="BGR")
    with col2:
        st.image(selfie_path, caption="Uploaded selfie", channels="BGR")

    if not verified:
        st.error("❌ Face verification failed — the selfie doesn't match the ID card.")
        return
    st.success("✅ Face verified.")

    with st.spinner("Reading card text (OCR)..."):
        raw_text = core.extract_text(core.CONTOUR_FILE)
        info = core.extract_information(raw_text)

    if not info["ID"]:
        st.error("Couldn't read a valid PAN number off the card. Try re-uploading a sharper image.")
        return

    st.subheader("Extracted details")
    st.json(info)

    try:
        if database.is_duplicate(info["ID"]):
            st.warning(f"A record with ID {info['ID']} already exists.")
            st.dataframe(database.fetch_record(info["ID"]))
            return

        with st.spinner("Saving record..."):
            embedding = core.get_face_embedding(selfie_path)
            info["Embedding"] = embedding
            database.insert_record(info)
        st.success(f"🎉 New user registered with ID {info['ID']}.")

    except ConnectionError as e:
        st.error(str(e))
    except Exception as e:
        logger.error(f"Unexpected error saving record: {e}")
        st.error(f"Something went wrong while saving the record: {e}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main():
    configure_page()
    id_type = sidebar_section()
    st.title(f"eKYC Verification — {id_type} Card")

    id_card_file = st.file_uploader("Upload ID card photo", type=["jpg", "jpeg", "png"])
    selfie_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])

    if st.button("Verify", type="primary", disabled=not (id_card_file and selfie_file)):
        run_ekyc_pipeline(id_card_file, selfie_file)
    elif not (id_card_file and selfie_file):
        st.info("Upload both an ID card photo and a selfie to begin.")


if __name__ == "__main__":
    main()
