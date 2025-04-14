import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import io
import os
import tempfile
from google import genai
from google.genai import types

# ---- Configurations ----
st.set_page_config(layout="wide")
st.title("📖 OCR Comparison")

# Set tesseract path if necessary
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
except Exception as e:
    st.error(f"Tesseract executable not configured correctly: {e}")
    st.stop()

# Initialize session
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'tesseract_text' not in st.session_state:
    st.session_state.tesseract_text = ""
if 'gemini_text' not in st.session_state:
    st.session_state.gemini_text = ""
if 'image_key' not in st.session_state:
    st.session_state.image_key = 0

# Function for Gemini OCR
def extract_with_gemini(image):
    client = genai.Client(api_key="AIzaSyBZWbaJTgKicwCX91M0DfK6jv7onEUALzg")  # Replace with your actual key
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_path = temp_file.name

        uploaded_file = client.files.upload(file=temp_path)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(
                        text="Extract text from the book spine exactly as it appears.  "
                             "Return only the raw book's titles in form of numbered list without any other comments. "
                             "Just give the titles without any other pre-text like `Here are the book titles from the image, exactly as they appear:.....` or `Here are the book titles from the image:` or any other text EVER"
                    ),
                ],
            )
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )

        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("📷 Capture Image")
    img_file = st.camera_input("Take a photo", key=f"camera_{st.session_state.image_key}")
    if img_file is not None:
        image = Image.open(io.BytesIO(img_file.getvalue()))
        st.session_state.captured_image = image
        st.image(image, caption="Captured Image", use_container_width=True)

        # --- Tesseract OCR ---
        with st.spinner("🔍 Extracting with Tesseract..."):
            try:
                tesseract_text = pytesseract.image_to_string(image)
                st.session_state.tesseract_text = tesseract_text
            except Exception as e:
                st.session_state.tesseract_text = f"Error: {e}"

        # --- Gemini OCR ---
        with st.spinner("🔍 Extracting..."):
            gemini_text = extract_with_gemini(image)
            st.session_state.gemini_text = gemini_text

with col2:
    st.header("🔎 OCR Comparison")

    if st.session_state.captured_image:
        col_tesseract, col_gemini  = st.columns(2)
        
        with col_tesseract:
            st.subheader("🧠 Model 1 Result")
            st.text_area("OCR Output", st.session_state.tesseract_text, height=300)

        with col_gemini:
            st.subheader("🌐 Model 2 Result")
            st.text_area("OCR Output", st.session_state.gemini_text, height=300)
    else:
        st.info("Capture an image to begin comparison.")

# --- Reset Button ---
if st.button("🔄 Reset"):
    st.session_state.captured_image = None
    st.session_state.tesseract_text = ""
    st.session_state.gemini_text = ""
    st.session_state.image_key += 1
    st.rerun()
