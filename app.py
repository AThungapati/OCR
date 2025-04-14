import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import io

# --- Configuration ---

# --- Tesseract Path (IMPORTANT: Uncomment and set if Tesseract is not in your PATH) ---
# On Windows, it might be: 'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, it's often found automatically if installed via package manager.
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
except Exception as e:
    st.error(f"Tesseract executable not configured correctly: {e}")
    st.stop() # Stop execution if Tesseract path isn't set correctly (optional but recommended)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide") # Use wide layout for side-by-side view
st.title("📷 Live Webcam OCR")
st.write("Allow webcam access, capture an image containing text, and see the OCR result.")

# Initialize session state variables if they don't exist
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = ""
if 'image_key' not in st.session_state:
    st.session_state.image_key = 0 # Key to force camera reset if needed

# --- Main Logic ---

col1, col2 = st.columns(2) # Create two columns for layout

with col1:
    st.header("Webcam Feed")
    # The st.camera_input widget handles the live view and capture button
    # It returns an UploadedFile object when a picture is taken
    # Use a unique key that changes to potentially reset the camera state if needed
    img_file_buffer = st.camera_input(
        "Click 'Take Photo' below",
        key=f"camera_{st.session_state.image_key}"
        )

    if img_file_buffer is not None:
        # A new picture has been taken
        st.session_state.ocr_text = "" # Clear previous OCR text

        # Read the image data from the buffer
        bytes_data = img_file_buffer.getvalue()
        # Convert to a PIL Image object
        try:
            pil_image = Image.open(io.BytesIO(bytes_data))
            st.session_state.captured_image = pil_image # Store in session state
            st.info("✅ Image captured successfully!")

            # --- Perform OCR ---
            st.write("🔄 Running OCR...")
            try:
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(st.session_state.captured_image)
                st.session_state.ocr_text = text
                st.write("✅ OCR complete.")
            except pytesseract.TesseractNotFoundError:
                st.error("❌ Tesseract is not installed or not in your PATH.")
                st.error("Please install Tesseract and configure the path if necessary (see code comments).")
                st.session_state.captured_image = None # Clear image if OCR failed due to setup
            except Exception as e:
                st.error(f"❌ An error occurred during OCR: {e}")
                st.session_state.captured_image = None # Clear image on other OCR errors

        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")
            st.session_state.captured_image = None # Clear if image loading fails


with col2:
    st.header("Captured Image & OCR Result")
    if st.session_state.captured_image is not None:
        st.image(st.session_state.captured_image, caption='Captured Image', use_container_width=True)

        st.subheader("Extracted Text:")
        if st.session_state.ocr_text:
            st.text_area("OCR Output", st.session_state.ocr_text, height=250)
        elif img_file_buffer is not None: # Only show if capture was just attempted
             st.warning("No text detected or OCR process failed.")
        else:
             st.info("Capture an image to see the OCR result here.")

    else:
        st.info("Image will appear here after capturing.")

# Optional: Add a button to clear the captured image and OCR text
if st.button("Clear Image & Result"):
    st.session_state.captured_image = None
    st.session_state.ocr_text = ""
    st.session_state.image_key += 1 # Increment key to potentially reset camera
    st.rerun() # Rerun the app to reflect the changes

st.sidebar.info(
    """
    **How to Use:**
    1.  Allow webcam access when prompted.
    2.  Position text in front of the camera.
    3.  Click the 'Take Photo' button under the webcam feed.
    4.  The captured image and extracted text will appear on the right.
    5.  Click 'Clear Image & Result' to start over.
    """
)