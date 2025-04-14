import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2 # OpenCV
import os
import requests # To download model
import time # To add delay
import pytesseract # For OCR

# --- Configuration ---
EAST_MODEL_PATH = "frozen_east_text_detection.pb"
EAST_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
CONFIDENCE_THRESHOLD = 0.5 # EAST detection confidence
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
MAX_CAMERAS_TO_CHECK = 5
OCR_PADDING = 2 # Add padding around detected box before OCR

# --- Tesseract Path (IMPORTANT: Configure if needed) ---
# Uncomment and set the path if Tesseract is not in your system's PATH
# On Windows, use raw string: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, it might be: '/usr/bin/tesseract' or '/opt/homebrew/bin/tesseract'
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
except Exception as e:
    st.error(f"Tesseract executable not configured correctly: {e}")
    st.stop() # Stop execution if Tesseract path isn't set correctly (optional but recommended)

# --- Streamlit App Layout ---
# --- Helper Functions ---

@st.cache_data
def get_available_cameras(max_to_check):
    """Checks camera indices and returns a list of available ones."""
    available_cameras = []
    # st.write(f"Checking for cameras up to index {max_to_check - 1}...") # Reduce verbosity
    for i in range(max_to_check):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else None) # Use CAP_DSHOW on Windows for potential speedup
            if cap is not None and cap.isOpened():
                available_cameras.append(i)
                cap.release()
            elif cap is not None:
                 cap.release()
        except Exception:
            continue
    # st.write(f"Found cameras: {available_cameras}") # Reduce verbosity
    return available_cameras

def download_model(url, file_path):
    # (Keep the download_model function as before)
    if not os.path.exists(file_path):
        st.info(f"Downloading EAST model from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            st.success("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}"); st.stop()
        except Exception as e:
            st.error(f"An error occurred saving model: {e}"); st.stop()

def decode_predictions(scores, geometry, min_confidence):
    # (Keep the decode_predictions function as before)
    (numRows, numCols) = scores.shape[2:4]; rects = []; confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]; xData0 = geometry[0, 0, y]; xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]; xData3 = geometry[0, 3, y]; anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < min_confidence: continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0); angle = anglesData[x]
            cos = np.cos(angle); sin = np.sin(angle)
            h = xData0[x] + xData2[x]; w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w); startY = int(endY - h)
            rects.append((startX, startY, endX, endY)); confidences.append(scoresData[x])
    return (rects, confidences)

def detect_and_ocr_live(frame, net, orig_w, orig_h):
    """Processes frame for text detection, draws boxes, and performs OCR."""
    processed_frame = frame.copy()
    ocr_results = [] # List to store (box, text) tuples

    # --- Text Detection (EAST) ---
    resized_image = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    (H, W) = resized_image.shape[:2]
    rW = orig_w / float(W); rH = orig_h / float(H)
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    output_layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    try:
        (scores, geometry) = net.forward(output_layer_names)
    except cv2.error as e:
        # st.warning(f"Detection Error: {e}") # Can be noisy
        return processed_frame, ocr_results # Return original frame on error

    (rects, confidences) = decode_predictions(scores, geometry, CONFIDENCE_THRESHOLD)
    boxes_for_nms = [(x1, y1, x2-x1, y2-y1) for (x1, y1, x2, y2) in rects]
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # --- Draw Boxes and Perform OCR ---
    if len(indices) > 0:
        final_indices = indices.flatten()
        for i in final_indices:
            # Get original box coordinates before NMS conversion
            (startX_norm, startY_norm, endX_norm, endY_norm) = rects[i]

            # Scale box coordinates back to original frame size
            startX = int(startX_norm * rW)
            startY = int(startY_norm * rH)
            endX = int(endX_norm * rW)
            endY = int(endY_norm * rH)

            # Add padding for OCR (ensure coordinates stay within frame)
            pad = OCR_PADDING
            startX_pad = max(0, startX - pad)
            startY_pad = max(0, startY - pad)
            endX_pad = min(orig_w, endX + pad)
            endY_pad = min(orig_h, endY + pad)

            # Draw the bounding box on the processed frame
            cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # --- Extract ROI and OCR ---
            roi = frame[startY_pad:endY_pad, startX_pad:endX_pad]

            if roi.size > 0: # Check if ROI is valid
                try:
                    # Configuration: --psm 7 assumes a single text line (often good for EAST boxes)
                    # Increase --oem if needed (e.g., 1 for LSTM only)
                    # Can add language options: lang='eng' (default) or lang='eng+fra' etc.
                    ocr_config = r'--oem 3 --psm 7' # Default settings often work best initially
                    text = pytesseract.image_to_string(roi, config=ocr_config)

                    # Basic text cleaning
                    cleaned_text = "".join(c for c in text if c.isalnum() or c in ' .-:/').strip()

                    if cleaned_text:
                        # Draw text onto the processed frame above the box
                        cv2.putText(processed_frame, cleaned_text, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Red text
                        ocr_results.append(((startX, startY, endX, endY), cleaned_text))

                except Exception as ocr_error:
                    # Can log OCR errors if needed, but avoid flooding the UI
                    # print(f"OCR Error on ROI: {ocr_error}")
                    pass # Continue to next box if OCR fails on one

    return processed_frame, ocr_results

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("📷 Live Webcam Text Detection & OCR")
st.warning("⚠️ Live OCR is computationally intensive and may reduce frame rate significantly.")

# --- Download and Load Model ---
download_model(EAST_MODEL_URL, EAST_MODEL_PATH)
try:
    net = cv2.dnn.readNet(EAST_MODEL_PATH)
except cv2.error as e:
    st.error(f"Error loading OpenCV DNN model: {e}"); st.stop()

# --- Camera Selection ---
st.sidebar.header("Camera Options")
available_camera_indices = get_available_cameras(MAX_CAMERAS_TO_CHECK)
if not available_camera_indices:
    st.error("No available cameras found."); st.stop()

# --- Session State ---
if 'run_detection' not in st.session_state: st.session_state.run_detection = False
if 'video_capture' not in st.session_state: st.session_state.video_capture = None
if 'selected_camera' not in st.session_state: st.session_state.selected_camera = available_camera_indices[0]
if 'ocr_output_text' not in st.session_state: st.session_state.ocr_output_text = ""

def camera_changed():
    if st.session_state.run_detection:
        st.session_state.run_detection = False
        if st.session_state.video_capture is not None:
            st.session_state.video_capture.release()
            st.session_state.video_capture = None
        st.info("Camera changed. Press 'Start Detection'.")
        time.sleep(0.5)
        st.rerun()

selected_cam_index = st.sidebar.selectbox(
    "Select Camera Device:", options=available_camera_indices,
    format_func=lambda x: f"Camera {x}", key='selected_camera', on_change=camera_changed
)

# --- Controls ---
st.sidebar.header("Controls")
start_button_disabled = st.session_state.run_detection
stop_button_disabled = not st.session_state.run_detection

if st.sidebar.button("Start Detection & OCR", key="start", disabled=start_button_disabled):
    st.session_state.run_detection = True
    st.session_state.ocr_output_text = "" # Clear previous OCR text
    st.rerun()

if st.sidebar.button("Stop Detection & OCR", key="stop", disabled=stop_button_disabled):
    st.session_state.run_detection = False
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release()
        st.session_state.video_capture = None
    st.rerun()

# --- Main Display Areas ---
col1, col2 = st.columns([3, 1]) # Make video column wider

with col1:
    st.header("Live Feed with OCR")
    FRAME_WINDOW = st.image([]) # Placeholder for video frames

with col2:
    st.header("OCR Output")
    OCR_TEXT_AREA = st.text_area("Detected Text:", value=st.session_state.ocr_output_text, height=400, key="ocr_area")


# --- Main Processing Loop ---
if st.session_state.run_detection:
    if st.session_state.video_capture is None:
        try:
            st.info(f"Attempting to open Camera {st.session_state.selected_camera}...")
            # Try CAP_DSHOW on Windows for potentially better performance/compatibility
            api_preference = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            st.session_state.video_capture = cv2.VideoCapture(st.session_state.selected_camera, api_preference)
            st.session_state.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Try setting a lower resolution
            st.session_state.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not st.session_state.video_capture.isOpened():
                st.error(f"Error: Could not open Camera {st.session_state.selected_camera}.")
                st.session_state.run_detection = False; st.session_state.video_capture = None; st.rerun()
            else:
                 st.success(f"Camera {st.session_state.selected_camera} opened.")
        except Exception as e:
            st.error(f"Error initializing Camera {st.session_state.selected_camera}: {e}")
            st.session_state.run_detection = False; st.session_state.video_capture = None; st.rerun()

    if st.session_state.video_capture and st.session_state.video_capture.isOpened():
        # st.info("🚀 Running live detection & OCR...") # Can be repetitive
        cumulative_ocr_text = ""
        while st.session_state.run_detection:
            cap = st.session_state.video_capture
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame. Stopping."); st.session_state.run_detection = False; break

            (orig_h, orig_w) = frame.shape[:2]

            # Process frame for detection and OCR
            processed_frame_bgr, ocr_results = detect_and_ocr_live(frame, net, orig_w, orig_h)

            # Convert frame for display
            processed_frame_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)

            # Update video display
            FRAME_WINDOW.image(processed_frame_rgb)

            # Update OCR text area (accumulate text from the current frame)
            current_frame_text = "\n".join([text for _, text in ocr_results])
            # Update session state for the text area (doing it inside loop makes it responsive)
            # Only update if text actually changed to reduce rerenders (minor optimization)
            if current_frame_text != st.session_state.get("last_ocr_text", ""):
                 st.session_state.ocr_output_text = current_frame_text
                 st.session_state.last_ocr_text = current_frame_text # Store for comparison
                 # Manually trigger update for text_area (might not be needed if key is set)
                 # OCR_TEXT_AREA.text_area("Detected Text:", value=st.session_state.ocr_output_text, height=400, key="ocr_area") # Re-render?


            time.sleep(0.01) # Crucial delay

        # Cleanup after loop stops
        if st.session_state.video_capture:
             st.session_state.video_capture.release()
             st.session_state.video_capture = None
             if not st.session_state.run_detection: # Only if stopped intentionally
                 st.info("⏹️ Detection stopped. Camera released.")
                 FRAME_WINDOW.empty()


elif not st.session_state.run_detection:
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release(); st.session_state.video_capture = None
    st.info("💡 Select camera, press 'Start Detection & OCR'.")
    FRAME_WINDOW.empty()
    # Optionally clear text area when stopped
    # if st.session_state.ocr_output_text:
    #     st.session_state.ocr_output_text = ""
    #     OCR_TEXT_AREA.text_area("Detected Text:", value="", height=400, key="ocr_area")


st.sidebar.markdown("---")
st.sidebar.info("Using EAST model for detection and Tesseract for OCR.")
if 'tesseract_version' in locals():
     st.sidebar.info(f"Tesseract Version: {tesseract_version}")
st.sidebar.markdown(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD}")