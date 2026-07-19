import os
import time
import glob
import uuid
import cv2
import numpy as np
import streamlit as st
import insightface
from insightface.app import FaceAnalysis
# Add this helper to your load_models() function inside app.py
import urllib.request
if not os.path.exists('models/inswapper_128.onnx'):
    os.makedirs('models', exist_ok=True)
    urllib.request.urlretrieve(
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx", 
        "models/inswapper_128.onnx"
    )

# --- CONFIGURATION & DIRECTORIES ---
UPLOAD_DIR = "temp_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="AI Face Swapper Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom responsive CSS injection for mobile-first feel
st.markdown("""
    <style>
    .reportview-container .main .block-container { max-width: 800px; padding-top: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #FF4B4B; color: white; }
    @media (max-width: 640px) {
        .col-mobile { display: flex; flex-direction: column; }
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL INITIALIZATION (CACHED) ---
@st.cache_resource
def load_models():
    """Initializes Face Analysis and the Inswapper ONNX model."""
    # Initialize detector (uses CPU/GPU automatically depending on onnxruntime build)
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load the specific 128x128 face swapper model
    model_path = 'models/inswapper_128.onnx'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please download inswapper_128.onnx.")
        st.stop()
        
    swapper = insightface.model_zoo.get_model(model_path, download=False, check_cv2_onerror=False)
    return app, swapper

try:
    face_analyzer, face_swapper = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# --- POST-PROCESSING ENGINE ---
def post_process_blend(source_img, target_img, face_landmarks):
    """
    Applies Poisson Blending and local sharpening to eliminate harsh structural lines
    around the edges of the swapped face.
    """
    # Create a mask around the facial landmarks boundary
    hull = cv2.convexHull(np.array(face_landmarks, dtype=np.int32))
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Calculate bounding box to find the center for seamlessClone
    r = cv2.boundingRect(hull)
    center = ((r[0] + r[0] + r[2]) // 2, (r[1] + r[1] + r[3]) // 2)
    
    # Execute seamless Poisson cloning
    blended = cv2.seamlessClone(source_img, target_img, mask, center, cv2.NORMAL_CLONE)
    
    # Mild unsharp mask filter to restore lost high-frequency texture details
    gaussian = cv2.GaussianBlur(blended, (0, 0), 2.0)
    sharpened = cv2.addWeighted(blended, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def process_face_swap(source_bytes, target_bytes, swap_mode):
    """Decodes buffers, tracks multiple target faces, swaps, and saves the output."""
    # Convert image buffers to OpenCV matrices
    src_arr = np.frombuffer(source_bytes.read(), np.uint8)
    tgt_arr = np.frombuffer(target_bytes.read(), np.uint8)
    
    img_src = cv2.imdecode(src_arr, cv2.IMREAD_COLOR)
    img_tgt = cv2.imdecode(tgt_arr, cv2.IMREAD_COLOR)
    
    # Detect faces
    src_faces = face_analyzer.get(img_src)
    tgt_faces = face_analyzer.get(img_tgt)
    
    if not src_faces:
        raise ValueError("No visible face detected in the Source Image.")
    if not tgt_faces:
        raise ValueError("No visible faces detected in the Target Image.")
        
    # Always use the primary/largest detected face from the source image
    src_face = src_faces[0]
    
    # Copy target to apply modifications iteratively
    result_img = img_tgt.copy()
    
    # Process swapping based on User configuration
    if swap_mode == "Single (First Detected)":
        # Swap onto the first detected target face
        result_img = face_swapper.get(result_img, tgt_faces[0], src_face, paste_back=True)
        # Apply post-processing using the target face landmarks mapping
        result_img = post_process_blend(result_img, img_tgt, tgt_faces[0].landmark_2d_106)
    else:
        # Loop through all faces detected in the target image
        for tgt_face in tgt_faces:
            result_img = face_swapper.get(result_img, tgt_face, src_face, paste_back=True)
            result_img = post_process_blend(result_img, result_img, tgt_face.landmark_2d_106)
            
    # Generate unique caching file signature
    filename = f"swap_{uuid.uuid4().hex}.png"
    output_path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(output_path, result_img)
    
    return output_path

# --- UI / FRONTEND LAYOUT ---
st.title("🎭 AI Face Swapper Pro")
st.caption("High-Fidelity Context-Aware Face Swapping | Mobile-Responsive")

# Interactive configurations
swap_mode = st.radio("Target Selection Mode", ["Single (First Detected)", "All Detected Faces (Multi-Swap)"], horizontal=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Source Face")
    src_file = st.file_uploader("Upload image containing source face", type=["jpg", "jpeg", "png", "webp"], key="src")
    if src_file:
        st.image(src_file, use_column_width=True)

with col2:
    st.subheader("2. Target Canvas")
    tgt_file = st.file_uploader("Upload image to modify", type=["jpg", "jpeg", "png", "webp"], key="tgt")
    if tgt_file:
        st.image(tgt_file, use_column_width=True)

st.markdown("---")

if src_file and tgt_file:
    if st.button("Execute Face Swap"):
        with st.spinner("Analyzing structures and blending skin tones..."):
            try:
                output_image_path = process_face_swap(src_file, tgt_file, swap_mode)
                
                # Inline success layout
                st.success("Face swap completed successfully!")
                
                # Side-by-Side comparison views
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown("**Original Target Canvas**")
                    st.image(tgt_file, use_column_width=True)
                with res_col2:
                    st.markdown("**Swapped Result**")
                    st.image(output_image_path, use_column_width=True)
                
                # Download implementation
                with open(output_image_path, "rb") as file:
                    st.download_button(
                        label="💾 Download Swapped Image",
                        data=file,
                        file_name="ai_face_swap_result.png",
                        mime="image/png"
                    )
            except Exception as err:
                st.error(f"Processing Error: {err}")
else:
    st.info("💡 Please upload both a Source Face image and a Target Canvas image to begin.")
