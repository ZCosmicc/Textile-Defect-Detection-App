import streamlit as st
from PIL import Image
import numpy as np
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

def main():
    st.title("Textile Defect Detection System")
    
    # Load YOLO model
    if YOLO_AVAILABLE:
        try:
            model = YOLO('weights/bestYOLOv8.pt')
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading YOLO model: {str(e)}")
            model_loaded = False
    else:
        st.warning("YOLO is not installed. Only basic image upload will be available.")
        model_loaded = False
    
    # File uploader at the top
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Store our detection result in session state so it persists across reruns
    if 'detection_result' not in st.session_state:
        st.session_state.detection_result = None
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    # Input image display
    with col1:
        st.subheader("Input Image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, use_container_width=True)
    
    # Result display
    with col2:
        st.subheader("Result")
        if st.session_state.detection_result is not None:
            st.image(st.session_state.detection_result, use_container_width=True)
            st.caption("Detection Result")
    
    # Button below both columns
    if st.button("Detect Image", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload an image first")
        elif not YOLO_AVAILABLE or not model_loaded:
            st.warning("YOLO model is not available. Please check your installation.")
        else:
            with st.spinner("Detecting defects..."):
                try:
                    # Convert PIL Image to numpy array
                    image_array = np.array(image)
                    
                    # Extra check for channels
                    if len(image_array.shape) == 2 or (len(image_array.shape) == 3 and image_array.shape[2] == 1):
                        # Convert grayscale to RGB
                        image_array = np.stack((image_array,)*3, axis=-1) if len(image_array.shape) == 2 else np.concatenate([image_array]*3, axis=2)
                    
                    # Get detection results
                    results = model(image_array)
                    
                    # Store result in session state
                    st.session_state.detection_result = results[0].plot()
                    
                    # Rerun to update the UI with the result
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    # Add more debug info
                    if uploaded_file is not None:
                        img = Image.open(uploaded_file)
                        st.error(f"Image mode: {img.mode}, Size: {img.size}, Format: {img.format}")

if __name__ == "__main__":
    main()