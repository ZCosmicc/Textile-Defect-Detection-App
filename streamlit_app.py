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
    
    # Load YOLO model - Updated path to match your file name
    if YOLO_AVAILABLE:
        try:
            model = YOLO('weights/bestYOLOv8.pt')  # Changed from 'best.pt' to 'bestYOLOv8.pt'
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading YOLO model: {str(e)}")
            model_loaded = False
    else:
        st.warning("YOLO is not installed. Only basic image upload will be available.")
        model_loaded = False
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Add detect button
    if st.button("Detect Image"):
        if uploaded_file is None:
            st.warning("Please upload an image first")
        elif not YOLO_AVAILABLE or not model_loaded:
            st.warning("YOLO model is not available. Please check your installation.")
        else:
            with st.spinner("Detecting defects..."):
                try:
                    # Convert PIL Image to numpy array
                    image_array = np.array(image)
                    
                    # Get detection results
                    results = model(image_array)
                    
                    # Display results
                    with col2:
                        st.subheader("Result")
                        res_plotted = results[0].plot()
                        st.image(res_plotted, caption='Detection Result', use_container_width=True)
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")

if __name__ == "__main__":
    main()