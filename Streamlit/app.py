import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ============================================
# KONFIGURASI PAGE
# ============================================
st.set_page_config(
    page_title="Potato Disease Classification",
    page_icon="🥔",
    layout="centered"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load model sekali saja (cached)"""
    # Path dari folder Streamlit/ ke Saved_Models/
    model_path = os.path.join(os.path.dirname(__file__), "..", "Saved_Models", "1.keras")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_class_names():
    """Load class names"""
    class_path = os.path.join(os.path.dirname(__file__), "..", "Saved_Models", "class_names_1.txt")
    with open(class_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Load model dan class names
with st.spinner("Loading model..."):
    model = load_model()
    CLASS_NAMES = load_class_names()

# ============================================
# UI
# ============================================
st.title("🥔 Potato Disease Classification")
st.markdown("Upload gambar daun kentang untuk deteksi penyakit")

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    
    # Tombol prediksi
    if st.button("🔍 Prediksi"):
        with st.spinner("Sedang menganalisis..."):
            # Preprocessing
            image_resized = image.convert("RGB").resize((256, 256))
            img_array = np.expand_dims(np.array(image_resized), 0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = round(100 * float(np.max(predictions[0])), 2)
            
            # Tampilkan hasil
            st.success("Prediksi selesai!")
            
            # Card hasil
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Kelas", predicted_class.replace("Potato___", "").replace("_", " "))
            with col2:
                st.metric("Confidence", f"{confidence}%")
            
            # Progress bar confidence
            st.progress(int(confidence))
            
            # Detail semua kelas
            st.subheader("Detail Probabilitas:")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = round(100 * float(predictions[0][i]), 2)
                st.write(f"{class_name.replace('Potato___', '').replace('_', ' ')}: {prob}%")
                st.progress(int(prob))

# Footer
st.markdown("---")
st.markdown("Powered by TensorFlow & Streamlit")