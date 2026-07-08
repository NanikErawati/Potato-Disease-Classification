import streamlit as st
import onnxruntime as ort
import numpy as np  # Streamlit sudah include versi yang kompatibel
import os

# ============================================
# LOAD MODEL ONNX
# ============================================
@st.cache_resource
def load_model():
    """Load model ONNX sekali saja (cached)"""
    model_path = os.path.join(os.path.dirname(__file__), "..", "Saved_Models", "1.onnx")
    return ort.InferenceSession(model_path)

@st.cache_resource
def load_class_names():
    """Load class names"""
    class_path = os.path.join(os.path.dirname(__file__), "..", "Saved_Models", "class_names_1.txt")
    with open(class_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Load model dan class names
with st.spinner("Loading model..."):
    session = load_model()
    CLASS_NAMES = load_class_names()
    input_name = session.get_inputs()[0].name

# ============================================
# UI
# ============================================
st.title("🥔 Potato Disease Classification")
st.markdown("Upload gambar daun kentang untuk deteksi penyakit")

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar (Streamlit handle sendiri)
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)
    
    # Tombol prediksi
    if st.button("🔍 Prediksi"):
        with st.spinner("Sedang menganalisis..."):
            # Preprocessing pakai Streamlit + NumPy
            import io
            from PIL import Image  # Streamlit sudah include Pillow
            
            image = Image.open(uploaded_file)
            image_resized = image.convert("RGB").resize((256, 256))
            img_array = np.expand_dims(np.array(image_resized).astype(np.float32), 0)
            
            # Predict dengan ONNX
            outputs = session.run(None, {input_name: img_array})
            predictions = outputs[0]
            
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = round(100 * float(np.max(predictions[0])), 2)
            
            # Tampilkan hasil
            st.success("Prediksi selesai!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Kelas", predicted_class.replace("Potato___", "").replace("_", " "))
            with col2:
                st.metric("Confidence", f"{confidence}%")
            
            st.progress(int(confidence))
            
            st.subheader("Detail Probabilitas:")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = round(100 * float(predictions[0][i]), 2)
                st.write(f"{class_name.replace('Potato___', '').replace('_', ' ')}: {prob}%")
                st.progress(int(prob))

st.markdown("---")
st.markdown("Powered by ONNX Runtime & Streamlit")