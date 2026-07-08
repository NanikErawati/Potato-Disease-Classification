import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ============================================
# LOAD MODEL
# ============================================
# Path dari folder HuggingFace/ ke Saved_Models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Saved_Models", "1.keras")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "..", "Saved_Models", "class_names_1.txt")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded: {model.name}")
print(f"✅ Classes: {CLASS_NAMES}")

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict(image):
    """
    Prediksi gambar daun kentang
    image: PIL Image dari Gradio
    """
    # Preprocessing (sama persis dengan training)
    image = image.convert("RGB").resize((256, 256))
    img_array = np.expand_dims(np.array(image), 0)  # (1, 256, 256, 3)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)
    
    # Format output
    result_text = f"**{predicted_class.replace('Potato___', '').replace('_', ' ')}**\n\nConfidence: **{confidence}%**"
    
    # Detail semua kelas
    details = {}
    for i, class_name in enumerate(CLASS_NAMES):
        prob = round(100 * float(predictions[0][i]), 2)
        details[class_name.replace('Potato___', '').replace('_', ' ')] = prob
    
    return result_text, details

# ============================================
# GRADIO INTERFACE
# ============================================
with gr.Blocks(title="Potato Disease Classification") as demo:
    gr.Markdown("# 🥔 Potato Disease Classification")
    gr.Markdown("Upload gambar daun kentang untuk deteksi penyakit")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Gambar")
            predict_btn = gr.Button("🔍 Prediksi", variant="primary")
        
        with gr.Column():
            output_text = gr.Markdown(label="Hasil")
            output_json = gr.JSON(label="Detail Probabilitas")
    
    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_text, output_json]
    )
    
    gr.Markdown("---")
    gr.Markdown("Powered by TensorFlow & Gradio")

# Launch
demo.launch()