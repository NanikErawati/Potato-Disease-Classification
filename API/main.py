from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# ============================================
# CORS
# ============================================
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# KONFIGURASI PATH
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "Saved_Models")

# ============================================
# LOAD MODEL & CLASS NAMES DARI FILE
# ============================================

def load_model_file(model_name: str):
    """Load model .keras dari Saved_Models/"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    
    print(f"Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_class_names(model_name: str):
    """Load class_names dari file .txt yang disimpan saat training"""
    class_names_path = os.path.join(MODELS_DIR, f"class_names_{model_name}.txt")
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names tidak ditemukan: {class_names_path}")
    
    with open(class_names_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Load model dan class names
CURRENT_MODEL = load_model_file("1")
CLASS_NAMES = load_class_names("1")

print(f"Model loaded: {CURRENT_MODEL.name}")
print(f"Class names: {CLASS_NAMES}")

# ============================================
# ENDPOINTS
# ============================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="https://www.google.com/favicon.ico")

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

@app.get("/model")
async def get_current_model():
    """Lihat model yang sedang aktif"""
    return {
        "current_model": "1",
        "class_names": CLASS_NAMES,
        "available_models": [f.replace(".keras", "") for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
    }

@app.post("/model/load")
async def switch_model(model_name: str):
    """Ganti model yang sedang aktif"""
    global CURRENT_MODEL, CLASS_NAMES
    
    try:
        CURRENT_MODEL = load_model_file(model_name)
        CLASS_NAMES = load_class_names(model_name)
        
        return {
            "status": "success",
            "message": f"Model {model_name} berhasil diload",
            "model_name": model_name,
            "class_names": CLASS_NAMES
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

# ============================================
# PREDIKSI
# ============================================
def read_file_as_image(data) -> np.ndarray:
    """
    Baca file gambar dan preprocess:
    - Konversi ke RGB (3 channel)
    - Resize ke 256x256 (sama dengan training)
    """
    image = Image.open(BytesIO(data))
    
    # Konversi ke RGB (hilangkan alpha channel kalau ada)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ke 256x256
    image = image.resize((256, 256))
    
    return np.array(image)

def predict_image(model, img):
    """
    Prediksi single image - SAMA PERSIS dengan fungsi di training.ipynb
    """
    # Konversi ke array jika perlu
    if hasattr(img, 'numpy'):
        img = img.numpy()
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch
    
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)
    
    return predicted_class, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Baca gambar dari upload
    image = read_file_as_image(await file.read())
    
    # Prediksi menggunakan fungsi yang sama dengan training
    predicted_class, confidence = predict_image(CURRENT_MODEL, image)
    
    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)