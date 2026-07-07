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
# Di Render, folder render/ ini jadi root working directory
# Model ada di ../Saved_Models/ (naik 1 level)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../render/
PROJECT_DIR = os.path.dirname(BASE_DIR)                  # .../Potato-Disease-Classification/
MODELS_DIR = os.path.join(PROJECT_DIR, "Saved_Models")

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 PROJECT_DIR: {PROJECT_DIR}")
print(f"📁 MODELS_DIR: {MODELS_DIR}")

# ============================================
# LOAD MODEL & CLASS NAMES
# ============================================

def load_model_file(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model tidak ditemukan: {model_path}")
    
    print(f"✅ Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_class_names(model_name: str):
    class_path = os.path.join(MODELS_DIR, f"class_names_{model_name}.txt")
    
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"❌ Class names tidak ditemukan: {class_path}")
    
    with open(class_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Load saat startup
CURRENT_MODEL = load_model_file("1")
CLASS_NAMES = load_class_names("1")

print(f"🚀 Model loaded: {CURRENT_MODEL.name}")
print(f"🚀 Classes: {CLASS_NAMES}")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {"message": "Potato Disease API - Render Deployment"}

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

@app.get("/model")
async def get_model_info():
    models_list = [
        f.replace(".keras", "") 
        for f in os.listdir(MODELS_DIR) 
        if f.endswith(".keras")
    ]
    return {
        "current_model": "1",
        "class_names": CLASS_NAMES,
        "available_models": models_list
    }

@app.post("/model/load")
async def switch_model(model_name: str):
    global CURRENT_MODEL, CLASS_NAMES
    
    try:
        CURRENT_MODEL = load_model_file(model_name)
        CLASS_NAMES = load_class_names(model_name)
        return {
            "status": "success",
            "message": f"Model {model_name} loaded",
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
    image = Image.open(BytesIO(data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((256, 256))
    return np.array(image)

def predict_image(model, img):
    if hasattr(img, 'numpy'):
        img = img.numpy()
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)
    
    return predicted_class, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    predicted_class, confidence = predict_image(CURRENT_MODEL, image)
    
    return {
        "class": predicted_class,
        "confidence": confidence
    }

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)