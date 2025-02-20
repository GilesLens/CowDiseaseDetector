import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf

# Force TensorFlow to use CPU (Prevents cuBLAS issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Path to the model (Modify if using Render)
MODEL_PATH = "BaseModel.keras"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to preprocess image
def preprocess_image(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
def home():
    return {"message": "Welcome to the Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        label = "Lumpy Cow (1)" if prediction > 0.5 else "Healthy Cow (0)"
        
        return {"prediction": label}
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Main entry point for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)