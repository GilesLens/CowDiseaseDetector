from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2

path = "C:\\Users\\gilbe\\ML Deployment\\BaseModel.keras"

# Load the trained model
model = tf.keras.models.load_model(path)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (authorization, content-type, etc.)
)

# Function to preprocess image
def preprocess_image(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB ✅
    
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
def home():
    return {"message": "Welcome to the Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    label = "Lumpy Cow (1)" if prediction > 0.5 else "Healthy Cow (0)"
    
    return {"prediction": label}