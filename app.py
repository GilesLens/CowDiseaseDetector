import os
import uvicorn
import cv2
import numpy as np
import csv
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8
model = YOLO("yolov8n.pt")

# ========== HELPER FUNCTIONS ==========

def preprocess_image(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def save_to_csv(data_list):
    csv_file = "lesion_metadata_log.csv"
    fieldnames = [
        "image_file", "x1", "y1", "x2", "y2", "confidence", "stage",
        "cause", "onset", "texture", "color", "itching", "rain_exposure"
    ]
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for data in data_list:
            writer.writerow(data)

# ========== 1: PREDICT ENDPOINT ==========

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    results = model(img, conf=0.3)
    all_bboxes = []

    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = scores[i]

            stage = "Severe" if confidence > 0.8 else "Moderate" if confidence > 0.5 else "Mild"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{stage} ({confidence:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            all_bboxes.append({
                "image_file": file.filename,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": confidence, "stage": stage
            })

    if len(all_bboxes) == 0:
        return {"status": "healthy", "message": f"No lumps detected for {file.filename}"}

    # Save detection results for follow-up metadata collection
    with open("detection_results.csv", mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_bboxes[0].keys())
        writer.writeheader()
        writer.writerows(all_bboxes)

    return {"status": "lumpy", "message": f"{len(all_bboxes)} lumps detected", "file": file.filename}

# ========== 2: METADATA COLLECTION ENDPOINT ==========

@app.post("/collect-metadata")
async def collect_metadata(
    image_file: str = Form(...),
    cause: str = Form(...),
    onset: str = Form(...),
    texture: str = Form(...),
    color: str = Form(...),
    itching: str = Form(...),
    rain_exposure: str = Form(...)
):
    metadata = {
        "cause": cause,
        "onset": onset,
        "texture": texture,
        "color": color,
        "itching": itching,
        "rain_exposure": rain_exposure
    }

    # Load previous detection results and enrich with metadata
    enriched_data = []
    with open("detection_results.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image_file"] == image_file:
                row.update(metadata)
                enriched_data.append(row)

    save_to_csv(enriched_data)
    return {"status": "metadata saved", "file": image_file}

# ========== 3: DIAGNOSIS ENDPOINT ==========

@app.get("/diagnose")
def diagnose():
    diagnoses = []
    with open("lesion_metadata_log.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cause"] == "1":
                diagnosis = "Tick-borne"
            elif row["cause"] == "2":
                diagnosis = "Viral (possible Bovine Herpes or Pseudocowpox)"
            elif row["cause"] == "3":
                diagnosis = "Fungal (possible Ringworm)"
            elif row["cause"] == "4":
                diagnosis = "Allergy or Hypersensitivity"
            elif row["rain_exposure"] == "1":
                diagnosis = "Rain scald (Dermatophilosis)"
            else:
                diagnosis = "Other (requires further investigation)"

            diagnoses.append({
                "image_file": row["image_file"],
                "x1": row["x1"], "y1": row["y1"],
                "diagnosis": diagnosis
            })

    return {"diagnoses": diagnoses}

# ========== 4: SERVE HTML (Optional) ==========

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ========== MAIN ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)