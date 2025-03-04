import os
import cv2
import csv
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Setup FastAPI app
app = FastAPI()

# CORS (so frontend can talk to backend even if they are hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving (for processed images in /static/annotated)
app.mount("/static", StaticFiles(directory="static"), name="static")

# YOLOv8 model (make sure 'yolov8n.pt' is in the same folder as app.py)
model = YOLO("yolov8n.pt")

# Templating for frontend (like your index.html)
templates = Jinja2Templates(directory="templates")

# Ensure folders exist
os.makedirs("static/annotated", exist_ok=True)

# ===================== Route 1: Serve Frontend =====================

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ===================== Route 2: Predict (Detect & Annotate) =====================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(img, conf=0.3)

        all_bboxes = []

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = scores[i]

                # Severity classification based on confidence
                if confidence > 0.8:
                    stage = "Severe"
                elif confidence > 0.5:
                    stage = "Moderate"
                else:
                    stage = "Mild"

                # Draw bounding box & label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{stage} ({confidence:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                all_bboxes.append({
                    "image_file": file.filename,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": confidence, "stage": stage
                })

        if len(all_bboxes) == 0:
            return {"status": "healthy", "message": "No lumps detected."}

        # Save annotated image to /static/annotated
        annotated_path = f"static/annotated/{file.filename}"
        cv2.imwrite(annotated_path, img)

        # Save bounding box data to detection_results.csv
        with open("detection_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_bboxes[0].keys())
            writer.writeheader()
            writer.writerows(all_bboxes)

        return {"status": "lumpy", "message": f"{len(all_bboxes)} lumps detected", "filename": file.filename}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ===================== Route 3: Collect Metadata (Observation Form) =====================

@app.post("/collect-metadata")
async def collect_metadata(
    image_file: str = Form(...),
    onset: str = Form(...),
    texture: str = Form(...),
    color: str = Form(...),
    itching: str = Form(...),
    rain_exposure: str = Form(...)
):
    metadata = {
        "onset": onset,
        "texture": texture,
        "color": color,
        "itching": itching,
        "rain_exposure": rain_exposure
    }

    enriched_data = []
    with open("detection_results.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image_file"] == image_file:
                row.update(metadata)
                enriched_data.append(row)

    save_to_csv(enriched_data)

    return {"status": "metadata saved", "file": image_file}

# ===================== Route 4: Diagnose (Decision Tree) =====================

@app.get("/diagnose")
def diagnose():
    diagnoses = []
    with open("lesion_metadata_log.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diagnosis = "Other (needs further check)"

            if row["rain_exposure"] == "1" and row["texture"] == "2":
                diagnosis = "Rain scald (Dermatophilosis)"
            elif row["onset"] == "1" and row["texture"] == "1" and row["color"] == "1":
                diagnosis = "Viral (BHV-2 or Pseudocowpox)"
            elif row["onset"] == "2" and row["texture"] == "1" and row["color"] == "2":
                diagnosis = "Tick-borne (Possible Onchocercosis)"
            elif row["texture"] == "3" and row["itching"] == "1":
                diagnosis = "Allergy/Hypersensitivity"
            elif row["texture"] == "2" and row["color"] == "3":
                diagnosis = "Fungal (Ringworm)"

            diagnoses.append({
                "image_file": row["image_file"],
                "x1": row["x1"], "y1": row["y1"],
                "diagnosis": diagnosis
            })

    return {"diagnoses": diagnoses}

# ===================== Helper: Save to CSV =====================

def save_to_csv(data_list):
    with open("lesion_metadata_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data_list[0].keys())
        if os.stat("lesion_metadata_log.csv").st_size == 0:
            writer.writeheader()
        writer.writerows(data_list)

# ===================== Main =====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)