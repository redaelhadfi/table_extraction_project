# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import pandas as pd
import pytesseract # Import for exception handling

from app.table_processor import extract_table_from_image, compute_boxes_from_image
# Import models_loader to ensure models are loaded at startup
from app import models_loader


app = FastAPI(title="Table Extraction API")

# Mount static files (for CSS, JS if any)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    print("Application startup: Ensuring models are loaded.")
    try:
        models_loader.get_models_and_processor()
        print("Models verified on startup.")
    except Exception as e:
        print(f"Error during model loading on startup: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: FastAPIRequest):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract_table/")
async def api_extract_table(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        df = extract_table_from_image(pil_image) 
        
        if df.empty:
            return JSONResponse(content={"message": "No table found or table is empty.", "headers": [], "rows": []}, status_code=200)
            
        table_data_records = df.to_dict(orient='records')
        headers = df.columns.tolist()

        return JSONResponse(content={"headers": headers, "rows": table_data_records})

    except ConnectionError as e: 
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(status_code=500, detail="Tesseract is not installed or not found in your PATH.")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect_structures/")
async def api_detect_structures(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        boxes, labels, scores = compute_boxes_from_image(pil_image)
        
        structure_model_ref, _ = models_loader.get_models_and_processor()
        id2label = structure_model_ref.config.id2label

        detections = []
        for box, label_id, score in zip(boxes, labels, scores):
            detections.append({
                "box": [round(b, 2) for b in box],
                "label": id2label.get(label_id, "unknown"),
                "score": round(score, 3)
            })
        return JSONResponse(content={"detections": detections})
    except Exception as e:
        print(f"Error in detect_structures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# For debugging app.main directly:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
