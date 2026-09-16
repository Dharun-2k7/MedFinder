import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from backend.services.prediction_service import PredictionService

router = APIRouter()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

@router.post("/predict")
async def predict_pathology(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")
        
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        
    try:
        service = PredictionService()
        result = service.predict(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
