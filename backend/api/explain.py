import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from backend.services.explainability_service import ExplainabilityService

router = APIRouter()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

@router.post("/explain")
async def explain_similarity(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")
        
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        
    try:
        service = ExplainabilityService()
        heatmap_base64, target_class = service.generate_heatmap(image)
        return {
            "heatmap_base64": heatmap_base64,
            "target_class": target_class
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {e}")
