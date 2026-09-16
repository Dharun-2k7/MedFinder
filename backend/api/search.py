import time
import uuid
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from backend.schemas.response import SearchResponse, SearchResultItem
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_service import RetrievalService

router = APIRouter()

# Validate image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

@router.post("/search", response_model=SearchResponse)
async def search_similar_images(file: UploadFile = File(...)):
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    # 1. Validate File
    ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")
        
    try:
        content = await file.read()
        # Set a reasonable file size limit (e.g., 10MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
            
        image = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        
    # 2. Extract Embedding
    try:
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.embed_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {e}")
        
    # 3. Retrieve Similar
    try:
        retrieval_service = RetrievalService()
        results_dicts = retrieval_service.search(query_embedding, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
        
    # 4. Format Output
    results = [SearchResultItem(**res) for res in results_dicts]
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return SearchResponse(
        query_id=query_id,
        results=results,
        predictions=None, # To be implemented in Milestone 7
        processing_time_ms=processing_time_ms
    )
