import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Ensure backend can import ml modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import health, search, predict, explain, analytics
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_service import RetrievalService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and index into memory
    print("Starting up MedFinder AI API...")
    EmbeddingService()
    RetrievalService()
    yield
    # Shutdown
    print("Shutting down API...")

app = FastAPI(
    title="MedFinder AI API",
    description="Medical Image Similarity & Decision-Support Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(predict.router, prefix="/api", tags=["Predict"])
app.include_router(explain.router, prefix="/api", tags=["Explain"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])

from backend.api import image
app.include_router(image.router, prefix="/api", tags=["Image"])

# Serve images statically
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
app.mount("/data", StaticFiles(directory=data_path), name="data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
