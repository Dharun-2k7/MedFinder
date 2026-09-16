from pydantic import BaseModel
from typing import List, Optional

class SearchResultItem(BaseModel):
    id: int
    similarity_score: float
    finding: str
    split: str
    original_index: int

class SearchResponse(BaseModel):
    query_id: str
    results: List[SearchResultItem]
    predictions: Optional[dict] = None
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
