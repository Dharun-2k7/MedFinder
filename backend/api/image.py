import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import os

router = APIRouter()

# We cache the datasets in memory
_datasets = {}

def get_dataset(split):
    if split not in _datasets:
        from ml.datasets.medmnist_dataset import MedMNISTDataset
        dataset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        _datasets[split] = MedMNISTDataset(split=split, dataset_root=dataset_root, transform=lambda x: x) # no transform to get PIL image
    return _datasets[split]

@router.get("/image/{split}/{index}")
async def get_image(split: str, index: int):
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split")
        
    try:
        dataset = get_dataset(split)
        img, _ = dataset.dataset[index]
        
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {e}")
