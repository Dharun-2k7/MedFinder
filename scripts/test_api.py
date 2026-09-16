import requests
import io
from PIL import Image

def run_api_tests():
    print("Testing /api/health...")
    resp = requests.get("http://localhost:8000/api/health")
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    
    print("\nTesting /api/search with a dummy image...")
    # Create dummy image
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    files = {'file': ('dummy.jpg', img_byte_arr, 'image/jpeg')}
    resp = requests.post("http://localhost:8000/api/search", files=files)
    
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Query ID: {data.get('query_id')}")
        print(f"Processing Time (ms): {data.get('processing_time_ms')}")
        print("Top Result:")
        print(data.get('results')[0])
    else:
        print(resp.text)

if __name__ == "__main__":
    run_api_tests()
