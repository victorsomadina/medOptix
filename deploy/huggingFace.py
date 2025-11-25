from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.upload_file(
        path_or_fileobj="../model/sarimax_model.pkl",
        path_in_repo="sarimax_model.pkl",
        repo_id="Somad/test",
        repo_type="model",
    )
    
except Exception as e:
    print(f"Upload failed: {e}")
    raise
