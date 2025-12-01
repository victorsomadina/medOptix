from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

api = HfApi(token=os.getenv("HF_token"))

try: 
    api.upload_file(
        path_or_fileobj="../model/sarimax_model.pkl",
        path_in_repo="sarimax_model.pkl",
        repo_id="Somad/medOptix-Admission-Model",
        repo_type="model"
    )

except Exception as e:
    print(f"Upload failed: {str(e)}")