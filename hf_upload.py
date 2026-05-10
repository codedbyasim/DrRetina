from huggingface_hub import HfApi

import os

print("Starting upload to DrRetina Space...")
api = HfApi()

api.create_repo(
    repo_id="lablab-ai-amd-developer-hackathon/DrRetina",
    repo_type="space",
    space_sdk="gradio",
    token=os.environ.get("HF_TOKEN", ""),
    exist_ok=True
)

api.upload_folder(
    folder_path=".",
    repo_id="lablab-ai-amd-developer-hackathon/DrRetina",
    repo_type="space",
    token=os.environ.get("HF_TOKEN", ""),
    ignore_patterns=[
        "*.pth", "*.h5", "venv/*", ".git/*", "checkpoints/*", 
        "aptos2019-blindness-detection/*", "Dataset/*", 
        "__pycache__/*", "*.zip", "*.tar.gz", "hf_upload.py"
    ]
)
print("Upload complete!")
