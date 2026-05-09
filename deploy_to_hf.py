#!/usr/bin/env python3
"""
RetinAgent – Hugging Face Deployment Script
Uploads model weights to HF Model Hub + code to HF Space
"""

import os
from huggingface_hub import HfApi, login

# ─────────────────────────────────────────────────────────────────
# CONFIG – fill these in
# ─────────────────────────────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
SPACE_REPO    = "lablab-ai-amd-developer-hackathon/RetinoAgent"   # Space
MODEL_REPO    = "lablab-ai-amd-developer-hackathon/RetinoAgent-weights"  # Model repo (weights)
CHECKPOINT    = r"H:\RetinoAgent\checkpoints\best_model.pth"

# Files to upload to the Space
SPACE_FILES = [
    r"H:\RetinoAgent\app.py",
    r"H:\RetinoAgent\requirements.txt",
    r"H:\RetinoAgent\README.md",
    r"H:\RetinoAgent\ui.py",
    r"H:\RetinoAgent\backend.py",
    r"H:\RetinoAgent\agent.py",
    r"H:\RetinoAgent\hero_banner.png",
]

# ─────────────────────────────────────────────────────────────────
# STEP 1 – Login
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Logging into Hugging Face...")
login(token=HF_TOKEN)
api = HfApi()

# ─────────────────────────────────────────────────────────────────
# STEP 2 – Create model repo (if not exists) & upload weights (SKIPPED)
# ─────────────────────────────────────────────────────────────────
# print("\nStep 2: Creating model repo and uploading weights...")
# print(f"  Repo : {MODEL_REPO}")
# print(f"  File : {CHECKPOINT}  ({os.path.getsize(CHECKPOINT)/1e9:.2f} GB)")

# try:
#     api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, private=False)
#     print("  Repo created/verified OK")
# except Exception as e:
#     print(f"  Repo creation note: {e}")

# print("  Uploading best_model.pth — this takes a few minutes for ~1 GB ...")
# url = api.upload_file(
#     path_or_fileobj=CHECKPOINT,
#     path_in_repo="best_model.pth",
#     repo_id=MODEL_REPO,
#     repo_type="model",
#     token=HF_TOKEN,
# )
# print(f"  Uploaded OK  -> {url}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 – Upload code files to the Space
# ─────────────────────────────────────────────────────────────────
print("\nStep 3: Uploading code files to Space...")
for local_path in SPACE_FILES:
    filename = os.path.basename(local_path)
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {filename}")
        continue
    url = api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=SPACE_REPO,
        repo_type="space",
        token=HF_TOKEN,
    )
    print(f"  {filename} OK  -> {url}")

# ─────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Deployment complete!")
print(f"   Space  : https://huggingface.co/spaces/{SPACE_REPO}")
print(f"   Model  : https://huggingface.co/models/{MODEL_REPO}")
print("\nNOTE: Space will rebuild automatically — check logs in ~2-3 min.")
print("      Set FEATHERLESS_API_KEY secret in Space Settings for LLM reports.")
