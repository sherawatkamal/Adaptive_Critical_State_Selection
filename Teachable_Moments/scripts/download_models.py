#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download, HfApi
from dotenv import load_dotenv
import sys

def setup_environment():
    """Load environment variables and ensure directories exist."""
    load_dotenv()
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("tmp", exist_ok=True)

def get_hf_token():
    """Retrieve HF_TOKEN from environment."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in environment variables.")
        print("Please ensure it is set in your .env file.")
    return token

def download_model(model_id, base_output_dir="models", cache_dir="tmp"):
    """
    Download a model from Hugging Face.
    
    Args:
        model_id (str): The Hugging Face model ID (e.g., 'Qwen/Qwen3-8B').
        base_output_dir (str): Local directory to store the model.
        cache_dir (str): Directory for temporary cache files.
    """
    token = get_hf_token()
    
    # Construct local path: models/Qwen3-8B
    # We strip the organization name for the local folder to keep it clean, 
    # or keep it if preferred. Let's keep the model name.
    model_name = model_id.split("/")[-1]
    local_dir = os.path.join(base_output_dir, model_name)
    
    print(f"Prepare to download {model_id}...")
    print(f"  - Destination: {local_dir}")
    print(f"  - Cache: {cache_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # We want actual files in the model folder
            token=token,
            cache_dir=cache_dir
        )
        print(f"✅ Successfully downloaded {model_id} to {local_dir}")
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download models for Teachable Moments experiments.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen3-8B", 
        help="Hugging Face model ID to download (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="models", 
        help="Base directory for models (default: models)"
    )
    parser.add_argument(
        "--cache", 
        type=str, 
        default="tmp", 
        help="Cache directory (default: tmp)"
    )
    
    args = parser.parse_args()
    
    setup_environment()
    download_model(args.model, args.output, args.cache)

if __name__ == "__main__":
    main()
