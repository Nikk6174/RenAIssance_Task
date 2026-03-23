import os
from huggingface_hub import login, HfApi

def main():
    print("=========================================")
    print(" Hugging Face Model Uploader")
    print("=========================================")
    print("We are bypassing the broken command-line tools")
    print("by using the Python library directly.\n")
    
    # 1. Login interactively
    print("Step 1: Login")
    login()  # This will prompt you for your token securely
    
    api = HfApi()
    username = api.whoami()["name"]
    print(f"\nWelcome, {username}!")
    
    # 2. Upload TrOCR
    trocr_repo = f"{username}/historical-spanish-trocr"
    print(f"\nStep 2: Uploading TrOCR to {trocr_repo}...")
    try:
        api.create_repo(repo_id=trocr_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path="models/trocr",
            repo_id=trocr_repo,
            repo_type="model",
            commit_message="Initial TrOCR model upload"
        )
        print("✅ TrOCR upload complete!")
    except Exception as e:
        print(f"❌ Failed to upload TrOCR: {e}")

    # 3. Upload T5
    t5_repo = f"{username}/historical-spanish-t5"
    print(f"\nStep 3: Uploading T5 to {t5_repo}...")
    try:
        api.create_repo(repo_id=t5_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path="models/t5",
            repo_id=t5_repo,
            repo_type="model",
            commit_message="Initial T5 model upload"
        )
        print("✅ T5 upload complete!")
    except Exception as e:
        print(f"❌ Failed to upload T5: {e}")

if __name__ == "__main__":
    main()
