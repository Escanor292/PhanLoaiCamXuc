"""
Script để chuyển đổi giữa các Hugging Face repositories
"""

import json
from huggingface_hub import create_repo, upload_folder
import os


def switch_to_repo(repo_id):
    """Chuyển sang repo khác và upload model."""
    
    print(f"🔄 Switching to repository: {repo_id}")
    
    # Update config
    config = {
        "sharing_method": "huggingface",
        "huggingface_repo": repo_id,
        "gdrive_folder_id": None,
        "dropbox_token": None,
        "base_url": None
    }
    
    with open("model_sharing_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Config updated!")
    
    # Try to create repo
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist or need manual creation: {e}")
    
    # Upload model
    try:
        upload_folder(
            folder_path="saved_model",
            repo_id=repo_id,
            path_in_repo="current_model",
            commit_message="Upload current production model"
        )
        print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    """Main function."""
    
    print("🔄 HUGGING FACE REPOSITORY SWITCHER")
    print("=" * 50)
    
    # Available options
    repos = [
        "Escanor292/emotion-classification",  # Personal repo (working)
        "emotion-classification-vn/emotion-classification",  # Organization repo (need setup)
    ]
    
    print("Available repositories:")
    for i, repo in enumerate(repos, 1):
        print(f"{i}. {repo}")
    
    print("3. Custom repo")
    
    choice = input("\nSelect repository (1-3): ").strip()
    
    if choice == "1":
        repo_id = repos[0]
    elif choice == "2":
        repo_id = repos[1]
    elif choice == "3":
        repo_id = input("Enter custom repo (format: username/repo-name): ").strip()
    else:
        print("❌ Invalid choice!")
        return
    
    # Switch to selected repo
    success = switch_to_repo(repo_id)
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SWITCH SUCCESSFUL!")
        print("=" * 50)
        print(f"✅ Active repo: {repo_id}")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
        print("\n🚀 Team can now use:")
        print("python train_simple.py  # Auto-download from new repo")
    else:
        print("\n❌ Switch failed! Check permissions and try again.")


if __name__ == "__main__":
    main()