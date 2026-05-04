
import os
import sys
from huggingface_hub import HfApi, list_repo_tree

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_sharing import ModelSharing
from model_registry import ModelRegistry

def cleanup_hf_repo():
    sharing = ModelSharing()
    repo_id = sharing.config['huggingface_repo']
    api = HfApi()
    
    print(f"--- Step 1: Cleaning up Hugging Face Repo: {repo_id} ---")
    
    try:
        # List all items in the root of the repo
        items = list_repo_tree(repo_id)
        
        for item in items:
            # We want to delete folders that look like models (model_*) 
            # and potentially 'current_model' if we want a fresh start
            if item.path.startswith("model_") or item.path == "current_model":
                print(f"🗑️  Deleting from HF: {item.path}...")
                try:
                    api.delete_folder(
                        repo_id=repo_id,
                        path_in_repo=item.path,
                        commit_message=f"Cleanup old model {item.path}"
                    )
                    print(f"   ✓ Deleted {item.path}")
                except Exception as e:
                    print(f"   ⚠️  Could not delete {item.path}: {e}")
                    
    except Exception as e:
        print(f"❌ Error listing repo: {e}")
        return

    print("\n--- Step 2: Uploading the latest best model ---")
    sharing.sync_best_model()

if __name__ == "__main__":
    cleanup_hf_repo()
