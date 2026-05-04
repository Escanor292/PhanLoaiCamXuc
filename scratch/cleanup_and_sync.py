
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import ModelRegistry
from model_sharing import ModelSharing

print("--- Step 1: Cleaning up local registry ---")
registry = ModelRegistry()
# Trigger cleanup manually
best_model = registry.get_best_model()
if best_model:
    print(f"Keeping best model: {best_model['model_id']}")
    registry._remove_old_models(keep_model_id=best_model['model_id'])
else:
    print("No models found in registry.")

print("\n--- Step 2: Syncing to Hugging Face ---")
sharing = ModelSharing()
sharing.sync_best_model()
