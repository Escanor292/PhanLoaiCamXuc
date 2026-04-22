"""
Model Sharing System - Giải quyết vấn đề model quá lớn cho GitHub

Vấn đề: Model 419MB không thể push lên GitHub (giới hạn 100MB)
Giải pháp: Dùng cloud storage hoặc model hub để share model

Các phương án:
1. Hugging Face Model Hub (KHUYẾN NGHỊ)
2. Google Drive
3. Dropbox
4. AWS S3
5. Local network sharing
"""

import os
import json
import requests
from datetime import datetime
from model_registry import ModelRegistry


class ModelSharing:
    """Hệ thống chia sẻ model qua cloud storage."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.config_file = "model_sharing_config.json"
        self.load_config()
    
    def load_config(self):
        """Load cấu hình sharing."""
        default_config = {
            "sharing_method": "huggingface",  # huggingface, gdrive, dropbox
            "huggingface_repo": "your-username/emotion-classification",
            "gdrive_folder_id": None,
            "dropbox_token": None,
            "base_url": None
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Lưu cấu hình."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def upload_model(self, model_id):
        """Upload model lên cloud storage."""
        model_info = self.registry.get_model_info(model_id)
        if not model_info:
            print(f"❌ Model {model_id} not found!")
            return False
        
        model_path = model_info['path']
        
        if self.config['sharing_method'] == 'huggingface':
            return self._upload_to_huggingface(model_id, model_path)
        elif self.config['sharing_method'] == 'gdrive':
            return self._upload_to_gdrive(model_id, model_path)
        else:
            print(f"❌ Sharing method {self.config['sharing_method']} not implemented yet!")
            return False
    
    def _upload_to_huggingface(self, model_id, model_path):
        """Upload model lên Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder
            
            api = HfApi()
            repo_id = self.config['huggingface_repo']
            
            print(f"📤 Uploading to Hugging Face: {repo_id}")
            
            # Tạo repo nếu chưa có (organization repo)
            try:
                create_repo(repo_id, exist_ok=True, repo_type="model")
                print(f"✅ Repository ready: {repo_id}")
            except Exception as e:
                print(f"⚠️  Repository may already exist: {e}")
            
            # Upload toàn bộ folder model
            try:
                upload_folder(
                    folder_path=model_path,
                    repo_id=repo_id,
                    path_in_repo=model_id,
                    commit_message=f"Upload model {model_id}"
                )
                print(f"✅ Model {model_id} uploaded successfully!")
                print(f"🔗 View at: https://huggingface.co/{repo_id}/tree/main/{model_id}")
                return True
                
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                return False
            
        except ImportError:
            print("❌ huggingface_hub not installed. Run: pip install -U huggingface_hub")
            return False
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def download_model(self, model_id, target_path=None):
        """Download model từ cloud storage."""
        if target_path is None:
            target_path = f"model_registry/models/{model_id}"
        
        os.makedirs(target_path, exist_ok=True)
        
        if self.config['sharing_method'] == 'huggingface':
            return self._download_from_huggingface(model_id, target_path)
        else:
            print(f"❌ Download method {self.config['sharing_method']} not implemented yet!")
            return False
    
    def _download_from_huggingface(self, model_id, target_path):
        """Download model từ Hugging Face Hub."""
        try:
            from huggingface_hub import snapshot_download
            
            repo_id = self.config['huggingface_repo']
            
            print(f"📥 Downloading from Hugging Face: {repo_id}/{model_id}")
            
            # Download toàn bộ folder model
            try:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=f"{model_id}/*",
                    local_dir=os.path.dirname(target_path),
                    local_dir_use_symlinks=False
                )
                
                # Move files từ subfolder lên target_path
                downloaded_path = os.path.join(os.path.dirname(target_path), model_id)
                if os.path.exists(downloaded_path):
                    import shutil
                    if os.path.exists(target_path):
                        shutil.rmtree(target_path)
                    shutil.move(downloaded_path, target_path)
                
                print(f"✅ Model {model_id} downloaded successfully!")
                return True
                
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return False
            
        except ImportError:
            print("❌ huggingface_hub not installed. Run: pip install -U huggingface_hub")
            return False
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def sync_best_model(self):
        """Tự động sync model tốt nhất."""
        best_model = self.registry.get_best_model()
        if best_model:
            model_id = best_model['model_id']
            print(f"🔄 Syncing best model: {model_id}")
            return self.upload_model(model_id)
        else:
            print("❌ No best model found!")
            return False


def main():
    """CLI interface cho model sharing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Sharing System")
    parser.add_argument("action", choices=['upload', 'download', 'sync', 'config'])
    parser.add_argument("--model-id", help="Model ID to upload/download")
    parser.add_argument("--method", choices=['huggingface', 'gdrive'], help="Sharing method")
    parser.add_argument("--repo", help="Hugging Face repository")
    
    args = parser.parse_args()
    
    sharing = ModelSharing()
    
    if args.action == 'config':
        if args.method:
            sharing.config['sharing_method'] = args.method
        if args.repo:
            sharing.config['huggingface_repo'] = args.repo
        sharing.save_config()
        print("✅ Config updated!")
        
    elif args.action == 'upload':
        if not args.model_id:
            print("❌ --model-id required for upload!")
            return
        sharing.upload_model(args.model_id)
        
    elif args.action == 'download':
        if not args.model_id:
            print("❌ --model-id required for download!")
            return
        sharing.download_model(args.model_id)
        
    elif args.action == 'sync':
        sharing.sync_best_model()


if __name__ == "__main__":
    main()