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
            from huggingface_hub import HfApi, create_repo
            
            api = HfApi()
            repo_id = self.config['huggingface_repo']
            
            # Tạo repo nếu chưa có
            try:
                create_repo(repo_id, exist_ok=True)
            except:
                pass
            
            # Upload model files
            model_files = [
                'pytorch_model.bin',
                'tokenizer.json',
                'tokenizer_config.json',
                'training_config.json'
            ]
            
            for file_name in model_files:
                file_path = os.path.join(model_path, file_name)
                if os.path.exists(file_path):
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=f"{model_id}/{file_name}",
                        repo_id=repo_id
                    )
            
            print(f"✅ Model {model_id} uploaded to Hugging Face!")
            return True
            
        except ImportError:
            print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
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
            from huggingface_hub import hf_hub_download
            
            repo_id = self.config['huggingface_repo']
            
            model_files = [
                'pytorch_model.bin',
                'tokenizer.json', 
                'tokenizer_config.json',
                'training_config.json'
            ]
            
            for file_name in model_files:
                try:
                    downloaded_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{model_id}/{file_name}",
                        local_dir=target_path,
                        local_dir_use_symlinks=False
                    )
                    print(f"✅ Downloaded {file_name}")
                except Exception as e:
                    print(f"⚠️  Could not download {file_name}: {e}")
            
            print(f"✅ Model {model_id} downloaded!")
            return True
            
        except ImportError:
            print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
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