"""
Setup Hugging Face Login - Đơn giản hóa quá trình login
"""

import os
from huggingface_hub import login, whoami, create_repo


def setup_huggingface():
    """Setup Hugging Face authentication và tạo repo."""
    
    print("🔑 HUGGING FACE SETUP")
    print("=" * 50)
    
    # Bước 1: Kiểm tra đã login chưa
    try:
        user_info = whoami()
        print(f"✅ Đã login với tài khoản: {user_info['name']}")
        return True
    except Exception:
        print("❌ Chưa login Hugging Face")
    
    # Bước 2: Hướng dẫn tạo token
    print("\n📋 HƯỚNG DẪN TẠO TOKEN:")
    print("1. Vào: https://huggingface.co/settings/tokens")
    print("2. Chọn 'New token' → 'Fine-grained'")
    print("3. Cấp quyền 'Write access to contents of all repos'")
    print("4. Copy token (dạng: hf_xxxxxxxxxxxxxxxxxxxxx)")
    print()
    
    # Bước 3: Nhập token
    token = input("📝 Nhập token của bạn: ").strip()
    
    if not token.startswith('hf_'):
        print("❌ Token không hợp lệ! Token phải bắt đầu bằng 'hf_'")
        return False
    
    # Bước 4: Login
    try:
        login(token=token)
        user_info = whoami()
        print(f"✅ Login thành công! Tài khoản: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Login thất bại: {e}")
        return False


def create_team_repo():
    """Tạo repository cho team."""
    
    print("\n🏗️  TẠO REPOSITORY CHO TEAM")
    print("=" * 50)
    
    repo_id = "phanloaicamxuc-team/emotion-classification"
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository đã sẵn sàng: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"⚠️  Không thể tạo repo (có thể đã tồn tại): {e}")
        print(f"🔗 Kiểm tra tại: https://huggingface.co/{repo_id}")
        return True  # Vẫn OK nếu repo đã tồn tại


def main():
    """Main setup function."""
    
    print("🚀 SETUP MODEL SHARING CHO TEAM")
    print("=" * 60)
    
    # Setup login
    if not setup_huggingface():
        print("\n❌ Setup thất bại! Vui lòng thử lại.")
        return
    
    # Tạo repo
    if not create_team_repo():
        print("\n❌ Không thể tạo repository!")
        return
    
    print("\n" + "=" * 60)
    print("🎉 SETUP HOÀN TẤT!")
    print("=" * 60)
    print()
    print("✅ Hugging Face: Đã login")
    print("✅ Repository: phanloaicamxuc-team/emotion-classification")
    print("✅ Model sharing: Sẵn sàng")
    print()
    print("🔄 BƯỚC TIẾP THEO:")
    print("python model_sharing.py sync  # Upload model hiện tại")
    print()


if __name__ == "__main__":
    main()