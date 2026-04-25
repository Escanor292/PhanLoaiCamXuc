"""
Test Organization Access - Kiểm tra quyền truy cập organization
"""

from huggingface_hub import whoami, create_repo
import requests


def test_organization_access():
    """Test quyền truy cập vào organization."""
    
    print("🔍 TESTING ORGANIZATION ACCESS")
    print("=" * 50)
    
    # Test 1: Kiểm tra user hiện tại
    try:
        user_info = whoami()
        print(f"✅ Current user: {user_info['name']}")
        
        # Kiểm tra organizations
        orgs = user_info.get('orgs', [])
        if orgs:
            print(f"✅ Member of organizations: {', '.join([org['name'] for org in orgs])}")
            
            # Kiểm tra có emotion-classification-vn không
            org_names = [org['name'] for org in orgs]
            if 'emotion-classification-vn' in org_names:
                print("✅ Has access to emotion-classification-vn!")
                return True
            else:
                print("❌ No access to emotion-classification-vn")
        else:
            print("❌ Not member of any organizations")
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Test 2: Thử tạo repo trong organization
    print("\n🧪 Testing repository creation...")
    try:
        repo_id = "emotion-classification-vn/emotion-classification"
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Can create/access repo: {repo_id}")
        return True
    except Exception as e:
        print(f"❌ Cannot create repo: {e}")
        return False


def main():
    """Main test function."""
    
    print("🧪 ORGANIZATION ACCESS TEST")
    print("=" * 60)
    
    has_access = test_organization_access()
    
    print("\n" + "=" * 60)
    if has_access:
        print("🎉 SUCCESS: You have organization access!")
        print("✅ Ready to upload to emotion-classification-vn/emotion-classification")
        print("\n🚀 Next step:")
        print("python switch_repo.py  # Choose option 2")
    else:
        print("❌ NO ACCESS: Need organization permissions")
        print("\n📋 Required actions:")
        print("1. Add Escanor292 to emotion-classification-vn organization")
        print("2. Or create token with organization write permissions")
        print("3. See ADD_MEMBER_GUIDE.md for details")
    
    print("=" * 60)


if __name__ == "__main__":
    main()