import os
import sys
import subprocess
import platform

def check_python_version():
    print(f"[CHECK] Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("[WARN] Warning: Python 3.8+ is recommended.")
    else:
        print("[OK] Python version is OK.")

def check_torch():
    print("\n[CHECK] Checking PyTorch...")
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"[FAIL] PyTorch not found or DLL Error: {e}")
        if "DLL load failed" in str(e):
            print("\n💡 SUGGESTION: Install Microsoft Visual C++ Redistributable:")
            print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    except Exception as e:
        print(f"❌ Error importing torch: {e}")

def fix_permissions():
    if os.name != 'nt':
        return
    print("\n[FIX] Unblocking project files...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ps_cmd = f"Get-ChildItem -Path '{current_dir}' -Recurse | Unblock-File"
        result = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] All files unblocked successfully.")
        else:
            print(f"[WARN] PowerShell warning: {result.stderr}")
    except Exception as e:
        print(f"[WARN] Could not run PowerShell: {e}")

def check_drive_access():
    print("\n[CHECK] Checking Write Permissions...")
    try:
        test_file = 'test_write.tmp'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("[OK] Write access to current directory is OK.")
    except Exception as e:
        print(f"[FAIL] No write access: {e}")

def run_doctor():
    print("="*50)
    print("      WINDOWS ENVIRONMENT DOCTOR")
    print("="*50)
    print(f"OS: {platform.system()} {platform.release()}")
    
    check_python_version()
    check_drive_access()
    fix_permissions()
    check_torch()
    
    print("\n" + "="*50)
    print("      Doctor scan complete.")
    print("="*50)

if __name__ == "__main__":
    run_doctor()
