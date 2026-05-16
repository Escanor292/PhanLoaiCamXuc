@echo off
setlocal enabledelayedexpansion

title AI Emotion Classifier - Pro Edition (Portable)
echo ===================================================
echo KHOI DONG HE THONG PHAN LOAI CAM XUC (PRO)
echo ===================================================
echo.

:: 1. Detect Project Root
set "BASE_DIR=%~dp0"

:: Tim thu muc goc bang cach kiem tra file config.py
if exist "%BASE_DIR%config.py" (
    set "PROJECT_ROOT=%BASE_DIR%"
    set "SCRIPT_PATH=gui_app.py"
) else if exist "%BASE_DIR%..\config.py" (
    set "PROJECT_ROOT=%BASE_DIR%.."
    set "SCRIPT_PATH=Pro_Edition\gui_app.py"
) else if exist "%BASE_DIR%PhanLoaiCamXuc\config.py" (
    set "PROJECT_ROOT=%BASE_DIR%PhanLoaiCamXuc"
    set "SCRIPT_PATH=gui_app.py"
) else (
    set "PROJECT_ROOT=%BASE_DIR%"
    set "SCRIPT_PATH=gui_app.py"
)

cd /d "%PROJECT_ROOT%"

:: Kiem tra file chinh co ton tai khong
if not exist "%SCRIPT_PATH%" (
    REM Thu tim lai lan nua neu dang o trong Pro_Edition ma script cung o do
    if exist "gui_app.py" (
        set "SCRIPT_PATH=gui_app.py"
    ) else (
        echo [LOI] Khong tim thay file %SCRIPT_PATH% trong thu muc: %CD%
        echo Vui long dam bao thu muc du an va file .bat o dung vi tri.
        pause
        exit /b
    )
)

echo [HE THONG] Dang chay tai: %CD%

:: 2. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python! 
    echo Vui long cai dat Python tai https://www.python.org/
    pause
    exit /b
)

:: 3. Setup Virtual Environment
if not exist ".venv" (
    echo [HE THONG] Dang khoi tao moi truong ao (^.venv^)...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [LOI] Khong the tao .venv.
        pause
        exit /b
    )
)

:: 4. Activate and Install/Run
echo [HE THONG] Dang kiem tra thu vien...
call .venv\Scripts\activate.bat

:: Kiem tra neu can cai dat dependencies (Tu dong bo qua neu da co Torch)
if exist "requirements.txt" (
    python -c "import torch, transformers, customtkinter, PIL, requests" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [HE THONG] Dang kiem tra va cai dat thu vien (^lan dau^)...
        pip install -r requirements.txt
    ) else (
        echo [HE THONG] Da co du thu vien, bo qua buoc kiem tra.
    )
)

echo.
echo [HE THONG] Dang tai giao dien, vui long doi giay lat...
echo.

python "%SCRIPT_PATH%"

if %errorlevel% neq 0 (
    echo.
    echo [LOI] Chuong trinh da dung lai do co loi.
    pause
)

deactivate
