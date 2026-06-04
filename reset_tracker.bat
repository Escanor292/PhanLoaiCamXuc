@echo off
echo ========================================
echo RESET DATA TRACKER
echo ========================================
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  ⚠️  CANH BAO QUAN TRONG                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Thao tac nay se xoa lich su training!
echo.
echo SAU KHI XOA:
echo  • He thong se QUEN TAT CA du lieu da train
echo  • Co the train lai cac du lieu cu
echo  • Du lieu KHONG BI MAT - chi xoa metadata theo doi
echo  • Model da train van giu nguyen trong model_registry/
echo.
echo NEU BAN KHONG CHAC CHAN, HAY NHAN CTRL+C DE HUY!
echo.
pause

del "model_registry\data_tracker.json"

echo.
echo ========================================
echo DA XOA DATA TRACKER!
echo ========================================
echo He thong da quen du lieu da train.
echo Ban co the train lai bat ky du lieu nao.
echo.
pause
