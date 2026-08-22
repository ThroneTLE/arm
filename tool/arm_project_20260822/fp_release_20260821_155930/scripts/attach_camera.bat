@echo off
chcp 65001 >nul
echo Attaching camera to WSL (auto-detect busid)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0attach_camera.ps1"
echo.
echo Done! Now in WSL run: bash ~/start_camera.sh
pause
