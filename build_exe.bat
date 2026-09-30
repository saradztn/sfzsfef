@echo off
rem Build a single FBX2IFP.exe (no Python needed on the target PC)
cd /d "%~dp0"
py -m pip install --upgrade pyinstaller || python -m pip install --upgrade pyinstaller
py -m PyInstaller --noconfirm --onefile --windowed --name FBX2IFP --paths tools --paths ref --hidden-import validate_ifp tools\fbx2ifp_gui.py || python -m PyInstaller --noconfirm --onefile --windowed --name FBX2IFP --paths tools --paths ref --hidden-import validate_ifp tools\fbx2ifp_gui.py
echo.
echo Done: dist\FBX2IFP.exe
pause
