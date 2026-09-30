@echo off
rem FBX -> IFP converter (GTA SA / MTA:SA) - Windows launcher
cd /d "%~dp0"
where pyw >nul 2>nul && (start "" pyw "tools\fbx2ifp_gui.py" & exit /b)
where pythonw >nul 2>nul && (start "" pythonw "tools\fbx2ifp_gui.py" & exit /b)
echo Python 3 is not installed.
echo Download: https://www.python.org/downloads/windows/  (tick "Add python.exe to PATH")
pause
