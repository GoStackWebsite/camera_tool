@echo off
cd /d "%~dp0"
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Setup complete. Edit .env with your camera credentials, then run run.bat
pause
