@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating a local Python environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Python was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Could not install the Python packages.
  pause
  exit /b 1
)

echo.
echo Preparing the model. The first run downloads sounds and can take a while.
python scripts\prepare_and_train.py
if errorlevel 1 (
  echo Training did not finish.
  pause
  exit /b 1
)

echo.
echo Opening the test page at http://127.0.0.1:5000
echo Keep this window open while you test. Close it to stop the page.
python app_ui\app.py
pause
