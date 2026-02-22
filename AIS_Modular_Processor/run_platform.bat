@echo off
echo Starting AIS Modular Platform...
cd /d %~dp0

:: Try to setup DB (will skip if already exists, fail gracefully if blocked)
echo Initializing Database...
python db_setup.py

echo Starting FastAPI Backend...
echo Application will be available at: http://localhost:8000/app/
python api.py
pause
