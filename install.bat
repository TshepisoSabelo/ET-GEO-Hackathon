@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
    echo Virtual environment not found. Create one first.
    exit /b 1
)
