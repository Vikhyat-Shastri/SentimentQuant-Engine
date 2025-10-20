@echo off
REM Installation script for Windows
REM Fear & Greed Sentiment Analysis Engine

echo ========================================
echo Fear ^& Greed Sentiment Analysis Engine
echo Installation Script
echo ========================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.11 or higher.
    pause
    exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 5: Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo Step 6: Creating config file from template...
if not exist config\api_keys.yaml (
    copy config\api_keys_template.yaml config\api_keys.yaml
    echo Created config\api_keys.yaml - Please edit with your API keys
) else (
    echo config\api_keys.yaml already exists
)

echo.
echo Step 7: Creating necessary directories...
if not exist logs mkdir logs
if not exist models mkdir models

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit config\api_keys.yaml with your API credentials
echo 2. Activate virtual environment: venv\Scripts\activate
echo 3. Run the system: python main.py
echo.
echo For more information, see README.md
echo.
pause
