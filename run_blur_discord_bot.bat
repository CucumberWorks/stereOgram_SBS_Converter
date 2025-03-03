@echo off
echo Starting Blur Discord Bot...
echo.

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b
)

:: Check if the virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Run the Discord bot
python blur_discord_bot.py

:: If the bot crashes, don't immediately close the window
if %ERRORLEVEL% neq 0 (
    echo.
    echo The bot has crashed with error code %ERRORLEVEL%
    echo Check the error message above for details.
    pause
) 