@echo off
echo Starting stereOgram SBS Converter Discord Bot...
echo.

rem Change to the directory where this batch file is located
cd /d "%~dp0\.."

rem Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    goto end
)

rem Run the Discord bot
python stereogram_main.py --mode bot

:end
pause 