@echo off
echo Stereogram SBS3D Converter
echo =======================================
echo.
echo Starting the converter...
echo.
cd ..
python core\test_with_demo.py %*
echo.
echo Conversion complete. Check the results folder.
pause 