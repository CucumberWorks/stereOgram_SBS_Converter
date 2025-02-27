@echo off
echo Stereogram SBS3D Converter - GUI Interface
echo =======================================
echo.
echo Starting the web interface...
echo Note: Low Memory Mode is enabled by default for better compatibility with all GPUs.
echo You can select from three different model sizes (S, B, L) in the Initialize tab.
echo - Model S (smallest) is best for systems with limited resources or Discord bots
echo - Model B (base) offers a good balance between quality and speed (default)
echo - Model L (large) provides the highest quality but requires more resources
echo.
echo You can still select High Quality Mode in the Advanced Settings when processing images.
echo.
echo If you see errors about inpainting models not loading, the tool will fall back to basic
echo inpainting methods. This is normal and won't affect the core functionality.
echo.
python gradio_interface.py
echo.
echo Closing the web interface...
pause 