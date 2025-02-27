@echo off
echo Testing Stereo 3D Converter...

echo.
echo This will test the converter functionality without running the Discord bot.
echo It will use a demo image and save the result in the test_results folder.
echo.

cd ..
if exist demo_images (
    set "found_image="
    for %%F in (demo_images\*.jpg demo_images\*.png) do (
        set "found_image=%%F"
        goto found
    )
    
    :found
    if not defined found_image (
        echo No demo images found. Please add images to the demo_images folder.
        pause
        exit /b
    )
    
    echo Found demo image: %found_image%
    echo Running test with this image...
    echo.
    
    python tools\test_converter.py "%found_image%"
) else (
    echo Demo images folder not found.
    echo Please specify an image path:
    echo.
    
    set /p img_path="Enter image path: "
    
    if not defined img_path (
        echo No image path provided. Exiting.
        pause
        exit /b
    )
    
    python tools\test_converter.py "%img_path%"
)

echo.
echo Test complete. Check the test_results folder for output.
pause 