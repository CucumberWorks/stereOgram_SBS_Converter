"""
Discord Bot Debug Script
This script helps verify the configuration of your Discord bot before running.
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import gc
import time
import imageio
from dotenv import load_dotenv

# Add the parent directory to the Python path to access core module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter

def print_header(text):
    print("\n" + "=" * 50)
    print(f" {text}")
    print("=" * 50)

def check_dependencies():
    print_header("Checking Dependencies")
    required_packages = [
        "discord.py", "python-dotenv", "opencv-python", "numpy", 
        "torch", "PIL", "aiohttp"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.split(">=")[0].replace("python-", ""))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    return missing_packages

def check_token():
    print_header("Checking Discord Bot Token")
    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    
    if not token:
        print("❌ DISCORD_BOT_TOKEN not found in environment variables")
        print("  Make sure you've created a .env file with your token")
        return False
    
    if token == "YOUR_DISCORD_BOT_TOKEN" or token == "your_discord_bot_token_here":
        print("❌ DISCORD_BOT_TOKEN is still set to the placeholder value")
        print("  Replace it with your actual Discord bot token in the .env file")
        return False
    
    # Basic validation for token format (simple check, not comprehensive)
    if len(token.strip()) < 50 or "." not in token:
        print("⚠️ DISCORD_BOT_TOKEN format looks unusual (but might still work)")
    else:
        print("✅ DISCORD_BOT_TOKEN found and looks valid")
    
    return True

def check_converter():
    """Check if the converter module is available."""
    print_header("Checking StereogramSBS3DConverter")
    try:
        # Already imported at the top
        converter = StereogramSBS3DConverter(use_advanced_infill=False)
        print("✅ StereogramSBS3DConverter module found")
        return True
    except Exception as e:
        print(f"❌ Error when importing StereogramSBS3DConverter: {e}")
        return False

def check_pytorch():
    print_header("Checking PyTorch Configuration")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA is not available - the bot will run on CPU mode (slower)")
        return True, torch.cuda.is_available()
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False, False

def main():
    print_header("Discord Bot Debug Tool")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print("\n⚠️ Some packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
    
    # Check token
    token_valid = check_token()
    
    # Check converter
    converter_valid = check_converter()
    
    # Check PyTorch
    pytorch_valid, cuda_available = check_pytorch()
    
    # Final assessment
    print_header("Summary")
    if token_valid and converter_valid and not missing_packages:
        print("✅ All checks passed! You should be good to run your bot.")
        print("   Run it with: python discord_stereo_bot.py")
    else:
        print("⚠️ Some issues were found. Fix them before running the bot.")
    
    if not cuda_available:
        print("\n⚠️ Note: Running without GPU will be significantly slower")
        print("   Consider setting model_size to 'vits' and reducing max_resolution")

if __name__ == "__main__":
    main() 