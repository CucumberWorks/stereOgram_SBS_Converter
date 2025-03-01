import os
import sys
import platform
import torch
import numpy as np
from dotenv import load_dotenv
import pkg_resources

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)

def check_python():
    """Check Python version."""
    print_section("Python Environment")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Check if Python version is compatible
    major, minor, _ = platform.python_version_tuple()
    if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
        print("⚠️ WARNING: Python version is below 3.8. This may cause compatibility issues.")
    else:
        print("✅ Python version is compatible (3.8+)")

def check_cuda():
    """Check CUDA availability and version."""
    print_section("CUDA Environment")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        print(f"Available GPU(s): {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            try:
                # Get GPU memory in GB
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    Memory: {total_memory:.2f} GB")
                
                if total_memory < 4:
                    print("    ⚠️ WARNING: Less than 4GB VRAM detected. Consider using 'vits' model with low_memory_mode=True")
                elif total_memory < 8:
                    print("    ⚠️ WARNING: Less than 8GB VRAM detected. Consider using 'vits' model")
            except Exception as e:
                print(f"    ⚠️ Error getting memory info: {e}")
        return True
    else:
        print("⚠️ CUDA is not available - processing will be much slower")
        print("   The bot will use CPU mode but performance will be limited")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print_section("Dependencies")
    required_packages = [
        "torch", "numpy", "opencv-python", "pillow", "discord.py", 
        "tqdm", "dotenv", "matplotlib", "diffusers", "transformers"
    ]
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package} {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} is not installed")

def check_token():
    """Check if the Discord bot token is configured."""
    print_section("Discord Bot Configuration")
    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    
    if token:
        if token == "your_discord_bot_token_here":
            print("❌ DISCORD_BOT_TOKEN is set but appears to be the default value")
            print("   Please update your .env file with your actual bot token")
        else:
            masked_token = token[:6] + "..." + token[-4:]
            print(f"✅ DISCORD_BOT_TOKEN is set (value: {masked_token})")
        return True
    else:
        print("❌ DISCORD_BOT_TOKEN is not set in .env file")
        print("   Create a .env file in the project root with: DISCORD_BOT_TOKEN=your_token_here")
        return False

def check_models_directory():
    """Check if models directory exists and has any model files."""
    print_section("Model Weights")
    
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        print(f"ℹ️ Models directory does not exist: {models_dir}")
        print("   Directory will be created automatically when the bot first runs")
        return False
    
    models = [f for f in os.listdir(models_dir) if f.endswith(".pth") or f.endswith(".pt")]
    
    if models:
        print(f"✅ Found {len(models)} model file(s) in {models_dir}:")
        for model in models:
            model_path = os.path.join(models_dir, model)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   - {model} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"ℹ️ No model files found in {models_dir}")
        print("   Models will be downloaded automatically when the bot first runs")
        return False

def main():
    """Run all debugging checks."""
    print_section("stereOgram SBS Converter - Debug Mode")
    
    try:
        # System checks
        check_python()
        gpu_available = check_cuda()
        check_dependencies()
        token_valid = check_token()
        models_available = check_models_directory()
        
        # Display summary
        print_section("Summary")
        print(f"GPU Available: {'Yes' if gpu_available else 'No'}")
        print(f"Token Valid: {'Yes' if token_valid else 'No'}")
        print(f"Models Available: {'Yes' if models_available else 'No'}")
        
        if not token_valid:
            print("\nWARNING: Discord bot token is not configured correctly.")
        
        if not models_available:
            print("\nWARNING: Model files are missing. They will be downloaded automatically when running the application.")
        
        print("\nDebug completed successfully. If you're experiencing issues, please review the output above.")
        return 0
    except Exception as e:
        print(f"Error during debug: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 