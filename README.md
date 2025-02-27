# Stereo3D Discord Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Discord.py](https://img.shields.io/badge/discord.py-2.3.0+-blue.svg)](https://discordpy.readthedocs.io/)

A Discord bot that converts regular 2D images into stereogram 3D formats (side-by-side) using depth estimation with Depth Anything V2.

## Features

- **High-quality depth estimation** using Depth Anything V2
- Generation of **side-by-side 3D images** for VR/AR viewers
- **Wiggle GIF animation** for glasses-free 3D viewing
- **Advanced hole filling** for parallax gaps using AI-based inpainting
- **Discord bot integration** for easy conversion through Discord
- Preserves original pixels for highest quality results
- Supports various resolutions and quality settings

## Screenshots

*[Place screenshots here]*

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended (8GB+ VRAM)
- Discord Bot Token

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Stereo3D-Discord-Bot.git
   cd Stereo3D-Discord-Bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Discord bot token:
   ```
   DISCORD_BOT_TOKEN=your_discord_bot_token_here
   ```

4. Model weights:
   
   > **Important Note:** Model weights are NOT included in this repository as they are too large for GitHub. They will be automatically downloaded the first time you run the bot.
   
   The application will automatically download the required model weights when you first run it. However, if you're in an offline environment or prefer to download them manually, follow these steps:
   
   a. Create the directories for the model weights:
   ```
   mkdir -p models
   ```
   
   b. Download the ViT-B model weights (or other size you prefer):
   ```
   # For ViT-B (Base) model - recommended for most users
   curl -L https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -o models/depth_anything_v2_vitb.pth
   
   # For ViT-S (Small) model - faster but less accurate
   curl -L https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -o models/depth_anything_v2_vits.pth
   
   # For ViT-L (Large) model - most accurate but requires more VRAM
   curl -L https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -o models/depth_anything_v2_vitl.pth
   ```
   
   c. Model file sizes (for reference):
   - ViT-S: ~47 MB
   - ViT-B: ~376 MB
   - ViT-L: ~1.17 GB

## Usage

### Discord Bot

Run the Discord bot using:

```bash
python discord_stereo_bot.py
```

Or use the convenience batch file (Windows only):

```bash
run_discord_bot.bat
```

### Discord Bot Commands

- `!sbs` - Convert an attached image to SBS 3D format
- Use this command when replying to a message with an image or attach an image to your message

## Debugging

If you encounter issues, refer to the [Debugging Guide](docs/DEBUGGING_GUIDE.md) for troubleshooting steps.

## Configuration Options

The Discord bot can be configured by modifying parameters in the `discord_stereo_bot.py` file:

```python
converter = StereogramSBS3DConverter(
    use_advanced_infill=True,
    depth_model_type="depth_anything_v2",
    model_size="vits",  # Using smaller model for Discord bot (options: vits, vitb, vitl)
    max_resolution=4096,  # Limit resolution to avoid Discord payload issues
    low_memory_mode=True  # Use low memory processing
)
```

### Model Size Options:

- **ViT-S** (smallest/fastest) - For systems with limited resources, good for Discord bots
- **ViT-B** (medium/balanced) - Good for most systems, balances quality and speed
- **ViT-L** (largest/best quality) - For high-end systems with plenty of RAM and VRAM

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Depth Anything V2](https://github.com/depth-anything/depth-anything) for the depth estimation model
- [Discord.py](https://github.com/Rapptz/discord.py) for the Discord bot framework
- [Diffusers](https://github.com/huggingface/diffusers) for the inpainting model