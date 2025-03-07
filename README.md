# stereOgram SBS Converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Discord.py](https://img.shields.io/badge/discord.py-2.3.0+-blue.svg)](https://discordpy.readthedocs.io/)

A tool that converts regular 2D images into stereogram 3D formats (side-by-side) using depth estimation with Depth Anything V2. Includes both a user-friendly GUI and a Discord bot interface.

## Features

- **High-quality depth estimation** using Depth Anything V2
- **User-friendly web GUI** for easy image conversion
- Generation of **side-by-side 3D images** for VR/AR viewers
- **Wiggle GIF animation** for glasses-free 3D viewing
- **Advanced hole filling** for parallax gaps using AI-based inpainting
- **Discord bot integration** for easy conversion through Discord
- **Specialized anime processing** with anime face detection ([learn more](docs/ANIME_FACE_TRACKING.md))
- Preserves original pixels for highest quality results
- Supports various resolutions and quality settings

## Screenshots

*[Place screenshots here]*

## Requirements

- Python 3.8+
- CUDA-compatible GPU or Apple Silicon Mac recommended (8GB+ VRAM)
- Discord Bot Token (if using the Discord bot)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/CucumberWorks/stereOgram_SBS_Converter.git
   cd stereOgram_SBS_Converter
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   > macOS users: For help setting up a Python virtual environment, see the [macOS Virtual Environment Guide](docs/MACOS_VENV_GUIDE.md).

3. Create a `.env` file with your Discord Bot Token (if using Discord bot):
   ```
   cp .env.example .env
   # Edit the .env file with your token
   ```

4. Model weights:
   
   > **Important Note:** Model weights are NOT included in this repository as they are too large for GitHub. They will be automatically downloaded the first time you run the application.
   
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

### Web GUI

Run the web interface using:

```bash
python stereogram_main.py --mode ui
```

Or simply:

```bash
python stereogram_main.py
```

For Windows users, you can also use the included batch file:
```bash
run_stereogram_sbs3d_gui.bat
```

### Discord Bot

1. Make sure you have set up your Discord bot token in the `.env` file:
   ```
   DISCORD_BOT_TOKEN=your_actual_token_here
   ```
   
   > Need help getting a Discord bot token? See the [Discord Token Guide](docs/DISCORD_TOKEN_GUIDE.md).

2. Run the Discord bot:

   **For Windows:**
   ```bash
   python stereogram_main.py --mode bot
   ```
   Or use the included batch file:
   ```bash
   run_discord_bot.bat
   ```

   **For macOS/Linux:**
   ```bash
   python3 stereogram_main.py --mode bot
   ```
   
   **For macOS with SSL certificate issues:**
   ```bash
   SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 stereogram_main.py --mode bot
   ```

3. The bot will connect to Discord using your token and will be ready to process image conversion commands.

For detailed Discord bot setup and usage, refer to the [Discord Bot Guide](docs/DISCORD_BOT_README.md).

### Blur Discord Bot

The project also includes a specialized Discord bot for applying depth-based blur effects to images with precise focal point selection.

#### Features

- **Interactive Focal Point Selection**: Uses a grid-based system for precise selection of focus areas
- **Depth-Based Blur**: Applies blur based on the depth map, keeping selected areas in focus
- **Adjustable Blur Strength**: Customize the blur effect by replying with "+" or "-" symbols
- **Adjustable Blur Size**: Modify the maximum blur kernel size along with the strength
- **Session Management**: Maintains user sessions for a seamless experience
- **Direct Image Upload**: Easily start a new session by uploading an image

#### Setup

1. Make sure you have set up your Discord bot token in the `.env` file (same as the main Discord bot):
   ```
   DISCORD_BOT_TOKEN=your_actual_token_here
   ```

2. Run the Blur Discord bot:

   **For Windows:**
   ```bash
   run_blur_discord_bot.bat
   ```

   **For macOS/Linux:**
   ```bash
   ./run_blur_discord_bot.sh
   ```

#### Usage

1. **Start a session**: Upload an image to any channel where the bot has access
2. **Select focal point**:
   - The bot will respond with a grid overlaid on your image
   - Reply with a letter (A-I) to select a grid section
   - The bot will then show a refined grid with numbers (1-9)
   - Reply with a number to select the exact focal point

3. **Adjust blur effect**:
   - After receiving the blurred image, reply with:
     - `+` to increase blur strength (can stack, e.g., `+++` for larger increase)
     - `-` to decrease blur strength (can stack, e.g., `---` for larger decrease)
   - The bot will regenerate the image with the adjusted blur settings

4. **Start a new session**: Simply upload a new image at any time

#### Example Commands

- `+` - Slightly increase blur strength
- `+++` - Significantly increase blur strength
- `-` - Slightly decrease blur strength
- `--` - Moderately decrease blur strength

#### Troubleshooting

- If the bot doesn't respond to uploads, ensure it has proper permissions in the Discord channel
- For very large images, the bot may take longer to process or might fail due to Discord file size limitations
- If you encounter "Invalid selection" messages, try uploading the image again to start a fresh session

### Command Line Interface

Process a single image using:

```bash
python stereogram_main.py --mode cli --input path/to/your/image.jpg --output path/to/output/folder
```

### Test Converter

Run the test converter with sample images:

```bash
python stereogram_main.py --mode test
```

## Documentation

The project includes comprehensive documentation:

- [Application User Guide](docs/APP_GUIDE.md) - How to use the application interface
- [Setup Guide](docs/SETUP_GUIDE.md) - Complete setup instructions
- [Discord Bot Guide](docs/DISCORD_BOT_README.md) - How to set up and use the Discord bot
- [Discord Token Guide](docs/DISCORD_TOKEN_GUIDE.md) - Step-by-step guide to obtain a Discord bot token
- [macOS Virtual Environment Setup](docs/MACOS_VENV_GUIDE.md) - Guide for setting up a Python virtual environment on macOS
- [Anime Face Tracking Reference](docs/ANIME_FACE_TRACKING.md) - Information about anime face tracking integration
- [Debugging Guide](docs/DEBUGGING_GUIDE.md) - Troubleshooting common issues

## Project Structure

```
stereOgram_SBS_Converter/
├── core/                     # Core processing modules
│   ├── depth_anything_v2/    # Depth estimation model
│   ├── advanced_infill.py    # Advanced inpainting techniques
│   └── stereogram_sbs3d_converter.py  # Main converter class
├── ui/                       # User interface components
│   └── gradio_interface.py   # Gradio web interface
├── debug/                    # Debugging tools
│   └── debug_discord_bot.py  # Discord bot diagnostic utility
├── utils/                    # Utility modules
│   └── lut/                  # Look-up table implementations
├── tools/                    # Helper tools and scripts
│   └── test_converter.py     # Test converter utility
├── scripts/                  # Additional batch files and scripts
├── models/                   # Pre-trained model weights
├── demo_images/              # Sample images for testing
├── results/                  # Output folder for generated images
├── docs/                     # Documentation
├── stereogram_main.py        # Main entry point
├── run_stereogram_ui.py      # Direct UI launcher
├── run_discord_bot.bat       # Windows batch file for Discord bot (root for easy access)
└── run_stereogram_sbs3d_gui.bat # Windows batch file for GUI (root for easy access)
```

## Debugging

If you encounter issues, you can run the debug mode:

```bash
python stereogram_main.py --mode debug
```

This will run the diagnostic utilities in the debug directory, which can help identify common issues.

For macOS users experiencing SSL certificate verification errors, try:

```bash
SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") python stereogram_main.py --mode bot
```

For more detailed troubleshooting information, refer to the [Debugging Guide](docs/DEBUGGING_GUIDE.md).

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
