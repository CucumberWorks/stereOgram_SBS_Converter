# stereOgram SBS3D converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A tool that converts regular 2D images into stereogram 3D formats (anaglyph and side-by-side) using depth estimation with Depth Anything V2.

## Features

- **High-quality depth estimation** using Depth Anything V2
- Generation of **anaglyph (red-cyan) 3D images**
- Generation of **side-by-side 3D images** for VR/AR viewers
- **Wiggle GIF animation** for glasses-free 3D viewing
- **Advanced hole filling** for parallax gaps using AI-based inpainting
- **Discord bot integration** for easy sharing and conversion through Discord
- Preserves original pixels for highest quality results
- Supports various resolutions and quality settings

## Screenshots

*[Place screenshots here]*

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended (8GB+ VRAM)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/CucumberWorks/stereOgram_SBS3D_Converter.git
   cd stereOgram_SBS3D_Converter
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Model weights:
   
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

## Usage

Run the demo script to process sample images:

```bash
python test_with_demo.py
```

Or use the graphical interface for easier operation:

```bash
python gradio_interface.py
```

Alternatively, you can run the batch files (Windows only):
- `run_stereogram_sbs3d.bat` - Command-line version
- `run_stereogram_sbs3d_gui.bat` - Graphical interface version (recommended)

### Command-line options:

```bash
python test_with_demo.py [OPTIONS]
```

Options:
- `--medium-memory`: Use medium memory mode (recommended for most GPUs)
- `--low-memory`: Use low memory mode (for GPUs with limited VRAM)
- `--resolution=2160`: Set output resolution (720, 1080, 1440, 2160)
- `--patch-size=384`: Set patch size for inpainting (larger values = better quality, more VRAM)
- `--patch-overlap=128`: Set patch overlap for inpainting (larger values = better blending)
- `--steps=30`: Set number of inference steps for inpainting (more = better quality, slower)
- `--high-quality`: Enable high quality mode (better but slower processing)
- `--debug`: Enable debug mode with purple background to visualize holes

Example:
```bash
python test_with_demo.py --medium-memory --resolution=2160 --patch-size=384 --patch-overlap=128 --steps=30 --high-quality
```

## Interface Options

### Graphical User Interface

The Gradio web interface provides an easy-to-use GUI for converting images:

```bash
python gradio_interface.py
```

### Discord Bot

A Discord bot is available for converting images directly within Discord:

```bash
python discord_stereo_bot.py
```

Key features of the Discord bot:
- Convert images with a simple `!sbs` command
- Automatic depth map generation and 3D conversion
- Produces side-by-side 3D images, depth maps, and wiggle GIFs
- Optimized for Discord's file size limits
- Works with image attachments or replies

See the [Discord Bot Guide](DISCORD_BOT_README.md) for complete setup and usage instructions.

## Adding your own images

Place your images in the `demo_images` folder, and they will be automatically processed.
Results will be saved in the `results` folder.

## Changing the depth model

The tool currently uses the Depth Anything V2 model. You can choose between three model variants:
- **ViT-S** (smallest/fastest) - For systems with limited resources, good for Discord bots
- **ViT-B** (medium/balanced) - Good for most systems, balances quality and speed
- **ViT-L** (largest/best quality) - For high-end systems with plenty of RAM and VRAM

In the GUI, you can select the model size in the "Initialize" tab. The graphical interface defaults to Low Memory Mode for better compatibility with all GPUs.