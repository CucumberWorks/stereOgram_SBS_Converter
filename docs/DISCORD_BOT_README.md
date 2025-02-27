# Stereo 3D Discord Bot

This Discord bot automatically converts 2D images into side-by-side (SBS) 3D stereo format. Users can simply send an image or use a command to convert images into 3D format.

## Features

- Converts 2D images to SBS 3D format
- Works with message attachments or replies to messages with images
- Uses AI-powered depth estimation for high-quality 3D conversion
- Automatically handles inpainting to fill gaps in the 3D conversion
- Enhanced anti-banding for smooth color gradients
- High-precision color processing to preserve image quality
- Improved messaging format - SBS image sent first, extras sent separately
- Creates a wiggle GIF animation for glasses-free 3D viewing
- Generates colorized depth maps for visualization
- Smart file format selection based on image size
- Simple to use with a single command

## Setup

1. **Prerequisites**
   - Python 3.8 or higher
   - All dependencies listed in `requirements.txt`
   - CUDA-compatible GPU recommended (at least 4GB VRAM)

2. **Installation**
   ```bash
   # Clone the repository (if you haven't already)
   git clone https://github.com/yourusername/Stereo3D.git
   cd Stereo3D
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   - Create a `.env` file in the root directory of the project
   - Add your Discord bot token to the file:
   ```
   DISCORD_BOT_TOKEN=your_actual_token_here
   ```
   - The `.env` file is included in `.gitignore` to prevent accidentally committing your token
   - You can use the provided `.env.example` as a template

4. **Discord Bot Setup**
   - Create a Discord application at the [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a Bot under your application
   - Enable the "Message Content Intent" under Bot settings
   - Copy your bot token from the Bot settings page
   - Add your token to the `.env` file as described above
   - Use the OAuth2 URL Generator to create an invite link for your bot with "bot" scope and "Send Messages", "Read Message History", and "Attach Files" permissions

5. **Running the Bot**
   ```bash
   python discord_stereo_bot.py
   ```
   
   Alternatively, on Windows, you can use the included batch file:
   ```
   run_discord_bot.bat
   ```

## Usage

1. **Direct Command**
   - Attach an image to a message and type `!sbs` to convert it
   ```
   !sbs
   ```

2. **Reply Command**
   - Reply to a message containing an image and type `!sbs` to convert it

## Output Files

The bot generates several files for each conversion:

1. **SBS 3D Image** (sent first in a separate message)
   - Side-by-side 3D format for viewing with VR headsets or parallel/cross-eye viewing
   - Format: PNG (for smaller images) or high-quality JPEG (for larger images)

2. **Additional Files** (sent in a follow-up message)
   - **Wiggle GIF Animation**: Alternating left/right views for glasses-free 3D effect
   - **Depth Map**: Colorized visualization of the estimated depth

## Configuration

You can modify these settings in the `discord_stereo_bot.py` file:

- `BOT_PREFIX`: Change the command prefix (default: `!`)
- `COMMAND_NAME`: Change the command name (default: `sbs`)
- Stereo converter settings:
  - `use_advanced_infill`: Whether to use advanced AI inpainting (default: `True`)
  - `depth_model_type`: Type of depth model to use (default: `depth_anything_v2`)
  - `model_size`: Size of model to use - available options are:
    - `vits`: Small model (fastest, lowest VRAM usage)
    - `vitb`: Base model (balanced speed/quality)
    - `vitl`: Large model (highest quality, requires more VRAM)
  - `max_resolution`: Maximum resolution for processing (default: `4096`)
  - `dithering_level`: Strength of dithering to combat banding (default: `1.5`)
  - `high_color_quality`: Whether to use high-precision color processing (default: `True`)
  - Inpainting parameters: `steps`, `guidance_scale`, `patch_size`, `patch_overlap`

## Tips for Best Results

- **Image Quality**: Higher quality input images produce better results
- **Subjects**: Images with clear foreground-background separation work best
- **Processing Time**: Complex images may take longer to process
- **Memory Usage**: If your bot is running out of memory, consider:
  - Reducing `max_resolution` to `2048` or lower
  - Using a smaller model like `vits` (default)
  - Enabling `low_memory_mode=True`

## Troubleshooting

- If the bot fails to process an image, check that the image is in a supported format (JPG, PNG)
- For "413 Payload Too Large" errors, the bot will automatically reduce image size, but you may need to use smaller input images
- If you see "CUDA out of memory" errors, reduce the max_resolution or use a smaller model size
- Ensure your bot has proper permissions in the Discord server

## License

This project is licensed under the same license as the main Stereo3D project. 