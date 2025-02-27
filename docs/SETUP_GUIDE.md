# Stereo3D Discord Bot - Setup Guide

This guide will walk you through the complete setup of the Stereo3D Discord bot, from creating your Discord application to running the bot on your server.

## Prerequisites

Before you begin, make sure you have:

- Python 3.8+ installed on your system
- Git installed (to clone the repository)
- A Discord account with access to the [Discord Developer Portal](https://discord.com/developers/applications)
- (Recommended) A CUDA-compatible GPU with at least 8GB VRAM for optimal performance

## Step 1: Creating a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click the "New Application" button in the top right corner
3. Give your application a name (e.g., "Stereo3D Bot") and click "Create"
4. In the left sidebar, click on "Bot"
5. Click "Add Bot" and confirm with "Yes, do it!"
6. Under the "Privileged Gateway Intents" section, enable "Message Content Intent" (this is required for the bot to see and respond to messages with images)
7. Under the "TOKEN" section, click "Reset Token" and then "Copy" to copy your bot token
8. Keep this token secure! You'll need it in a later step

## Step 2: Inviting the Bot to Your Server

1. In the left sidebar, click on "OAuth2" and then "URL Generator"
2. Under "Scopes", select "bot"
3. Under "Bot Permissions", select the following permissions:
   - Send Messages
   - Send Messages in Threads
   - Attach Files
   - Read Message History
   - Read Messages/View Channels
4. Copy the generated URL from the bottom of the page
5. Paste this URL into your web browser, select your server, and authorize the bot

## Step 3: Setting Up the Bot Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Stereo3D-Discord-Bot.git
   cd Stereo3D-Discord-Bot
   ```

2. Create a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your bot token:
   ```
   DISCORD_BOT_TOKEN=your_bot_token_here
   ```
   Replace `your_bot_token_here` with the token you copied in Step 1.

## Step 4: Running the Bot

1. Start the bot:
   ```bash
   python discord_stereo_bot.py
   ```
   
   On Windows, you can also use the provided batch file:
   ```
   run_discord_bot.bat
   ```

2. You should see a message that your bot is online. The first time you run the bot, it will automatically download the necessary model weights.

## Step 5: Using the Bot

1. In any channel where the bot has access, type `!sbs` and attach an image
2. Alternatively, reply to a message that contains an image and type `!sbs`
3. The bot will process the image and return a side-by-side 3D version, along with a wiggle GIF

## Advanced Configuration

### Choosing a Different Model Size

You can modify the bot's configuration by editing the `discord_stereo_bot.py` file:

```python
converter = StereogramSBS3DConverter(
    use_advanced_infill=True,
    depth_model_type="depth_anything_v2",
    model_size="vits",  # Change this to "vitb" or "vitl" for different model sizes
    max_resolution=4096,
    low_memory_mode=True
)
```

Available model sizes:
- `vits`: Smallest model, fastest processing, lowest memory requirements (~47 MB)
- `vitb`: Medium model, balanced between quality and speed (~376 MB)
- `vitl`: Largest model, highest quality but slowest and requires more memory (~1.17 GB)

### Low Memory Mode

If you're running on a system with limited resources, make sure `low_memory_mode=True` is set in the configuration.

### Customizing the Command

You can change the command prefix and command name by setting these environment variables in your `.env` file:

```
BOT_PREFIX=!
COMMAND_NAME=sbs
```

## Troubleshooting

If you encounter issues with the bot, refer to the [Debugging Guide](DEBUGGING_GUIDE.md) for troubleshooting steps.

Common issues:
- "Privileged intent(s) provided is not enabled": Make sure you enabled the "Message Content Intent" in the Discord Developer Portal
- CUDA/GPU errors: Try setting `low_memory_mode=True` and using the smaller `vits` model
- Token errors: Make sure your bot token in the `.env` file is correct and hasn't been regenerated 