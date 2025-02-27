# Stereo 3D Discord Bot - Debugging Guide

This guide will help you troubleshoot common issues with the Stereo 3D Discord bot.

## Step 1: Run the Debug Script

First, run the debugging script to check your environment:

```bash
python debug_discord_bot.py
```

This will check your dependencies, token configuration, and PyTorch setup.

## Step 2: Test the Converter Separately

Before running the full Discord bot, test that the converter itself is working correctly:

```bash
python test_converter.py
```

Or use the convenience batch file:

```bash
run_test_converter.bat
```

This will:
1. Load a sample image
2. Initialize the converter
3. Convert the image to SBS format
4. Save the result and a depth map in the `test_results` folder

If this fails, the issue is with the converter itself, not the Discord integration.

## Step 3: Check for Common Issues

### Discord Bot Token Issues

- **Invalid Token**: Make sure your token in `.env` is correct and hasn't been regenerated in the Discord Developer Portal.
- **Format**: The token should be a long string with dots and look something like: `YOUR_DISCORD_BOT_TOKEN_HERE`.
- **Privileged Intents**: Ensure you've enabled the "Message Content Intent" in your bot settings in the Discord Developer Portal.

### Memory Issues

If you're experiencing out-of-memory errors:

1. Try the lower memory version of the bot:
   ```bash
   python discord_stereo_bot_low_memory.py
   ```

2. Reduce the model size in `discord_stereo_bot.py`:
   ```python
   converter = StereogramSBS3DConverter(
       model_size="vits",  # Smaller model
       max_resolution=1024,  # Lower resolution
       low_memory_mode=True
   )
   ```

### Performance Issues

- **Slow Processing**: SBS conversion is computationally intensive. Without a CUDA-capable GPU, processing will be much slower.
- **Timeouts**: Discord has a 15-minute timeout for bot responses. For very large images or slow systems, the bot might timeout.

## Step 4: Run the Bot

Once the debug checks pass and the converter test works:

```bash
python discord_stereo_bot.py
```

Or use the convenience batch file:

```bash
run_discord_bot.bat
```

## Step 5: Testing in Discord

1. Invite the bot to your server with the invite link from the Discord Developer Portal
2. Send an image with the command `!sbs`
3. The bot should process the image and send back an SBS version

## Common Error Messages

### "ModuleNotFoundError: No module named 'xxx'"

Run: `pip install -r requirements.txt` to install all dependencies.

### "Privileged intent(s) provided is not enabled"

Enable "Message Content Intent" in the Discord Developer Portal under your bot settings.

### "Error loading model" or CUDA-related errors

- Check that you have sufficient VRAM if using GPU
- Try using the low memory version of the bot
- Set `model_size="vits"` for the smallest model

### "Cannot run bot: Improper token"

Check your `.env` file and make sure the token is correct.

## Advanced Debugging

### Check Discord.py Logs

Add logging to see more details about what's happening:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Add this near the top of your `discord_stereo_bot.py` file.

### Memory Profiling

If you're having memory issues, you can use tools like `memory_profiler` to identify where memory is being used:

```bash
pip install memory_profiler
python -m memory_profiler discord_stereo_bot.py
```

## Getting Help

If you continue to have issues after following these steps, check the discord.py documentation or reach out to the Discord.py community for help with bot-specific issues.

For issues with the SBS converter functionality, check the main project documentation. 