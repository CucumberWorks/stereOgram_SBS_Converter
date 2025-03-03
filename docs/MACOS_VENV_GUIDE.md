# Setting Up a Python Virtual Environment on macOS

This guide will walk you through setting up a Python virtual environment on macOS for the stereOgram SBS Converter project, along with solutions to common macOS-specific issues.

## Prerequisites

Before you begin, make sure you have:
- macOS 10.14 (Mojave) or later
- Admin privileges on your Mac

## Step 1: Install Python

macOS comes with a system version of Python, but it's better to install a separate version for development:

### Using Homebrew (Recommended)

1. Install Homebrew if you don't have it already:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python 3:
   ```bash
   brew install python@3.10
   ```

3. Verify the installation:
   ```bash
   python3 --version
   ```

### Using the Official Installer

Alternatively, you can download the installer from the [Python website](https://www.python.org/downloads/macos/).

## Step 2: Create a Virtual Environment

1. Navigate to your project directory:
   ```bash
   cd stereOgram_SBS_Converter
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
   
   Your terminal prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active.

## Step 3: Install Dependencies

With your virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

For M1/M2 Mac users (Apple Silicon), some packages may need special installation:

```bash
pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## Step 4: Fix Common macOS Issues

### SSL Certificate Verification Issues

If you encounter SSL certificate verification errors when running the Discord bot:

1. Install certifi:
   ```bash
   pip install certifi
   ```

2. Run the bot with the SSL certificate path:
   ```bash
   SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") python stereogram_main.py --mode bot
   ```

### Metal Performance Shaders (MPS) for M1/M2 Macs

If you have an Apple Silicon Mac (M1/M2), you can enable MPS acceleration:

1. Make sure PyTorch 2.0+ is installed:
   ```bash
   pip install --upgrade torch torchvision
   ```

2. The converter will automatically detect and use MPS if available.

### Permission Issues

If you encounter permission errors:

1. Check file permissions:
   ```bash
   chmod -R 755 stereOgram_SBS_Converter
   ```

2. If you're unable to install packages, try:
   ```bash
   pip install --user -r requirements.txt
   ```

## Step 5: Deactivating the Virtual Environment

When you're done working on the project, deactivate the virtual environment:

```bash
deactivate
```

## Working with Virtual Environments in Visual Studio Code

If you're using Visual Studio Code:

1. Open the Command Palette (Cmd+Shift+P)
2. Search for "Python: Select Interpreter"
3. Select the Python interpreter in your virtual environment (usually shows as `.venv` or `venv`)

## Troubleshooting

### Package Installation Failures

If you encounter issues installing packages:

1. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

2. Try installing with the `--no-cache-dir` option:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

### "Command not found: python3"

If you get "command not found" errors:

1. Check if Python is installed:
   ```bash
   which python3
   ```

2. If it's installed in a non-standard location, add it to your PATH in `~/.zshrc` or `~/.bash_profile`:
   ```bash
   export PATH="/usr/local/opt/python/libexec/bin:$PATH"
   ```

3. Reload your terminal configuration:
   ```bash
   source ~/.zshrc   # or source ~/.bash_profile
   ```

### Disk Space Issues

Check your available disk space:

```bash
df -h
```

The downloaded model files can be large, so make sure you have enough space (at least 2GB recommended).

### Can't Start the Discord Bot

If the Discord bot won't start:

1. Check for SSL certificate issues (see above)
2. Verify your `.env` file contains a valid token
3. Make sure Discord's Message Content Intent is enabled in the Developer Portal 