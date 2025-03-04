#!/bin/bash

echo "Starting stereOgram SBS Converter Discord Bot..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in your PATH."
    echo "Please install Python from https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if SSL certificate verification might be an issue on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected, applying SSL certificate fix..."
    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
fi

# Run the Discord bot
python3 stereogram_main.py --mode bot

# If the bot crashes, don't immediately close the window
if [ $? -ne 0 ]; then
    echo ""
    echo "The bot has crashed with error code $?"
    echo "Check the error message above for details."
    read -p "Press Enter to exit..."
fi 