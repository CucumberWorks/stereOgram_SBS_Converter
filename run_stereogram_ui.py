#!/usr/bin/env python3
"""
Stereogram SBS Converter UI Launcher
-----------------------------------
This script launches the Gradio web interface for the Stereogram SBS Converter.
"""

import os
import sys

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the UI
if __name__ == "__main__":
    from ui.gradio_interface import run_interface
    run_interface() 