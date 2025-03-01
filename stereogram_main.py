#!/usr/bin/env python3
"""
Stereogram SBS Converter Main Entry Point
-----------------------------------------
This script serves as the main entry point for the Stereogram SBS Converter application.
It allows users to choose between different modes of operation.
"""

import os
import sys
import argparse


def main():
    """Main entry point for the stereogram SBS converter."""
    parser = argparse.ArgumentParser(
        description="Stereogram SBS Converter - Convert regular images to stereogram SBS 3D format"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["ui", "test", "bot", "cli", "debug"], 
        default="ui",
        help="Operation mode: ui (Gradio web interface), test (run test converter), bot (run Discord bot), cli (command line interface), debug (run debug tools)"
    )
    
    parser.add_argument(
        "--input", 
        type=str,
        help="Input image path for CLI mode"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Output directory for CLI mode"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        default=True,
        help="Share the Gradio UI publicly (for UI mode)"
    )
    
    args = parser.parse_args()
    
    # Ensure the current directory is in the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    if args.mode == "ui":
        print("Starting Gradio web interface...")
        from ui.gradio_interface import run_interface
        return run_interface(share=args.share)
        
    elif args.mode == "test":
        print("Running test converter...")
        from tools.test_converter import main as test_main
        return test_main()
        
    elif args.mode == "bot":
        print("Starting Discord bot...")
        import discord_stereo_bot
        return discord_stereo_bot.main()
        
    elif args.mode == "debug":
        print("Starting Discord bot in debug mode...")
        from debug.debug_discord_bot import main as debug_main
        return debug_main()
        
    elif args.mode == "cli":
        if not args.input:
            print("Error: Input image required for CLI mode")
            return 1
            
        output_dir = args.output or "results"
        print(f"Processing image {args.input} to directory {output_dir}...")
        from tools.test_converter import test_converter
        result = test_converter(args.input, output_dir)
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main()) 