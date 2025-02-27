import os
import sys
import subprocess
import importlib.util
import urllib.request
import json
import tkinter as tk
from tkinter import messagebox

APP_NAME = "stereOgram SBS3D Converter"
VERSION = "1.0.0"

def check_models_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs("models", exist_ok=True)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def show_splash_screen():
    """Show a splash screen while loading dependencies"""
    root = tk.Tk()
    root.title(APP_NAME)
    
    # Center the window
    window_width = 500
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Add content
    tk.Label(root, text=APP_NAME, font=("Arial", 24, "bold")).pack(pady=20)
    tk.Label(root, text=f"Version {VERSION}", font=("Arial", 12)).pack()
    
    status_label = tk.Label(root, text="Initializing...", font=("Arial", 10))
    status_label.pack(pady=20)
    
    # Add a progress bar
    progress = tk.Canvas(root, width=400, height=20, bg="white")
    progress.pack(pady=10)
    
    root.update()
    return root, status_label, progress

def update_status(root, label, message):
    """Update status message"""
    label.config(text=message)
    root.update()

def update_progress(root, canvas, value, max_value):
    """Update progress bar"""
    canvas.delete("progress")
    width = 400 * (value / max_value)
    canvas.create_rectangle(0, 0, width, 20, fill="green", tags="progress")
    root.update()

def main():
    # Create a splash screen
    splash, status_label, progress_bar = show_splash_screen()
    
    try:
        # Initialize environment
        update_status(splash, status_label, "Initializing environment...")
        update_progress(splash, progress_bar, 1, 4)
        
        # Check models directory
        check_models_directory()
        update_status(splash, status_label, "Checking dependencies...")
        update_progress(splash, progress_bar, 2, 4)
        
        # Launch the GUI
        update_status(splash, status_label, "Starting application...")
        update_progress(splash, progress_bar, 3, 4)
        
        # Import and run the gradio interface
        update_status(splash, status_label, "Loading GUI...")
        update_progress(splash, progress_bar, 4, 4)
        
        # Close splash screen after a short delay
        splash.after(1000, splash.destroy)
        splash.mainloop()
        
        # Import the gradio interface module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import gradio_interface
        
        # Run the interface
        gradio_interface.build_interface().launch(share=False, inbrowser=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        splash.destroy()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 