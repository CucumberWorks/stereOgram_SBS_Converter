import discord
from discord.ext import commands
import os
import io
import aiohttp
import asyncio
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gc
import time
import torch
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from io import BytesIO
# Import Discord UI components
from discord.ui import Button, View

# Import the converter from the core module for using the blur functionality
# This bot leverages the sophisticated depth-based blur algorithm from the core converter
# for high-quality depth of field effects with proper bokeh simulation
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter

# Load environment variables
load_dotenv()

# Translation dictionaries for multilingual support
TRANSLATIONS = {
    "en": {
        "select_grid_cell": "**Select a grid cell** by clicking a button below:",
        "select_sub_grid": "**Select a precise focal point** within cell {}:",
        "adjust_blur": "**Adjust the blur effect** using the buttons below:",
        "processing": "Processing your image... ⏳",
        "not_your_session": "This is not your blur session. Only the user who started it can interact with these buttons.",
        "done": "Done! Here's your image with the background blurred:",
        "language_english": "English",
        "language_japanese": "Japanese",
        "increase_blur": "Stronger Blur (+)",
        "decrease_blur": "Lighter Blur (-)",
        "increase_blur_more": "Much Stronger Blur (++)",
        "decrease_blur_more": "Much Lighter Blur (--)",
        "finished": "Finished",
        "active_session": "You already have an active blur session. Please use the buttons in your active session or wait for it to expire.",
        "attach_image": "Please attach an image to use this command!",
        "valid_image": "Please attach a valid image file (PNG, JPG, JPEG, WEBP, BMP)!",
        "download_failed": "Failed to download the image. Please try again!",
        "processing_wait": "Processing... Please wait"
    },
    "ja": {
        "select_grid_cell": "**グリッドセルを選択**してください：",
        "select_sub_grid": "セル{}内の**焦点を選択**してください：",
        "adjust_blur": "下のボタンで**ぼかし効果を調整**してください：",
        "processing": "画像を処理中です... ⏳",
        "not_your_session": "これはあなたのセッションではありません。開始したユーザーのみが操作できます。",
        "done": "完了！背景をぼかした画像です：",
        "language_english": "英語",
        "language_japanese": "日本語",
        "increase_blur": "強いぼかし (+)",
        "decrease_blur": "弱いぼかし (-)",
        "increase_blur_more": "さらに強いぼかし (++)",
        "decrease_blur_more": "さらに弱いぼかし (--)",
        "finished": "完了",
        "active_session": "すでにアクティブなぼかしセッションがあります。現在のセッションのボタンを使用するか、期限切れになるまでお待ちください。",
        "attach_image": "このコマンドを使用するには画像を添付してください！",
        "valid_image": "有効な画像ファイル（PNG、JPG、JPEG、WEBP、BMP）を添付してください！",
        "download_failed": "画像のダウンロードに失敗しました。もう一度お試しください！",
        "processing_wait": "処理中... お待ちください"
    }
}

# Blur configuration parameters - centralized for easy adjustment
BLUR_CONFIG = {
    # Default initial values
    "DEFAULT_BLUR_STRENGTH": 1.3,      # Default blur strength
    "DEFAULT_MAX_BLUR_SIZE": 35,       # Default max blur kernel size (must be odd number)
    "DEFAULT_FOCAL_THICKNESS": 0.25,   # Default focal plane thickness (smaller = narrower focus)
    "DEFAULT_APERTURE_SHAPE": "octagon", # Shape of bokeh (circle, hexagon, octagon)
    "DEFAULT_HIGHLIGHT_BOOST": 1.5,    # Default highlight boost factor
    "DEFAULT_EDGE_SMOOTHNESS": 2.0,    # Default edge smoothness factor
    "DEFAULT_CHROMATIC_ABERRATION": 0.0, # Default chromatic aberration (0 = disabled)
    "DEFAULT_ENABLE_DITHERING": True,  # Enable dithering by default to prevent banding
    
    # Adjustment increments for blur buttons
    "BLUR_INCREMENT_SMALL": 0.25,       # Small increment for blur strength
    "BLUR_INCREMENT_LARGE": 0.75,       # Large increment for blur strength
    "BLUR_SIZE_INCREMENT_SMALL": 6,    # Small increment for blur size (must be even to keep sizes odd)
    "BLUR_SIZE_INCREMENT_LARGE": 18,    # Large increment for blur size (must be even to keep sizes odd)
    "FOCAL_THICKNESS_INCREMENT_SMALL": 0.025,  # Small adjustment for focal thickness
    "FOCAL_THICKNESS_INCREMENT_LARGE": 0.075,  # Large adjustment for focal thickness
    
    # Limits
    "MIN_BLUR_STRENGTH": 0.16,          # Minimum blur strength
    "MAX_BLUR_STRENGTH": 10.0,         # Maximum blur strength
    "MIN_FOCAL_THICKNESS": 0.02,       # Minimum focal thickness
    "MAX_FOCAL_THICKNESS": 0.5,        # Maximum focal thickness
    "MIN_BLUR_SIZE": 1,                # Minimum blur kernel size
    "MAX_BLUR_SIZE": 81,               # Maximum blur kernel size
    "MIN_EDGE_SMOOTHNESS": 0.5,        # Minimum edge smoothness value
    "MAX_EDGE_SMOOTHNESS": 2.0,        # Maximum edge smoothness value
    
    # Session timeout (in seconds)
    "SESSION_TIMEOUT": 600             # 10 minutes
}

# Helper function to get translated text
def get_text(key, language, *args):
    """Get translated text for a given key and language with optional formatting args"""
    if language not in TRANSLATIONS or key not in TRANSLATIONS[language]:
        # Fallback to English if translation not found
        text = TRANSLATIONS["en"].get(key, key)
    else:
        text = TRANSLATIONS[language][key]
    
    # Apply any formatting arguments
    if args:
        text = text.format(*args)
    
    return text

# Configuration
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Load token from .env file
BOT_PREFIX = "!"
COMMAND_NAME = "blur"

# Initialize the bot with intents
intents = discord.Intents.default()
intents.message_content = True  # Required to see message content
bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

# We'll initialize the converter only when needed to save memory
converter = None

# Dictionary to store user sessions
user_sessions = {}

def clear_converter():
    """Clear the converter to free up memory."""
    global converter
    if converter is not None:
        converter.clear_vram_cache()
        del converter
        converter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def end_user_session(user_id):
    """End a user's session cleanly."""
    if user_id in user_sessions:
        # Clean up any large data
        session = user_sessions[user_id]
        session.original_image = None
        session.grid_image = None
        session.checkerboard_image = None
        session.sub_grid_image = None
        # Remove from sessions dictionary
        del user_sessions[user_id]

class BlurSession:
    """Class to manage user session state for blur operations."""
    
    def __init__(self, user_id, original_image, original_message=None):
        """Initialize a blur session for a user."""
        self.user_id = user_id
        self.original_image = original_image
        self.grid_image = None
        self.grid_cells = {}
        self.checkerboard_image = None
        self.selected_cell = None
        self.sub_grid_image = None
        self.sub_grid_cells = {}
        self.focal_point = None
        self.blur_strength = BLUR_CONFIG["DEFAULT_BLUR_STRENGTH"]  # Default blur strength
        self.max_blur_size = BLUR_CONFIG["DEFAULT_MAX_BLUR_SIZE"]  # Default max blur size (kernel size)
        self.focal_thickness = BLUR_CONFIG["DEFAULT_FOCAL_THICKNESS"]  # Default focal plane thickness
        self.aperture_shape = BLUR_CONFIG["DEFAULT_APERTURE_SHAPE"]  # Default bokeh shape
        self.highlight_boost = BLUR_CONFIG["DEFAULT_HIGHLIGHT_BOOST"]  # Default highlight boost
        self.edge_smoothness = BLUR_CONFIG["DEFAULT_EDGE_SMOOTHNESS"]  # Default edge smoothness
        self.chromatic_aberration = BLUR_CONFIG["DEFAULT_CHROMATIC_ABERRATION"]  # Default chromatic aberration
        self.enable_dithering = BLUR_CONFIG["DEFAULT_ENABLE_DITHERING"]  # Default dithering setting
        self.last_interaction = time.time()
        self.expiry_time = time.time() + BLUR_CONFIG["SESSION_TIMEOUT"]  # Session timeout
        self.selection_message_id = None
        self.result_message_id = None
        self.selection_stage = 1   # 1: Main grid selection, 2: Sub-grid selection
        self.view_message = None   # To store the message containing button views
        self.view_id = None        # To store the ID of the view message
        self.original_message = original_message  # Reference to the original command response message
        self.loading_image = None  # Placeholder for loading image
        self.language = "en"       # Default language is English

# Function to periodically clean up expired sessions
async def clean_expired_sessions():
    """Remove expired user sessions to prevent memory leaks."""
    while True:
        current_time = time.time()
        expired_users = []
        
        for user_id, session in user_sessions.items():
            if current_time > session.expiry_time:
                expired_users.append(user_id)
        
        for user_id in expired_users:
            print(f"Session expired for user ID {user_id}")
            end_user_session(user_id)
        
        await asyncio.sleep(60)  # Check every minute

# Main Grid Selection View
class MainGridView(View):
    def __init__(self, session, ctx, timeout=600):
        super().__init__(timeout=timeout)
        self.session = session
        self.ctx = ctx
        self.add_selection_buttons()
        self.add_language_buttons()
    
    def add_selection_buttons(self):
        # Get all available grid cells from the session
        grid_cells = list(self.session.grid_cells.keys())
        
        # Group grid cells by row
        rows = {}
        for cell in grid_cells:
            # Cell format is typically like 'A1', 'B2', etc.
            # Extract row number
            row_identifier = cell[1:]  # Everything after the first character is the row number
            
            if row_identifier not in rows:
                rows[row_identifier] = []
            rows[row_identifier].append(cell)
        
        # Sort row identifiers and sort cells within each row
        sorted_row_identifiers = sorted(rows.keys())
        
        # Track Discord UI row (0-4)
        discord_row = 0
        
        # Add buttons row by row
        for row_identifier in sorted_row_identifiers:
            # Sort cells within this row (e.g., A1, B1, C1)
            rows[row_identifier].sort()
            
            # Add buttons for this row
            for cell_code in rows[row_identifier]:
                # Create button for this cell
                button = Button(
                    label=cell_code, 
                    style=discord.ButtonStyle.primary if int(row_identifier) % 2 == 0 else discord.ButtonStyle.secondary, 
                    custom_id=f"main_grid_{cell_code}",
                    row=discord_row
                )
                
                # Adding the callback function
                button.callback = self.button_callback
                self.add_item(button)
            
            # Move to next Discord UI row
            discord_row += 1
            # Ensure we don't exceed Discord's limit of 5 rows
            if discord_row >= 4:  # Reduced from 5 to leave room for language buttons
                break
    
    def add_language_buttons(self):
        # Add language selection buttons in the last row (row 4)
        # English button
        en_button = Button(
            label=get_text("language_english", self.session.language),
            style=discord.ButtonStyle.success if self.session.language == "en" else discord.ButtonStyle.secondary,
            custom_id="language_en",
            row=4  # Last row
        )
        en_button.callback = self.language_button_callback
        self.add_item(en_button)
        
        # Japanese button
        ja_button = Button(
            label=get_text("language_japanese", self.session.language),
            style=discord.ButtonStyle.success if self.session.language == "ja" else discord.ButtonStyle.secondary,
            custom_id="language_ja",
            row=4  # Last row
        )
        ja_button.callback = self.language_button_callback
        self.add_item(ja_button)
    
    async def language_button_callback(self, interaction):
        # Only the user who started the session can interact with buttons
        if interaction.user.id != self.session.user_id:
            await interaction.response.send_message(
                get_text("not_your_session", self.session.language),
                ephemeral=True
            )
            return
        
        # Get the selected language from the button's custom_id
        selected_language = interaction.data["custom_id"].split("_")[-1]
        
        # Update the session's language
        self.session.language = selected_language
        
        # Update the session's last interaction time
        self.session.last_interaction = time.time()
        self.session.expiry_time = time.time() + BLUR_CONFIG["SESSION_TIMEOUT"]
        
        # Acknowledge the interaction
        await interaction.response.defer(ephemeral=True)
        
        # Create a new view with updated language
        new_view = MainGridView(self.session, self.ctx)
        
        # Update the existing message with the new view
        await self.session.view_message.edit(
            content=f"{self.ctx.author.mention} {get_text('select_grid_cell', self.session.language)}",
            view=new_view
        )
    
    async def button_callback(self, interaction):
        # Only the user who started the session can interact with buttons
        if interaction.user.id != self.session.user_id:
            # For unauthorized users, send an ephemeral message instead of editing the main message
            await interaction.response.send_message(
                get_text("not_your_session", self.session.language),
                ephemeral=True
            )
            return
        
        # Get the selected cell from the button's custom_id
        # Access from the component instead of interaction directly
        cell_code = interaction.data["custom_id"].split("_")[-1]
        self.session.selected_cell = cell_code
        
        # Update the session's last interaction time
        self.session.last_interaction = time.time()
        self.session.expiry_time = time.time() + BLUR_CONFIG["SESSION_TIMEOUT"]
        
        # Create and display sub-grid
        await self.create_and_show_sub_grid(interaction, cell_code)
    
    async def create_and_show_sub_grid(self, interaction, cell_code):
        # Acknowledge the interaction immediately to prevent timeout
        # Use ephemeral=True to hide the "thinking" message
        await interaction.response.defer(ephemeral=True)
        
        # Get the selected cell's boundaries
        cell_coords = self.session.grid_cells[cell_code]
        
        # Create the sub-grid image
        sub_grid_image, sub_grid_cells = create_sub_grid(
            self.session.original_image.copy(), 
            cell_coords, 
            cell_code
        )
        
        # Store the sub-grid cells in the session
        self.session.sub_grid_cells = sub_grid_cells
        
        # Save the sub-grid image to a temporary file with compression
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_sub_grid_{cell_code}.png") as temp_file:
            sub_grid_path = temp_file.name
            sub_grid_image.save(sub_grid_path, format="PNG", optimize=True, compress_level=6)
        
        # Create a Discord file object
        sub_grid_file = discord.File(sub_grid_path, filename=f"sub_grid_{cell_code}.png")
        
        # Create a new view for the sub-grid selection
        sub_grid_view = SubGridView(self.session, self.ctx)
        
        # Edit the original message instead of the interaction response
        await self.session.view_message.edit(
            content=f"{self.ctx.author.mention} {get_text('select_sub_grid', self.session.language, cell_code)}",
            attachments=[sub_grid_file],
            view=sub_grid_view
        )
        
        # Stop listening for interactions with this view
        self.stop()

# Sub Grid Selection View
class SubGridView(View):
    def __init__(self, session, ctx, timeout=600):
        super().__init__(timeout=timeout)
        self.session = session
        self.ctx = ctx
        self.add_selection_buttons()
    
    def add_selection_buttons(self):
        # Create a 3x3 grid of numbered buttons
        # Using the numpad-style layout to match create_sub_grid (7-8-9, 4-5-6, 1-2-3)
        
        # Row 1 (buttons 7-8-9)
        for i, num in enumerate([7, 8, 9]):
            button = Button(
                label=str(num),
                style=discord.ButtonStyle.primary,
                custom_id=f"sub_grid_{num}",
                row=0
            )
            button.callback = self.button_callback
            self.add_item(button)
        
        # Row 2 (buttons 4-5-6)
        for i, num in enumerate([4, 5, 6]):
            button = Button(
                label=str(num),
                style=discord.ButtonStyle.primary,
                custom_id=f"sub_grid_{num}",
                row=1
            )
            button.callback = self.button_callback
            self.add_item(button)
        
        # Row 3 (buttons 1-2-3)
        for i, num in enumerate([1, 2, 3]):
            button = Button(
                label=str(num),
                style=discord.ButtonStyle.primary,
                custom_id=f"sub_grid_{num}",
                row=2
            )
            button.callback = self.button_callback
            self.add_item(button)
    
    async def button_callback(self, interaction):
        # Only the user who started the session can interact with buttons
        if interaction.user.id != self.session.user_id:
            await interaction.response.send_message(
                get_text("not_your_session", self.session.language),
                ephemeral=True
            )
            return
        
        # Get the selected sub-cell from the button's custom_id
        sub_cell = interaction.data["custom_id"].split("_")[-1]
        
        # Update the session's last interaction time
        self.session.last_interaction = time.time()
        self.session.expiry_time = time.time() + BLUR_CONFIG["SESSION_TIMEOUT"]
        
        # Apply the blur effect with the selected sub-cell
        await self.apply_blur_with_focal_point(interaction, sub_cell)
    
    async def apply_blur_with_focal_point(self, interaction, sub_cell):
        # Acknowledge the interaction immediately to prevent timeout
        # Use ephemeral=True to hide the "thinking" message
        await interaction.response.defer(ephemeral=True)
        
        # Immediately show loading image before calculating anything
        # This ensures users get immediate feedback when they click a button
        if self.session.loading_image:
            # Save loading image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                loading_path = temp_file.name
                self.session.loading_image.save(loading_path, format="PNG", optimize=True, compress_level=6)
                
            # Create file for Discord
            loading_file = discord.File(loading_path, filename="loading.png")
            
            # Update the original message with the loading image immediately
            await self.session.view_message.edit(
                content=f"{self.ctx.author.mention} {get_text('processing', self.session.language)}",
                attachments=[loading_file],
                view=None  # Remove buttons during processing
            )
        
        try:
            # Get the coordinates from the sub-grid cell - use string key to match how it's stored
            # Debug the available keys in sub_grid_cells
            print(f"DEBUG: Selected sub_cell: {sub_cell}")
            print(f"DEBUG: Available sub_grid_cells keys: {list(self.session.sub_grid_cells.keys())}")
            
            # Use the string key directly without converting to int
            sub_cell_coords = self.session.sub_grid_cells[sub_cell]
            
            # Get image dimensions for normalization
            h, w = self.session.original_image.shape[:2]
            
            # Calculate the center point of the sub-cell as the focal point
            # The sub_cell_coords contains (x1, y1, x2, y2) - need to calculate center
            x1, y1, x2, y2 = sub_cell_coords
            center_x = (x1 + x2) / 2 / w  # Normalize to 0-1 range
            center_y = (y1 + y2) / 2 / h  # Normalize to 0-1 range
            
            # Debug the coordinates
            print(f"DEBUG: Sub cell coords: {sub_cell_coords}")
            print(f"DEBUG: Calculated center point: ({center_x:.4f}, {center_y:.4f})")
            
            # Store the focal point in the session
            self.session.focal_point = (center_x, center_y)
            
            # Now update the message with more specific information
            if self.session.loading_image:
                await self.session.view_message.edit(
                    content=f"{self.ctx.author.mention} {get_text('processing', self.session.language)}",
                    # No need to update the attachment as we're just changing the text
                )
            
            # Apply blur with initial settings - returns (blurred_image, depth_map)
            blurred_result, depth_map = await apply_focal_blur(
                self.session.original_image.copy(),
                center_x, center_y,  # Already normalized
                session=self.session,
                blur_strength=self.session.blur_strength,
                max_blur_size=self.session.max_blur_size,
                focal_thickness=self.session.focal_thickness
            )
            
            # Convert the blurred image to PIL format for saving
            blurred_image = Image.fromarray(cv2.cvtColor(blurred_result, cv2.COLOR_BGR2RGB))
            
            # Save the updated blurred image
            self.session.blurred_image = blurred_image
            
            # Save as temporary file with compression
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                blur_path = temp_file.name
                # Save with compression to reduce file size
                blurred_image.save(blur_path, format="PNG", optimize=True, compress_level=6)
            
            # Create file for Discord
            blur_file = discord.File(blur_path, filename="blurred_image_adjusted.png")
            
            # Create adjustment view
            adjustment_view = BlurAdjustmentView(self.session, self.ctx)
            
            # Edit the original message with the new blurred image
            await self.session.view_message.edit(
                content=f"{self.ctx.author.mention} {get_text('adjust_blur', self.session.language)}",
                attachments=[blur_file],
                view=adjustment_view
            )
            
            # Stop listening for interactions with this view
            self.stop()
            
        except Exception as e:
            # Handle errors by editing the original message
            try:
                await self.session.view_message.edit(
                    content=f"❌ {str(e)}\n{get_text('processing', self.session.language)}",
                    attachments=[],
                    view=None
                )
            except Exception as edit_error:
                # Log the error but don't create a new message
                print(f"Error during focal point processing and couldn't edit message: {str(e)}\nEdit error: {str(edit_error)}")
                # We don't use followup.send here to avoid creating new messages

# Blur Adjustment View
class BlurAdjustmentView(View):
    def __init__(self, session, ctx, timeout=600):
        super().__init__(timeout=timeout)
        self.session = session
        self.ctx = ctx
        self.add_adjustment_buttons()
    
    def add_adjustment_buttons(self):
        # Add buttons for blur adjustment in a better layout
        # Row 1: Blur strength controls
        button_more_blur = Button(
            label=get_text("increase_blur", self.session.language),
            style=discord.ButtonStyle.primary,
            custom_id="blur_more",
            row=0
        )
        button_more_blur.callback = self.increase_blur
        self.add_item(button_more_blur)
        
        button_less_blur = Button(
            label=get_text("decrease_blur", self.session.language),
            style=discord.ButtonStyle.primary,
            custom_id="blur_less",
            row=0
        )
        button_less_blur.callback = self.decrease_blur
        self.add_item(button_less_blur)
        
        # Row 2: More extreme blur controls
        button_much_more_blur = Button(
            label=get_text("increase_blur_more", self.session.language),
            style=discord.ButtonStyle.secondary,
            custom_id="blur_much_more",
            row=1
        )
        button_much_more_blur.callback = self.increase_blur_more
        self.add_item(button_much_more_blur)
        
        button_much_less_blur = Button(
            label=get_text("decrease_blur_more", self.session.language),
            style=discord.ButtonStyle.secondary,
            custom_id="blur_much_less",
            row=1
        )
        button_much_less_blur.callback = self.decrease_blur_more
        self.add_item(button_much_less_blur)
        
        # Row 3: Done button
        button_done = Button(
            label=get_text("finished", self.session.language),
            style=discord.ButtonStyle.success,
            custom_id="blur_done",
            row=2
        )
        button_done.callback = self.done_callback
        self.add_item(button_done)
    
    # Callback methods for the buttons
    async def increase_blur(self, interaction):
        await self.adjust_blur(
            interaction, 
            BLUR_CONFIG["BLUR_INCREMENT_SMALL"], 
            BLUR_CONFIG["BLUR_SIZE_INCREMENT_SMALL"], 
            -BLUR_CONFIG["FOCAL_THICKNESS_INCREMENT_SMALL"]
        )  # Increase blur strength and size, slightly decrease focal thickness
        
    async def decrease_blur(self, interaction):
        await self.adjust_blur(
            interaction, 
            -BLUR_CONFIG["BLUR_INCREMENT_SMALL"], 
            -BLUR_CONFIG["BLUR_SIZE_INCREMENT_SMALL"], 
            BLUR_CONFIG["FOCAL_THICKNESS_INCREMENT_SMALL"]
        )  # Decrease blur strength and size, slightly increase focal thickness
        
    async def increase_blur_more(self, interaction):
        await self.adjust_blur(
            interaction, 
            BLUR_CONFIG["BLUR_INCREMENT_LARGE"], 
            BLUR_CONFIG["BLUR_SIZE_INCREMENT_LARGE"], 
            -BLUR_CONFIG["FOCAL_THICKNESS_INCREMENT_LARGE"]
        )  # Increase blur strength and size more, decrease focal thickness more
        
    async def decrease_blur_more(self, interaction):
        await self.adjust_blur(
            interaction, 
            -BLUR_CONFIG["BLUR_INCREMENT_LARGE"], 
            -BLUR_CONFIG["BLUR_SIZE_INCREMENT_LARGE"], 
            BLUR_CONFIG["FOCAL_THICKNESS_INCREMENT_LARGE"]
        )  # Decrease blur strength and size more, increase focal thickness more
        
    async def done_callback(self, interaction):
        # Acknowledge the interaction immediately with ephemeral=True to hide the "thinking" message
        await interaction.response.defer(ephemeral=True)
        
        # Get the blurred image bytes from the current attachment
        attachment = interaction.message.attachments[0]
        
        # Update the original message with disabled buttons and completion message
        await self.session.view_message.edit(
            content=f"{self.ctx.author.mention} {get_text('done', self.session.language)}",
            attachments=[interaction.message.attachments[0]],
            view=None  # Remove all buttons
        )
        
        # Clean up the user's session
        end_user_session(self.session.user_id)
    
    async def adjust_blur(self, interaction, strength_adjustment, size_adjustment, thickness_adjustment):
        # Only the user who started the session can interact with buttons
        if interaction.user.id != self.session.user_id:
            # For unauthorized users, send an ephemeral message instead of editing the main message
            await interaction.response.send_message(
                get_text("not_your_session", self.session.language),
                ephemeral=True
            )
            return
        
        # Acknowledge the interaction immediately to prevent timeout
        # Use ephemeral=True to hide the "thinking" message
        await interaction.response.defer(ephemeral=True)
        
        # Immediately show loading image before calculating anything
        # This ensures users get immediate feedback when they click a button
        if self.session.loading_image:
            # Save loading image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                loading_path = temp_file.name
                self.session.loading_image.save(loading_path, format="PNG", optimize=True, compress_level=6)
                
            # Create file for Discord
            loading_file = discord.File(loading_path, filename="loading.png")
            
            # Update the original message with the loading image immediately
            await self.session.view_message.edit(
                content=f"{self.ctx.author.mention} {get_text('processing', self.session.language)}",
                attachments=[loading_file],
                view=None  # Remove buttons during processing
            )
        
        # Apply the adjustments
        old_blur_strength = self.session.blur_strength
        old_focal_thickness = self.session.focal_thickness
        old_max_blur_size = self.session.max_blur_size
        
        self.session.blur_strength = max(
            BLUR_CONFIG["MIN_BLUR_STRENGTH"], 
            min(BLUR_CONFIG["MAX_BLUR_STRENGTH"], 
                self.session.blur_strength + strength_adjustment)
        )
        self.session.max_blur_size = max(
            BLUR_CONFIG["MIN_BLUR_SIZE"], 
            min(BLUR_CONFIG["MAX_BLUR_SIZE"], 
                self.session.max_blur_size + size_adjustment)
        )
        # Ensure max_blur_size is always odd (required for Gaussian blur kernel)
        if self.session.max_blur_size % 2 == 0:
            self.session.max_blur_size += 1
            
        self.session.focal_thickness = max(
            BLUR_CONFIG["MIN_FOCAL_THICKNESS"], 
            min(BLUR_CONFIG["MAX_FOCAL_THICKNESS"], 
                self.session.focal_thickness + thickness_adjustment)
        )
        
        # Debug output for parameter changes
        print(f"DEBUG: Blur parameters changed:")
        print(f"  blur_strength: {old_blur_strength:.2f} -> {self.session.blur_strength:.2f}")
        print(f"  max_blur_size: {old_max_blur_size} -> {self.session.max_blur_size}")
        print(f"  focal_thickness: {old_focal_thickness:.3f} -> {self.session.focal_thickness:.3f}")
        
        # Now update the message with the specific values being processed
        if self.session.loading_image:
            debug_info = f"Blur: {self.session.blur_strength:.1f}, Size: {self.session.max_blur_size}, Focus: {self.session.focal_thickness:.3f}"
            await self.session.view_message.edit(
                content=f"{self.ctx.author.mention} {get_text('processing', self.session.language)} ({debug_info})",
                # No need to update the attachment as we're just changing the text
            )
        
        try:
            # Get normalized focal point from session
            norm_focal_x, norm_focal_y = self.session.focal_point
            
            # Apply blur with new settings - returns (blurred_image, depth_map)
            blurred_result, depth_map = await apply_focal_blur(
                self.session.original_image.copy(),
                norm_focal_x, norm_focal_y,  # Already normalized
                session=self.session,
                blur_strength=self.session.blur_strength,
                max_blur_size=self.session.max_blur_size,
                focal_thickness=self.session.focal_thickness
            )
            
            # Convert the blurred image to PIL format for saving
            blurred_image = Image.fromarray(cv2.cvtColor(blurred_result, cv2.COLOR_BGR2RGB))
            
            # Save the updated blurred image
            self.session.blurred_image = blurred_image
            
            # Save as temporary file with compression
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                blur_path = temp_file.name
                # Save with compression to reduce file size
                blurred_image.save(blur_path, format="PNG", optimize=True, compress_level=6)
            
            # Create file for Discord
            blur_file = discord.File(blur_path, filename="blurred_image_adjusted.png")
            
            # Add debug information to the UI
            debug_info = f"Blur: {self.session.blur_strength:.1f}, Size: {self.session.max_blur_size}, Focus: {self.session.focal_thickness:.3f}"
            
            # Edit the original message with the new blurred image
            await self.session.view_message.edit(
                content=f"{self.ctx.author.mention} {get_text('adjust_blur', self.session.language)} ({debug_info})",
                attachments=[blur_file],
                view=self  # Restore the view with buttons
            )
            
        except Exception as e:
            # Handle errors by editing the original message
            try:
                await self.session.view_message.edit(
                    content=f"❌ {str(e)}\n{get_text('processing', self.session.language)}",
                    attachments=[],
                    view=None
                )
            except Exception as edit_error:
                # Log the error but don't create a new message
                print(f"Error during blur adjustment and couldn't edit message: {str(e)}\nEdit error: {str(edit_error)}")

@bot.event
async def on_ready():
    """Event fired when the bot is ready."""
    print(f"{bot.user.name} is online!")
    print(f"Bot ID: {bot.user.id}")
    print("------")
    
    # Start the session cleanup task
    bot.loop.create_task(clean_expired_sessions())

async def download_image(url):
    """Download an image from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            return None

async def initialize_converter():
    """Initialize the depth estimation model."""
    global converter
    
    # Print GPU info
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"CUDA available: {device_count} device(s), using {device_name}")
    else:
        print("CUDA not available, using CPU")
    
    # Initialize the converter with appropriate settings
    converter = StereogramSBS3DConverter(
        use_advanced_infill=False,  # No need for infill with blur
        depth_model_type="depth_anything_v2",
        model_size="vitl",  
        max_resolution=1536,  # Reduced to avoid Discord's payload size limits
        low_memory_mode=False  # Use low memory processing
    )
    print("Depth model initialized")

def create_checkerboard_grid(image, max_blocks=5):
    """
    Divide an image into a checkerboard grid with a maximum of max_blocks on the longer edge.
    Label columns with letters and rows with numbers.
    
    Args:
        image: numpy array of the image
        max_blocks: maximum number of blocks on the longer edge, limited to 5 for Discord UI compatibility
    
    Returns:
        grid_image: PIL Image with the grid overlay
        grid_cells: dictionary mapping cell codes (e.g., 'A1') to their bounding boxes
    """
    height, width = image.shape[:2]
    
    # Override max_blocks to ensure we never exceed 5 columns (Discord's UI limit)
    max_blocks = min(max_blocks, 5)
    
    # Determine the number of blocks based on aspect ratio
    if width >= height:
        num_cols = max_blocks
        num_rows = max(3, int(num_cols * height / width))
    else:
        num_rows = max_blocks
        num_cols = max(3, int(num_rows * width / height))
    
    # Ensure num_cols never exceeds 5 for Discord button compatibility
    # and is at least 3 for better grid granularity
    num_cols = min(max(num_cols, 3), 5)
    
    # Also ensure we have at least 3 rows
    num_rows = max(num_rows, 3)
    
    # Calculate cell dimensions
    cell_width = width // num_cols
    cell_height = height // num_rows
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fallback to default if not available
    large_font_size = max(36, min(width, height) // 18)  # Increased font size for better visibility
    small_font_size = large_font_size // 1.5
    
    # Try to load bold fonts with fallbacks
    try:
        # Try to load bold fonts first
        font = ImageFont.truetype("/System/Library/Fonts/SFNSDisplay-Bold.otf", large_font_size)  # Arial Bold
        small_font = ImageFont.truetype("/System/Library/Fonts/SFNSDisplay-Bold.otf", small_font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arialbd.ttf", large_font_size)
            small_font = ImageFont.truetype("arialbd.ttf", small_font_size)
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", large_font_size)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", small_font_size)
            except IOError:
                try:
                    # Regular fonts if bold not available
                    font = ImageFont.truetype("arial.ttf", large_font_size)
                    small_font = ImageFont.truetype("arial.ttf", small_font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", large_font_size)
                        small_font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", small_font_size)
                    except IOError:
                        font = ImageFont.load_default()
                        small_font = font
    
    # Create a dictionary to store grid cells
    grid_cells = {}
    
    # Add a semi-transparent overlay for better grid visibility
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw the grid and labels
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate cell boundaries
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            # Store the cell boundaries
            cell_code = f"{chr(65 + col)}{row + 1}"  # A1, B1, etc.
            grid_cells[cell_code] = (x1, y1, x2, y2)
            
            # Draw semi-transparent cell background for better visibility
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 40))
            
            # Draw thicker cell boundaries
            overlay_draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 230), width=4)
            
            # Draw the cell code in the center
            # Check if font has getbbox method (newer PIL versions)
            if hasattr(font, "getbbox"):
                text_bbox = font.getbbox(cell_code)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            # Fallback for older PIL versions
            elif hasattr(font, "getsize"):
                text_width, text_height = font.getsize(cell_code)
            elif hasattr(draw, "textsize"):
                text_width, text_height = draw.textsize(cell_code, font=font)
            else:
                # Rough estimate if all else fails
                text_width, text_height = len(cell_code) * 10, 20
                
            text_x = x1 + (cell_width - text_width) // 2
            text_y = y1 + (cell_height - text_height) // 2
            
            # Draw text with shadow for better visibility
            # Calculate shadow thickness based on image size
            shadow_thickness = max(1, min(pil_image.width, pil_image.height) // 600)  # Scale shadow with image size
            shadow_count = max(2, min(pil_image.width, pil_image.height) // 400)  # Scale number of shadow points with image size
            
            shadow_positions = []
            for offset_x in range(-shadow_count, shadow_count + 1):
                for offset_y in range(-shadow_count, shadow_count + 1):
                    # Skip the center position (that will be the main text)
                    if abs(offset_x) + abs(offset_y) > 0 and abs(offset_x) + abs(offset_y) < shadow_count * 1.5:
                        # Scale the offset by shadow thickness
                        scaled_x = offset_x * shadow_thickness
                        scaled_y = offset_y * shadow_thickness
                        shadow_positions.append((scaled_x, scaled_y))
            
            # Draw thicker shadow
            for offset_x, offset_y in shadow_positions:
                overlay_draw.text(
                    (text_x + offset_x, text_y + offset_y), 
                    cell_code, 
                    fill=(0, 0, 0, 225), 
                    font=font
                )
            
            # Draw main text in bright color with higher contrast - multiple passes for thickness
            overlay_draw.text((text_x, text_y), cell_code, fill=(255, 255, 0, 255), font=font)
            # Additional passes with scaled offsets for boldness
            bold_thickness = max(1, min(pil_image.width, pil_image.height) // 1000)  # Scale bold effect with image size
            overlay_draw.text((text_x + bold_thickness, text_y), cell_code, fill=(255, 255, 0, 255), font=font)
            overlay_draw.text((text_x - bold_thickness, text_y), cell_code, fill=(255, 255, 0, 255), font=font)
            overlay_draw.text((text_x, text_y + bold_thickness), cell_code, fill=(255, 255, 0, 255), font=font)
            overlay_draw.text((text_x, text_y - bold_thickness), cell_code, fill=(255, 255, 0, 255), font=font)
    
    # Add column headers at the top
    for col in range(num_cols):
        col_letter = chr(65 + col)
        # Get text dimensions
        if hasattr(small_font, "getbbox"):
            bbox = small_font.getbbox(col_letter)
            text_width = bbox[2] - bbox[0]
        elif hasattr(small_font, "getsize"):
            text_width, _ = small_font.getsize(col_letter)
        else:
            text_width, _ = overlay_draw.textsize(col_letter, font=small_font)
            
        x_pos = col * cell_width + (cell_width - text_width) // 2
        
        # Calculate shadow thickness based on image size
        shadow_thickness = max(1, min(width, height) // 800)  # Scale shadow with image size
        shadow_count = max(2, min(width, height) // 500)  # Scale number of shadow points with image size
        
        # Draw black shadow for better readability
        shadow_positions = []
        for offset_x in range(-shadow_count, shadow_count + 1):
            for offset_y in range(-shadow_count, shadow_count + 1):
                # Skip the center position (that will be the main text)
                if abs(offset_x) + abs(offset_y) > 0 and abs(offset_x) + abs(offset_y) < shadow_count * 1.5:
                    # Scale the offset by shadow thickness
                    scaled_x = offset_x * shadow_thickness
                    scaled_y = offset_y * shadow_thickness
                    shadow_positions.append((scaled_x, scaled_y))
        
        # Draw thicker shadow
        for offset_x, offset_y in shadow_positions:
            overlay_draw.text((x_pos + offset_x, 5 + offset_y), col_letter, fill=(0, 0, 0, 200), font=small_font)
            
        # Draw main text with multiple passes for boldness
        overlay_draw.text((x_pos, 5), col_letter, fill=(255, 255, 255, 230), font=small_font)
        # Additional passes with scaled offsets for boldness
        bold_thickness = max(1, min(width, height) // 1200)  # Scale bold effect with image size
        overlay_draw.text((x_pos + bold_thickness, 5), col_letter, fill=(255, 255, 255, 230), font=small_font)
        overlay_draw.text((x_pos - bold_thickness, 5), col_letter, fill=(255, 255, 255, 230), font=small_font)
        overlay_draw.text((x_pos, 5 + bold_thickness), col_letter, fill=(255, 255, 255, 230), font=small_font)
        overlay_draw.text((x_pos, 5 - bold_thickness), col_letter, fill=(255, 255, 255, 230), font=small_font)
    
    # Add row headers on the left
    for row in range(num_rows):
        row_num = str(row + 1)
        # Get text dimensions
        if hasattr(small_font, "getbbox"):
            bbox = small_font.getbbox(row_num)
            text_width = bbox[2] - bbox[0]
        elif hasattr(small_font, "getsize"):
            text_width, _ = small_font.getsize(row_num)
        else:
            text_width, _ = overlay_draw.textsize(row_num, font=small_font)
            
        y_pos = row * cell_height + (cell_height - small_font_size) // 2
        
        # Calculate shadow thickness based on image size
        shadow_thickness = max(1, min(width, height) // 800)  # Scale shadow with image size
        shadow_count = max(2, min(width, height) // 500)  # Scale number of shadow points with image size
        
        # Draw black shadow for better readability
        shadow_positions = []
        for offset_x in range(-shadow_count, shadow_count + 1):
            for offset_y in range(-shadow_count, shadow_count + 1):
                # Skip the center position (that will be the main text)
                if abs(offset_x) + abs(offset_y) > 0 and abs(offset_x) + abs(offset_y) < shadow_count * 1.5:
                    # Scale the offset by shadow thickness
                    scaled_x = offset_x * shadow_thickness
                    scaled_y = offset_y * shadow_thickness
                    shadow_positions.append((scaled_x, scaled_y))
        
        # Draw thicker shadow
        for offset_x, offset_y in shadow_positions:
            overlay_draw.text((5 + offset_x, y_pos + offset_y), row_num, fill=(0, 0, 0, 200), font=small_font)
        # Additional passes with scaled offsets for boldness
        bold_thickness = max(1, min(width, height) // 1200)  # Scale bold effect with image size
        overlay_draw.text((5 + bold_thickness, y_pos), row_num, fill=(255, 255, 255, 230), font=small_font)
        overlay_draw.text((5 - bold_thickness, y_pos), row_num, fill=(255, 255, 255, 230), font=small_font)
        overlay_draw.text((5, y_pos + bold_thickness), row_num, fill=(255, 255, 255, 230), font=small_font)
        overlay_draw.text((5, y_pos - bold_thickness), row_num, fill=(255, 255, 255, 230), font=small_font)
    
    # Composite the overlay with the original image
    # Convert image to RGBA if it's not already
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
        
    result_image = Image.alpha_composite(pil_image, overlay)
    # Convert back to RGB for easier handling
    result_image = result_image.convert('RGB')
    
    return result_image, grid_cells

def create_sub_grid(image, cell_coords, cell_code):
    """
    Create a 3x3 numbered grid within the selected cell for more precise selection.
    
    Args:
        image: The original image (numpy array)
        cell_coords: The coordinates of the selected cell (x1, y1, x2, y2)
        cell_code: The selected cell code (e.g., "A1")
    
    Returns:
        sub_grid_image: PIL Image with the 3x3 grid overlay
        sub_grid_cells: Dictionary mapping numbers 1-9 to their coordinates
    """
    # Extract the selected cell region
    x1, y1, x2, y2 = cell_coords
    cell_width = x2 - x1
    cell_height = y2 - y1
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Create an overlay for the sub-grid
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Darken the areas outside the selected cell
    overlay_draw.rectangle([0, 0, pil_image.width, pil_image.height], fill=(0, 0, 0, 150))
    overlay_draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))  # Clear the selected cell area
    
    # Calculate the size of each sub-cell
    sub_cell_width = cell_width // 3
    sub_cell_height = cell_height // 3
    
    # Font setup - try to use bold fonts first
    font_size = max(32, min(sub_cell_width, sub_cell_height) // 3)
    try:
        # Try bold fonts first
        font = ImageFont.truetype("arialbd.ttf", font_size)  # Arial Bold
    except IOError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSDisplay-Bold.otf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except IOError:
                try:
                    # Regular fonts if bold not available
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()
    
    # Create a dictionary to store sub-grid cells (1-9)
    sub_grid_cells = {}
    
    # Numpad-style layout (7-8-9, 4-5-6, 1-2-3)
    numpad_layout = [
        [7, 8, 9],
        [4, 5, 6],
        [1, 2, 3]
    ]
    
    # Draw the 3x3 grid
    for row in range(3):
        for col in range(3):
            # Calculate sub-cell boundaries
            sub_x1 = x1 + col * sub_cell_width
            sub_y1 = y1 + row * sub_cell_height
            sub_x2 = sub_x1 + sub_cell_width
            sub_y2 = sub_y1 + sub_cell_height
            
            number = numpad_layout[row][col]
            sub_grid_cells[str(number)] = (sub_x1, sub_y1, sub_x2, sub_y2)
            
            # Draw sub-cell
            overlay_draw.rectangle([sub_x1, sub_y1, sub_x2, sub_y2], outline=(255, 255, 255, 255), width=3)
            
            # Center the number in the sub-cell
            if hasattr(font, "getbbox"):
                text_bbox = font.getbbox(str(number))
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            elif hasattr(font, "getsize"):
                text_width, text_height = font.getsize(str(number))
            elif hasattr(overlay_draw, "textsize"):
                text_width, text_height = overlay_draw.textsize(str(number), font=font)
            else:
                text_width, text_height = len(str(number)) * 10, 20
            
            text_x = sub_x1 + (sub_cell_width - text_width) // 2
            text_y = sub_y1 + (sub_cell_height - text_height) // 2
            
            # Draw text with shadow for better visibility
            # Calculate shadow thickness based on image size
            shadow_thickness = max(1, min(pil_image.width, pil_image.height) // 600)  # Scale shadow with image size
            shadow_count = max(2, min(pil_image.width, pil_image.height) // 400)  # Scale number of shadow points with image size
            
            shadow_positions = []
            for offset_x in range(-shadow_count, shadow_count + 1):
                for offset_y in range(-shadow_count, shadow_count + 1):
                    # Skip the center position (that will be the main text)
                    if abs(offset_x) + abs(offset_y) > 0 and abs(offset_x) + abs(offset_y) < shadow_count * 1.5:
                        # Scale the offset by shadow thickness
                        scaled_x = offset_x * shadow_thickness
                        scaled_y = offset_y * shadow_thickness
                        shadow_positions.append((scaled_x, scaled_y))
            
            # Draw thicker shadow
            for offset_x, offset_y in shadow_positions:
                overlay_draw.text(
                    (text_x + offset_x, text_y + offset_y), 
                    str(number), 
                    fill=(0, 0, 0, 230), 
                    font=font
                )
            
            # Draw main text with multiple passes for boldness
            overlay_draw.text((text_x, text_y), str(number), fill=(255, 255, 0, 255), font=font)
            # Additional passes with scaled offsets for boldness
            bold_thickness = max(1, min(pil_image.width, pil_image.height) // 1000)  # Scale bold effect with image size
            overlay_draw.text((text_x + bold_thickness, text_y), str(number), fill=(255, 255, 0, 255), font=font)
            overlay_draw.text((text_x - bold_thickness, text_y), str(number), fill=(255, 255, 0, 255), font=font)
            overlay_draw.text((text_x, text_y + bold_thickness), str(number), fill=(255, 255, 0, 255), font=font)
            overlay_draw.text((text_x, text_y - bold_thickness), str(number), fill=(255, 255, 0, 255), font=font)
    
    # Add title at the top of the image
    title_text = f"Selected cell {cell_code}: Choose a number (1-9) for precise focus"
    title_font_size = max(28, min(pil_image.width, pil_image.height) // 30)
    try:
        # Try to load bold fonts first
        title_font = ImageFont.truetype("arialbd.ttf", title_font_size)  # Arial Bold
    except IOError:
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/SFNSDisplay-Bold.otf", title_font_size)
        except IOError:
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_font_size)
            except IOError:
                title_font = font  # Fallback to the previously loaded font
    
    # Draw title with thicker shadow for better visibility
    title_y = 20
    
    # Calculate shadow thickness based on image size
    shadow_thickness = max(1, min(pil_image.width, pil_image.height) // 700)  # Scale shadow with image size
    shadow_count = max(2, min(pil_image.width, pil_image.height) // 450)  # Scale number of shadow points with image size
    
    # Create more shadow positions for bolder text
    shadow_positions = []
    for offset_x in range(-shadow_count, shadow_count + 1):
        for offset_y in range(-shadow_count, shadow_count + 1):
            # Skip the center position (that will be the main text)
            if abs(offset_x) + abs(offset_y) > 0 and abs(offset_x) + abs(offset_y) < shadow_count * 1.5:
                # Scale the offset by shadow thickness
                scaled_x = offset_x * shadow_thickness
                scaled_y = offset_y * shadow_thickness
                shadow_positions.append((scaled_x, scaled_y))
    
    # Draw thicker shadow
    for offset_x, offset_y in shadow_positions:
        overlay_draw.text(
            (pil_image.width // 2 - 200 + offset_x, title_y + offset_y),
            title_text,
            fill=(0, 0, 0, 230),
            font=title_font
        )
    
    # Draw the main text with multiple passes for boldness
    overlay_draw.text(
        (pil_image.width // 2 - 200, title_y),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font
    )
    # Additional passes with scaled offsets for boldness
    bold_thickness = max(1, min(pil_image.width, pil_image.height) // 1100)  # Scale bold effect with image size
    overlay_draw.text(
        (pil_image.width // 2 - 200 + bold_thickness, title_y),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font
    )
    overlay_draw.text(
        (pil_image.width // 2 - 200 - bold_thickness, title_y),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font
    )
    overlay_draw.text(
        (pil_image.width // 2 - 200, title_y + bold_thickness),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font
    )
    overlay_draw.text(
        (pil_image.width // 2 - 200, title_y - bold_thickness),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font
    )
    
    # Composite the overlay onto the image
    result_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    
    return result_image.convert('RGB'), sub_grid_cells

async def apply_focal_blur(image, focal_x, focal_y, session=None, blur_strength=None, max_blur_size=None, focal_thickness=None, region=None):
    """
    Apply depth-based blur to an image using a specified focal point.
    
    Args:
        image: The input image
        focal_x: X-coordinate of focal point (normalized 0-1)
        focal_y: Y-coordinate of focal point (normalized 0-1)
        session: BlurSession object containing blur parameters (optional)
        blur_strength: Strength of blur effect (uses session or default from BLUR_CONFIG if None)
        max_blur_size: Maximum blur kernel size (uses session or default from BLUR_CONFIG if None)
        focal_thickness: Thickness of the focal plane (uses session or default from BLUR_CONFIG if None)
        region: Region of interest for region-based depth sampling
        
    Returns:
        Tuple of (blurred image, depth map)
    """
    # Use session values if available, otherwise use provided values or defaults
    if session is not None:
        blur_strength = blur_strength if blur_strength is not None else session.blur_strength
        max_blur_size = max_blur_size if max_blur_size is not None else session.max_blur_size
        focal_thickness = focal_thickness if focal_thickness is not None else session.focal_thickness
        aperture_shape = session.aperture_shape
        highlight_boost = session.highlight_boost
        enable_dithering = session.enable_dithering
        chromatic_aberration = session.chromatic_aberration
        edge_smoothness = session.edge_smoothness
    else:
        # Use default values from config if no session and no values provided
        blur_strength = blur_strength if blur_strength is not None else BLUR_CONFIG["DEFAULT_BLUR_STRENGTH"]
        max_blur_size = max_blur_size if max_blur_size is not None else BLUR_CONFIG["DEFAULT_MAX_BLUR_SIZE"]
        focal_thickness = focal_thickness if focal_thickness is not None else BLUR_CONFIG["DEFAULT_FOCAL_THICKNESS"]
        aperture_shape = BLUR_CONFIG["DEFAULT_APERTURE_SHAPE"]
        highlight_boost = BLUR_CONFIG["DEFAULT_HIGHLIGHT_BOOST"]
        enable_dithering = BLUR_CONFIG["DEFAULT_ENABLE_DITHERING"]
        chromatic_aberration = BLUR_CONFIG["DEFAULT_CHROMATIC_ABERRATION"]
        edge_smoothness = BLUR_CONFIG["DEFAULT_EDGE_SMOOTHNESS"]
        
    # Initialize converter if it's not already initialized
    global converter
    if converter is None:
        await initialize_converter()
    
    # Generate depth map
    depth_map = converter.estimate_depth(image)
    
    # Normalize coordinates to image dimensions
    h, w = image.shape[:2]
    y_coord = int(focal_y * h)
    x_coord = int(focal_x * w)
    
    # Limit coordinates to valid range
    y_coord = max(0, min(y_coord, h - 1))
    x_coord = max(0, min(x_coord, w - 1))
    
    # Sample depth at focal point or region
    if region is not None:
        # Extract region coordinates
        x1 = max(0, int(region['x1'] * w))
        y1 = max(0, int(region['y1'] * h))
        x2 = min(w-1, int(region['x2'] * w))
        y2 = min(h-1, int(region['y2'] * h))
        
        # Extract the depth values in the region
        region_depths = depth_map[y1:y2, x1:x2]
        
        # Use median depth in the region as focal depth (more robust than average)
        focal_depth = np.median(region_depths)
    else:
        # Use single point if no region provided
        focal_depth = depth_map[y_coord, x_coord]
    
    # Debug output
    print(f"DEBUG: apply_focal_blur - focal_depth: {focal_depth:.4f}, focal_thickness: {focal_thickness:.4f}, blur_strength: {blur_strength:.2f}, max_blur_size: {max_blur_size}")
    
    # Use the core converter's apply_depth_based_blur method
    # This leverages the more sophisticated algorithm with proper depth of field
    result_bgr = converter.apply_depth_based_blur(
        image,                  # Original image
        depth_map,              # Depth map we just generated
        focal_distance=focal_depth,  # Use the sampled depth as the focal distance
        focal_thickness=focal_thickness,
        blur_strength=blur_strength,
        max_blur_size=max_blur_size,
        aperture_shape=aperture_shape,     # Using session or default aperture shape
        highlight_boost=highlight_boost,    # Using session or default highlight boost
        enable_dithering=enable_dithering,  # Using session or default dithering setting
        chromatic_aberration=chromatic_aberration,  # Using session or default chromatic aberration
        edge_smoothness=edge_smoothness     # Using session or default edge smoothness
    )
    
    # Return the result and depth map
    return result_bgr, depth_map

def create_loading_image(image, language="en"):
    """
    Create a heavily blurred version of the original image to use as a loading indicator.
    
    Args:
        image: The original image (numpy array)
        language: Language code to use for text (default: "en")
        
    Returns:
        PIL Image with heavy blur applied
    """
    # Create a very blurred version by applying a large Gaussian blur
    blurred = cv2.GaussianBlur(image, (61, 61), 30)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    
    # Add a "Processing..." text overlay
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fallback to default if not available
    font_size = max(36, min(pil_image.width, pil_image.height) // 15)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
    
    # Draw text with shadow for better visibility
    text = get_text("processing_wait", language)
    
    # Get text size
    if hasattr(font, "getbbox"):
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    elif hasattr(font, "getsize"):
        text_width, text_height = font.getsize(text)
    else:
        text_width, text_height = len(text) * 10, 20
        
    # Calculate position to center the text
    text_x = (pil_image.width - text_width) // 2
    text_y = (pil_image.height - text_height) // 2
    
    # Draw text with shadow
    shadow_offset = 3
    for offset_x, offset_y in [(-shadow_offset, -shadow_offset), (-shadow_offset, shadow_offset), 
                               (shadow_offset, -shadow_offset), (shadow_offset, shadow_offset)]:
        draw.text((text_x + offset_x, text_y + offset_y), text, fill=(0, 0, 0, 225), font=font)
    
    # Draw the main text
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    return pil_image

@bot.command(name=COMMAND_NAME)
async def blur_command(ctx):
    """Command to apply out-of-focus blur to an image."""
    # First, create a response message that we'll edit for all subsequent communications
    # This ensures we're always editing the same message instead of creating new ones
    response_msg = await ctx.reply(get_text("processing", "en"))
    
    # Check if the user already has an active session
    if ctx.author.id in user_sessions and not ctx.message.attachments:
        session = user_sessions[ctx.author.id]
        # Update the session expiry
        session.last_interaction = time.time()
        session.expiry_time = time.time() + BLUR_CONFIG["SESSION_TIMEOUT"]
        
        # Inform the user they already have an active session by editing our response
        await response_msg.edit(
            content=get_text("active_session", session.language)
        )
        return
    
    # For users without an existing session, default to their system language or English
    # We can get the user's locale from Discord (if available)
    user_language = "en"
    if hasattr(ctx, "locale") and ctx.locale:
        if ctx.locale.startswith("ja"):
            user_language = "ja"
    
    if not ctx.message.attachments:
        await response_msg.edit(content=get_text("attach_image", user_language))
        return
    
    # Check if the attached file is an image
    attachment = ctx.message.attachments[0]
    if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        await response_msg.edit(content=get_text("valid_image", user_language))
        return
    
    # Update the processing message
    await response_msg.edit(content=get_text("processing", user_language))
    
    try:
        # Download the image
        image_data = await download_image(attachment.url)
        if not image_data:
            await response_msg.edit(content=f"{ctx.author.mention} {get_text('download_failed', user_language)}")
            return
        
        # Convert to numpy array
        np_image = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # Create the checkerboard grid
        grid_image, grid_cells = create_checkerboard_grid(image, max_blocks=6)
        
        # Save the grid image to a BytesIO object with compression
        grid_bytes = io.BytesIO()
        grid_image.save(grid_bytes, format='PNG', optimize=True, compress_level=9)
        grid_bytes.seek(0)
        
        # Create a new session for the user
        user_sessions[ctx.author.id] = BlurSession(
            ctx.author.id,
            image.copy(),
            response_msg
        )
        session = user_sessions[ctx.author.id]
        session.grid_image = grid_image
        session.grid_cells = grid_cells
        session.language = user_language  # Set the initial language based on user's locale
        
        # Create a loading image and store it in the session
        loading_image = create_loading_image(image.copy(), user_language)
        session.loading_image = loading_image
        
        # Create the main grid view with buttons for all valid cells
        main_grid_view = MainGridView(session, ctx)
        
        # Edit the processing message with the grid and buttons instead of creating a new one
        await response_msg.edit(
            content=f"{ctx.author.mention} {get_text('select_grid_cell', session.language)}",
            attachments=[discord.File(fp=grid_bytes, filename="grid.png")],
            view=main_grid_view
        )
        
        # Store the updated processing message in the session
        session.view_message = response_msg
        
    except Exception as e:
        await response_msg.edit(content=f"{ctx.author.mention} An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

@bot.event
async def on_message(message):
    """Only process bot commands, all interactions now use buttons."""
    # Skip messages from the bot itself
    if message.author == bot.user:
        return
    
    # Process commands (like !blur) but don't handle text replies
    if message.content.startswith(BOT_PREFIX):
        await bot.process_commands(message)
        return
    
    # All other interactions are now handled through buttons only

def main():
    """Main function to start the bot."""
    if not TOKEN:
        print("Error: No Discord bot token found. Please set the DISCORD_BOT_TOKEN environment variable.")
        print("You can create a .env file with DISCORD_BOT_TOKEN=your_token_here")
        return 1
    
    # Start the bot
    bot.run(TOKEN)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 