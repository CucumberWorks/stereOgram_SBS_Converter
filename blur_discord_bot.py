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

# Import the converter from the core module for using the blur functionality
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter

# Load environment variables
load_dotenv()

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

class BlurSession:
    """Class to manage user session state for blur operations."""
    
    def __init__(self, user_id, original_image):
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
        self.blur_strength = 2.5  # Default blur strength
        self.max_blur_size = 31   # Default max blur size (kernel size)
        self.last_interaction = time.time()
        self.expiry_time = time.time() + 600  # 10 minutes expiry
        self.selection_message_id = None
        self.result_message_id = None
        self.selection_stage = 1   # 1: Main grid selection, 2: Sub-grid selection

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
    if converter is None:
        # Initialize with smaller model for faster processing and less memory usage
        converter = StereogramSBS3DConverter(
            use_advanced_infill=False,
            depth_model_type="depth_anything_v2",
            model_size="vitl",  # Using smallest model for Discord bot
            max_resolution=2048,  # Limit resolution to avoid memory issues
            low_memory_mode=True  # Use low memory processing
        )

def create_checkerboard_grid(image, max_blocks=6):
    """
    Divide an image into a checkerboard grid with a maximum of max_blocks on the longer edge.
    Label columns with letters and rows with numbers.
    
    Args:
        image: numpy array of the image
        max_blocks: maximum number of blocks on the longer edge
    
    Returns:
        grid_image: PIL Image with the grid overlay
        grid_cells: dictionary mapping cell codes (e.g., 'A1') to their bounding boxes
    """
    height, width = image.shape[:2]
    
    # Determine the number of blocks based on aspect ratio
    if width >= height:
        num_cols = max_blocks
        num_rows = max(1, int(num_cols * height / width))
    else:
        num_rows = max_blocks
        num_cols = max(1, int(num_rows * width / height))
    
    # Calculate cell dimensions
    cell_width = width // num_cols
    cell_height = height // num_rows
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fallback to default if not available
    large_font_size = max(32, min(width, height) // 20)  # Increased font size for better visibility
    small_font_size = large_font_size // 2
    
    # Try to load fonts with fallbacks
    try:
        font = ImageFont.truetype("arial.ttf", large_font_size)
        small_font = ImageFont.truetype("arial.ttf", small_font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", large_font_size)
            small_font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", small_font_size)
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", large_font_size)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", small_font_size)
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
            elif hasattr(draw, "textsize"):
                text_width, text_height = draw.textsize(cell_code, font=font)
            else:
                # Rough estimate if all else fails
                text_width, text_height = len(cell_code) * 10, 20
                
            text_x = x1 + (cell_width - text_width) // 2
            text_y = y1 + (cell_height - text_height) // 2
            
            # Draw text with better visibility
            # Draw black shadow/outline for better readability
            for offset_x, offset_y in [(-3, -3), (-3, 0), (-3, 3), (0, -3), (0, 3), (3, -3), (3, 0), (3, 3)]:
                overlay_draw.text((text_x + offset_x, text_y + offset_y), cell_code, fill=(0, 0, 0, 225), font=font)
            
            # Draw main text in bright color with higher contrast
            overlay_draw.text((text_x, text_y), cell_code, fill=(255, 255, 0, 255), font=font)
    
    # Add column headers at the top
    for col in range(num_cols):
        col_letter = chr(65 + col)
        # Get text dimensions
        if hasattr(small_font, "getbbox"):
            bbox = small_font.getbbox(col_letter)
            text_width = bbox[2] - bbox[0]
        else:
            text_width, _ = overlay_draw.textsize(col_letter, font=small_font)
            
        x_pos = col * cell_width + (cell_width - text_width) // 2
        # Draw black shadow for better readability
        overlay_draw.text((x_pos-1, 5-1), col_letter, fill=(0, 0, 0, 200), font=small_font)
        overlay_draw.text((x_pos+1, 5-1), col_letter, fill=(0, 0, 0, 200), font=small_font)
        overlay_draw.text((x_pos-1, 5+1), col_letter, fill=(0, 0, 0, 200), font=small_font)
        overlay_draw.text((x_pos+1, 5+1), col_letter, fill=(0, 0, 0, 200), font=small_font)
        # Draw main text
        overlay_draw.text((x_pos, 5), col_letter, fill=(255, 255, 255, 230), font=small_font)
    
    # Add row headers on the left
    for row in range(num_rows):
        row_num = str(row + 1)
        # Get text dimensions
        if hasattr(small_font, "getbbox"):
            bbox = small_font.getbbox(row_num)
            text_width = bbox[2] - bbox[0]
        else:
            text_width, _ = overlay_draw.textsize(row_num, font=small_font)
            
        y_pos = row * cell_height + (cell_height - small_font_size) // 2
        # Draw black shadow for better readability
        overlay_draw.text((5-1, y_pos-1), row_num, fill=(0, 0, 0, 200), font=small_font)
        overlay_draw.text((5+1, y_pos-1), row_num, fill=(0, 0, 0, 200), font=small_font)
        overlay_draw.text((5-1, y_pos+1), row_num, fill=(0, 0, 0, 200), font=small_font)
        overlay_draw.text((5+1, y_pos+1), row_num, fill=(0, 0, 0, 200), font=small_font)
        # Draw main text
        overlay_draw.text((5, y_pos), row_num, fill=(255, 255, 255, 230), font=small_font)
    
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
    
    # Font setup
    font_size = max(28, min(sub_cell_width, sub_cell_height) // 3)
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
            elif hasattr(overlay_draw, "textsize"):
                text_width, text_height = overlay_draw.textsize(str(number), font=font)
            else:
                text_width, text_height = len(str(number)) * 10, 20
            
            text_x = sub_x1 + (sub_cell_width - text_width) // 2
            text_y = sub_y1 + (sub_cell_height - text_height) // 2
            
            # Draw text with shadow for better visibility
            for offset_x, offset_y in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 0)]:
                overlay_draw.text(
                    (text_x + offset_x, text_y + offset_y), 
                    str(number), 
                    fill=(0, 0, 0, 230) if (offset_x, offset_y) != (0, 0) else (255, 255, 0, 255), 
                    font=font
                )
    
    # Add title at the top of the image
    title_text = f"Selected cell {cell_code}: Choose a number (1-9) for precise focus"
    title_font_size = max(24, min(pil_image.width, pil_image.height) // 30)
    try:
        title_font = ImageFont.truetype("arial.ttf", title_font_size)
    except IOError:
        title_font = font
    
    # Draw title with shadow
    title_y = 20
    for offset_x, offset_y in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        overlay_draw.text(
            (pil_image.width // 2 - 200 + offset_x, title_y + offset_y),
            title_text,
            fill=(0, 0, 0, 230),
            font=title_font
        )
    overlay_draw.text(
        (pil_image.width // 2 - 200, title_y),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font
    )
    
    # Composite the overlay onto the image
    result_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    
    return result_image.convert('RGB'), sub_grid_cells

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

@bot.command(name=COMMAND_NAME)
async def blur_command(ctx):
    """Command to apply out-of-focus blur to an image."""
    # Check if the user already has an active session
    if ctx.author.id in user_sessions and not ctx.message.attachments:
        session = user_sessions[ctx.author.id]
        # Update the session expiry
        session.last_interaction = time.time()
        session.expiry_time = time.time() + 600
        
        # Inform the user they already have an active session
        await ctx.send(
            f"{ctx.author.mention} You already have an active blur session. "
            f"Please respond with a grid cell code like **A1** or wait for it to expire."
        )
        return
    
    if not ctx.message.attachments:
        await ctx.send(f"{ctx.author.mention} Please attach an image to use this command!")
        return
    
    # Check if the attached file is an image
    attachment = ctx.message.attachments[0]
    if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        await ctx.send(f"{ctx.author.mention} Please attach a valid image file (PNG, JPG, JPEG, WEBP, BMP)!")
        return
    
    # If user has an existing session, replace it with a new one
    if ctx.author.id in user_sessions:
        await ctx.send(f"{ctx.author.mention} Starting a new blur session with your uploaded image...")
        end_user_session(ctx.author.id)
    
    # Send initial processing message
    processing_msg = await ctx.send(f"{ctx.author.mention} Processing your image... ⏳")
    
    try:
        # Download the image
        image_data = await download_image(attachment.url)
        if not image_data:
            await processing_msg.edit(content=f"{ctx.author.mention} Failed to download the image. Please try again!")
            return
        
        # Convert to numpy array
        np_image = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # Create the checkerboard grid
        grid_image, grid_cells = create_checkerboard_grid(image, max_blocks=6)
        
        # Save the grid image to a BytesIO object
        grid_bytes = io.BytesIO()
        grid_image.save(grid_bytes, format='PNG')
        grid_bytes.seek(0)
        
        # Create a new session for the user
        user_sessions[ctx.author.id] = BlurSession(
            ctx.author.id,
            image.copy()
        )
        session = user_sessions[ctx.author.id]
        session.grid_image = grid_image
        session.grid_cells = grid_cells
        
        # Create a more visible grid
        grid_file = discord.File(grid_bytes, filename='grid.png')
        await processing_msg.delete()
        
        # Send a more detailed instruction message with the grid image
        await ctx.send(
            f"{ctx.author.mention} Here's your image with a grid overlay.\n"
            f"**Instructions:**\n"
            f"1. Look at the grid and decide which area you want to keep in focus\n"
            f"2. Reply with a grid cell code (like **A1** or **B2**) to set that area as the focal point\n"
            f"3. Your selection will expire in 10 minutes if not used",
            file=grid_file
        )
        
    except Exception as e:
        await processing_msg.edit(content=f"{ctx.author.mention} An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

async def apply_focal_blur(image, focal_x, focal_y, blur_strength=3.5, max_blur_size=31, region=None):
    """
    Apply depth-based blur to an image using a specified focal point.
    
    Args:
        image: The input image
        focal_x: X-coordinate of focal point (normalized 0-1)
        focal_y: Y-coordinate of focal point (normalized 0-1)
        blur_strength: Strength of blur effect (default: 3.5)
        max_blur_size: Maximum blur kernel size (default: 31)
        region: Region of interest for region-based depth sampling
        
    Returns:
        Tuple of (blurred image, depth map)
    """
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
        
        # Don't modify the original image
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
    else:
        # Use single point if no region provided
        focal_depth = depth_map[y_coord, x_coord]
    
    # Parameters for depth-based blur
    focal_thickness = 0.15
    blur_strength = blur_strength  # Higher values = stronger blur
    
    # Convert to RGB for processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process as float
    rgb_float = rgb_image.astype(np.float32) / 255.0
    
    # Apply depth-based blur
    blurred = np.zeros_like(rgb_float)
    
    # Create depth mask (0 = in focus, 1 = out of focus)
    depth_diff = np.abs(depth_map - focal_depth)
    focus_mask = np.clip(depth_diff / focal_thickness, 0, 1)
    
    # Apply variable blur based on focus_mask
    for i in range(3):  # RGB channels
        channel = rgb_float[:,:,i]
        blurred_channel = channel.copy()
        
        # Apply blur with sizes based on out-of-focus amount
        for blur_size in range(3, max_blur_size + 1, 2):  # Odd numbers from 3 to max_blur_size
            # Scale the blur strength
            blur_weight = (blur_size - 1) / (max_blur_size - 1)
            
            # Areas that need this level of blur
            level_mask = (focus_mask >= blur_weight * 0.8) & (focus_mask <= blur_weight * 1.2)
            
            if np.any(level_mask):
                # Apply blur at this level
                temp_blurred = cv2.GaussianBlur(channel, (blur_size, blur_size), 0)
                blurred_channel[level_mask] = temp_blurred[level_mask]
        
        blurred[:,:,i] = blurred_channel
    
    # Linear interpolation between original and blurred based on focus_mask * blur_strength
    blend_weights = np.clip(focus_mask * blur_strength, 0, 1)
    blend_weights = blend_weights[:,:,np.newaxis]  # Add channel dimension
    result = (1 - blend_weights) * rgb_float + blend_weights * blurred
    
    # Convert back to uint8
    result_uint8 = (result * 255).astype(np.uint8)
    
    # Convert back to BGR for OpenCV
    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
    
    # Remove all markers - no visual feedback
    # marker_size = max(10, min(w, h) // 50)  # Adaptive marker size
    # line_thickness = max(2, marker_size // 5)
    
    # Draw crosshair
    # cv2.line(result_bgr, 
    #         (x_coord - marker_size, y_coord), 
    #         (x_coord + marker_size, y_coord), 
    #         (0, 255, 255), line_thickness)  # Horizontal yellow line
    # cv2.line(result_bgr, 
    #         (x_coord, y_coord - marker_size), 
    #         (x_coord, y_coord + marker_size), 
    #         (0, 255, 255), line_thickness)  # Vertical yellow line
    
    # Draw circle around crosshair
    # cv2.circle(result_bgr, (x_coord, y_coord), marker_size + 2, (0, 255, 255), line_thickness // 2)
    
    return result_bgr, depth_map

@bot.event
async def on_message(message):
    """Handle messages for selecting grid cells, sub-grid numbers, and blur adjustments."""
    # Skip messages from the bot itself
    if message.author == bot.user:
        return
    
    # Check if this is a command - if so, let the command handler process it
    if message.content.startswith(BOT_PREFIX):
        await bot.process_commands(message)
        return
    
    # REMOVED: No longer handling image attachments here to start sessions automatically
    # Now only processing images when they come with the !blur command
    
    # Handle blur strength adjustment replies
    if message.reference and message.reference.message_id:
        # Check if the message is a reply to a bot message
        try:
            # Get the referenced message
            referenced_message = await message.channel.fetch_message(message.reference.message_id)
            
            # Only process if it's a reply to one of our bot's messages
            if referenced_message.author == bot.user:
                content = message.content.strip().lower()
                
                # Check if user has an active session
                if message.author.id in user_sessions:
                    session = user_sessions[message.author.id]
                    
                    # Check if this is a reply to the result message
                    if session.result_message_id == referenced_message.id:
                        processing_msg = None
                        
                        # Handle blur strength/size adjustment with flexible input
                        if "+" in content or "-" in content:
                            # Count the number of + and - symbols
                            plus_count = content.count("+")
                            minus_count = content.count("-")
                            
                            # Determine adjustment amount based on symbol count
                            strength_adjustment = 0
                            size_adjustment = 0
                            
                            if plus_count > 0 and minus_count == 0:
                                # Increasing blur - adjust both strength and size
                                if plus_count <= 3:
                                    strength_adjustment = plus_count * 1.0  # 1.0 per + (doubled from 0.5)
                                    size_adjustment = plus_count * 4        # 4 per + (doubled from 2)
                                else:
                                    strength_adjustment = 3.0 + (plus_count - 3) * 0.5  # Diminishing returns past +++ (doubled)
                                    size_adjustment = 12 + (plus_count - 3) * 2         # Diminishing returns past +++ (doubled)
                            elif minus_count > 0 and plus_count == 0:
                                # Decreasing blur - adjust both strength and size
                                if minus_count <= 3:
                                    strength_adjustment = minus_count * -1.0  # -1.0 per - (doubled from -0.5)
                                    size_adjustment = minus_count * -4        # -4 per - (doubled from -2)
                                else:
                                    strength_adjustment = -3.0 + (minus_count - 3) * -0.5  # Diminishing returns past --- (doubled)
                                    size_adjustment = -12 + (minus_count - 3) * -2         # Diminishing returns past --- (doubled)
                            
                            # Send processing message
                            adjustment_sign = "+" if strength_adjustment > 0 else ""
                            processing_msg = await message.channel.send(
                                f"{message.author.mention} Adjusting blur ({adjustment_sign}{strength_adjustment:.1f})... ⏳"
                            )
                            
                            # Update blur parameters (ensure they stay in reasonable ranges)
                            session.blur_strength = max(0.5, min(10.0, session.blur_strength + strength_adjustment))
                            session.max_blur_size = max(11, min(63, int(session.max_blur_size + size_adjustment)))
                            # Ensure max_blur_size is odd
                            if session.max_blur_size % 2 == 0:
                                session.max_blur_size += 1
                            
                            try:
                                # Get the focal point coordinates
                                # The new format is (focal_point_str, focal_x, focal_y, region)
                                # Handle both old and new format for backward compatibility
                                if len(session.focal_point) >= 4:
                                    _, focal_x, focal_y, region = session.focal_point
                                else:
                                    _, focal_x, focal_y = session.focal_point
                                    region = None
                                
                                # Apply updated depth-based blur
                                blurred_image, _ = await apply_focal_blur(
                                    session.original_image,
                                    focal_x,
                                    focal_y,
                                    session.blur_strength,
                                    session.max_blur_size,
                                    region
                                )
                                
                                # Convert the blurred image to bytes for sending
                                blurred_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
                                with BytesIO() as image_binary:
                                    blurred_pil.save(image_binary, 'PNG')
                                    image_binary.seek(0)
                                    
                                    # Delete processing message
                                    if processing_msg:
                                        await processing_msg.delete()
                                    
                                    # Send the updated blurred image
                                    focal_point_str = session.focal_point[0]
                                    result_message = await message.channel.send(
                                        f"{message.author.mention} Here's your image with **blur strength {session.blur_strength:.1f}** and **size {session.max_blur_size}** focused at {focal_point_str}:\n"
                                        f"*Reply with multiple + or - symbols to adjust blur effect (e.g., +, ++, +++, -, --, ---, etc.)*",
                                        file=discord.File(fp=image_binary, filename=f"blurred_{focal_point_str}_s{session.blur_strength:.1f}_k{session.max_blur_size}.png")
                                    )
                                    
                                    # Update the result message ID
                                    session.result_message_id = result_message.id
                                    
                                    # Update session expiry time
                                    session.last_interaction = time.time()
                                    session.expiry_time = time.time() + 600
                            except Exception as e:
                                if processing_msg:
                                    await processing_msg.delete()
                                await message.channel.send(
                                    f"{message.author.mention} An error occurred while adjusting blur: {str(e)}"
                                )
                                import traceback
                                traceback.print_exc()
                        
                        return  # Skip further processing
        except (discord.NotFound, discord.HTTPException):
            # Message not found or other Discord API error, continue with normal processing
            pass
    
    # Check if the user has an active session
    if message.author.id in user_sessions:
        session = user_sessions[message.author.id]
        session.last_interaction = time.time()  # Update last interaction time
        content = message.content.strip().upper()
        
        # Step 1: Main grid selection
        if session.selection_stage == 1 and content in session.grid_cells:
            # Send processing message with user mention
            processing_msg = await message.channel.send(
                f"{message.author.mention} You selected cell {content}. Creating sub-grid for precise selection... ⏳"
            )
            
            try:
                # Store the selected cell
                session.selected_cell = content
                cell_coords = session.grid_cells[content]
                
                # Create the sub-grid
                sub_grid_image, sub_grid_cells = create_sub_grid(
                    session.original_image, 
                    cell_coords,
                    content
                )
                
                # Store the sub-grid information
                session.sub_grid_image = sub_grid_image
                session.sub_grid_cells = sub_grid_cells
                
                # Convert the image to bytes for sending
                with BytesIO() as image_binary:
                    sub_grid_image.save(image_binary, 'PNG')
                    image_binary.seek(0)
                    
                    # Send the sub-grid image
                    await processing_msg.delete()
                    await message.channel.send(
                        f"{message.author.mention} Cell {content} selected! Now choose a number (1-9) for precise focus:",
                        file=discord.File(fp=image_binary, filename=f"subgrid_{content}.png")
                    )
                
                # Update to the second selection stage
                session.selection_stage = 2
                
            except Exception as e:
                await message.channel.send(
                    f"{message.author.mention} An error occurred: {str(e)}"
                )
                import traceback
                traceback.print_exc()
                
                # If an error occurred, clean up the session
                end_user_session(message.author.id)
                
        # Step 2: Sub-grid selection
        elif session.selection_stage == 2 and content in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            processing_msg = await message.channel.send(
                f"{message.author.mention} You selected position {content} within cell {session.selected_cell}. Processing your image with depth-based blur... ⏳"
            )
            
            try:
                # Get the precise coordinates for the focal point
                sub_cell_coords = session.sub_grid_cells[content]
                x1, y1, x2, y2 = sub_cell_coords
                focal_x = (x1 + x2) // 2
                focal_y = (y1 + y2) // 2
                
                # Calculate normalized coordinates (0-1 range) for consistent handling
                h, w = session.original_image.shape[:2]
                norm_focal_x = focal_x / w
                norm_focal_y = focal_y / h
                
                # Also store the sub-cell region for region-based depth sampling
                norm_region = {
                    'x1': x1 / w,
                    'y1': y1 / h,
                    'x2': x2 / w,
                    'y2': y2 / h
                }
                
                # Store the final focal point
                focal_point = f"{session.selected_cell}-{content}"  # e.g., "A1-5"
                session.focal_point = (focal_point, norm_focal_x, norm_focal_y, norm_region)
                
                # Apply depth-based blur
                blurred_image, depth_map = await apply_focal_blur(
                    session.original_image,
                    norm_focal_x,
                    norm_focal_y,
                    session.blur_strength,
                    session.max_blur_size,
                    norm_region
                )
                
                # Convert the blurred image to bytes for sending
                blurred_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
                with BytesIO() as image_binary:
                    blurred_pil.save(image_binary, 'PNG')
                    image_binary.seek(0)
                    
                    # Send the blurred image
                    await processing_msg.delete()
                    result_message = await message.channel.send(
                        f"{message.author.mention} Here's your image with **blur strength {session.blur_strength:.1f}** and **size {session.max_blur_size}** focused at {focal_point}:\n"
                        f"*Reply with multiple + or - symbols to adjust blur effect (e.g., +, ++, +++, -, --, ---, etc.)*",
                        file=discord.File(fp=image_binary, filename=f"blurred_{focal_point}_s{session.blur_strength:.1f}_k{session.max_blur_size}.png")
                    )
                    
                    # Store the result message ID for later reference
                    session.result_message_id = result_message.id
                    
                    # Keep the session active for further blur adjustments
                    # Don't delete the session to allow for blur strength adjustments
                
            except Exception as e:
                await message.channel.send(
                    f"{message.author.mention} An error occurred: {str(e)}"
                )
                import traceback
                traceback.print_exc()
                
                # If an error occurred, clean up the session
                end_user_session(message.author.id)
        
        # Invalid input
        elif (session.selection_stage == 1 and content not in session.grid_cells) or \
             (session.selection_stage == 2 and content not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            # Don't show invalid selection messages for messages with image attachments
            # as these will be handled by the image upload code above
            if not message.attachments:
                if session.selection_stage == 1:
                    await message.channel.send(
                        f"{message.author.mention} Invalid cell selection. Please select a valid cell (e.g., A1, B2) from the grid."
                    )
                else:
                    await message.channel.send(
                        f"{message.author.mention} Invalid selection. Please choose a number from 1 to 9 for precise focus within cell {session.selected_cell}."
                    )
    
    # Process commands
    await bot.process_commands(message)

def main():
    # Check if token is available
    if not TOKEN:
        print("Error: No Discord bot token found. Please set DISCORD_BOT_TOKEN in your .env file.")
        return
    
    try:
        # Start the bot
        bot.run(TOKEN)
    except Exception as e:
        print(f"Error starting the bot: {e}")

if __name__ == "__main__":
    main() 