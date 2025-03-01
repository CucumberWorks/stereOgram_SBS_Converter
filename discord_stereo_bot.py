import discord
from discord.ext import commands
import os
import io
import aiohttp
import asyncio
import tempfile
import cv2
import numpy as np
from PIL import Image
import gc
import time
import imageio
from dotenv import load_dotenv

# Add the current directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the converter from the core module
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter

# Load environment variables
load_dotenv()

# Configuration
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Load token from .env file
BOT_PREFIX = "!"
COMMAND_NAME = "sbs"

# Initialize the bot with intents
intents = discord.Intents.default()
intents.message_content = True  # Required to see message content
bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

# We'll initialize the converter only when needed to save memory
converter = None

@bot.event
async def on_ready():
    """Event fired when the bot is ready."""
    print(f"{bot.user.name} is online!")
    print(f"Bot ID: {bot.user.id}")
    print("------")

async def download_image(url):
    """Download an image from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            return None

async def initialize_converter():
    """Initialize the SBS converter only when needed."""
    global converter
    if converter is None:
        converter = StereogramSBS3DConverter(
            use_advanced_infill=True,
            depth_model_type="depth_anything_v2",
            model_size="vits",  # Using vits model as requested (smaller but faster)
            max_resolution=4096,  # Limit resolution to avoid Discord payload issues
            low_memory_mode=True  # Use full quality processing
        )
        # Configure high quality color processing with enhanced anti-banding
        converter.set_color_quality(
            high_color_quality=True,
            apply_dithering=True,
            dithering_level=1.5  # Stronger dithering to combat the banding issues
        )
        
        # Set inpainting parameters matching Gradio's high quality defaults
        converter.set_inpainting_params(
            steps=20,
            guidance_scale=7.5,
            patch_size=128,
            patch_overlap=32,
            high_quality=True
        )
    return converter

def create_wiggle_gif(left_view, right_view, output_path, duration=0.15):
    """Create a wiggle GIF alternating between left and right views to show 3D effect"""
    # Make sure both images have the same dimensions
    h_left, w_left = left_view.shape[:2]
    h_right, w_right = right_view.shape[:2]
    
    if h_left != h_right or w_left != w_right:
        # Resize right view to match left view
        right_view = cv2.resize(right_view, (w_left, h_left), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert from BGR to RGB for GIF
    left_view_rgb = cv2.cvtColor(left_view, cv2.COLOR_BGR2RGB)
    right_view_rgb = cv2.cvtColor(right_view, cv2.COLOR_BGR2RGB)
    
    # Calculate a reduced size to avoid Discord's payload limits
    # Discord has an 8MB file size limit for most servers
    max_width = 1080
    if w_left > max_width:
        # Calculate scale factor to reduce size
        scale = max_width / w_left
        new_width = int(w_left * scale)
        new_height = int(h_left * scale)
        
        # Resize both frames
        left_view_rgb = cv2.resize(left_view_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        right_view_rgb = cv2.resize(right_view_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        print(f"Resized GIF frames from {w_left}x{h_left} to {new_width}x{new_height} to avoid payload limits")
    
    # Create GIF with alternating frames
    frames = [left_view_rgb, right_view_rgb]
    
    # Use lower quality settings for GIF to reduce file size
    imageio.mimsave(output_path, frames, duration=duration, loop=0, quantizer='nq', palettesize=128)
    
    return output_path

def clear_converter():
    """Clear the converter from memory."""
    global converter
    if converter is not None:
        try:
            # Try to free VRAM if method exists
            if hasattr(converter, 'clear_vram_cache'):
                converter.clear_vram_cache()
            
            # Delete the converter instance
            del converter
            converter = None
            
            # Force garbage collection
            gc.collect()
            
            # If using PyTorch, empty CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("Cleared converter from memory")
        except Exception as e:
            print(f"Error clearing converter: {e}")

@bot.command(name=COMMAND_NAME)
async def convert_to_sbs(ctx):
    """Convert an attached image to SBS 3D format."""
    # Check if there's an attachment
    if not ctx.message.attachments:
        # Check if the message is a reply to a message with an attachment
        if ctx.message.reference and ctx.message.reference.resolved:
            original_msg = ctx.message.reference.resolved
            if original_msg.attachments:
                attachment = original_msg.attachments[0]
            else:
                await ctx.reply("No image found! Please attach an image or reply to a message with an image.")
                return
        else:
            await ctx.reply("No image found! Please attach an image or reply to a message with an image.")
            return
    else:
        attachment = ctx.message.attachments[0]
    
    # Check if the attachment is an image
    if not attachment.content_type or not attachment.content_type.startswith('image'):
        await ctx.reply("The attachment is not an image!")
        return
    
    # Send processing message
    processing_msg = await ctx.reply("🔄 Processing your image into SBS 3D format... This may take a minute.")
    
    try:
        # Initialize converter (only when needed)
        conv = await initialize_converter()
        
        # Download the image
        image_bytes = await download_image(attachment.url)
        if not image_bytes:
            await processing_msg.edit(content="Failed to download the image.")
            return
        
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image using the exact same pipeline as the Gradio interface
        
        # Make sure we're working with BGR (OpenCV format)
        if img.shape[2] == 4:  # If RGBA, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        await processing_msg.edit(content="🔄 Analyzing image and generating depth map...")
        
        # Get original dimensions and aspect ratio
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        # If image is very large, resize it to avoid processor and memory issues
        max_dimension = 2048  # Maximum dimension for depth processing
        if h > max_dimension or w > max_dimension:
            if h > w:
                new_h = max_dimension
                new_w = int(max_dimension * aspect_ratio)
            else:
                new_w = max_dimension
                new_h = int(max_dimension / aspect_ratio)
            
            # Make dimensions divisible by 8 for stable diffusion compatibility
            new_w = new_w - (new_w % 8)
            new_h = new_h - (new_h % 8)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized input image from {w}x{h} to {new_w}x{new_h} to avoid processing issues")
            
            # Update dimensions
            h, w = img.shape[:2]
            aspect_ratio = w / h
        
        # Convert to float32 early for better precision (matching Gradio)
        img_float = img.astype(np.float32)
        
        # Generate depth map (use original image for best results)
        depth_map = conv.estimate_depth(img)
        
        # Colorize depth map for visualization
        depth_colored = conv.visualize_depth(depth_map)
        
        # Create temporary files with appropriate extension
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as gif_file:
            gif_path = gif_file.name
            
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as depth_file:
            depth_path = depth_file.name
            
        # Determine processing resolution (as in Gradio interface)
        resolution = 1080  # Target height as in Gradio
        high_quality = True  # Use high quality processing
        
        # Calculate processing resolution based on Gradio's logic
        if resolution == 720:
            process_res_factor = 480 if not high_quality else 600
        elif resolution == 1080:
            process_res_factor = 720 if not high_quality else 900
        elif resolution == 1440:
            process_res_factor = 800 if not high_quality else 1000
        elif resolution == 2160:
            process_res_factor = 960 if not high_quality else 1200
        else:
            process_res_factor = 720 if not high_quality else 900  # Default
            
        await processing_msg.edit(content="🔄 Preparing image for 3D conversion...")
            
        # Calculate processing dimensions
        proc_h = process_res_factor
        proc_w = int(proc_h * aspect_ratio)
        # Make divisible by 8 (for stable diffusion inpainting)
        proc_w = proc_w - (proc_w % 8)
        proc_h = proc_h - (proc_h % 8)
        
        # Resize image for processing
        proc_img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Resize depth map to match processing resolution
        depth_map_resized = cv2.resize(depth_map, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        
        await processing_msg.edit(content="🔄 Generating stereo views from depth map...")
        
        # Using the same stereo views generation logic as Gradio
        shift_factor = 0.03  # Default from Gradio interface
        left_view, right_view, left_holes, right_holes = conv.generate_stereo_views(
            proc_img, depth_map_resized, shift_factor=shift_factor
        )
        
        # Fill holes in the stereo views using exactly the same method as Gradio
        if conv.use_advanced_infill:
            await processing_msg.edit(content="🔄 Filling in missing areas with AI inpainting...")
            if np.sum(left_holes) > 0:
                try:
                    left_view = conv.fill_holes_preserving_originals(
                        proc_img,  # Original image
                        left_view, 
                        left_holes,
                        depth_map_resized,
                        shift_factor=shift_factor,
                        is_left_view=True
                    )
                except Exception as e:
                    print(f"Error using preserving method for left view: {e}, falling back to advanced infill")
                    left_view = conv.advanced_infill(left_view, left_holes, efficient_mode=True)
            
            if np.sum(right_holes) > 0:
                try:
                    right_view = conv.fill_holes_preserving_originals(
                        proc_img,  # Original image
                        right_view, 
                        right_holes,
                        depth_map_resized,
                        shift_factor=shift_factor,
                        is_left_view=False
                    )
                except Exception as e:
                    print(f"Error using preserving method for right view: {e}, falling back to advanced infill")
                    right_view = conv.advanced_infill(right_view, right_holes, efficient_mode=True)
        
        # Apply the same enhanced anti-banding as in Gradio
        await processing_msg.edit(content="🔄 Applying anti-banding treatment...")
        try:
            if hasattr(conv, '_enhanced_anti_banding') and conv.high_color_quality:
                left_view = conv._enhanced_anti_banding(left_view)
                right_view = conv._enhanced_anti_banding(right_view)
            else:
                left_view = conv._enhance_image_quality(left_view)
                right_view = conv._enhance_image_quality(right_view)
        except Exception as e:
            print(f"Error in anti-banding: {e}")
            # Continue without anti-banding if it fails
        
        # Determine output resolution, ensuring reasonable size for Discord
        max_height = 1080  # Maximum height to avoid Discord payload issues
        target_h = min(resolution, max_height)
        target_w = int(target_h * aspect_ratio)
        
        # Make width divisible by 2 for even dimensions
        target_w = target_w - (target_w % 2)
        
        await processing_msg.edit(content="🔄 Creating final SBS 3D image and GIF...")
        
        # Resize to output resolution
        left_view_resize = cv2.resize(left_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        right_view_resize = cv2.resize(right_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Generate side-by-side 3D output - use target height for both sides
        # Use the Gradio SBS generation logic
        h_left, w_left = left_view_resize.shape[:2]
        sbs_width = target_w * 2  # Full width of SBS image
        sbs_3d = np.hstack((left_view_resize, right_view_resize))
        
        # Check final image sizes to avoid Discord's payload limit (8MB for most servers)
        sbs_size_estimate = sbs_3d.nbytes / (1024 * 1024)  # Size in MB
        if sbs_size_estimate > 7.5:  # Leave some margin for overhead
            # Need to reduce the size
            scale_factor = min(1.0, np.sqrt(7.5 / sbs_size_estimate))
            new_width = int(sbs_width * scale_factor)
            new_height = int(target_h * scale_factor)
            
            # Resize the SBS image
            sbs_3d = cv2.resize(sbs_3d, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized SBS output from {sbs_width}x{target_h} to {new_width}x{new_height} to avoid payload limits")
            
            # Also resize the depth map
            depth_colored = cv2.resize(depth_colored, (new_width//2, new_height), interpolation=cv2.INTER_AREA)
        
        # Save the outputs with high quality
        # Save depth visualization
        cv2.imwrite(depth_path, depth_colored, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # Save SBS image as high quality PNG (if small enough) or JPEG
        if sbs_size_estimate < 4:  # Use PNG for smaller images
            sbs_png_path = tmp_path
            cv2.imwrite(sbs_png_path, sbs_3d, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            tmp_path = sbs_png_path
            sbs_extension = '.png'
        else:  # Use JPEG for larger images with high quality
            sbs_jpg_path = tmp_path.replace('.png', '.jpg')
            cv2.imwrite(sbs_jpg_path, sbs_3d, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tmp_path = sbs_jpg_path
            sbs_extension = '.jpg'
        
        # Create wiggle GIF animation
        create_wiggle_gif(left_view_resize, right_view_resize, gif_path, duration=0.2)
        
        # Check if any files are too large (Discord limit is 8MB)
        max_filesize = 8 * 1024 * 1024  # 8MB in bytes
        files_to_send = []
        
        # Check SBS file size and add to list if it's within limits
        sbs_filesize = os.path.getsize(tmp_path)
        if sbs_filesize <= max_filesize:
            with open(tmp_path, 'rb') as f:
                sbs_file = discord.File(f, filename=f"sbs_{attachment.filename.split('.')[0]}{sbs_extension}")
                files_to_send.append(sbs_file)
        else:
            # Try more aggressive compression if still too large
            higher_compression_path = tmp_path.replace(sbs_extension, '_compressed.jpg')
            cv2.imwrite(higher_compression_path, sbs_3d, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if os.path.getsize(higher_compression_path) <= max_filesize:
                with open(higher_compression_path, 'rb') as f:
                    sbs_file = discord.File(f, filename=f"sbs_{attachment.filename.split('.')[0]}.jpg")
                    files_to_send.append(sbs_file)
            else:
                print(f"SBS file too large even after compression: {sbs_filesize/1024/1024:.2f}MB")
        
        # Check GIF file size
        gif_filesize = os.path.getsize(gif_path)
        if gif_filesize <= max_filesize:
            with open(gif_path, 'rb') as f:
                gif_file = discord.File(f, filename=f"wiggle_{attachment.filename.split('.')[0]}.gif")
                files_to_send.append(gif_file)
        else:
            print(f"GIF file too large: {gif_filesize/1024/1024:.2f}MB")
            
        # Check depth file size
        depth_filesize = os.path.getsize(depth_path)
        if depth_filesize <= max_filesize:
            with open(depth_path, 'rb') as f:
                depth_file = discord.File(f, filename=f"depth_{attachment.filename.split('.')[0]}{os.path.splitext(depth_path)[1]}")
                files_to_send.append(depth_file)
        else:
            print(f"Depth file too large: {depth_filesize/1024/1024:.2f}MB")
        
        if files_to_send:
            # Split files into SBS image and extras
            sbs_files = [f for f in files_to_send if f.filename.startswith("sbs_")]
            extra_files = [f for f in files_to_send if not f.filename.startswith("sbs_")]
            
            # Send SBS image first
            if sbs_files:
                await ctx.reply("✅ Here's your SBS 3D image:", files=sbs_files)
                
                # Send extras in a separate message if any exist
                if extra_files:
                    await ctx.send("And here are the extras (depth map and wiggle GIF):", files=extra_files)
            else:
                # If no SBS file, send all together
                await ctx.reply("✅ Here are your 3D conversion results:", files=files_to_send)
        else:
            await ctx.reply("❌ Could not create files within Discord's size limits. Try with a smaller image.")
        
        # Clean up the temporary files
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(gif_path):
                os.remove(gif_path)
            if os.path.exists(depth_path):
                os.remove(depth_path)
            # Clean up any additional files created
            if os.path.exists(tmp_path.replace(sbs_extension, '_compressed.jpg')):
                os.remove(tmp_path.replace(sbs_extension, '_compressed.jpg'))
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
        
        # Delete the processing message
        await processing_msg.delete()
        
    except Exception as e:
        await processing_msg.edit(content=f"❌ Error processing image: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clear the converter to free memory
        clear_converter()

@bot.event
async def on_message(message):
    """Event handler for all messages to process commands."""
    # Don't respond to our own messages
    if message.author == bot.user:
        return
    
    # Process commands
    await bot.process_commands(message)

# Main function to run the bot
def main():
    # Check if token is available
    if not TOKEN:
        print("Error: Discord bot token not found in environment variables.")
        print("Please set DISCORD_BOT_TOKEN in your .env file.")
        return 1
    
    try:
        # Run the Discord bot
        bot.run(TOKEN)
    except Exception as e:
        print(f"Error running the bot: {e}")
        if "Improper token" in str(e):
            print("\nYour Discord token appears to be invalid. Please check:")
            print("1. The token in your .env file is correct")
            print("2. The token hasn't been regenerated in the Discord Developer Portal")
            print("3. The bot has been added to a server")
        elif "Privileged intent" in str(e):
            print("\nYou need to enable Message Content Intent in your Discord Developer Portal:")
            print("1. Go to https://discord.com/developers/applications")
            print("2. Select your application")
            print("3. Go to Bot section")
            print("4. Enable 'Message Content Intent' under 'Privileged Gateway Intents'")
        return 1
    finally:
        # Make sure to clean up resources
        clear_converter()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 