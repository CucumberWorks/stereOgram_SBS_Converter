import os
import gradio as gr
import cv2
import numpy as np
import torch
import time
from PIL import Image
import imageio
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter
from core.advanced_infill import AdvancedInfillTechniques

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Initialize the converter with default settings
converter = None

def initialize_converter(depth_model_type, model_size, use_advanced_infill, max_resolution, low_memory_mode):
    global converter
    try:
        converter = StereogramSBS3DConverter(
            use_advanced_infill=use_advanced_infill,
            depth_model_type=depth_model_type,
            model_size=model_size,
            max_resolution=int(max_resolution),
            low_memory_mode=low_memory_mode
        )
        
        # Report GPU info if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # VRAM in GB
            return f"Using GPU: {gpu_name} with {vram:.1f} GB VRAM\nInitialized with {model_size} model"
        else:
            return "No GPU detected - using CPU mode"
    except Exception as e:
        message = f"Error initializing converter: {str(e)}\n"
        message += "The tool will still attempt to function with limited capabilities."
        print(message)
        return message

def generate_3d_image(left, right):
    """Generate red-cyan anaglyph image from left and right views"""
    # Make sure both images have the same height
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    if h_left != h_right:
        # Resize the smaller image to match the height of the larger one
        if h_left < h_right:
            aspect_ratio = w_left / h_left
            new_height = h_right
            new_width = int(new_height * aspect_ratio)
            left = cv2.resize(left, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            aspect_ratio = w_right / h_right
            new_height = h_left
            new_width = int(new_height * aspect_ratio)
            right = cv2.resize(right, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Get dimensions of resized images
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    # Create a blank output image
    output_width = max(w_left, w_right)
    output = np.zeros((h_left, output_width, 3), dtype=np.uint8)
    
    # Convert images to BGR for processing if they aren't already
    if len(left.shape) == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if len(right.shape) == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    
    # Create red-cyan anaglyph exactly as test_with_demo.py does
    # OpenCV uses BGR order
    output[:, :, 0] = right[:, :, 0]  # Blue channel from right eye
    output[:, :, 1] = right[:, :, 1]  # Green channel from right eye
    output[:, :, 2] = left[:, :, 2]   # Red channel from left eye
    
    return output

def generate_sbs_3d(left, right):
    """Generate side-by-side 3D image from left and right views"""
    # Make sure both images have the same height
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    if h_left != h_right:
        # Resize the smaller image to match the height of the larger one
        if h_left < h_right:
            aspect_ratio = w_left / h_left
            new_height = h_right
            new_width = int(new_height * aspect_ratio)
            left = cv2.resize(left, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            aspect_ratio = w_right / h_right
            new_height = h_left
            new_width = int(new_height * aspect_ratio)
            right = cv2.resize(right, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Get dimensions of resized images
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    # Create a blank output image (side-by-side)
    output_width = w_left + w_right
    output = np.zeros((h_left, output_width, 3), dtype=np.uint8)
    
    # Convert images to BGR if needed (for consistency with OpenCV)
    if len(left.shape) == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if len(right.shape) == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    
    # Place left and right views side by side
    output[:, 0:w_left] = left
    output[:, w_left:w_left+w_right] = right
    
    return output

def create_wiggle_gif(left_view, right_view, duration=0.15):
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
    
    # Create temporary file path
    os.makedirs("results", exist_ok=True)
    timestamp = int(time.time())
    gif_path = f"results/wiggle_{timestamp}.gif"
    
    # Create GIF with alternating frames
    frames = [left_view_rgb, right_view_rgb]
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)
    
    return gif_path

def process_image(image, shift_factor, resolution, patch_size, patch_overlap, steps, cfg_scale, high_quality, debug_mode):
    global converter
    
    if converter is None:
        return None, None, None, None, None, "Converter not initialized. Please initialize first."
    
    try:
        # Convert to numpy array (OpenCV format)
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
            
        # Make sure we're working with BGR (OpenCV format)
        if img.shape[2] == 4:  # If RGBA, convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:  # If RGB (from Gradio), convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        # Determine processing resolution
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
        
        # Generate progress message
        progress_message = f"Processing image ({w}x{h}) with shift factor {shift_factor:.2f}..."
        
        # Use original image for depth estimation (no resizing)
        depth_img = img.copy()
        
        # Generate depth map
        start_time = time.time()
        progress_message += f"\nGenerating depth map..."
        depth_map = converter.estimate_depth(depth_img)
        progress_message += f"\nDepth map generated in {time.time() - start_time:.2f} seconds"
        
        # Colorize depth map for visualization
        depth_colored = converter.visualize_depth(depth_map)
        
        # Determine processing resolution
        proc_h = process_res_factor
        proc_w = int(proc_h * aspect_ratio)
        # Make divisible by 8
        proc_w = proc_w - (proc_w % 8)
        proc_h = proc_h - (proc_h % 8)
        
        # Resize image for processing
        proc_img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Resize depth map to match processing resolution
        depth_map_resized = cv2.resize(depth_map, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        
        # Set inpainting parameters
        converter.set_inpainting_params(
            steps=steps,
            guidance_scale=cfg_scale,
            patch_size=patch_size,
            patch_overlap=patch_overlap
        )
        
        # Set high_quality directly if possible
        try:
            converter.high_quality = high_quality
        except:
            pass
        
        # Generate stereo views
        progress_message += f"\nGenerating stereo views with {int(shift_factor * proc_w)}px shift..."
        start_time = time.time()
        
        if debug_mode:
            # Use debug visualization mode with purple background
            left_view, right_view, left_holes, right_holes = converter.generate_stereo_views_debug(
                proc_img, depth_map_resized, shift_factor=shift_factor
            )
        else:
            # Use normal mode
            left_view, right_view, left_holes, right_holes = converter.generate_stereo_views(
                proc_img, depth_map_resized, shift_factor=shift_factor
            )
        
        progress_message += f"\nStereo views generated in {time.time() - start_time:.2f} seconds"
        
        # Fill holes in the stereo views
        if converter.use_advanced_infill:
            # Apply advanced inpainting
            progress_message += f"\nFilling holes using advanced inpainting..."
            start_time = time.time()
            
            if np.sum(left_holes) > 0:
                progress_message += f"\nFilling left view holes ({np.sum(left_holes)} pixels)..."
                left_inpainted = converter.fill_holes_preserving_originals(
                    proc_img,  # Original image
                    left_view, 
                    left_holes,
                    depth_map_resized,
                    shift_factor=shift_factor,
                    is_left_view=True
                )
                left_view = left_inpainted
            
            if np.sum(right_holes) > 0:
                progress_message += f"\nFilling right view holes ({np.sum(right_holes)} pixels)..."
                right_inpainted = converter.fill_holes_preserving_originals(
                    proc_img,  # Original image
                    right_view, 
                    right_holes,
                    depth_map_resized,
                    shift_factor=shift_factor,
                    is_left_view=False
                )
                right_view = right_inpainted
                
            progress_message += f"\nHole filling completed in {time.time() - start_time:.2f} seconds"
        
        # Determine output resolution
        target_h = resolution
        target_w = int(target_h * aspect_ratio)
        
        # Make width divisible by 2 for even dimensions
        target_w = target_w - (target_w % 2)
        
        # Resize to output resolution
        left_view_resize = cv2.resize(left_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        right_view_resize = cv2.resize(right_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Generate anaglyph output
        anaglyph = generate_3d_image(left_view_resize, right_view_resize)
        
        # Generate side-by-side 3D output
        sbs_3d = generate_sbs_3d(left_view_resize, right_view_resize)
        
        # Create wiggle GIF
        wiggle_gif_path = create_wiggle_gif(left_view_resize, right_view_resize)
        
        progress_message += f"\nProcessing complete. Final output resolution: {target_w}x{target_h}"
        
        # Convert OpenCV images (BGR) to PIL for Gradio display (RGB)
        depth_colored_pil = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        anaglyph_pil = Image.fromarray(cv2.cvtColor(anaglyph, cv2.COLOR_BGR2RGB))
        sbs_3d_pil = Image.fromarray(cv2.cvtColor(sbs_3d, cv2.COLOR_BGR2RGB))
        
        # Save images to results directory
        timestamp = int(time.time())
        os.makedirs("results", exist_ok=True)
        
        # Save outputs with timestamp - using same format as test_with_demo.py
        depth_path = f"results/depth_{timestamp}.jpg"
        anaglyph_path = f"results/anaglyph_{timestamp}.jpg"
        sbs_path = f"results/sbs_{timestamp}.jpg"
        
        # Save files using cv2 to maintain correct colors in BGR format
        # This matches how test_with_demo.py saves files
        cv2.imwrite(depth_path, depth_colored)
        cv2.imwrite(anaglyph_path, anaglyph)
        cv2.imwrite(sbs_path, sbs_3d)
        
        progress_message += f"\nImages saved to results directory with timestamp {timestamp}."
        
        return depth_colored_pil, anaglyph_pil, sbs_3d_pil, wiggle_gif_path, [depth_path, anaglyph_path, sbs_path, wiggle_gif_path], progress_message
        
    except Exception as e:
        error_message = f"Error processing image: {str(e)}\nPlease try a different image or reinitialize the converter."
        print(error_message)
        return None, None, None, None, None, error_message

def build_interface():
    with gr.Blocks(title="stereOgram SBS3D converter") as interface:
        gr.Markdown("# stereOgram SBS3D converter")
        gr.Markdown("Convert regular 2D images into stereo 3D formats using depth estimation")
        
        with gr.Tab("Initialize"):
            with gr.Row():
                with gr.Column():
                    depth_model = gr.Dropdown(
                        choices=["depth_anything_v2"], 
                        value="depth_anything_v2", 
                        label="Depth Model"
                    )
                    model_size = gr.Dropdown(
                        choices=["vits", "vitb", "vitl"], 
                        value="vitb", 
                        label="Model Size", 
                        info="vits = smallest/fastest, vitb = medium/balanced, vitl = largest/best quality"
                    )
                    use_advanced = gr.Checkbox(
                        value=True, 
                        label="Use Advanced Inpainting", 
                        info="Higher quality but more GPU memory"
                    )
                    max_res = gr.Slider(
                        minimum=1024, 
                        maximum=8192, 
                        value=8192, 
                        step=1024, 
                        label="Max Resolution", 
                        info="Maximum resolution for depth estimation"
                    )
                    low_memory = gr.Checkbox(
                        value=True, 
                        label="Low Memory Mode", 
                        info="For GPUs with limited VRAM"
                    )
                    init_btn = gr.Button("Initialize Converter")
                
                with gr.Column():
                    init_output = gr.Textbox(label="Initialization Status")
                    clear_cache_btn = gr.Button("Clear VRAM Cache")
        
        with gr.Tab("Convert"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Accordion("Basic Settings", open=True):
                        resolution = gr.Dropdown(
                            choices=[720, 1080, 1440, 2160], 
                            value=1080, 
                            label="Output Resolution", 
                            info="Output resolution height (pixels)"
                        )
                        shift_factor = gr.Slider(
                            minimum=0.01, 
                            maximum=0.2, 
                            value=0.03, 
                            step=0.005, 
                            label="Shift Factor", 
                            info="Stereo shift amount (0.01-0.2)"
                        )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        patch_size = gr.Slider(
                            minimum=64, 
                            maximum=512, 
                            value=384, 
                            step=32, 
                            label="Patch Size", 
                            info="Size of patches for inpainting"
                        )
                        patch_overlap = gr.Slider(
                            minimum=16, 
                            maximum=256, 
                            value=128, 
                            step=16, 
                            label="Patch Overlap", 
                            info="Overlap between patches"
                        )
                        steps = gr.Slider(
                            minimum=10, 
                            maximum=50, 
                            value=30, 
                            step=5, 
                            label="Inference Steps", 
                            info="More steps = better quality, slower"
                        )
                        cfg_scale = gr.Slider(
                            minimum=1.0, 
                            maximum=15.0, 
                            value=7.5, 
                            step=0.5, 
                            label="Guidance Scale", 
                            info="How closely to follow the prompt"
                        )
                        high_quality = gr.Checkbox(
                            value=False, 
                            label="High Quality Mode", 
                            info="Better quality but uses more VRAM and processing time"
                        )
                        debug_mode = gr.Checkbox(
                            value=False, 
                            label="Debug Mode", 
                            info="Show purple background to visualize holes"
                        )
                    
                    with gr.Row():
                        convert_btn = gr.Button("Convert to 3D", variant="primary")
                        clear_btn = gr.Button("Clear Images")
                    progress = gr.Textbox(label="Progress")
                
            with gr.Row():
                with gr.Column():
                    depth_map = gr.Image(label="Depth Map")
                    
                with gr.Column():
                    anaglyph = gr.Image(label="Red-Cyan Anaglyph 3D")
                    
                with gr.Column():
                    sbs = gr.Image(label="Side-by-Side 3D")
            
            with gr.Row():
                wiggle_gif = gr.Image(label="Wiggle 3D (No Glasses Required)", show_download_button=True)
            
            download_files = gr.File(label="Download Results", file_count="multiple", visible=True)
                
        # Define function to clear VRAM cache
        def clear_vram_cache():
            global converter
            if converter is not None:
                result = converter.clear_vram_cache()
                return result
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                return "VRAM cache cleared (converter not initialized)"
            else:
                return "No GPU detected"
        
        # Define function to clear images
        def clear_images():
            return None, None, None, None, None, "Images cleared"
            
        # Connect buttons to functions
        init_btn.click(
            initialize_converter, 
            inputs=[depth_model, model_size, use_advanced, max_res, low_memory], 
            outputs=[init_output]
        )
        
        clear_cache_btn.click(
            clear_vram_cache,
            inputs=[],
            outputs=[init_output]
        )
        
        convert_btn.click(
            process_image, 
            inputs=[
                input_image, shift_factor, resolution, patch_size, 
                patch_overlap, steps, cfg_scale, high_quality, debug_mode
            ], 
            outputs=[depth_map, anaglyph, sbs, wiggle_gif, download_files, progress]
        )
        
        clear_btn.click(
            clear_images,
            inputs=[],
            outputs=[depth_map, anaglyph, sbs, wiggle_gif, download_files, progress]
        )
        
        # Set up examples
        examples_dir = os.path.join(os.getcwd(), "demo_images")
        if os.path.exists(examples_dir):
            example_files = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=[input_image],
                    outputs=[depth_map, anaglyph, sbs, wiggle_gif, download_files, progress],
                    fn=lambda x: process_image(x, 0.03, 1080, 384, 128, 30, 7.5, False, False)
                )
    
    return interface

# Launch the Gradio app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=True) 