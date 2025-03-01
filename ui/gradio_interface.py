import os
import gradio as gr
from gradio import EventData, SelectData
import cv2
import numpy as np
import torch
import time
from PIL import Image
import imageio
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter
from core.advanced_infill import AdvancedInfillTechniques
import gc
import multiprocessing as mp
from functools import partial
import numba
import importlib
import subprocess
import sys

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Initialize the converter with default settings
converter = None
stored_depth_map = None  # Will store the depth map after first generation
stored_depth_colored = None  # Will store the colored visualization
cached_depth_gray = None  # Will store the grayscale depth for faster focal plane updates
cached_depth_height = None  # Will store the dimensions for faster processing
cached_depth_width = None
base_depth_image = None  # Store the base color depth image without overlays
original_input_image = None  # Store the original input image for overlay

# Create global variables to store cached visualization elements
cached_diagonal_pattern = None
cached_legend_template = None

# Anime face detector variables
anime_face_detector_loaded = False
anime_face_cascade = None
anime_detector_model_path = os.path.join("models", "lbpcascade_animeface.xml")

# Visual indicators tracking
last_clicked_point = None  # Store the last clicked point (x, y)
last_detected_faces = None  # Store the last detected faces with format (x, y, w, h, depth_val)
last_face_center = None  # Store the average face center point
selected_face_idx = None  # Store which face was selected as primary focus

# Default converter settings
default_depth_model = "depth_anything_v2"
default_model_size = "vitb"
default_use_advanced = False
default_max_res = 8192
default_low_memory = True

# Auto-initialize converter
converter = None

# Check if anime face detector is available
def check_anime_face_detector():
    return os.path.exists(anime_detector_model_path)

# Install anime face detector if needed
def install_anime_face_detector():
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        if not os.path.exists(anime_detector_model_path):
            print(f"Downloading anime face detector model to {anime_detector_model_path}...")
            import urllib.request
            import ssl
            
            # Create a context that doesn't verify certificates for macOS
            # This is needed because macOS Python often has issues with SSL certificates
            context = ssl._create_unverified_context()
            
            try:
                # Download the lbpcascade_animeface.xml from nagadomi's GitHub repo
                model_url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
                urllib.request.urlretrieve(model_url, anime_detector_model_path, context=context)
                print(f"Downloaded anime face detector model to {anime_detector_model_path}")
            except Exception as url_error:
                print(f"Error with urllib.request: {url_error}")
                # Fallback to subprocess curl which works better on macOS
                print("Trying with curl instead...")
                subprocess.check_call(["curl", "-L", "-o", anime_detector_model_path, model_url])
                print(f"Downloaded anime face detector model to {anime_detector_model_path} using curl")
                
            return "Anime face detector downloaded successfully. Ready to use!"
        else:
            return "Anime face detector model already exists. Ready to use!"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error installing anime face detector: {error_trace}")
        return f"Error downloading anime face detector: {str(e)}\nTry downloading manually from https://github.com/nagadomi/lbpcascade_animeface"

# Initialize anime face detector
def init_anime_face_detector():
    global anime_face_cascade, anime_face_detector_loaded
    try:
        # Check if model exists
        if not os.path.exists(anime_detector_model_path):
            print("Anime face detector model not found. Installing automatically...")
            install_result = install_anime_face_detector()
            print(install_result)
            
            # Check again if installation was successful
            if not os.path.exists(anime_detector_model_path):
                return "Failed to install anime face detector model. Please try manual installation."
            
        print(f"Loading anime face detector from {anime_detector_model_path}")
        
        # Initialize the cascade classifier
        anime_face_cascade = cv2.CascadeClassifier(anime_detector_model_path)
        
        # Check if cascade loaded successfully
        if anime_face_cascade.empty():
            return "Failed to load anime face detector. The XML file may be corrupted or invalid."
        
        anime_face_detector_loaded = True
        print("Anime face detector loaded successfully")
        return "Anime face detector initialized successfully! Ready to use."
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error initializing anime face detector: {error_trace}")
        return f"Error initializing anime face detector: {str(e)}"

# Auto-load anime face detector at module import time
try:
    if not anime_face_detector_loaded:
        print("Auto-initializing anime face detector on startup...")
        init_result = init_anime_face_detector()
        print(init_result)
except Exception as e:
    print(f"Error during auto-initialization of anime face detector: {e}")

# Auto-initialize converter at module import time
try:
    print("Auto-initializing converter with default settings...")
    converter_init_result = initialize_converter(
        default_depth_model, 
        default_model_size, 
        default_use_advanced, 
        default_max_res, 
        default_low_memory
    )
    print(converter_init_result)
except Exception as e:
    print(f"Error during auto-initialization of converter: {e}")

# Detect anime faces using the cascade classifier
def detect_anime_faces(image):
    """Detect anime faces in the image using lbpcascade_animeface"""
    global anime_face_cascade, anime_face_detector_loaded
    
    # Initialize detector if not already done
    if not anime_face_detector_loaded or anime_face_cascade is None:
        print("Auto-initializing anime face detector...")
        result = init_anime_face_detector()
        print(result)
        if not anime_face_detector_loaded:
            print("Failed to initialize anime face detector")
            return []
    
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Make sure image is in the right format (RGB/BGR and dtype)
    if image.dtype != np.uint8:
        print(f"Converting image from {image.dtype} to uint8")
        image = (image * 255).astype(np.uint8)
        
    # Convert to BGR if in RGB format (Gradio uses RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if the image is RGB and convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    try:
        print(f"Running anime face detection on image with shape {img_bgr.shape}")
        
        # Detect faces using the anime cascade classifier with adjusted parameters
        faces = anime_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Detected {len(faces)} anime faces")
        
        # If no faces found, try with more lenient parameters
        if len(faces) == 0:
            faces = anime_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"Second attempt: Detected {len(faces)} anime faces")
        
        # Log detected faces
        for (x, y, w, h) in faces:
            print(f"  Face at ({x}, {y}) with size {w}x{h}")
        
        return faces
    except Exception as e:
        print(f"Error in anime face detection: {e}")
        import traceback
        traceback.print_exc()
        return []

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
        elif torch.backends.mps.is_available():
            message = f"Using Apple MPS (Metal Performance Shaders) with {model_size} model"
            if use_advanced_infill:
                message += "\n\nNote: If you encounter errors with advanced inpainting on Apple Silicon,"
                message += "\nconsider disabling advanced inpainting, or edit core/stereogram_sbs3d_converter.py"
                message += "\nto uncomment the line: self.use_advanced_infill = False"
            return message
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

def generate_sbs_3d(left_view, right_view):
    """Generate side-by-side 3D image from left and right views"""
    # Get dimensions
    h, w = left_view.shape[:2]
    
    # Create side-by-side image
    sbs_3d = np.zeros((h, w * 2, 3), dtype=np.uint8)
    sbs_3d[:, :w, :] = left_view
    sbs_3d[:, w:, :] = right_view
    
    return sbs_3d

def generate_depth_map(image):
    """Generate depth map once and store it"""
    global converter, stored_depth_map, stored_depth_colored, cached_depth_gray, cached_depth_height, cached_depth_width, base_depth_image, original_input_image
    
    if converter is None:
        return None, "Converter not initialized. Please initialize first."
    
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
        
        # Store original input image for visualization overlay
        original_input_image = img.copy()
        
        # Generate depth map
        start_time = time.time()
        stored_depth_map = converter.estimate_depth(img)
        stored_depth_colored = converter.visualize_depth(stored_depth_map)
        base_depth_image = stored_depth_colored.copy()  # Store a copy of the original colored depth map
        processing_time = time.time() - start_time
        
        # Pre-calculate grayscale depth for faster focal plane visualization
        if stored_depth_map.max() > 1.0:
            normalized_depth = stored_depth_map / stored_depth_map.max()
        else:
            normalized_depth = stored_depth_map.copy()
            
        cached_depth_gray = (normalized_depth * 255).astype(np.uint8)
        cached_depth_height, cached_depth_width = cached_depth_gray.shape[:2]
        
        # Create an initial visualization with the focal plane at default values
        focal_viz, _ = update_focal_plane_visualization(0.5, 0.1)
        
        # Return the visualization with the focal plane overlay
        return focal_viz, f"Depth map generated in {processing_time:.2f} seconds"
    
    except Exception as e:
        error_message = f"Error generating depth map: {str(e)}"
        print(error_message)
        return None, error_message

def create_diagonal_pattern(height, width, line_spacing=10, line_thickness=1, color=(255, 255, 255)):
    """Create a diagonal line pattern image for out-of-focus areas"""
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw diagonal lines
    for i in range(-height, width + height, line_spacing):
        cv2.line(pattern, (i, 0), (i + height, height), color, line_thickness)
    
    return pattern

# Numba-accelerated function for ultra-fast color mapping
@numba.njit(parallel=True, fastmath=True, cache=True)
def apply_color_mapping_numba(depth_array, lower_bound_val, upper_bound_val, focal_distance_val):
    """JIT-compiled function for ultra-fast color mapping"""
    h, w = depth_array.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in numba.prange(h):
        for j in range(w):
            depth_val = depth_array[i, j]
            
            # In-focus area (white)
            if lower_bound_val <= depth_val <= upper_bound_val:
                result[i, j, 0] = 255  # B
                result[i, j, 1] = 255  # G
                result[i, j, 2] = 255  # R
                
                # Exact focal point (yellow)
                if abs(depth_val - focal_distance_val) <= 1:
                    result[i, j, 0] = 0    # B
                    result[i, j, 1] = 255  # G
                    result[i, j, 2] = 255  # R
            
            # Near field (red)
            elif depth_val < lower_bound_val:
                intensity = 255 - (depth_val * 255 // lower_bound_val) if lower_bound_val > 0 else 255
                result[i, j, 2] = intensity  # R
            
            # Far field (blue)
            else:  # depth_val > upper_bound_val
                max_depth = 255
                intensity = ((depth_val - upper_bound_val) * 255 // (max_depth - upper_bound_val)) if max_depth > upper_bound_val else 255
                result[i, j, 0] = intensity  # B
                
    return result

def process_chunk(chunk_data):
    """Process a chunk of the depth map in parallel"""
    chunk, start_row, lower_bound_val, upper_bound_val, focal_distance_val = chunk_data
    return apply_color_mapping_numba(chunk, lower_bound_val, upper_bound_val, focal_distance_val), start_row

def update_focal_plane_visualization(first_arg=0.5, second_arg=0.1, depth_map=None):
    """Depth-aware visualization with color overlay on original image:
    - Far areas (depth=0) are blue tinted
    - Near areas (depth=1) are red tinted
    - In-focus areas retain original image colors
    """
    global stored_depth_map, cached_depth_gray, base_depth_image, original_input_image
    global last_clicked_point, last_detected_faces, last_face_center, selected_face_idx
    
    # Check if first_arg is a depth map (numpy array)
    if isinstance(first_arg, np.ndarray):
        depth_gray = first_arg
        focal_distance = second_arg
        focal_thickness = depth_map if depth_map is not None else 0.1
    else:
        # Normal usage
        focal_distance = first_arg
        focal_thickness = second_arg
        depth_gray = depth_map if depth_map is not None else cached_depth_gray
    
    if depth_gray is None and (stored_depth_map is None or cached_depth_gray is None):
        return None, "Please generate depth map first"
    
    try:
        start_time = time.time()
        
        # Calculate focal plane boundaries
        half_thickness = focal_thickness / 2.0
        lower_bound = max(0, focal_distance - half_thickness)
        upper_bound = min(1.0, focal_distance + half_thickness)
        
        # Convert to 0-255 range for direct comparison
        lower_bound_val = int(lower_bound * 255)
        upper_bound_val = int(upper_bound * 255)
        
        # Check if we need to create a new LUT or can reuse cached one
        if not hasattr(update_focal_plane_visualization, 'last_params') or \
           update_focal_plane_visualization.last_params != (lower_bound_val, upper_bound_val):
            
            # Create a 256×3 overlay LUT (for B,G,R format)
            # This will be used to determine which areas get which tint
            lut = np.zeros((256, 3), dtype=np.uint8)
            
            # REVERSED DEPTH INTERPRETATION: 0=far (blue), 1=near (red)
            
            # Blue for far field (depth values closer to 0)
            if lower_bound_val > 0:
                # Create a blue tint that fades as it approaches the in-focus zone
                blue_mask = np.zeros((lower_bound_val, 3), dtype=np.uint8)
                blue_mask[:, 0] = np.linspace(180, 60, lower_bound_val, dtype=np.uint8)  # Blue channel
                lut[:lower_bound_val] = blue_mask
            
            # No tint for in-focus area (keep original image colors)
            lut[lower_bound_val:upper_bound_val+1] = [0, 0, 0]  # No tint
            
            # Red for near field (depth values closer to 1)
            if upper_bound_val < 255:
                # Create a red tint that intensifies as it gets further from the in-focus zone
                red_range = 255 - upper_bound_val
                if red_range > 0:
                    red_mask = np.zeros((red_range, 3), dtype=np.uint8)
                    red_mask[:, 2] = np.linspace(60, 180, red_range, dtype=np.uint8)  # Red channel
                    lut[upper_bound_val+1:] = red_mask
            
            # Cache the LUT for reuse
            update_focal_plane_visualization.cached_lut = lut
            update_focal_plane_visualization.last_params = (lower_bound_val, upper_bound_val)
        else:
            # Reuse the cached LUT
            lut = update_focal_plane_visualization.cached_lut
        
        # Use the original input image as the base for visualization
        if original_input_image is not None:
            # Resize original image to match depth map dimensions if needed
            if (original_input_image.shape[0] != depth_gray.shape[0] or
                original_input_image.shape[1] != depth_gray.shape[1]):
                base_image = cv2.resize(original_input_image, 
                                     (depth_gray.shape[1], depth_gray.shape[0]), 
                                     interpolation=cv2.INTER_LANCZOS4)
            else:
                base_image = original_input_image.copy()
        else:
            # Fallback to depth visualization if original image is not available
            if base_depth_image is None:
                base_image = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
            else:
                base_image = base_depth_image.copy()
        
        # Apply the color overlay
        # Get the tint mask for each pixel from the LUT
        tint_mask = lut[depth_gray]
        
        # Create a mask for areas that should be tinted (non-zero in the tint_mask)
        areas_to_tint = np.any(tint_mask > 0, axis=2)
        
        # Apply the tint by blending the base image with the tint
        result = base_image.copy()
        
        # Apply tint only to areas outside the focal plane
        for c in range(3):  # For each color channel
            # Add the tint to the original image in the corresponding areas
            result[:,:,c] = np.where(areas_to_tint, 
                                     np.clip(base_image[:,:,c] * 0.7 + tint_mask[:,:,c], 0, 255).astype(np.uint8), 
                                     base_image[:,:,c])
        
        # Re-add visual indicators if they exist
        if last_clicked_point is not None or last_detected_faces is not None:
            # Get dimensions for size calculations
            height, width = depth_gray.shape[:2]
            
            # Re-draw clicked point if it exists
            if last_clicked_point is not None:
                x_img, y_img = last_clicked_point
                # Calculate size based on image dimensions
                marker_size_ratio = 0.015  # 1.5% of the smaller dimension
                circle_size = max(10, int(min(width, height) * marker_size_ratio))
                line_thickness = max(2, int(min(width, height) * 0.003))  # 0.3% of smaller dimension
                
                # Draw red dot
                cv2.circle(result, (x_img, y_img), circle_size, (0, 0, 255), -1)  # Filled red circle
                cv2.circle(result, (x_img, y_img), circle_size, (255, 255, 255), line_thickness)  # White outline
            
            # Re-draw face detection boxes if they exist
            if last_detected_faces is not None:
                # Calculate line thickness based on image dimensions
                line_thickness = max(2, int(min(width, height) * 0.003))  # 0.3% of smaller dimension
                
                # Draw rectangles for each detected face - highlight selected face in magenta, others in grey
                for idx, (face_x, face_y, face_w, face_h, _) in enumerate(last_detected_faces):
                    # If this is the selected face or there's only one face, use magenta
                    if selected_face_idx == idx or selected_face_idx is None or len(last_detected_faces) == 1:
                        rect_color = (255, 0, 255)  # Magenta for selected/primary face
                    else:
                        rect_color = (128, 128, 128)  # Grey for non-selected faces
                    
                    # Draw a hollow rectangle with thick border around the whole face
                    cv2.rectangle(result, 
                                 (face_x, face_y),
                                 (face_x + face_w, face_y + face_h),
                                 rect_color, line_thickness * 2)  # Border twice as thick
                
                # Yellow diamond removed - no need to display the average focal point anymore
        
        # Convert to RGB for Gradio display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        processing_time = time.time() - start_time
        return result_rgb, f"Focal plane updated in {processing_time*1000:.1f} ms"
    
    except Exception as e:
        error_message = f"Error updating focal plane: {str(e)}"
        print(error_message)
        return None, error_message

def generate_depth_with_focal_plane(image, focal_distance=0.5):
    """Generate depth map and visualize the focal plane"""
    global stored_depth_map
    
    # If we don't have a stored depth map or this is a new image, generate it
    if stored_depth_map is None:
        depth_img, status = generate_depth_map(image)
        if depth_img is None:
            return None, status
            
    # Update the focal plane visualization
    return update_focal_plane_visualization(focal_distance)

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

def process_image(image, shift_factor, resolution, patch_size, patch_overlap, steps, cfg_scale, high_quality, debug_mode,
                 apply_depth_blur=False, focal_distance=0.5, focal_thickness=0.1, blur_intensity=0.2, blur_radius=0.4):
    global converter, stored_depth_map, stored_depth_colored
    
    if converter is None:
        return None, None, None, None, None, "Converter not initialized. Please initialize first."
    
    if stored_depth_map is None:
        return None, None, None, None, None, "Please generate depth map first"
    
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
        
        # Use the stored depth map - don't regenerate
        progress_message += f"\nUsing pre-generated depth map..."
        depth_map = stored_depth_map
        
        # Use current visualization for depth map display
        current_viz, _ = update_focal_plane_visualization(focal_distance, focal_thickness)
        # Convert from RGB back to BGR for processing
        depth_colored = cv2.cvtColor(current_viz, cv2.COLOR_RGB2BGR)
        
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
        
        # Create 2D image with depth blur
        depth_blurred_2d = None
        if apply_depth_blur:
            # Convert normalized blur parameters to actual values
            # Blur intensity: 0-1 to 0.1-5.0
            blur_strength = 0.1 + (blur_intensity * 4.9)
            
            # Blur radius: 0-1 to 3-51 (odd numbers only)
            max_blur_size = int(3 + (blur_radius * 48))
            if max_blur_size % 2 == 0:  # Ensure odd number
                max_blur_size = max_blur_size + 1
            
            progress_message += f"\nApplying depth-based blur (focal distance: {focal_distance:.2f}, thickness: {focal_thickness:.2f}, intensity: {blur_intensity:.2f}, radius: {blur_radius:.2f})..."
            start_time = time.time()
            
            # Apply blur to the original 2D image based on depth map
            depth_blurred_2d = converter.apply_depth_based_blur(
                proc_img.copy(), depth_map_resized,
                focal_distance=focal_distance,
                focal_thickness=focal_thickness,
                blur_strength=blur_strength,
                max_blur_size=max_blur_size
            )
            
            # Apply blur to both views based on the depth map for the stereo images
            left_view = converter.apply_depth_based_blur(
                left_view, depth_map_resized,
                focal_distance=focal_distance,
                focal_thickness=focal_thickness,
                blur_strength=blur_strength, 
                max_blur_size=max_blur_size
            )
            
            right_view = converter.apply_depth_based_blur(
                right_view, depth_map_resized,
                focal_distance=focal_distance,
                focal_thickness=focal_thickness,
                blur_strength=blur_strength,
                max_blur_size=max_blur_size
            )
            
            progress_message += f"\nDepth-based blur applied in {time.time() - start_time:.2f} seconds"
        else:
            # If depth blur is not applied, use the original image as the output
            depth_blurred_2d = proc_img.copy()
        
        # Determine output resolution
        target_h = resolution
        target_w = int(target_h * aspect_ratio)
        
        # Make width divisible by 2 for even dimensions
        target_w = target_w - (target_w % 2)
        
        # Resize to output resolution
        left_view_resize = cv2.resize(left_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        right_view_resize = cv2.resize(right_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        depth_blurred_2d_resize = cv2.resize(depth_blurred_2d, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Generate anaglyph output
        anaglyph = generate_3d_image(left_view_resize, right_view_resize)
        
        # Generate side-by-side 3D output
        sbs_3d = generate_sbs_3d(left_view_resize, right_view_resize)
        
        # Create wiggle GIF
        wiggle_gif_path = create_wiggle_gif(left_view_resize, right_view_resize)
        
        progress_message += f"\nProcessing complete. Final output resolution: {target_w}x{target_h}"
        
        # Convert OpenCV images (BGR) to PIL for Gradio display (RGB)
        # Only take the depth image part without the legend
        if cached_depth_height is not None:
            depth_colored_top = depth_colored[:cached_depth_height]
            depth_colored_pil = Image.fromarray(cv2.cvtColor(depth_colored_top, cv2.COLOR_BGR2RGB))
        else:
            depth_colored_pil = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        
        depth_blurred_2d_pil = Image.fromarray(cv2.cvtColor(depth_blurred_2d_resize, cv2.COLOR_BGR2RGB))
        anaglyph_pil = Image.fromarray(cv2.cvtColor(anaglyph, cv2.COLOR_BGR2RGB))
        sbs_3d_pil = Image.fromarray(cv2.cvtColor(sbs_3d, cv2.COLOR_BGR2RGB))
        
        # Save images to results directory
        timestamp = int(time.time())
        os.makedirs("results", exist_ok=True)
        
        # Save outputs with timestamp - using same format as test_with_demo.py
        depth_path = f"results/depth_{timestamp}.jpg"
        depth_blur_path = f"results/depth_blur_2d_{timestamp}.jpg"
        anaglyph_path = f"results/anaglyph_{timestamp}.jpg"
        sbs_path = f"results/sbs_{timestamp}.jpg"
        
        # Save files using cv2 to maintain correct colors in BGR format
        cv2.imwrite(depth_path, depth_colored)
        cv2.imwrite(depth_blur_path, depth_blurred_2d_resize)
        cv2.imwrite(anaglyph_path, anaglyph)
        cv2.imwrite(sbs_path, sbs_3d)
        
        progress_message += f"\nImages saved to results directory with timestamp {timestamp}."
        
        output_files = [depth_path, anaglyph_path, sbs_path, wiggle_gif_path, depth_blur_path]
        
        return depth_colored_pil, depth_blurred_2d_pil, anaglyph_pil, sbs_3d_pil, wiggle_gif_path, output_files, progress_message
        
    except Exception as e:
        error_message = f"Error processing image: {str(e)}\nPlease try a different image or reinitialize the converter."
        print(error_message)
        import traceback
        traceback.print_exc() 
        return None, None, None, None, None, None, error_message

def clear_torch_cache():
    """Clear PyTorch CUDA cache"""
    if converter is not None:
        result = converter.clear_vram_cache()
        return result
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        return "VRAM cache cleared (converter not initialized)"
    elif torch.backends.mps.is_available():
        gc.collect()
        return "MPS memory cleared (garbage collection)"
    else:
        return "No GPU detected"

def clear_stored_depth_map():
    global stored_depth_map, stored_depth_colored, cached_depth_gray, cached_depth_height, cached_depth_width, base_depth_image, original_input_image
    global last_clicked_point, last_detected_faces, last_face_center, selected_face_idx
    
    stored_depth_map = None
    stored_depth_colored = None
    cached_depth_gray = None
    cached_depth_height = None
    cached_depth_width = None
    base_depth_image = None
    original_input_image = None
    
    # Clear visual indicator tracking variables
    last_clicked_point = None
    last_detected_faces = None
    last_face_center = None
    selected_face_idx = None
    
    return None, None, None, None, None, None, "Cleared stored depth map due to new image", gr.update(interactive=False, value="Need Generate Depth Map")

def set_focal_distance_from_click(focal_distance, focal_thickness, evt: gr.SelectData):
    """Handler for depth map click events to set focal plane"""
    global last_clicked_point, last_detected_faces, last_face_center, selected_face_idx
    
    try:
        print(f"Function called with focal_distance={focal_distance}, focal_thickness={focal_thickness}")
        print(f"Event data type: {type(evt)}")
        
        # Check if we have a cached depth map
        if cached_depth_gray is None:
            return focal_distance, None, "Please generate a depth map first."
            
        # Get coordinates from SelectData object
        if not hasattr(evt, "index"):
            return focal_distance, None, "Invalid event data: no coordinates found."
            
        # SelectData.index contains (x, y) pixel coordinates
        x_img, y_img = evt.index
        print(f"Selected coordinates (pixels): x_img={x_img}, y_img={y_img}")
        
        # Get depth map dimensions
        height, width = cached_depth_gray.shape
        print(f"Depth map dimensions: {width}x{height}")
            
        # Ensure coordinates are within bounds
        x_img = max(0, min(x_img, width-1))
        y_img = max(0, min(y_img, height-1))
        
        # Get depth value at click point
        depth_val = cached_depth_gray[y_img, x_img]
        print(f"Depth value at click point: {depth_val}")
        
        # Set new focal distance (normalized to 0-1 range to match slider)
        new_focal_distance = float(depth_val) / 255.0
        print(f"New focal distance (0-1 range): {new_focal_distance}")
        
        # Round to nearest step of 0.05 to match slider steps
        new_focal_distance = round(new_focal_distance * 20) / 20
        print(f"Rounded focal distance: {new_focal_distance}")
        
        # Store the clicked point for persistence
        last_clicked_point = (x_img, y_img)
        
        # Clear any previous face detection indicators
        last_detected_faces = None
        last_face_center = None
        selected_face_idx = None
        
        # Update visualization
        depth_map_with_plane, status = update_focal_plane_visualization(new_focal_distance, focal_thickness)
        
        if depth_map_with_plane is None:
            return focal_distance, None, f"Error updating visualization: {status}"
            
        # Return the new focal distance as the first value to update the slider
        print(f"Returning new focal distance: {new_focal_distance}")
        return new_focal_distance, depth_map_with_plane, f"Focal plane updated to depth: {new_focal_distance:.2f} (clicked at {x_img},{y_img})"
        
    except Exception as e:
        print(f"Error in set_focal_distance_from_click: {e}")
        import traceback
        traceback.print_exc() 
        return focal_distance, None, f"Error: {str(e)}"

# Define a function to check if process_btn should be interactive
def check_depth_map_exists():
    """Check if a depth map has been generated and return appropriate interactivity state"""
    global stored_depth_map
    return stored_depth_map is not None

def detect_faces(image, detector_type="regular"):
    """Detect faces in the image using specified detector"""
    # Use anime face detector if selected
    if detector_type == "anime":
        return detect_anime_faces(image)
    
    # Regular face detection (OpenCV Haar Cascades)
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Convert to BGR if in RGB format (Gradio uses RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if the image is RGB and convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Try different face cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces with adjusted parameters for better detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,        # Reduced from 1.3 to detect more faces
        minNeighbors=3,         # Reduced from 5 to be less strict
        minSize=(20, 20),       # Smaller minimum size to detect smaller faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces found, try with even more lenient parameters
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If still no faces, try another classifier
        if len(faces) == 0:
            alt_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            faces = alt_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(10, 10),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
    
    return faces  # Returns array of (x, y, w, h) tuples for each face

def face_track_focal_point(focal_distance, focal_thickness, face_detector_type="anime"):
    """Set the focal plane to detected faces in the original image"""
    global original_input_image, cached_depth_gray, cached_depth_height, cached_depth_width, anime_face_detector_loaded
    global last_clicked_point, last_detected_faces, last_face_center, selected_face_idx
    
    # Check if we have a cached depth map and original image
    if cached_depth_gray is None or original_input_image is None:
        return focal_distance, None, "Please generate a depth map first."
    
    try:
        # Debug info
        print(f"Original image shape: {original_input_image.shape}")
        print(f"Original image type: {original_input_image.dtype}")
        print(f"Using face detector: {face_detector_type}")
        
        # Check if anime detector is needed but not initialized
        if face_detector_type == "anime" and not anime_face_detector_loaded:
            # Try to initialize
            result = init_anime_face_detector()
            if not anime_face_detector_loaded:
                # Get current visualization instead of returning None
                current_viz, _ = update_focal_plane_visualization(focal_distance, focal_thickness)
                return focal_distance, current_viz, "Anime face detector is not initialized. Please go to the Initialize tab and set it up first."
        
        # Ensure image is properly formatted for face detection
        # OpenCV expects uint8 images
        if original_input_image.dtype != np.uint8:
            print(f"Converting image from {original_input_image.dtype} to uint8")
            original_input_image = (original_input_image * 255).astype(np.uint8)
        
        # Detect faces in the original image
        faces = detect_faces(original_input_image, face_detector_type)
        print(f"Detected {len(faces)} faces")
        
        # Create a visualization with face rectangles for debugging
        debug_img = original_input_image.copy()
        for (x, y, w, h) in faces:
            # Draw rectangle around each face
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw center point
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Save debug image
        debug_path = os.path.join("results", "face_debug.jpg")
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved face detection debug image to {debug_path}")
        
        if len(faces) == 0:
            # Get current visualization instead of returning None
            current_viz, _ = update_focal_plane_visualization(focal_distance, focal_thickness)
            detector_name = "anime face" if face_detector_type == "anime" else "face"
            return focal_distance, current_viz, f"No {detector_name}s detected in the image. Debug image saved to results/face_debug.jpg"
            
        # Get the depth map dimensions
        height, width = cached_depth_gray.shape
        
        # Get the original image dimensions
        orig_h, orig_w = original_input_image.shape[:2]
        
        # For each face, calculate the center point and get corresponding depth value
        face_depths = []
        face_locations = []
        face_sizes = []
        
        for (x, y, w, h) in faces:
            # Calculate center of face
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Scale coordinates to depth map dimensions
            depth_x = int(face_center_x * width / orig_w)
            depth_y = int(face_center_y * height / orig_h)
            
            # Scale face dimensions to depth map dimensions
            depth_w = int(w * width / orig_w)
            depth_h = int(h * height / orig_h)
            
            # Ensure coordinates are within bounds
            depth_x = max(0, min(depth_x, width-1))
            depth_y = max(0, min(depth_y, height-1))
            
            # Get depth value at face center point
            depth_val = cached_depth_gray[depth_y, depth_x]
            face_depths.append(depth_val)
            
            # Store the actual face rectangle coordinates in depth map scale
            face_locations.append((depth_x - depth_w//2, depth_y - depth_h//2, depth_w, depth_h, depth_val))
            
            # Store face size (area) for prioritization
            face_sizes.append(depth_w * depth_h)
            
            print(f"Face at ({face_center_x}, {face_center_y}) has depth value: {depth_val}, size: {w*h}")
        
        # Instead of averaging all faces, prioritize the most likely face
        # Prioritize largest face, then the one with highest depth value (closest)
        if face_depths:
            # Create a combined score for each face (size * 0.8 + depth_val * 0.2)
            # This gives more weight to size but still considers depth
            max_size = max(face_sizes) if face_sizes else 1
            max_depth = max(face_depths) if face_depths else 1
            
            # Normalize values to 0-1 range
            normalized_sizes = [size / max_size for size in face_sizes]
            normalized_depths = [depth / max_depth for depth in face_depths]
            
            # Calculate combined scores
            combined_scores = [size * 0.8 + depth * 0.2 for size, depth in zip(normalized_sizes, normalized_depths)]
            
            # Find the face with the highest score
            best_face_idx = combined_scores.index(max(combined_scores))
            
            # Store which face was selected for visualization
            selected_face_idx = best_face_idx
            
            # Use the depth of the best face
            best_depth = face_depths[best_face_idx]
            new_focal_distance = float(best_depth) / 255.0
            
            # Round to nearest step of 0.05 to match slider steps
            new_focal_distance = round(new_focal_distance * 20) / 20
            print(f"Setting new focal distance to {new_focal_distance} (based on highest priority face)")
            
            # Store face detection results for persistence
            # We'll still store all faces for visualization, but prioritize one for focal distance
            last_detected_faces = face_locations
            
            # Calculate and store the best face center point
            best_face = face_locations[best_face_idx]
            best_face_center_x = best_face[0] + best_face[2]//2
            best_face_center_y = best_face[1] + best_face[3]//2
            last_face_center = (best_face_center_x, best_face_center_y)
            
            # Clear any previous click point
            last_clicked_point = None
            
            # Update visualization
            depth_map_with_plane, status = update_focal_plane_visualization(new_focal_distance, focal_thickness)
            
            if depth_map_with_plane is None:
                # Get current visualization instead of returning None
                current_viz, _ = update_focal_plane_visualization(focal_distance, focal_thickness)
                return focal_distance, current_viz, f"Error updating visualization: {status}"
                
            # Return the new focal distance and visualization
            detector_name = "anime face" if face_detector_type == "anime" else "face"
            total_faces = len(faces)
            if total_faces > 1:
                message = f"Focal plane set to most prominent {detector_name}: {new_focal_distance:.2f} ({total_faces} faces detected, using the largest/closest)"
            else:
                message = f"Focal plane set to {detector_name} depth: {new_focal_distance:.2f}"
            
            return new_focal_distance, depth_map_with_plane, message
        else:
            # Get current visualization instead of returning None
            current_viz, _ = update_focal_plane_visualization(focal_distance, focal_thickness)
            return focal_distance, current_viz, "Error processing face depths. Debug image saved to results/face_debug.jpg"
            
    except Exception as e:
        print(f"Error in face_track_focal_point: {e}")
        import traceback
        traceback.print_exc() 
        # Get current visualization instead of returning None
        current_viz, _ = update_focal_plane_visualization(focal_distance, focal_thickness)
        return focal_distance, current_viz, f"Error: {str(e)}"

def build_interface():
    # Add custom CSS for orange buttons and green process button
    css = """
    .orange-button button {
        background-color: #FF9300 !important;
    }
    .orange-button button:hover {
        background-color: #FFB347 !important;
    }
    .green-button button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .green-button button:hover {
        background-color: #6EC071 !important;
    }
    .green-button button[disabled] {
        background-color: #E0E0E0 !important;
        color: #A0A0A0 !important;
        font-weight: normal !important;
    }
    """
    
    with gr.Blocks(title="stereOgram SBS3D converter", css=css) as interface:
        # Top-level UI
        gr.Markdown("# stereOgram SBS3D converter")
        gr.Markdown("Convert regular 2D images into stereo 3D formats using depth estimation")
        
        # Tab layout
        with gr.Tabs() as tabs:
            # Initialize tab - simplified
            with gr.Tab("Advanced Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Settings")
                        gr.Markdown("The system is initialized automatically with default settings. You can change these settings here if needed.")
                        
                        with gr.Accordion("Model Options", open=False):
                            depth_model = gr.Dropdown(
                                choices=["depth_anything_v2"], 
                                value=default_depth_model, 
                                label="Depth Model"
                            )
                            model_size = gr.Dropdown(
                                choices=["vits", "vitb", "vitl"], 
                                value=default_model_size, 
                                label="Model Size", 
                                info="vits = smallest/fastest, vitb = medium/balanced, vitl = largest/best quality"
                            )
                            use_advanced = gr.Checkbox(
                                value=default_use_advanced, 
                                label="Use Advanced Inpainting", 
                                info="Higher quality but more GPU memory"
                            )
                            max_res = gr.Slider(
                                minimum=1024, 
                                maximum=8192, 
                                value=default_max_res, 
                                step=1024, 
                                label="Max Resolution", 
                                info="Maximum resolution for depth estimation"
                            )
                            low_memory = gr.Checkbox(
                                value=default_low_memory, 
                                label="Low Memory Mode", 
                                info="For GPUs with limited VRAM"
                            )
                            init_btn = gr.Button("Reinitialize with New Settings", variant="secondary", elem_classes="orange-button")
                    
                    with gr.Column():
                        init_output = gr.Textbox(label="Status", value="System initialized automatically with default settings")
                        clear_cache_btn = gr.Button("Clear VRAM Cache", variant="secondary", elem_classes="orange-button")
                        
                        # Add section for anime face detector
                        gr.Markdown("### Anime Face Detector")
                        anime_detector_status = gr.Textbox(label="Status", value="Initialized automatically at startup")
                        
                        with gr.Accordion("Face Detector Tools", open=False):
                            gr.Markdown("""
                            **Only needed if automatic initialization fails:**
                            1. The anime face detector is loaded automatically on startup
                            2. If errors occur, use these buttons to reinstall/reinitialize
                            """)
                            install_anime_face_btn = gr.Button("Reinstall Anime Face Detector", variant="secondary", elem_classes="orange-button")
                            init_anime_face_btn = gr.Button("Reinitialize Anime Face Detector", variant="secondary", elem_classes="orange-button")
            
            # Convert tab
            with gr.Tab("Convert"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="Input Image", type="pil")
                        gen_depth_btn = gr.Button("Generate Depth Map", variant="primary", elem_classes="orange-button")
                        
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
                        
                        with gr.Accordion("Configure Blur & Generate Output", open=True):
                            apply_depth_blur = gr.Checkbox(
                                value=True,
                                label="Apply Depth-Based Blur",
                                info="Adds depth of field effect based on the depth map"
                            )
                            blur_intensity = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.2,
                                step=0.05,
                                label="Blur Intensity",
                                info="Controls how strongly the blur effect is applied (0=subtle, 1=strong)"
                            )
                            blur_radius = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.4,
                                step=0.05,
                                label="Blur Radius",
                                info="Controls how far the blur spreads (0=tight, 1=wide)"
                            )
                            
                            with gr.Accordion("Advanced Processing Settings", open=False):
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
                                    value=True, 
                                    label="High Quality Mode", 
                                    info="Better anti-banding, smoother output"
                                )
                                debug_mode = gr.Checkbox(
                                    value=False, 
                                    label="Debug Mode", 
                                    info="Show occlusion areas with purple"
                                )
                            
                            # Create the button in disabled state initially with conditional text
                            process_btn = gr.Button(
                                value="Need Generate Depth Map", 
                                variant="primary", 
                                elem_classes="green-button", 
                                interactive=False
                            )
                        
                        clear_btn = gr.Button("Clear Images", variant="secondary", elem_classes="orange-button")
                    
                    with gr.Column(scale=1):
                        depth_map = gr.Image(
                            label="Depth Map Visualization", 
                            elem_id="depth_map_viz",
                            show_download_button=True,
                            show_label=True,
                            interactive=True,
                            height=400,
                            width=500,
                            type="numpy",
                            sources=["upload"]
                        )
                        
                        # Make the instructions more prominent
                        depth_click_instructions = gr.Markdown("### 👆 Click on the depth map to set the focal plane at that depth")
                        
                        # Focal plane controls directly under depth map
                        with gr.Accordion("Focal Plane Adjustments", open=True):
                            focal_distance = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Focal Distance",
                                info="Distance to keep in focus (0=farthest, 1=nearest)"
                            )
                            
                            focal_thickness = gr.Slider(
                                minimum=0.05,
                                maximum=0.5,
                                value=0.1,
                                step=0.05,
                                label="Focal Thickness",
                                info="Thickness of the in-focus region"
                            )
                            
                            # Face detector type selector - changed default to "anime"
                            face_detector_type = gr.Radio(
                                choices=["anime", "regular"],
                                value="anime",
                                label="Face Detector Type",
                                info="Choose between anime face detection and regular (human) face detection"
                            )
                            
                            # Add Face Tracking button
                            face_track_btn = gr.Button(
                                value="Track Faces", 
                                variant="secondary", 
                                elem_classes="orange-button"
                            )
                        
                        depth_status = gr.Textbox(label="Status")
                        
                        output_tabs = gr.Tabs()
                        with output_tabs:
                            with gr.TabItem("2D with Depth Blur"):
                                depth_blur_2d = gr.Image(label="2D with Depth Blur")
                            with gr.TabItem("Side-by-Side"):
                                sbs = gr.Image(label="Side-by-Side Stereo")
                            with gr.TabItem("Red-Cyan Anaglyph"):
                                anaglyph = gr.Image(label="Red-Cyan Anaglyph")
                            with gr.TabItem("Wiggle GIF"):
                                wiggle_gif = gr.Image(label="Wiggle GIF")
                        
                        download_files = gr.File(label="Download Results")
                        progress = gr.Textbox(label="Processing Status")
        
        # Connect event handlers
        init_btn.click(
            initialize_converter, 
            inputs=[depth_model, model_size, use_advanced, max_res, low_memory], 
            outputs=[init_output]
        )
        
        clear_cache_btn.click(
            clear_torch_cache, 
            inputs=[],
            outputs=[init_output]
        )
        
        # Add handlers for anime face detector
        install_anime_face_btn.click(
            install_anime_face_detector,
            inputs=[],
            outputs=[anime_detector_status]
        )
        
        init_anime_face_btn.click(
            init_anime_face_detector,
            inputs=[],
            outputs=[anime_detector_status]
        )
        
        # Update the gen_depth_btn click event to enable the process_btn and change its text
        gen_depth_btn.click(
            generate_depth_map,
            inputs=[input_image],
            outputs=[depth_map, depth_status]
        ).then(
            lambda: (gr.update(interactive=True, value="Generate Stereo Image")),  # Use gr.update() instead of direct tuple
            inputs=None,
            outputs=[process_btn]
        )
        
        focal_distance.change(
            update_focal_plane_visualization,
            inputs=[focal_distance, focal_thickness],
            outputs=[depth_map, depth_status]
        )
        
        focal_thickness.change(
            update_focal_plane_visualization,
            inputs=[focal_distance, focal_thickness],
            outputs=[depth_map, depth_status]
        )
        
        # Use select event with the updated function
        depth_map.select(
            fn=set_focal_distance_from_click,
            inputs=[focal_distance, focal_thickness],
            outputs=[focal_distance, depth_map, depth_status],
            show_progress=False  # Faster response in 5.19.0
        )
        
        # Connect face track button with detector type
        face_track_btn.click(
            face_track_focal_point,
            inputs=[focal_distance, focal_thickness, face_detector_type],
            outputs=[focal_distance, depth_map, depth_status]
        )
        
        # Define a local version of clear_stored_depth_map for the UI
        def _clear_stored_depth_map():
            global stored_depth_map, stored_depth_colored, cached_depth_gray, cached_depth_height, cached_depth_width, base_depth_image, original_input_image
            stored_depth_map = None
            stored_depth_colored = None
            cached_depth_gray = None
            cached_depth_height = None
            cached_depth_width = None
            base_depth_image = None
            original_input_image = None
            return None, None, None, None, None, None, "Cleared stored depth map due to new image", gr.update(interactive=False, value="Need Generate Depth Map")
        
        process_btn.click(
            process_image, 
            inputs=[
                input_image, shift_factor, resolution, patch_size, 
                patch_overlap, steps, cfg_scale, high_quality, debug_mode,
                apply_depth_blur, focal_distance, focal_thickness, blur_intensity, blur_radius
            ], 
            outputs=[depth_map, depth_blur_2d, anaglyph, sbs, wiggle_gif, download_files, progress]
        )
        
        # Update the clear_btn click event to also disable the process_btn and update its text
        clear_btn.click(
            _clear_stored_depth_map,
            inputs=[],
            outputs=[depth_map, depth_blur_2d, anaglyph, sbs, wiggle_gif, download_files, progress, process_btn]
        )
        
        # Update the input_image change event to also disable the process_btn and update its text
        input_image.change(
            _clear_stored_depth_map,
            inputs=[],
            outputs=[depth_map, depth_blur_2d, anaglyph, sbs, wiggle_gif, download_files, progress, process_btn]
        )
    
    return interface

# Launch the Gradio app
def run_interface(share=True):
    try:
        interface = build_interface()
        interface.launch(share=share)
        return 0
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(run_interface()) 