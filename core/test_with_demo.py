import os
import argparse
import glob
import cv2
import numpy as np
import time
import torch
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter
from core.advanced_infill import AdvancedInfillTechniques

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D images from 2D images")
    parser.add_argument("--input-dir", type=str, default="demo_images", help="Input directory containing images")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--shift-factor", type=float, default=0.03, help="Shift factor for 3D effect (0.01-0.2)")
    parser.add_argument("--depth-model", type=str, default="depth_anything_v2", choices=["depth_anything_v2", "depth_anything", "zmde", "dn"], help="Depth model to use")
    parser.add_argument("--low-memory", action="store_true", help="Low memory mode")
    parser.add_argument("--medium-memory", action="store_true", help="Medium memory mode")
    parser.add_argument("--resolution", type=int, default=1080, help="Output resolution height (720, 1080, 1440, 2160)")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size for inpainting")
    parser.add_argument("--patch-overlap", type=int, default=32, help="Patch overlap for inpainting")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--high-quality", action="store_true", help="High quality mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode with purple background visualization")
    return parser.parse_args()

def generate_3d_image(left, right, shift=3):
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
    
    # Convert images to BGR for processing
    if len(left.shape) == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if len(right.shape) == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    
    # Create red-cyan anaglyph
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
    
    # Convert images to BGR if needed
    if len(left.shape) == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if len(right.shape) == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    
    # Place left and right views side by side
    output[:, 0:w_left] = left
    output[:, w_left:w_left+w_right] = right
    
    return output

def process_demo_images(args, debug_mode=False):
    """Process all images in the demo directory"""
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Determine resolution factors
    if args.resolution == 720:
        process_res_factor = 480 if not args.high_quality else 600
    elif args.resolution == 1080:
        process_res_factor = 720 if not args.high_quality else 900
    elif args.resolution == 1440:
        process_res_factor = 800 if not args.high_quality else 1000
    elif args.resolution == 2160:
        process_res_factor = 960 if not args.high_quality else 1200
    else:
        process_res_factor = 720  # Default
    
    # Adjust for memory constraints
    if args.low_memory:
        process_res_factor = max(process_res_factor // 2, 384)
        print(f"Low memory mode - using processing resolution {process_res_factor}p")
    elif args.medium_memory:
        process_res_factor = max(int(process_res_factor * 0.75), 480)
        print(f"Medium memory mode - using processing resolution {process_res_factor}p")
    else:
        print(f"Using processing resolution {process_res_factor}p")
    
    if debug_mode:
        print("Debug mode enabled - using purple background visualization")
    
    # Check for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        
        # Adjust processing resolution based on GPU memory
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # VRAM in GB
        print(f"GPU memory: {vram:.1f} GB")
        
        # Automatically adjust for high-end GPUs
        if vram >= 20.0 and not args.low_memory and not args.medium_memory:
            if args.high_quality:
                print("High-end GPU detected - using maximum quality settings")
                process_res_factor = min(int(process_res_factor * 1.5), 1600)
            else:
                process_res_factor = min(int(process_res_factor * 1.2), 1200)
            print(f"Adjusted processing resolution to {process_res_factor}p")
    else:
        print("No GPU detected - using CPU mode")
    
    # Get all image paths
    img_paths = glob.glob(os.path.join(args.input_dir, "*.jpg")) + \
                glob.glob(os.path.join(args.input_dir, "*.png")) + \
                glob.glob(os.path.join(args.input_dir, "*.jpeg"))
    
    # Sort images by name
    img_paths.sort()
    print(f"Found {len(img_paths)} images in {args.input_dir}")
    
    # Initialize converter (this loads the models)
    use_advanced = not args.low_memory  # Use advanced mode unless low memory
    converter = StereogramSBS3DConverter(
        use_advanced_infill=use_advanced, 
        depth_model_type=args.depth_model,
        max_resolution=8192,  # Use a very high max_resolution to prevent resizing during depth estimation
        low_memory_mode=args.low_memory
    )
    
    # Process each image
    for img_path in img_paths:
        # Get filename without extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing {img_name}...")
        
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        # Save original image
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_original.jpg"), img)
        
        # Use original image for depth estimation (no resizing)
        depth_img = img.copy()
        
        # Generate depth map
        start_time = time.time()
        depth_map = converter.estimate_depth(depth_img)
        print(f"Depth map generated in {time.time() - start_time:.2f} seconds")
        
        # Save depth map
        depth_colored = converter.visualize_depth(depth_map)
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_depth.jpg"), depth_colored)
        
        # Determine processing resolution
        # We want the height to be close to process_res_factor
        # and we want the width to be divisible by 8 for the model
        proc_h = process_res_factor
        proc_w = int(proc_h * aspect_ratio)
        # Make divisible by 8
        proc_w = proc_w - (proc_w % 8)
        proc_h = proc_h - (proc_h % 8)
        
        # Resize image for processing
        proc_img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Resize depth map to match processing resolution
        depth_map_resized = cv2.resize(depth_map, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        
        # Generate stereo views
        shift_pixels = int(args.shift_factor * proc_w)
        print(f"Generating stereo views with {shift_pixels}px shift...")
        
        start_time = time.time()
        if debug_mode:
            # Use debug visualization mode with purple background
            left_view, right_view, left_holes, right_holes = converter.generate_stereo_views_debug(
                proc_img, depth_map_resized, shift_factor=args.shift_factor
            )
        else:
            # Use normal mode
            left_view, right_view, left_holes, right_holes = converter.generate_stereo_views(
                proc_img, depth_map_resized, shift_factor=args.shift_factor
            )
        
        print(f"Stereo views generated in {time.time() - start_time:.2f} seconds")
        
        # Save hole masks and views with holes for debugging
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_left_holes.png"), (left_holes*255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_right_holes.png"), (right_holes*255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_left_view_with_holes.jpg"), left_view)
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_right_view_with_holes.jpg"), right_view)
        
        # Generate basic anaglyph (without hole filling)
        basic_anaglyph = generate_3d_image(left_view, right_view, shift=args.shift_factor)
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_basic_shift{int(args.shift_factor*100)}.jpg"), basic_anaglyph)
        
        # Fill holes in the stereo views
        if use_advanced:
            # Use advanced method to fill holes
            print("Using advanced inpainting for hole filling...")
            start_time = time.time()
            
            # Set inpainting parameters
            converter.set_inpainting_params(
                steps=args.steps,
                guidance_scale=args.cfg_scale,
                patch_size=args.patch_size,
                patch_overlap=args.patch_overlap,
                high_quality=args.high_quality
            )
            
            # Do context-aware inpainting for small holes first
            threshold_small = 1000  # Adjust this threshold as needed
            
            # Count small holes
            small_left_holes = np.sum(cv2.connectedComponents(
                (left_holes*255).astype(np.uint8), connectivity=8
            )[0] > 1 and left_holes.size < threshold_small)
            
            small_right_holes = np.sum(cv2.connectedComponents(
                (right_holes*255).astype(np.uint8), connectivity=8
            )[0] > 1 and right_holes.size < threshold_small)
            
            # Apply context-aware filling to small holes
            if small_left_holes > 0 or small_right_holes > 0:
                print(f"Applying context-aware inpainting to small holes...")
                # Create kernel for morphological operations
                kernel = np.ones((5, 5), np.uint8)
                
                # Process left view
                if np.sum(left_holes) > 0:
                    # Dilate holes slightly for better blending
                    left_holes_dilated = cv2.dilate(left_holes.astype(np.uint8), kernel)
                    
                    # Use inpaint to fill small holes - this uses nearby pixels
                    left_view = cv2.inpaint(
                        left_view, 
                        left_holes_dilated, 
                        inpaintRadius=3, 
                        flags=cv2.INPAINT_TELEA
                    )
                
                # Process right view
                if np.sum(right_holes) > 0:
                    # Dilate holes slightly for better blending
                    right_holes_dilated = cv2.dilate(right_holes.astype(np.uint8), kernel)
                    
                    # Use inpaint to fill small holes - this uses nearby pixels
                    right_view = cv2.inpaint(
                        right_view, 
                        right_holes_dilated, 
                        inpaintRadius=3, 
                        flags=cv2.INPAINT_TELEA
                    )
            
            # Apply advanced diffusion-based inpainting
            if np.sum(left_holes) > 0:
                print(f"Filling left view holes ({np.sum(left_holes)} pixels)...")
                left_inpainted = converter.fill_holes_preserving_originals(
                    proc_img,  # Original image
                    left_view, 
                    left_holes,
                    depth_map_resized,
                    shift_factor=args.shift_factor,
                    is_left_view=True
                )
                left_view = left_inpainted
            
            if np.sum(right_holes) > 0:
                print(f"Filling right view holes ({np.sum(right_holes)} pixels)...")
                right_inpainted = converter.fill_holes_preserving_originals(
                    proc_img,  # Original image
                    right_view, 
                    right_holes,
                    depth_map_resized,
                    shift_factor=args.shift_factor,
                    is_left_view=False
                )
                right_view = right_inpainted
                
            print(f"Hole filling completed in {time.time() - start_time:.2f} seconds")
            
            # Generate advanced anaglyph (with hole filling)
            advanced_anaglyph = generate_3d_image(left_view, right_view, shift=args.shift_factor)
            cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_advanced_shift{int(args.shift_factor*100)}.jpg"), advanced_anaglyph)
        
        # Determine output resolution
        target_h = args.resolution
        target_w = int(target_h * aspect_ratio)
        
        # Make width divisible by 2 for even dimensions
        target_w = target_w - (target_w % 2)
        
        # Generate depth-aware anaglyph (using depth map for enhanced effect)
        print("Generating depth-aware anaglyph...")
        start_time = time.time()
        
        # Resize to output resolution
        left_view_resize = cv2.resize(left_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        right_view_resize = cv2.resize(right_view, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Generate final anaglyph output
        depth_aware_anaglyph = generate_3d_image(left_view_resize, right_view_resize, shift=args.shift_factor)
        
        # Generate side-by-side 3D output
        print("Generating side-by-side 3D...")
        sbs_3d = generate_sbs_3d(left_view_resize, right_view_resize)
        
        # Save final results
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_depth_aware_shift{int(args.shift_factor*100)}.jpg"), depth_aware_anaglyph)
        cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_sbs3d_shift{int(args.shift_factor*100)}.jpg"), sbs_3d)
        print(f"Final outputs ({target_w}x{target_h}) saved in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    process_demo_images(args, debug_mode=args.debug) 