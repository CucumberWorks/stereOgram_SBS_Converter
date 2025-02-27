import os
import sys
import cv2
import numpy as np
import time

# Add the parent directory to the Python path to access core module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.stereogram_sbs3d_converter import StereogramSBS3DConverter

def test_converter(image_path, output_folder="test_results"):
    """Test the stereo converter with a single image."""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create output path
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{base_name}_sbs{ext}")
    
    print(f"Input image: {image_path}")
    print(f"Output will be saved to: {output_path}")
    print("-" * 40)
    
    # Load image
    print("Loading image...")
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}!")
        return False
    
    print(f"Image loaded successfully. Size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize converter
    print("Initializing converter...")
    start_time = time.time()
    
    try:
        converter = StereogramSBS3DConverter(
            use_advanced_infill=True,
            depth_model_type="depth_anything_v2",
            model_size="vitb",  # Use vitb for balance of quality & performance
            max_resolution=2048,  # Limit max resolution
            low_memory_mode=False  # Set to True if memory issues occur
        )
        init_time = time.time() - start_time
        print(f"Converter initialized in {init_time:.2f} seconds")
        
        # Set inpainting parameters
        converter.set_inpainting_params(
            steps=20,
            guidance_scale=7.5, 
            patch_size=128, 
            patch_overlap=32,
            high_quality=True
        )
    except Exception as e:
        print(f"Error initializing converter: {e}")
        return False
    
    # Process image
    print("Processing image to SBS stereo format...")
    start_time = time.time()
    
    try:
        stereo_sbs, left_view, right_view, depth_map = converter.generate_sbs_stereo(
            image,
            output_path=output_path,
            shift_factor=0.05,  # Adjust stereo separation factor
            efficient_mode=True  # Use efficient mode for faster processing
        )
        process_time = time.time() - start_time
        print(f"Processing completed in {process_time:.2f} seconds")
        
        # Save depth map for analysis
        depth_map_path = os.path.join(output_folder, f"{base_name}_depth{ext}")
        print(f"Saving depth map to: {depth_map_path}")
        # Normalize depth map for visualization
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(depth_map_path, depth_norm)
        
        print("\nOutput files:")
        print(f"- SBS stereo image: {output_path}")
        print(f"- Depth map: {depth_map_path}")
        print("\nSuccessfully converted image to SBS stereo format!")
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_converter.py <image_path>")
        print("Example: python test_converter.py demo_images/example.jpg")
        
        # Check if demo_images folder exists
        demo_folder = "demo_images"
        if os.path.exists(demo_folder) and os.path.isdir(demo_folder):
            print("\nAvailable demo images:")
            demo_images = [f for f in os.listdir(demo_folder) 
                          if os.path.isfile(os.path.join(demo_folder, f)) 
                          and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if demo_images:
                for img in demo_images:
                    print(f"  - {os.path.join(demo_folder, img)}")
                
                # Use the first demo image if available
                image_path = os.path.join(demo_folder, demo_images[0])
                print(f"\nNo image specified, using first demo image: {image_path}")
            else:
                print("  No demo images found.")
                return
        else:
            return
    else:
        image_path = sys.argv[1]
    
    # Run test
    test_converter(image_path)

if __name__ == "__main__":
    main() 