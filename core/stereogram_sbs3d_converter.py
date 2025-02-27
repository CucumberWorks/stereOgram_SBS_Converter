import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as transforms
import sys
import os
import gc

class StereogramSBS3DConverter:
    def __init__(self, use_advanced_infill=True, depth_model_type="depth_anything_v2", model_size="vitb", max_resolution=4096, low_memory_mode=False):
        # Initialize parameters
        self.use_advanced_infill = use_advanced_infill
        self.depth_model_type = depth_model_type
        self.model_size = model_size
        self.max_resolution = max_resolution
        self.low_memory_mode = low_memory_mode
        self.high_color_quality = True  # Always enable high color quality
        
        # Initialize resources when needed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        self.inpaint_model = None
        self.inpaint_steps = 20
        self.inpaint_guidance_scale = 7.5
        self.inpaint_patch_size = 128
        self.inpaint_patch_overlap = 32
        
        # Default enhancement parameters
        self.apply_dithering = True  # Enable dithering by default
        self.dithering_level = 1.0   # Default dithering strength
        
        # Initialize depth estimation model
        self._init_depth_model()
        
        # Initialize inpainting model if using advanced infill
        if use_advanced_infill:
            self._init_inpainting_model()
    
    def _init_depth_model(self):
        """Initialize depth estimation model based on selected type"""
        if self.depth_model_type == "midas":
            # MiDaS depth estimation model
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            # MiDaS transformation
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = self.midas_transforms.small_transform
        elif self.depth_model_type == "depth_anything":
            # Load Depth Anything model (legacy V1)
            try:
                # Add depth_anything to path if not already present
                depth_anything_path = os.path.join(os.getcwd(), "depth_anything")
                if os.path.exists(depth_anything_path) and depth_anything_path not in sys.path:
                    sys.path.append(depth_anything_path)
                
                # Import the model
                from depth_anything.dpt import DepthAnything
                
                # Use the largest model for highest quality
                self.depth_model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
                self.depth_model.to(self.device)
                self.depth_model.eval()
                
                # Define the transform for Depth Anything
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5),
                ])
                
                print("Loaded Depth Anything ViT-L model - highest quality version")
            except Exception as e:
                print(f"Failed to load Depth Anything: {e}")
                print("Falling back to MiDaS model")
                # Fallback to MiDaS
                self.depth_model_type = "midas"
                self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                self.depth_model.to(self.device)
                self.depth_model.eval()
                
                # MiDaS transformation
                self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = self.midas_transforms.small_transform
        elif self.depth_model_type == "depth_anything_v2":
            # Load Depth Anything V2 model (highest quality)
            try:
                # Add depth_anything_v2 to path if not already present
                depth_anything_v2_path = os.path.dirname(os.path.abspath(__file__))
                if depth_anything_v2_path not in sys.path:
                    sys.path.append(depth_anything_v2_path)
                
                # Import the model
                from depth_anything_v2.dpt import DepthAnythingV2
                
                # Define model configuration based on model size
                # Using ViT-B model for better memory efficiency (can be changed to vits or vitl)
                encoder = self.model_size  # Use the model_size parameter from initialization
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
                }
                
                # Initialize the model with proper configuration
                if encoder not in model_configs:
                    print(f"Model size '{encoder}' not available for Depth Anything V2.")
                    if encoder == 'vitg':
                        print("Giant (vitg) model is not publicly released yet.")
                    print("Falling back to MiDaS model")
                    
                    # Fallback to MiDaS
                    self.depth_model_type = "midas"
                    self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                    self.depth_model.to(self.device)
                    self.depth_model.eval()
                    
                    # MiDaS transformation
                    self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                    self.transform = self.midas_transforms.small_transform
                    return  # Exit the method early
                
                self.depth_model = DepthAnythingV2(**model_configs[encoder])
                
                # Check if model weights exist, if not, download them
                model_path = os.path.join(os.getcwd(), "models")
                os.makedirs(model_path, exist_ok=True)
                weight_path = os.path.join(model_path, f"depth_anything_v2_{encoder}.pth")
                
                if not os.path.exists(weight_path):
                    print(f"Downloading Depth Anything V2 {encoder} model weights...")
                    import urllib.request
                    
                    # URL for the model weights based on the encoder type
                    base_url = "https://huggingface.co/depth-anything/Depth-Anything-V2"
                    model_urls = {
                        'vits': f"{base_url}-Small/resolve/main/depth_anything_v2_vits.pth",
                        'vitb': f"{base_url}-Base/resolve/main/depth_anything_v2_vitb.pth",
                        'vitl': f"{base_url}-Large/resolve/main/depth_anything_v2_vitl.pth"
                    }
                    
                    if encoder in model_urls:
                        # Download the weights
                        urllib.request.urlretrieve(model_urls[encoder], weight_path)
                        print(f"Downloaded weights to {weight_path}")
                    else:
                        print(f"No pre-built URL for {encoder} model, please download manually")
                
                # Load the weights
                try:
                    self.depth_model.load_state_dict(torch.load(weight_path, map_location=self.device))
                    self.depth_model.to(self.device)
                    self.depth_model.eval()
                    print(f"Loaded Depth Anything V2 {encoder} model")
                except Exception as e:
                    print(f"Error loading model weights: {e}")
                    raise
                
            except Exception as e:
                print(f"Failed to load Depth Anything V2: {e}")
                print("Falling back to MiDaS model")
                # Fallback to MiDaS
                self.depth_model_type = "midas"
                self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                self.depth_model.to(self.device)
                self.depth_model.eval()
                
                # MiDaS transformation
                self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = self.midas_transforms.small_transform
    
    def _init_inpainting_model(self):
        """Initialize advanced inpainting model using Stable Diffusion"""
        try:
            if self.low_memory_mode:
                print("Using low memory mode for inpainting - loading model with reduced precision")
                # Load the model with reduced precision and memory optimizations
                self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    revision="fp16" if self.device.type == "cuda" else "main",
                    safety_checker=None  # Skip safety checker to save memory
                )
                
                # Enable memory efficient attention if xformers is available
                try:
                    from diffusers.utils import is_xformers_available
                    if is_xformers_available():
                        self.inpaint_model.enable_xformers_memory_efficient_attention()
                        print("Using xformers for memory-efficient attention")
                    else:
                        print("xformers not available, using standard attention")
                except ImportError:
                    print("Could not check for xformers, using standard attention")
                
                # Use model offloading to save VRAM
                self.inpaint_model.enable_sequential_cpu_offload()
                print("Enabled sequential CPU offloading for model weights")
            else:
                # Use standard loading
                self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                # Disable safety checker to prevent NSFW filtering
                self.inpaint_model.safety_checker = None
                self.inpaint_model.to(self.device)
        except Exception as e:
            print(f"Error initializing inpainting model: {e}")
            print("Will fall back to basic inpainting")
            self.use_advanced_infill = False
    
    def extract_hole_patches(self, image, hole_mask, context_size=64):
        """Extract patches containing holes with some context around them"""
        # Convert to the right format for connected components
        hole_mask_uint8 = hole_mask.astype(np.uint8) * 255
        
        # Find connected components in the hole mask
        num_labels, labels = cv2.connectedComponents(hole_mask_uint8, connectivity=4)
        
        # If too many components, merge them by dilation first
        if num_labels > 1000:
            print(f"Too many hole components ({num_labels}), merging...")
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(hole_mask_uint8, kernel, iterations=2)
            num_labels, labels = cv2.connectedComponents(dilated_mask, connectivity=4)
            print(f"Reduced to {num_labels} components")
        
        # Group small nearby holes to reduce component count
        if num_labels > 100:
            kernel = np.ones((7, 7), np.uint8)
            dilated_mask = cv2.dilate(hole_mask_uint8, kernel, iterations=1)
            num_labels, labels = cv2.connectedComponents(dilated_mask, connectivity=4)
            print(f"Further reduced to {num_labels} components")
            
        patches = []
        patch_positions = []
        
        # Process the holes in larger chunks by dividing the image into a grid
        if num_labels > 200:
            print("Using grid-based approach for too many holes")
            height, width = hole_mask.shape[:2]
            grid_size = 256  # Size of grid cells
            
            for y in range(0, height, grid_size):
                for x in range(0, width, grid_size):
                    # Define grid cell with some overlap
                    x_start = max(0, x - context_size)
                    y_start = max(0, y - context_size)
                    x_end = min(width, x + grid_size + context_size)
                    y_end = min(height, y + grid_size + context_size)
                    
                    # Extract cell mask and check if it has holes
                    cell_mask = hole_mask[y_start:y_end, x_start:x_end]
                    
                    if np.any(cell_mask):
                        cell = image[y_start:y_end, x_start:x_end].copy()
                        patches.append((cell, cell_mask))
                        patch_positions.append((x_start, y_start, x_end, y_end))
            
            return patches, patch_positions
            
        # Normal approach for reasonable number of components
        for label in range(1, num_labels):  # Skip 0 (background)
            # Get bounding box for this hole
            hole_component = (labels == label).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(hole_component)
            
            # Skip extremely small holes
            if w < 2 or h < 2:
                continue
                
            # Add context around the hole
            x_start = max(0, x - context_size)
            y_start = max(0, y - context_size)
            x_end = min(image.shape[1], x + w + context_size)
            y_end = min(image.shape[0], y + h + context_size)
            
            # Make sure the patch is at least a minimum size for the model
            min_size = 64
            if x_end - x_start < min_size:
                padding = min_size - (x_end - x_start)
                x_start = max(0, x_start - padding // 2)
                x_end = min(image.shape[1], x_end + padding // 2)
                
            if y_end - y_start < min_size:
                padding = min_size - (y_end - y_start)
                y_start = max(0, y_start - padding // 2)
                y_end = min(image.shape[0], y_end + padding // 2)
            
            # Extract the patch and its mask
            patch = image[y_start:y_end, x_start:x_end].copy()
            patch_mask = hole_mask[y_start:y_end, x_start:x_end].copy()
            
            # Only add if the patch has holes
            if np.any(patch_mask):
                patches.append((patch, patch_mask))
                patch_positions.append((x_start, y_start, x_end, y_end))
        
        return patches, patch_positions
    
    def estimate_depth(self, image):
        """Estimate depth map from input image"""
        # Prepare image for depth estimation
        if isinstance(image, np.ndarray):
            # Convert from OpenCV BGR to RGB
            if image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = image
            
        # Convert PIL Image to numpy array before applying the transform
        img_array = np.array(pil_image)
        
        if self.depth_model_type == "depth_anything_v2":
            # Process with Depth Anything V2
            # Use original image size without resizing
            pil_image_resized = pil_image
            
            # Use infer_image method from DepthAnythingV2
            raw_image = np.array(pil_image_resized)
            # Convert RGB to BGR for the model
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            
            # Get depth map using the model's inference method
            # Input size must be divisible by 14
            width, height = pil_image.size
            max_side = max(width, height)
            input_size = min(1120, max(518, (max_side // 14) * 14))  # Use larger input size for better quality
            print(f"Using input size {input_size} for Depth Anything V2 inference")
            depth_map = self.depth_model.infer_image(raw_image, input_size=input_size)
            
        elif self.depth_model_type == "depth_anything":
            # Process with Depth Anything
            # Use original image without resizing
            pil_image_resized = pil_image
                
            # Apply transform and move to device
            img_input = self.transform(pil_image_resized).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get depth prediction
                depth_map = self.depth_model(img_input)
                
                # Resize to original resolution if needed
                width, height = pil_image.size
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(1).squeeze(0)
                
            # Convert to numpy
            depth_map = depth_map.cpu().numpy()
            
        else:
            # Process with MiDaS (original method)
            img_input = self.transform(img_array).to(self.device)
            
            with torch.no_grad():
                depth_map = self.depth_model(img_input)
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=pil_image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_map = depth_map.cpu().numpy()
        
        # Normalize depth map to 0-1 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Apply additional smoothing to further reduce harsh transitions
        depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
        
        return depth_normalized
    
    def generate_stereo_views(self, image, depth_map, shift_factor=0.05):
        """Generate left and right views based on depth map"""
        height, width = depth_map.shape[:2]
        
        # For better color precision, convert to float32 early in the process
        image_float = image.astype(np.float32)
        
        # Create coordinate maps
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Create left and right views as float32 to preserve color precision
        left_view = np.zeros_like(image_float)
        right_view = np.zeros_like(image_float)
        
        # Generate masks to track pixels that have been filled
        left_mask = np.zeros((height, width), dtype=bool)
        right_mask = np.zeros((height, width), dtype=bool)
        
        # Apply strong blur to depth map for transitions
        smoothed_depth = cv2.GaussianBlur(depth_map, (9, 9), 0)
        
        # Calculate shifts based on smoothed depth map to prevent harsh transitions
        shifts = (smoothed_depth * shift_factor * width).astype(int)
        
        # Generate views (back to front)
        for d in range(int(np.max(shifts)), -1, -1):
            mask = (shifts == d)
            
            # For right view: shift left
            x_coords_right = np.clip(x_coords - d, 0, width-1)
            # For left view: shift right
            x_coords_left = np.clip(x_coords + d, 0, width-1)
            
            # Fill right view (where not already filled)
            fill_mask_right = mask & ~right_mask[y_coords, x_coords_right]
            right_view[y_coords[fill_mask_right], x_coords_right[fill_mask_right]] = image_float[y_coords[fill_mask_right], x_coords[fill_mask_right]]
            right_mask[y_coords[fill_mask_right], x_coords_right[fill_mask_right]] = True
            
            # Fill left view (where not already filled)
            fill_mask_left = mask & ~left_mask[y_coords, x_coords_left]
            left_view[y_coords[fill_mask_left], x_coords_left[fill_mask_left]] = image_float[y_coords[fill_mask_left], x_coords[fill_mask_left]]
            left_mask[y_coords[fill_mask_left], x_coords_left[fill_mask_left]] = True
        
        # Perform gap filling - preprocess holes to identify problematic edge boundaries
        kernel_small = np.ones((3, 3), np.uint8)
        left_holes = ~left_mask
        right_holes = ~right_mask
        
        # Dilate and erode to identify boundary regions
        left_edge_mask = cv2.dilate(left_holes.astype(np.uint8), kernel_small, iterations=2) - cv2.erode(left_holes.astype(np.uint8), kernel_small, iterations=1)
        right_edge_mask = cv2.dilate(right_holes.astype(np.uint8), kernel_small, iterations=2) - cv2.erode(right_holes.astype(np.uint8), kernel_small, iterations=1)
        
        # Process and smooth boundary regions first
        # For left view boundaries
        for y, x in zip(*np.where(left_edge_mask > 0)):
            if not left_mask[y, x]:  # Only process holes
                # Search in larger neighborhood for valid pixels (5x5 window)
                valid_pixels = []
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and left_mask[ny, nx]):
                            valid_pixels.append(left_view[ny, nx])
                
                if valid_pixels:
                    # Use median filter for more robust edge filling
                    # Use actual float median instead of converting to uint8
                    left_view[y, x] = np.median(valid_pixels, axis=0)
                    left_mask[y, x] = True
        
        # For right view boundaries
        for y, x in zip(*np.where(right_edge_mask > 0)):
            if not right_mask[y, x]:  # Only process holes
                # Search in larger neighborhood for valid pixels
                valid_pixels = []
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and right_mask[ny, nx]):
                            valid_pixels.append(right_view[ny, nx])
                
                if valid_pixels:
                    # Use median filter for more robust edge filling
                    # Use actual float median instead of converting to uint8
                    right_view[y, x] = np.median(valid_pixels, axis=0)
                    right_mask[y, x] = True
        
        # Apply a joint bilateral filter to smooth transitions while preserving edges
        # Keep working in float32 to minimize banding
        if np.any(left_mask):
            # Use the d parameter to limit the computational radius for better performance
            sigma_s = 60 if not self.low_memory_mode else 40
            sigma_r = 0.4 if not self.low_memory_mode else 0.3
            # Use bilateral filter which better preserves edges than edgePreservingFilter
            # and minimizes banding in smooth areas
            left_view = cv2.bilateralFilter(left_view, d=9, sigmaColor=sigma_r*100, sigmaSpace=sigma_s)
            
        if np.any(right_mask):
            sigma_s = 60 if not self.low_memory_mode else 40
            sigma_r = 0.4 if not self.low_memory_mode else 0.3
            right_view = cv2.bilateralFilter(right_view, d=9, sigmaColor=sigma_r*100, sigmaSpace=sigma_s)
        
        # Convert back to uint8 at the very end to minimize rounding errors
        left_view_uint8 = np.clip(left_view, 0, 255).astype(np.uint8)
        right_view_uint8 = np.clip(right_view, 0, 255).astype(np.uint8)
        
        # Update holes after boundary processing
        left_holes = ~left_mask
        right_holes = ~right_mask
        
        return left_view_uint8, right_view_uint8, left_holes, right_holes
    
    def basic_infill(self, image, hole_mask):
        """Basic inpainting using OpenCV's inpainting algorithm"""
        # Convert hole_mask to uint8 format required by inpaint
        hole_mask_uint8 = hole_mask.astype(np.uint8) * 255
        
        # Apply inpainting
        return cv2.inpaint(image, hole_mask_uint8, 3, cv2.INPAINT_TELEA)
    
    def _process_single_patch(self, image, hole_mask):
        """Process a single image patch with stable diffusion inpainting"""
        # Convert image to RGB PIL format
        if image.shape[2] == 3 and image.dtype == np.uint8:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(image)
        
        # Convert hole mask to PIL
        mask_pil = Image.fromarray((hole_mask * 255).astype(np.uint8))
        
        # Store original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Check if we need to resize
        orig_size = img_pil.size
        need_resize = img_pil.width > self.max_resolution or img_pil.height > self.max_resolution
        
        if need_resize:
            # Calculate new size while maintaining aspect ratio
            aspect_ratio = img_pil.width / img_pil.height
            if img_pil.width > img_pil.height:
                new_width = self.max_resolution
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = self.max_resolution
                new_width = int(new_height * aspect_ratio)
            
            # Resize image and mask
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
        
        try:
            # Run inpainting with a generic prompt
            result = self.inpaint_model(
                prompt="a complete image with natural background",
                image=img_pil,
                mask_image=mask_pil,
                guidance_scale=7.5,
                num_inference_steps=20
            ).images[0]
            
            # Resize back if needed
            if need_resize:
                result = result.resize(orig_size, Image.LANCZOS)
            
            # Convert back to numpy
            result_np = np.array(result)
            if image.shape[2] == 3 and image.dtype == np.uint8:
                result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # Make sure the result shape matches the original image shape
            if result_np.shape[:2] != (orig_height, orig_width):
                print(f"Resizing result from {result_np.shape} to {image.shape}")
                result_np = cv2.resize(result_np, (orig_width, orig_height))
            
            return result_np
            
        except Exception as e:
            print(f"Error in inpainting: {str(e)}")
            print(f"Falling back to basic inpainting for this patch")
            return self.basic_infill(image, hole_mask)
            
    def advanced_infill(self, img, mask, efficient_mode=False):
        """Fill in holes using advanced inpainting with Stable Diffusion."""
        # Make sure we're working with RGB images for stable-diffusion
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make sure mask is binary (0 or 255)
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        # Create PIL images
        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(binary_mask)
        
        if efficient_mode:
            # Use efficient mode with smaller patch size and fewer steps
            inpainted_image = self.inpaint_diffusion(
                img_pil, mask_pil, 
                steps=self.inpainting_steps // 2, 
                guidance_scale=self.inpainting_guidance_scale,
                patch_size=self.inpainting_patch_size // 2,
                patch_overlap=self.inpainting_patch_overlap // 2,
                high_quality=False
            )
        else:
            # Use full quality with specified parameters
            inpainted_image = self.inpaint_diffusion(
                img_pil, mask_pil, 
                steps=self.inpainting_steps, 
                guidance_scale=self.inpainting_guidance_scale,
                patch_size=self.inpainting_patch_size,
                patch_overlap=self.inpainting_patch_overlap,
                high_quality=self.inpainting_high_quality
            )
        
        # Convert back to BGR for OpenCV operations
        inpainted_bgr = cv2.cvtColor(np.array(inpainted_image), cv2.COLOR_RGB2BGR)
        
        return inpainted_bgr

    def fill_holes_preserving_originals(self, original_img, stereo_view, holes_mask, depth_map, 
                                      shift_factor=0.03, is_left_view=True):
        """Fill holes in stereo view while preserving the original image content where possible.
        
        This method is used by the Gradio interface for better quality results.
        
        Args:
            original_img: Original non-shifted image
            stereo_view: Left or right view with potential holes
            holes_mask: Binary mask indicating holes (1 where holes exist)
            depth_map: The depth map used for generating the stereo views
            shift_factor: The stereo shift amount used (default: 0.03)
            is_left_view: Whether this is the left view (True) or right view (False)
            
        Returns:
            The stereo view with holes filled
        """
        # If no holes, return the original view
        if np.sum(holes_mask) == 0:
            return stereo_view
            
        # Make a copy and convert to float32 to avoid precision loss
        result = stereo_view.astype(np.float32)
        original_float = original_img.astype(np.float32)
        
        # Convert binary mask to uint8 format needed for inpainting (0 or 255)
        binary_mask = np.where(holes_mask > 0, 255, 0).astype(np.uint8)
        
        # For areas where the shift is small (based on depth), we can use original content
        # Calculate shift amounts based on depth and direction
        h, w = depth_map.shape[:2]
        shift_direction = -1 if is_left_view else 1  # -1 for left view, 1 for right view
        max_shift_px = int(shift_factor * w)  # Maximum shift in pixels
        
        # Create a shift map: how many pixels each point moved
        shift_map = np.zeros_like(depth_map, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                # Calculate shift for this pixel based on depth
                depth_val = depth_map[y, x]
                shift_amount = shift_direction * depth_val * max_shift_px
                shift_map[y, x] = abs(shift_amount)  # Store absolute shift amount
        
        # Normalize shift map to 0-1 range
        if np.max(shift_map) > 0:
            shift_map = shift_map / np.max(shift_map)
        
        # Create threshold mask: areas with small shifts where original content can be used
        small_shift_threshold = 0.2  # Pixels with shifts < 20% of max can use original content
        small_shift_mask = (shift_map < small_shift_threshold).astype(np.uint8) * 255
        
        # Combine the holes mask with small shift mask to get areas to fill from original
        fill_from_original = cv2.bitwise_and(binary_mask, small_shift_mask)
        
        # For remaining holes, use advanced inpainting
        remaining_holes = cv2.bitwise_and(binary_mask, cv2.bitwise_not(fill_from_original))
        
        # Fill small-shift holes from original image
        if np.sum(fill_from_original) > 0:
            # Create a mask for the original image pixels to copy
            original_mask = fill_from_original.astype(bool)
            result[original_mask] = original_float[original_mask]
            
            # Apply a small amount of blending around these filled areas to prevent hard borders
            kernel = np.ones((3,3), np.uint8)
            blend_border = cv2.dilate(fill_from_original, kernel, iterations=1) - fill_from_original
            blend_border_mask = blend_border.astype(bool)
            
            if np.any(blend_border_mask):
                # Create a blended version in these border areas
                for y, x in zip(*np.where(blend_border_mask)):
                    # Get neighboring pixels from both original and result
                    neighbors_original = []
                    neighbors_result = []
                    
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if fill_from_original[ny, nx]:
                                    neighbors_original.append(original_float[ny, nx])
                                elif not binary_mask[ny, nx]:  # Not a hole
                                    neighbors_result.append(result[ny, nx])
                    
                    if neighbors_original and neighbors_result:
                        # Blend values from original and result
                        avg_original = np.mean(neighbors_original, axis=0)
                        avg_result = np.mean(neighbors_result, axis=0)
                        result[y, x] = (avg_original + avg_result) / 2
        
        # Fill remaining holes with advanced inpainting if any exist
        if np.sum(remaining_holes) > 0:
            # Convert remaining_holes to proper mask format
            remaining_holes_mask = remaining_holes.astype(np.uint8)
            
            # Convert back to uint8 for inpainting
            result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
            
            # Apply advanced inpainting to remaining holes
            inpainted = self.advanced_infill(result_uint8, remaining_holes_mask, efficient_mode=False)
            
            # Convert inpainted back to float32
            inpainted_float = inpainted.astype(np.float32)
            
            # Update result with inpainted content
            result[remaining_holes_mask > 0] = inpainted_float[remaining_holes_mask > 0]
        
        # Convert back to uint8 as the final step
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def set_inpainting_params(self, steps=20, guidance_scale=7.5, patch_size=128, patch_overlap=32, high_quality=True):
        """Set the inpainting parameters.
        
        Args:
            steps: Number of diffusion steps (higher = more quality but slower)
            guidance_scale: How closely to follow the prompt (higher = more fidelity)
            patch_size: Size of patches for inpainting (larger = better context but more memory)
            patch_overlap: Overlap between patches (larger = smoother transitions)
            high_quality: Whether to use high quality settings
        """
        self.inpainting_steps = steps
        self.inpainting_guidance_scale = guidance_scale
        self.inpainting_patch_size = patch_size
        self.inpainting_patch_overlap = patch_overlap
        self.inpainting_high_quality = high_quality
        
    def set_color_quality(self, high_color_quality=True, apply_dithering=False, dithering_level=1.0):
        """Configure color quality settings.
        
        Args:
            high_color_quality: Whether to use high bit depth processing to avoid banding
            apply_dithering: Whether to apply dithering to reduce banding in gradients
            dithering_level: Strength of dithering (1.0 = normal, higher = stronger)
        """
        self.high_color_quality = high_color_quality
        self.apply_dithering = apply_dithering
        self.dithering_level = dithering_level
        
    def _apply_dithering(self, img):
        """Apply dithering to reduce color banding while preserving details"""
        # Convert to float32 for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Enhance local contrast slightly to counteract later blurring
        img_enhanced = np.clip((img_float - 0.5) * 1.05 + 0.5, 0, 1)
        
        # Add noise based on dithering level
        noise_amplitude = 1.0 / 255.0 * self.dithering_level
        noise = np.random.normal(0, noise_amplitude, img_enhanced.shape)
        img_dithered = np.clip(img_enhanced + noise, 0, 1)
        
        # Apply very slight blur to blend the noise pattern
        # For low memory mode, ensure this is a very efficient operation
        if not self.low_memory_mode:
            img_dithered = cv2.GaussianBlur(img_dithered, (0, 0), 0.3)
        
        # Convert back to uint8
        return (img_dithered * 255).astype(np.uint8)

    def _enhance_image_quality(self, img):
        """Enhance image quality with focus on minimizing banding"""
        # Skip enhancement if disabled
        if not self.high_color_quality:
            return img
        
        # Apply dithering if enabled
        if self.apply_dithering:
            img = self._apply_dithering(img)
        
        return img

    def _enhanced_anti_banding(self, img):
        """Apply advanced anti-banding techniques with 16-bit color precision"""
        # Convert to float32 for higher bit depth processing
        img_float = img.astype(np.float32) / 255.0
        
        # Check for areas prone to banding (gradients with subtle changes)
        # Calculate local variance to identify gradient areas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        local_var = cv2.GaussianBlur(gray**2, (5, 5), 0) - cv2.GaussianBlur(gray, (5, 5), 0)**2
        
        # Create a mask for areas with low variance (subtle gradients prone to banding)
        banding_prone = (local_var < 0.001) & (local_var > 0.00001)
        
        # Apply stronger dithering only to banding-prone areas
        if np.any(banding_prone):
            # Create a 3-channel mask
            mask_3ch = np.stack([banding_prone]*3, axis=2)
            
            # Generate high-quality blue noise instead of random noise
            # (Simulating blue noise with filtered white noise)
            noise = np.random.normal(0, 0.5/255.0 * self.dithering_level, img_float.shape)
            blue_noise = cv2.GaussianBlur(noise, (0, 0), 1.5) - cv2.GaussianBlur(noise, (0, 0), 0.5)
            blue_noise *= 2.0 * self.dithering_level  # Amplify the effect
            
            # Apply the blue noise selectively to banding-prone areas
            dithered = np.clip(img_float + blue_noise * mask_3ch.astype(np.float32), 0, 1)
            
            # Blend result with original based on mask
            result = img_float * (1 - mask_3ch.astype(np.float32) * 0.7) + dithered * mask_3ch.astype(np.float32) * 0.7
        else:
            # No banding-prone areas detected
            result = img_float
            
        # Apply a very light blur to smooth the noise while preserving details
        if not self.low_memory_mode:
            result = cv2.bilateralFilter(result, d=5, sigmaColor=0.01, sigmaSpace=5)
        
        # Re-apply any lost local contrast
        if not self.low_memory_mode:
            gray_new = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            detail = gray - cv2.GaussianBlur(gray, (5, 5), 0)
            result_enhanced = result + np.stack([detail] * 3, axis=2) * 0.2
            result = np.clip(result_enhanced, 0, 1)
            
        # Convert back to uint8
        return (result * 255).astype(np.uint8)

    def generate_sbs_stereo(self, image_path, output_path=None, shift_factor=0.05, efficient_mode=True):
        """Generate side-by-side stereo 3D image from 2D image"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        # Use higher precision internal processing (float32)
        image_float = image.astype(np.float32)
        
        # Estimate depth
        depth_map = self.estimate_depth(image)
        
        # Generate stereo views
        left_view, right_view, left_holes, right_holes = self.generate_stereo_views(
            image, depth_map, shift_factor
        )
        
        # Apply inpainting to fill holes
        if self.use_advanced_infill:
            print("Applying advanced inpainting...")
            if np.any(left_holes):
                print(f"  Inpainting left view with {np.sum(left_holes)} hole pixels")
                # Try to use fill_holes_preserving_originals for better quality
                try:
                    left_view = self.fill_holes_preserving_originals(
                        image, left_view, left_holes, depth_map,
                        shift_factor=shift_factor, is_left_view=True
                    )
                except Exception as e:
                    print(f"Error using preserving method: {e}, falling back to standard advanced infill")
                    left_view = self.advanced_infill(left_view, left_holes, efficient_mode)
                    
            if np.any(right_holes):
                print(f"  Inpainting right view with {np.sum(right_holes)} hole pixels")
                # Try to use fill_holes_preserving_originals for better quality
                try:
                    right_view = self.fill_holes_preserving_originals(
                        image, right_view, right_holes, depth_map,
                        shift_factor=shift_factor, is_left_view=False
                    )
                except Exception as e:
                    print(f"Error using preserving method: {e}, falling back to standard advanced infill")
                    right_view = self.advanced_infill(right_view, right_holes, efficient_mode)
        else:
            # Fallback to basic inpainting if advanced is not available
            if np.any(left_holes):
                left_view = self.basic_infill(left_view, left_holes)
            if np.any(right_holes):
                right_view = self.basic_infill(right_view, right_holes)
        
        # Apply color quality enhancement with stronger anti-banding
        try:
            # Skip anti-banding if high_color_quality is False
            if self.high_color_quality:
                if hasattr(self, '_enhanced_anti_banding'):
                    left_view = self._enhanced_anti_banding(left_view)
                    right_view = self._enhanced_anti_banding(right_view)
                else:
                    left_view = self._enhance_image_quality(left_view)
                    right_view = self._enhance_image_quality(right_view)
        except Exception as e:
            print(f"Enhanced anti-banding failed: {e}, falling back to basic enhancement")
            left_view = self._enhance_image_quality(left_view)
            right_view = self._enhance_image_quality(right_view)
        
        # Combine into side-by-side stereo image
        stereo_sbs = np.hstack((left_view, right_view))
        
        # Save if output path provided
        if output_path:
            # Apply PNG compression for best quality
            if output_path.lower().endswith('.png'):
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 0 = no compression
                cv2.imwrite(output_path, stereo_sbs, compression_params)
            else:
                cv2.imwrite(output_path, stereo_sbs)
        
        return stereo_sbs, left_view, right_view, depth_map
    
    def visualize_depth(self, depth_map):
        """Create a colored visualization of the depth map"""
        # Apply color map to create a visual representation
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )
        return depth_colored
        
    def visualize_results(self, image, stereo_sbs, left_view, right_view, depth_map):
        """Visualize the conversion process results"""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("Depth Map")
        plt.imshow(depth_map, cmap='plasma')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("Left View")
        plt.imshow(cv2.cvtColor(left_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("Right View")
        plt.imshow(cv2.cvtColor(right_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("Side-by-Side Stereo")
        plt.imshow(cv2.cvtColor(stereo_sbs, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_stereo_views_debug(self, image, depth_map, shift_factor=0.05, bg_color=(128, 0, 128)):
        """
        Generate left and right views with a debug colored background.
        This version starts with a colored background and overlays pixels from far to near,
        making it easy to see where holes are (they remain the background color).
        
        Args:
            image: Input image
            depth_map: Depth map (normalized 0-1)
            shift_factor: Amount of parallax shift (0.03-0.1 typical)
            bg_color: Background color to use (default: purple)
            
        Returns:
            left_view, right_view, left_holes, right_holes
        """
        height, width = depth_map.shape[:2]
        
        # Create coordinate maps
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Create left and right views with colored background
        # Convert bg_color to a 3-channel array of the specified color
        bg_color_array = np.array(bg_color, dtype=np.uint8)
        left_view = np.ones_like(image) * bg_color_array
        right_view = np.ones_like(image) * bg_color_array
        
        # Generate masks to track pixels that have been filled
        left_mask = np.zeros((height, width), dtype=bool)
        right_mask = np.zeros((height, width), dtype=bool)
        
        # Calculate shifts based on depth map
        shifts = (depth_map * shift_factor * width).astype(int)
        
        # Process depths from farthest to nearest (opposite of normal)
        # This ensures that closer objects occlude farther ones
        depth_levels = np.unique(shifts)
        depth_levels.sort()  # Sort from smallest shift (far) to largest (near)
        
        print(f"Debug mode: Processing {len(depth_levels)} depth levels from far to near")
        
        # Generate views (far to near)
        for d in depth_levels:
            # Get mask of pixels at this depth
            mask = (shifts == d)
            
            # For right view: shift left 
            x_coords_right = np.clip(x_coords - d, 0, width-1)
            # For left view: shift right
            x_coords_left = np.clip(x_coords + d, 0, width-1)
            
            # Fill right view (where not already filled)
            fill_mask_right = mask & ~right_mask[y_coords, x_coords_right]
            right_view[y_coords[fill_mask_right], x_coords_right[fill_mask_right]] = image[y_coords[fill_mask_right], x_coords[fill_mask_right]]
            right_mask[y_coords[fill_mask_right], x_coords_right[fill_mask_right]] = True
            
            # Fill left view (where not already filled)
            fill_mask_left = mask & ~left_mask[y_coords, x_coords_left]
            left_view[y_coords[fill_mask_left], x_coords_left[fill_mask_left]] = image[y_coords[fill_mask_left], x_coords[fill_mask_left]]
            left_mask[y_coords[fill_mask_left], x_coords_left[fill_mask_left]] = True
        
        # Holes are where the mask is still False
        left_holes = ~left_mask
        right_holes = ~right_mask
        
        print(f"Debug: Left view has {np.sum(left_holes)} purple holes, Right view has {np.sum(right_holes)} purple holes")
        
        return left_view, right_view, left_holes, right_holes
    
    def fill_holes(self, img_view, holes_mask, process_only_holes=False):
        """
        Fill holes in the image using inpainting.
        
        Args:
            img_view: The view image with holes
            holes_mask: Boolean mask where True indicates holes
            process_only_holes: Whether to only process hole regions
            
        Returns:
            The filled image
        """
        print(f"Filling holes in view ({np.sum(holes_mask)} pixels)...")
        
        if np.sum(holes_mask) == 0:
            return img_view
        
        try:
            # Try to use advanced infill if available
            from core.advanced_infill import AdvancedInfillTechniques
            advanced_infill = AdvancedInfillTechniques(
                max_resolution=self.max_resolution,
                patch_size=self.inpaint_patch_size, 
                patch_overlap=self.inpaint_patch_overlap,
                batch_size=1,
                inference_steps=self.inpaint_steps
            )
            
            # Process hole regions
            if process_only_holes:
                result = advanced_infill._process_only_hole_regions(img_view, holes_mask)
            else:
                result = advanced_infill.process_image(
                    img_view, 
                    holes_mask, 
                    patch_size=self.inpaint_patch_size, 
                    patch_overlap=self.inpaint_patch_overlap, 
                    high_quality=self.high_color_quality
                )
            return result
        except Exception as e:
            print(f"Advanced infill failed: {e}, falling back to basic inpainting")
            # Fall back to basic inpainting
            return self.basic_infill(img_view, holes_mask)
            
    def generate_sbs_3d(self, img, depth_map, output_size=(1920, 1080), offset_factor=0.05, fill_method="basic"):
        """Generate side-by-side 3D from image and depth map"""
        # Create left and right views with specified offset
        left_view, right_view = self._generate_stereoscopic_pair(img, depth_map, offset_factor, fill_method)
        
        # Resize views to maintain target aspect ratio and resolution
        h, w = output_size
        left_view_resized = cv2.resize(left_view, (w//2, h))
        right_view_resized = cv2.resize(right_view, (w//2, h))
        
        # Combine views side by side
        sbs_3d = np.hstack((left_view_resized, right_view_resized))
        
        return sbs_3d
        
    def clear_vram_cache(self):
        """Clear GPU VRAM cache to free up memory"""
        if torch.cuda.is_available():
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Optionally force garbage collection
            gc.collect()
            return "VRAM cache cleared successfully"
        else:
            return "No CUDA device available, nothing to clear"
    
    def inpaint_diffusion(self, img_pil, mask_pil, steps=20, guidance_scale=7.5, 
                        patch_size=128, patch_overlap=32, high_quality=True):
        """Apply diffusion-based inpainting using the advanced_infill module.
        
        This method is used to fill holes in stereo views with realistic content.
        
        Args:
            img_pil: PIL Image with holes to be filled
            mask_pil: PIL Image mask where white (255) indicates holes
            steps: Number of diffusion steps (higher = better quality, slower)
            guidance_scale: How closely to follow the prompt (7.5 is good default)
            patch_size: Size of patches for inpainting
            patch_overlap: Overlap between patches
            high_quality: Whether to use high quality settings
            
        Returns:
            PIL Image with holes filled in
        """
        if not hasattr(self, 'advanced_infiller'):
            # Import here to avoid loading models unnecessarily
            try:
                from core.advanced_infill import AdvancedInfillTechniques
                self.advanced_infiller = AdvancedInfillTechniques(
                    max_resolution=self.max_resolution,
                    patch_size=patch_size,
                    patch_overlap=patch_overlap,
                    inference_steps=steps
                )
            except ImportError as e:
                print(f"Error loading advanced infill module: {e}")
                # Fall back to using OpenCV inpainting
                img_np = np.array(img_pil)
                mask_np = np.array(mask_pil)
                inpainted = cv2.inpaint(img_np, mask_np, 3, cv2.INPAINT_TELEA)
                return Image.fromarray(inpainted)
                
        try:
            # Convert to numpy if advanced_infiller expects numpy arrays
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)
            
            # Check if mask is binary
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0
                
            # Check if multimodel_ensemble_infill method exists and use it
            if hasattr(self.advanced_infiller, 'multimodel_ensemble_infill'):
                inpainted = self.advanced_infiller.multimodel_ensemble_infill(
                    img_np, mask_np, efficient_mode=not high_quality
                )
            # Otherwise try the _full_image_depth_aware_infill method
            elif hasattr(self.advanced_infiller, '_full_image_depth_aware_infill'):
                inpainted = self.advanced_infiller._full_image_depth_aware_infill(
                    img_np, mask_np, prompt="photo realistic"
                )
            # If neither exists, use sdxl_inpaint if it exists
            elif hasattr(self.advanced_infiller, 'sdxl_inpaint'):
                # Need to convert back to PIL for the SDXL pipeline
                result = self.advanced_infiller.sdxl_inpaint(
                    prompt="photo realistic",
                    image=img_pil,
                    mask_image=mask_pil,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps
                ).images[0]
                return result
            else:
                # Fall back to OpenCV inpainting
                inpainted = cv2.inpaint(img_np, (mask_np*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            
            # Convert back to PIL if we got numpy array
            if isinstance(inpainted, np.ndarray):
                return Image.fromarray(inpainted)
            return inpainted
            
        except Exception as e:
            print(f"Error in inpaint_diffusion: {e}")
            # Fall back to OpenCV inpainting
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)
            inpainted = cv2.inpaint(img_np, (mask_np).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            return Image.fromarray(inpainted)

    def set_color_quality(self, high_color_quality=True, apply_dithering=True, dithering_level=1.0):
        """Set parameters for color quality enhancement"""
        self.high_color_quality = high_color_quality
        self.apply_dithering = apply_dithering
        self.dithering_level = dithering_level
        return self 