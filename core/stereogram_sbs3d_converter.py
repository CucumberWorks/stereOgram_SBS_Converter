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
    def __init__(self, use_advanced_infill=False, depth_model_type="depth_anything_v2", model_size="vitb", max_resolution=4096, low_memory_mode=False):
        # Initialize parameters
        self.use_advanced_infill = use_advanced_infill
        self.depth_model_type = depth_model_type
        self.model_size = model_size
        self.max_resolution = max_resolution
        self.low_memory_mode = low_memory_mode
        self.high_color_quality = True  # Always enable high color quality
        
        # Initialize resources when needed
        # Check if MPS is available (Apple Silicon) and handle it specially
        if torch.backends.mps.is_available():
            print("Using Apple MPS (Metal Performance Shaders)")
            # For MPS, we need to ensure consistent tensor types
            self.device = torch.device("mps")
            self.use_cpu_for_model_load = True
            
            # On some MPS setups, advanced inpainting causes problems
            # If you want to completely disable advanced inpainting on MPS, uncomment this line:
            # self.use_advanced_infill = False
            # print("Disabled advanced inpainting on MPS for stability")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.use_cpu_for_model_load = False
            
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
        # For MPS compatibility, we may need to load the model on CPU first
        device_for_loading = torch.device("cpu") if self.use_cpu_for_model_load else self.device
        
        if self.depth_model_type == "midas":
            # MiDaS depth estimation model
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.depth_model.to(device_for_loading)
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
                self.depth_model.to(device_for_loading)
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
                self.depth_model.to(device_for_loading)
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
                    self.depth_model.to(device_for_loading)
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
                
                # Load the weights - for MPS, we need to load on CPU first
                try:
                    if self.use_cpu_for_model_load:
                        print("Loading model weights on CPU first for MPS compatibility")
                        self.depth_model.load_state_dict(torch.load(weight_path, map_location="cpu"))
                        self.depth_model.to(device_for_loading)  # Keep on CPU for now
                    else:
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
                self.depth_model.to(device_for_loading)
                self.depth_model.eval()
                
                # MiDaS transformation
                self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = self.midas_transforms.small_transform
    
    def _init_inpainting_model(self):
        """Initialize inpainting model for advanced hole filling"""
        if self.use_advanced_infill:
            try:
                # Import here to avoid circular imports
                import torch
                
                # Use device for loading
                device_for_loading = torch.device("cpu") if self.use_cpu_for_model_load else self.device
                
                # Load the stable diffusion inpainting model
                try:
                    # First try stable-diffusion-2-inpainting
                    model_path = "stabilityai/stable-diffusion-2-inpainting"
                    print(f"Loading inpainting model from {model_path}")
                    
                    # For MPS compatibility, we need to avoid certain optimizations
                    use_auth_token = False
                    extra_kwargs = {}
                    
                    # Check for torch version to handle compatibility
                    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
                    if torch_version >= (2, 0) and torch.backends.mps.is_available():
                        print("Using torch 2.0+ with MPS - applying special handling")
                        extra_kwargs["variant"] = "fp32"  # Force fp32 for MPS
                    
                    self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        use_auth_token=use_auth_token,
                        **extra_kwargs
                    )
                    
                except Exception as e:
                    # If that fails, try the older model
                    print(f"Error loading SD2 model: {e}, trying fallback model")
                    model_path = "runwayml/stable-diffusion-inpainting"
                    print(f"Loading fallback inpainting model from {model_path}")
                    self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        use_auth_token=False
                    )
                
                # Disable safety checker to save memory
                if hasattr(self.inpaint_model, "safety_checker") and self.inpaint_model.safety_checker is not None:
                    self.inpaint_model.safety_checker = None
                    print("Disabled safety checker to save memory")
                
                # Move to proper device
                if device_for_loading == torch.device("mps"):
                    # Special handling for MPS
                    print("Setting up model for MPS")
                    self.inpaint_model.to("mps")
                else:
                    self.inpaint_model.to(device_for_loading)
                
                # Use float16 if not using CPU and low memory mode is enabled
                if device_for_loading.type != "cpu" and self.low_memory_mode and device_for_loading.type != "mps":
                    print("Using float16 for inpainting model to reduce memory usage")
                    try:
                        # Only convert to float16 if cuda is available and supports it
                        if torch.cuda.is_available():
                            import torch.backends.cuda
                            if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
                                self.inpaint_model.to(torch.float16)
                                print("Converted inpainting model to float16")
                    except Exception as e:
                        print(f"Could not convert to float16: {e}")
                
                print(f"Inpainting model loaded and set to device: {device_for_loading}")
            except Exception as e:
                print(f"Error loading inpainting model: {e}")
                import traceback
                traceback.print_exc()
                print("Advanced inpainting will not be available")
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
            
            try:
                # For MPS compatibility, when using MPS, potentially move model to device
                if self.use_cpu_for_model_load and torch.backends.mps.is_available():
                    # At inference time, move the model to MPS
                    print("Moving model to MPS for inference")
                    self.depth_model.to(self.device)
                
                # Now call inference
                depth_map = self.depth_model.infer_image(raw_image, input_size=input_size)
                
                # If needed, move model back to CPU for next run to prevent type mismatch
                if self.use_cpu_for_model_load and torch.backends.mps.is_available():
                    print("Moving model back to CPU after inference")
                    self.depth_model.to(torch.device("cpu"))
                    torch.mps.empty_cache()
                    
            except RuntimeError as e:
                if "Input type" in str(e) and "weight type" in str(e) and torch.backends.mps.is_available():
                    print("Detected MPS tensor type mismatch, retrying with model on CPU")
                    # Move model to CPU for execution
                    self.depth_model.to(torch.device("cpu"))
                    # Run inference on CPU
                    depth_map = self.depth_model.infer_image(raw_image, input_size=input_size)
                else:
                    # Re-raise other exceptions
                    raise
            
        elif self.depth_model_type == "depth_anything":
            # Process with Depth Anything
            # Use original image without resizing
            pil_image_resized = pil_image
                
            # Apply transform and move to device
            img_input = self.transform(pil_image_resized).unsqueeze(0)
            
            # Handle MPS compatibility
            if self.use_cpu_for_model_load and torch.backends.mps.is_available():
                # Use CPU for inference if needed for compatibility
                img_input = img_input.to(torch.device("cpu"))
                self.depth_model.to(torch.device("cpu"))
            else:
                img_input = img_input.to(self.device)
            
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
            
            # Move model back to proper device if needed
            if self.use_cpu_for_model_load and torch.backends.mps.is_available():
                self.depth_model.to(torch.device("cpu"))  # Keep on CPU for next use
                torch.mps.empty_cache()
            
        else:
            # Process with MiDaS (original method)
            img_input = self.transform(img_array)
            
            # Handle MPS compatibility
            if self.use_cpu_for_model_load and torch.backends.mps.is_available():
                # Use CPU for inference if needed for compatibility
                img_input = img_input.to(torch.device("cpu"))
                self.depth_model.to(torch.device("cpu"))
            else:
                img_input = img_input.to(self.device)
            
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
            
            # Move model back to proper device if needed
            if self.use_cpu_for_model_load and torch.backends.mps.is_available():
                self.depth_model.to(torch.device("cpu"))  # Keep on CPU for next use
                torch.mps.empty_cache()
        
        # Normalize depth map to 0-1 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Apply additional smoothing to further reduce harsh transitions
        depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
        
        return depth_normalized
    
    def generate_stereo_views(self, image, depth_map, shift_factor=0.05, apply_depth_blur=False, 
                             focal_distance=0.5, focal_thickness=0.1, blur_strength=1.0, max_blur_size=21,
                             aperture_shape='circle', highlight_boost=1.5, enable_dithering=True, chromatic_aberration=2.0,
                             edge_smoothness=2.0):
        """Generate left and right views based on depth map
        
        This enhanced version applies depth-based blur using parallax-shifted depth maps.
        
        Args:
            image: Input image
            depth_map: Depth map (0-1 range)
            shift_factor: Amount of shift for stereo effect
            apply_depth_blur: Whether to apply depth-based blur
            focal_distance: Distance to keep in focus (0-1, where 0 is closest, 1 is farthest)
            focal_thickness: Thickness of the in-focus region (0-1)
            blur_strength: Strength of the blur effect (multiplier)
            max_blur_size: Maximum kernel size for blur
            aperture_shape: Shape of the bokeh ('circle', 'hexagon', 'octagon')
            highlight_boost: How much to boost bright areas (1.0 = no boost)
            enable_dithering: Whether to apply dithering to prevent banding
            chromatic_aberration: Amount of color channel separation (0.0 = none)
            edge_smoothness: Controls the smoothness of transitions at depth boundaries (0.5-2.0)
        """
        height, width = depth_map.shape[:2]
        
        # For better color precision, convert to float32 early in the process
        image_float = image.astype(np.float32)
        
        # Create coordinate maps
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # STEP 1: CALCULATE PIXEL SHIFTS BASED ON DEPTH
        # Apply strong blur to depth map for transitions
        smoothed_depth = cv2.GaussianBlur(depth_map, (9, 9), 0)
        
        # Calculate shifts based on smoothed depth map to prevent harsh transitions
        shifts = (smoothed_depth * shift_factor * width).astype(int)
        
        # STEP 2: CREATE BASIC STEREO VIEWS WITHOUT BLUR
        # Create basic left and right views without blur
        left_view = np.zeros_like(image_float)
        right_view = np.zeros_like(image_float)
        
        # Create parallax-shifted depth maps for left and right views
        left_depth = np.zeros_like(depth_map, dtype=np.float32)
        right_depth = np.zeros_like(depth_map, dtype=np.float32)
        
        # Generate masks to track pixels that have been filled
        left_mask = np.zeros((height, width), dtype=bool)
        right_mask = np.zeros((height, width), dtype=bool)
        
        # Get unique depth values and sort them from far to near
        unique_shifts = np.unique(shifts)
        unique_shifts.sort()  # Sort from smallest shift (far) to largest (near)
        
        # Generate stereo views and shifted depth maps (back to front)
        for d in unique_shifts:
            # Create a mask for pixels at this depth level
            depth_mask = (shifts == d)
            
            # If no pixels at this depth, skip
            if not np.any(depth_mask):
                continue
                
            # For right view: shift left
            # For left view: shift right
            for y, x in zip(*np.where(depth_mask)):
                # Calculate shifted x-coordinates
                left_x = min(width-1, x + d)
                right_x = max(0, x - d)
                
                # Fill left view if not already filled
                if not left_mask[y, left_x]:
                    left_view[y, left_x] = image_float[y, x]
                    left_depth[y, left_x] = depth_map[y, x]
                    left_mask[y, left_x] = True
                
                # Fill right view if not already filled
                if not right_mask[y, right_x]:
                    right_view[y, right_x] = image_float[y, x]
                    right_depth[y, right_x] = depth_map[y, x]
                    right_mask[y, right_x] = True
        
        # Fill holes in the stereo views and depth maps
        if np.any(~left_mask):
            left_holes = ~left_mask
            left_view, left_depth = self._fill_view_holes(left_view, left_depth, left_holes)
            left_mask = np.ones_like(left_mask)  # Update mask after filling
        
        if np.any(~right_mask):
            right_holes = ~right_mask
            right_view, right_depth = self._fill_view_holes(right_view, right_depth, right_holes)
            right_mask = np.ones_like(right_mask)  # Update mask after filling
        
        # If not applying depth blur, convert to uint8 and return the basic stereo views
        if not apply_depth_blur:
            left_view_uint8 = np.clip(left_view, 0, 255).astype(np.uint8)
            right_view_uint8 = np.clip(right_view, 0, 255).astype(np.uint8)
            # Recompute holes after hole filling
            left_holes = ~left_mask
            right_holes = ~right_mask
            return left_view_uint8, right_view_uint8, left_holes, right_holes
        
        # STEP 3: APPLY DEPTH BLUR USING PARALLAX-SHIFTED DEPTH MAPS
        # Convert to linear light space for physically accurate color mixing
        gamma = 2.2
        left_linear = np.power(left_view / 255.0, gamma)
        right_linear = np.power(right_view / 255.0, gamma)
        
        # Process left view with depth blur using its parallax-shifted depth map
        left_blurred = self._apply_depth_blur_to_view(
            left_linear, 
            left_depth, 
            focal_distance, 
            focal_thickness, 
            blur_strength, 
            max_blur_size,
            aperture_shape,
            highlight_boost,
            enable_dithering,
            chromatic_aberration,
            edge_smoothness
        )
        
        # Process right view with depth blur using its parallax-shifted depth map
        right_blurred = self._apply_depth_blur_to_view(
            right_linear, 
            right_depth, 
            focal_distance, 
            focal_thickness, 
            blur_strength, 
            max_blur_size,
            aperture_shape,
            highlight_boost,
            enable_dithering,
            chromatic_aberration,
            edge_smoothness
        )
        
        # Convert back to uint8
        left_view_uint8 = np.clip(left_blurred * 255.0, 0, 255).astype(np.uint8)
        right_view_uint8 = np.clip(right_blurred * 255.0, 0, 255).astype(np.uint8)
        
        # No holes after processing
        left_holes = np.zeros_like(left_mask)
        right_holes = np.zeros_like(right_mask)
        
        return left_view_uint8, right_view_uint8, left_holes, right_holes
    
    def _fill_view_holes(self, view, depth_map, holes_mask):
        """Fill holes in view and depth map using nearest neighbor approach"""
        height, width = depth_map.shape[:2]
        filled_view = view.copy()
        filled_depth = depth_map.copy()
        
        for y, x in zip(*np.where(holes_mask)):
            # Find nearest valid pixel
            valid_pixels = []
            search_radius = 5
            for dy in range(-search_radius, search_radius+1):
                for dx in range(-search_radius, search_radius+1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not holes_mask[ny, nx]:
                        dist = np.sqrt(dy*dy + dx*dx)
                        valid_pixels.append((filled_view[ny, nx], filled_depth[ny, nx], dist))
            
            if valid_pixels:
                # Use nearest valid pixel
                valid_pixels.sort(key=lambda x: x[2])
                filled_view[y, x] = valid_pixels[0][0]
                filled_depth[y, x] = valid_pixels[0][1]
        
        return filled_view, filled_depth
    
    def _apply_depth_blur_to_view(self, image, depth_map, focal_distance, focal_thickness, blur_strength, max_blur_size,
                                   aperture_shape='circle', highlight_boost=1.5, enable_dithering=True, chromatic_aberration=0.0,
                                   edge_smoothness=1.0):
        """Apply depth-based blur to a view using its parallax-shifted depth map
        
        Args:
            image: Image in linear light space (0-1 range)
            depth_map: Parallax-shifted depth map (0-1 range)
            focal_distance: Distance to keep in focus (0-1)
            focal_thickness: Thickness of the in-focus region (0-1)
            blur_strength: Strength of the blur effect (multiplier)
            max_blur_size: Maximum kernel size for blur
            aperture_shape: Shape of the bokeh ('circle', 'hexagon', 'octagon')
            highlight_boost: How much to boost bright areas (1.0 = no boost)
            enable_dithering: Whether to apply dithering to prevent banding
            chromatic_aberration: Amount of color channel separation (0.0 = none)
            edge_smoothness: Controls the smoothness of transitions at depth boundaries (0.5-2.0)
            
        Returns:
            Blurred image in linear light space (0-1 range)
        """
        height, width = depth_map.shape[:2]
        result = np.zeros_like(image)
        
        # Detect edges in the depth map for special treatment
        depth_edges = cv2.Canny(
            (depth_map * 255).astype(np.uint8), 
            threshold1=10, 
            threshold2=50
        )
        
        # Create a mask around depth edges
        edge_mask = cv2.dilate(
            depth_edges, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 
            iterations=int(edge_smoothness * 2)
        ).astype(bool)
        
        # Apply bilateral filtering to the depth map for smoother transitions while preserving edges
        depth_map_smooth = cv2.bilateralFilter(
            depth_map, 
            d=int(9 * edge_smoothness),
            sigmaColor=0.1,
            sigmaSpace=int(7 * edge_smoothness)
        )
        
        # Apply additional smoothing at the edges
        if edge_smoothness > 0.5:
            edge_smoothed = cv2.GaussianBlur(
                depth_map_smooth, 
                (int(7 * edge_smoothness) | 1, int(7 * edge_smoothness) | 1),
                0
            )
            blend_factor = np.minimum(1.0, edge_smoothness / 1.5)
            depth_map_smooth[edge_mask] = (
                depth_map_smooth[edge_mask] * (1 - blend_factor) + 
                edge_smoothed[edge_mask] * blend_factor
            )
        
        # Calculate focal plane boundaries for determining blur amount
        half_thickness = focal_thickness / 2.0
        lower_bound = max(0, focal_distance - half_thickness)
        upper_bound = min(1.0, focal_distance + half_thickness)
        
        # Create mask for in-focus region (no blur)
        in_focus_mask = np.logical_and(
            depth_map_smooth >= lower_bound,
            depth_map_smooth <= upper_bound
        )
        
        # Calculate focus distance (how far each pixel is from the in-focus region)
        focus_distance = np.zeros_like(depth_map_smooth)
        
        # Pixels closer than focal plane (foreground)
        foreground_mask = depth_map_smooth < lower_bound
        if np.any(foreground_mask):
            focus_distance[foreground_mask] = lower_bound - depth_map_smooth[foreground_mask]
        
        # Pixels further than focal plane (background)
        background_mask = depth_map_smooth > upper_bound
        if np.any(background_mask):
            focus_distance[background_mask] = depth_map_smooth[background_mask] - upper_bound
        
        # Apply additional smoothing to focus distance map at edges
        if edge_smoothness > 0.5:
            focus_distance_smoothed = cv2.GaussianBlur(
                focus_distance, 
                (int(5 * edge_smoothness) | 1, int(5 * edge_smoothness) | 1),
                0
            )
            blend_factor = np.minimum(1.0, edge_smoothness / 1.5)
            focus_distance[edge_mask] = (
                focus_distance[edge_mask] * (1 - blend_factor) + 
                focus_distance_smoothed[edge_mask] * blend_factor
            )
        
        # Apply dithering to the focus distance to prevent banding
        if enable_dithering:
            # Scale noise by edge smoothness for better transitions
            noise_scale = 0.005 * np.clip(edge_smoothness, 0.5, 2.0)
            noise = np.random.normal(0, noise_scale, focus_distance.shape)
            focus_distance = np.clip(focus_distance + noise, 0, 1)
        
        # Scale the focus distance to get blur kernel sizes
        # Multiply by blur_strength to control effect intensity
        # Ensure odd kernel sizes (required by kernels)
        blur_sizes = np.clip(2 * np.ceil(focus_distance * blur_strength * max_blur_size) + 1, 1, max_blur_size).astype(int)
        
        # Apply additional smoothing to blur sizes at edges
        if edge_smoothness > 0.5:
            blur_sizes_float = blur_sizes.astype(np.float32)
            blur_sizes_smoothed = cv2.GaussianBlur(
                blur_sizes_float, 
                (int(5 * edge_smoothness) | 1, int(5 * edge_smoothness) | 1),
                0
            )
            blur_sizes_smoothed = 2 * np.ceil(blur_sizes_smoothed / 2) - 1
            
            blend_factor = np.minimum(1.0, edge_smoothness / 1.5)
            blur_sizes_float[edge_mask] = (
                blur_sizes_float[edge_mask] * (1 - blend_factor) + 
                blur_sizes_smoothed[edge_mask] * blend_factor
            )
            blur_sizes = np.clip(blur_sizes_float, 1, max_blur_size).astype(int)
        
        # Set blur size to 1 (no blur) for in-focus regions
        blur_sizes[in_focus_mask] = 1
        
        # Function to create bokeh kernel of given shape and size with anti-aliasing
        def create_bokeh_kernel(size, shape='circle', aa_factor=4):
            if size <= 1:
                return np.ones((1, 1), dtype=np.float32)
            
            # For anti-aliasing, create a higher resolution kernel and then downsample
            aa_size = size * aa_factor
            r = aa_size // 2
            
            # Create a coordinate grid ensuring the dimensions are exactly (aa_size, aa_size)
            y = np.linspace(-r, r, aa_size)
            x = np.linspace(-r, r, aa_size)
            xx, yy = np.meshgrid(x, y)
            
            # Initialize mask with correct dimensions
            mask = np.zeros((aa_size, aa_size), dtype=np.float32)
            
            if shape == 'circle':
                # Circular kernel with anti-aliasing
                dist_squared = xx**2 + yy**2
                r_squared = r**2
                
                # Create soft edge
                soft_edge_width = r * 0.1
                inner_r_squared = (r - soft_edge_width)**2
                outer_r_squared = (r + soft_edge_width)**2
                
                # Create base mask (all ones inside inner circle)
                mask = np.ones((aa_size, aa_size), dtype=np.float32)
                
                # Apply soft edge in transition area
                edge_region = (dist_squared > inner_r_squared) & (dist_squared < outer_r_squared)
                if np.any(edge_region):
                    edge_values = 1.0 - (np.sqrt(dist_squared[edge_region]) - (r - soft_edge_width)) / (2 * soft_edge_width)
                    mask[edge_region] = np.clip(edge_values, 0, 1)
                
                # Set outside to zero
                outside_region = (dist_squared >= outer_r_squared)
                mask[outside_region] = 0
                
            elif shape == 'hexagon' or shape == 'octagon':
                # Simplified polygon implementation using OpenCV
                sides = 6 if shape == 'hexagon' else 8
                angles = np.linspace(0, 2*np.pi, sides+1)[:-1]
                
                # Create polygon points
                polygon_points = []
                for angle in angles:
                    x_point = int(r * np.cos(angle) + r)
                    y_point = int(r * np.sin(angle) + r)
                    polygon_points.append([x_point, y_point])
                
                # Create binary mask
                binary_mask = np.zeros((aa_size, aa_size), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [np.array(polygon_points)], 1)
                
                # Apply Gaussian blur for soft edges
                mask = cv2.GaussianBlur(binary_mask.astype(np.float32), 
                                      (int(aa_factor * 0.5) | 1, int(aa_factor * 0.5) | 1), 0)
                
                # Normalize center to 1.0
                if mask[r, r] > 0:
                    mask = mask / mask[r, r]
                
            else:
                # Default to circle
                dist_squared = xx**2 + yy**2
                r_squared = r**2
                
                soft_edge_width = r * 0.1
                inner_r_squared = (r - soft_edge_width)**2
                outer_r_squared = (r + soft_edge_width)**2
                
                # Create base mask
                mask = np.ones((aa_size, aa_size), dtype=np.float32)
                
                # Apply soft edge in transition area
                edge_region = (dist_squared > inner_r_squared) & (dist_squared < outer_r_squared)
                if np.any(edge_region):
                    edge_values = 1.0 - (np.sqrt(dist_squared[edge_region]) - (r - soft_edge_width)) / (2 * soft_edge_width)
                    mask[edge_region] = np.clip(edge_values, 0, 1)
                
                # Set outside to zero
                outside_region = (dist_squared >= outer_r_squared)
                mask[outside_region] = 0
            
            # Downsample to target size
            if aa_factor > 1:
                mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
            
            # Normalize kernel
            kernel_sum = np.sum(mask)
            if kernel_sum > 0:
                mask /= kernel_sum
            
            return mask
        
        # Get unique blur sizes
        unique_blur_sizes = np.unique(blur_sizes)
        
        # If all pixels are in focus or blur strength is zero, return original image
        if len(unique_blur_sizes) == 1 and unique_blur_sizes[0] == 1:
            return image
        
        # Copy in-focus areas directly
        result[in_focus_mask] = image[in_focus_mask]
        
        # Pre-compute bokeh kernels for different sizes
        bokeh_kernels = {}
        
        # Prepare highlight image for bokeh effect
        if highlight_boost > 1.0:
            # Extract highlights by applying gamma correction to emphasize bright areas
            highlights = np.power(image, highlight_boost)
            # Normalize to keep the range in check
            highlights = highlights / np.maximum(1e-5, np.max(highlights))
        else:
            highlights = image.copy()
        
        # For each unique blur size, apply appropriate blur and copy to result
        for blur_size in unique_blur_sizes:
            if blur_size == 1:
                continue  # Skip in-focus areas (already handled)
                
            # Create mask for this blur size
            blur_mask = blur_sizes == blur_size
            if not np.any(blur_mask):
                continue
                
            # Get or create the bokeh kernel for this blur size
            if blur_size not in bokeh_kernels:
                bokeh_kernels[blur_size] = create_bokeh_kernel(blur_size, aperture_shape, aa_factor=4)
            
            # Apply bokeh blur with appropriate kernel
            blurred = cv2.filter2D(image, -1, bokeh_kernels[blur_size])
            
            # Apply chromatic aberration if enabled
            if chromatic_aberration > 0:
                # Apply different blur sizes to R, G, B channels to simulate chromatic aberration
                r_size = max(1, int(blur_size * (1 + chromatic_aberration)))
                b_size = max(1, int(blur_size * (1 - chromatic_aberration)))
                
                # Get or create kernels
                if r_size not in bokeh_kernels:
                    bokeh_kernels[r_size] = create_bokeh_kernel(r_size, aperture_shape, aa_factor=4)
                if b_size not in bokeh_kernels:
                    bokeh_kernels[b_size] = create_bokeh_kernel(b_size, aperture_shape, aa_factor=4)
                
                # Apply different blur to R and B channels
                r_blurred = cv2.filter2D(image[:,:,0], -1, bokeh_kernels[r_size])
                b_blurred = cv2.filter2D(image[:,:,2], -1, bokeh_kernels[b_size])
                
                # Replace channels
                blurred[:,:,0] = r_blurred
                blurred[:,:,2] = b_blurred
            
            # Apply highlight boost for bokeh effect
            if highlight_boost > 1.0:
                # Use a larger kernel for highlights to create beautiful bokeh spots
                highlight_size = min(max_blur_size, blur_size + 4)
                if highlight_size not in bokeh_kernels:
                    bokeh_kernels[highlight_size] = create_bokeh_kernel(highlight_size, aperture_shape, aa_factor=4)
                
                highlights_blurred = cv2.filter2D(highlights, -1, bokeh_kernels[highlight_size])
                
                # Add highlights to blurred image for the bokeh effect
                blurred = np.maximum(blurred, highlights_blurred)
            
            # Copy the blurred pixels to the result for this blur level
            result[blur_mask] = blurred[blur_mask]
        
        # Apply a final bilateral filter to edges for smoother transitions
        if edge_smoothness > 0.8:
            # Create a filtered version for edges
            edge_filtered = cv2.bilateralFilter(
                result,
                d=int(5 * edge_smoothness),
                sigmaColor=0.05,
                sigmaSpace=int(5 * edge_smoothness)
            )
            
            # Only blend at the edges
            blend_factor = np.minimum(1.0, (edge_smoothness - 0.8) / 1.0)
            result[edge_mask] = (
                result[edge_mask] * (1 - blend_factor) +
                edge_filtered[edge_mask] * blend_factor
            )
        
        # Apply subtle dithering to prevent banding
        if enable_dithering:
            # Add very subtle noise to prevent banding in gradients
            noise = np.random.normal(0, 0.002, result.shape)
            result = np.clip(result + noise, 0, 1)
        
        return result
    
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
        
    def set_color_quality(self, high_color_quality=True, apply_dithering=True, dithering_level=1.0):
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
        """Enhance image quality with better color precision"""
        # Apply bilateral filter to reduce noise while preserving edges
        # Parameters tuned for good balance of performance and quality
        enhanced = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        return enhanced
        
    def apply_depth_based_blur(self, image, depth_map, focal_distance=0.5, focal_thickness=0.1, blur_strength=1.0, max_blur_size=21, 
                             aperture_shape='circle', highlight_boost=1.5, enable_dithering=True, chromatic_aberration=0.0,
                             edge_smoothness=1.0):
        """
        Apply realistic depth-of-field blur based on the depth map
        
        This enhanced version:
        1. Processes depth layers from back to front for realistic occlusion
        2. Uses true disk/aperture-shaped kernels instead of Gaussian for realistic bokeh
        3. Enhances bright highlights to simulate lens bokeh effects
        4. Creates a "spread" effect for out-of-focus areas with ultra-smooth transitions
        5. Simulates natural bokeh-style blurring with exponential transparency falloff
        6. Uses linear light space for physically accurate color blending
        7. Applies dithering to prevent banding artifacts
        8. Optional chromatic aberration simulation
        9. Advanced edge detection for smoother transitions at depth boundaries
        
        Args:
            image: Original image to apply blur to
            depth_map: Depth map (0-1 range)
            focal_distance: Distance to keep in focus (0-1, where 0 is closest, 1 is farthest) 
            focal_thickness: Thickness of the in-focus region (0-1)
            blur_strength: Strength of the blur effect (multiplier)
            max_blur_size: Maximum kernel size for blur
            aperture_shape: Shape of the bokeh ('circle', 'hexagon', 'octagon')
            highlight_boost: How much to boost bright areas (1.0 = no boost)
            enable_dithering: Whether to apply dithering to prevent banding
            chromatic_aberration: Amount of color channel separation (0.0 = none)
            edge_smoothness: Controls the smoothness of transitions at depth boundaries (0.5-2.0)
            
        Returns:
            Image with realistic depth-based blur applied
        """
        # Normalize depth map to 0-1 range if needed
        if depth_map.max() > 1.0:
            normalized_depth = depth_map / depth_map.max()
        else:
            normalized_depth = depth_map.copy()
        
        # IMPROVED EDGE HANDLING: Use a more sophisticated edge detection
        # Instead of Canny, use gradient magnitude for more continuous edge detection
        sobelx = cv2.Sobel(normalized_depth, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(normalized_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradient magnitude to create a continuous edge weight map
        if gradient_magnitude.max() > 0:
            edge_weight = gradient_magnitude / gradient_magnitude.max()
        else:
            edge_weight = gradient_magnitude
            
        # Apply smoothing on the edge weight map
        edge_weight = cv2.GaussianBlur(edge_weight, (9, 9), 0)
        
        # Create a gradually decreasing edge influence mask
        edge_influence = np.clip(edge_weight * edge_smoothness * 4, 0, 1)
        
        # Preserve depth edges by applying stronger bilateral filtering
        normalized_depth = cv2.bilateralFilter(
            normalized_depth, 
            d=int(9 * edge_smoothness),  # Diameter of pixel neighborhood
            sigmaColor=0.1,  # Filter sigma in color space
            sigmaSpace=int(7 * edge_smoothness)  # Filter sigma in coordinate space
        )
        
        # Apply additional adaptive smoothing at the edges based on gradient
        if edge_smoothness > 0.5:
            # Create increasingly smoothed versions
            s1 = cv2.GaussianBlur(normalized_depth, (5, 5), 0)
            s2 = cv2.GaussianBlur(normalized_depth, (9, 9), 0)
            s3 = cv2.GaussianBlur(normalized_depth, (15, 15), 0)
            
            # Blend based on edge magnitude - stronger edges get more smoothing
            # This creates a progressive falloff effect
            edge_mask_strong = edge_influence > 0.7
            edge_mask_medium = (edge_influence > 0.4) & (edge_influence <= 0.7)
            edge_mask_light = (edge_influence > 0.1) & (edge_influence <= 0.4)
            
            # Apply progressive smoothing based on edge strength
            if np.any(edge_mask_strong):
                normalized_depth[edge_mask_strong] = s3[edge_mask_strong]
            if np.any(edge_mask_medium):
                normalized_depth[edge_mask_medium] = s2[edge_mask_medium]
            if np.any(edge_mask_light):
                normalized_depth[edge_mask_light] = s1[edge_mask_light]
            
        # Calculate focal plane boundaries
        half_thickness = focal_thickness / 2.0
        lower_bound = max(0, focal_distance - half_thickness)
        upper_bound = min(1.0, focal_distance + half_thickness)
        
        # Create mask for in-focus region (no blur) with soft transition
        in_focus_distance = np.maximum(
            lower_bound - normalized_depth, 
            normalized_depth - upper_bound
        )
        in_focus_distance = np.clip(in_focus_distance, 0, half_thickness)
        in_focus_factor = 1.0 - (in_focus_distance / half_thickness)
        in_focus_mask = in_focus_factor >= 0.95  # Hard cutoff for true in-focus areas
        
        # Calculate focus distance (how far each pixel is from the in-focus region)
        focus_distance = np.zeros_like(normalized_depth)
        
        # Pixels closer than focal plane (foreground)
        foreground_mask = normalized_depth < lower_bound
        if np.any(foreground_mask):
            focus_distance[foreground_mask] = lower_bound - normalized_depth[foreground_mask]
            
        # Pixels further than focal plane (background)
        background_mask = normalized_depth > upper_bound
        if np.any(background_mask):
            focus_distance[background_mask] = normalized_depth[background_mask] - upper_bound
        
        # Apply extra smoothing to the focus distance map at edges
        if edge_smoothness > 0.5:
            focus_distance_smoothed = cv2.GaussianBlur(
                focus_distance, 
                (int(7 * edge_smoothness) | 1, int(7 * edge_smoothness) | 1),  # Ensure odd kernel size
                0
            )
            
            # Blend based on edge influence
            blend_factor = np.clip(edge_influence * 1.5, 0, 1.0)
            focus_distance = (1 - blend_factor) * focus_distance + blend_factor * focus_distance_smoothed
        
        # Apply dithering to the focus distance to prevent banding
        if enable_dithering:
            # Add noise scaled by edge smoothness
            noise_scale = 0.01 * np.clip(edge_smoothness, 0.5, 2.0)
            noise = np.random.normal(0, noise_scale, focus_distance.shape)
            focus_distance = np.clip(focus_distance + noise, 0, 1)
        
        # Scale the focus distance to get blur kernel sizes with smooth progression
        # REVERTED: Remove the enhanced blur strength multiplier
        blur_sizes_float = focus_distance * blur_strength * max_blur_size
        
        # Apply additional smoothing to blur sizes at edges
        if edge_smoothness > 0.5:
            blur_sizes_smoothed = cv2.GaussianBlur(
                blur_sizes_float, 
                (int(5 * edge_smoothness) | 1, int(5 * edge_smoothness) | 1),
                0
            )
            
            # Use the edge influence mask to blend original and smoothed blur sizes
            blend_factor = np.clip(edge_influence * 1.8, 0, 1.0)
            blur_sizes_float = (1 - blend_factor) * blur_sizes_float + blend_factor * blur_sizes_smoothed
        
        # REVERTED: Remove minimum blur enhancement
        # Ensure odd kernel sizes for convolution operations
        blur_sizes = 2 * np.ceil(blur_sizes_float / 2) + 1
        blur_sizes = np.clip(blur_sizes, 1, max_blur_size).astype(int)
        
        # Set blur size to 1 (no blur) for in-focus regions
        blur_sizes[in_focus_mask] = 1
        
        # Initialize output image with alpha channel for smoother blending
        h, w, c = image.shape
        result = np.zeros((h, w, 4), dtype=np.float32)
        
        # Get unique depth values and sort them (from far to near)
        depth_bins = int(50 * np.clip(edge_smoothness, 0.5, 2.5))  # More bins for smoother transitions
        depth_values = np.linspace(1.0, 0.0, depth_bins)  # From far (1.0) to near (0.0)
        
        # Convert to linear light space for physically accurate color mixing
        gamma = 2.2
        float_image = image.astype(np.float32)
        linear_image = np.power(float_image / 255.0, gamma)
        
        # Prepare highlight image for bokeh effect
        if highlight_boost > 1.0:
            # Extract highlights by applying gamma correction to emphasize bright areas
            highlights = np.power(linear_image, highlight_boost)
            # Normalize to keep the range in check, but handle potential division by zero
            max_val = np.maximum(1e-5, np.max(highlights))
            highlights = highlights / max_val
        else:
            highlights = linear_image.copy()
        
        # First, add in-focus areas as the base layer (these are always visible)
        # Convert to 4 channels (RGB + alpha)
        linear_image_rgba = np.zeros((h, w, 4), dtype=np.float32)
        linear_image_rgba[:,:,:3] = linear_image
        linear_image_rgba[:,:,3] = 1.0  # Full opacity
        
        # For smoother transitions, use the in-focus factor instead of a hard mask
        result[:,:,:3] = linear_image
        result[:,:,3] = 1.0  # Initialize all pixels as visible
        
        # Create a mask for truly empty (unfilled) regions
        unfilled_mask = np.ones((h, w), dtype=bool)
        
        # Create a cumulative opacity mask to handle progressive transparency blending
        cumulative_opacity = np.zeros((h, w), dtype=np.float32)
        
        # Pre-compute bokeh kernels for different sizes
        bokeh_kernels = {}
        
        # Function to create bokeh kernel of a given shape and size with anti-aliasing
        def create_bokeh_kernel(size, shape='circle', aa_factor=4):
            if size <= 1:
                return np.ones((1, 1), dtype=np.float32)
            
            # For anti-aliasing, create a higher resolution kernel and then downsample
            aa_size = size * aa_factor
            r = aa_size // 2
            
            # Create a coordinate grid ensuring the dimensions are exactly (aa_size, aa_size)
            y = np.linspace(-r, r, aa_size)
            x = np.linspace(-r, r, aa_size)
            xx, yy = np.meshgrid(x, y)
            
            # Initialize mask with correct dimensions
            mask = np.zeros((aa_size, aa_size), dtype=np.float32)
            
            if shape == 'circle':
                # Circular kernel with anti-aliasing
                dist_squared = xx**2 + yy**2
                r_squared = r**2
                
                # Create soft edge with wider transition
                soft_edge_width = r * 0.15  # Increased edge width for smoother transition
                inner_r_squared = (r - soft_edge_width)**2
                outer_r_squared = (r + soft_edge_width)**2
                
                # Create base mask (all ones inside inner circle)
                mask = np.ones((aa_size, aa_size), dtype=np.float32)
                
                # Apply soft edge in transition area using a smoother falloff
                edge_region = (dist_squared > inner_r_squared) & (dist_squared < outer_r_squared)
                if np.any(edge_region):
                    # Use a sinusoidal falloff for smoother transition
                    t = (np.sqrt(dist_squared[edge_region]) - (r - soft_edge_width)) / (2 * soft_edge_width)
                    edge_values = 0.5 * (1 + np.cos(np.pi * t))
                    mask[edge_region] = np.clip(edge_values, 0, 1)
                
                # Set outside to zero
                outside_region = (dist_squared >= outer_r_squared)
                mask[outside_region] = 0
                
            elif shape == 'hexagon' or shape == 'octagon':
                # Use optimized polygon implementation
                sides = 6 if shape == 'hexagon' else 8
                angles = np.linspace(0, 2*np.pi, sides+1)[:-1]
                
                # Create polygon points with a slightly larger radius for the base shape
                base_points = []
                for angle in angles:
                    x_point = int(r * 1.05 * np.cos(angle) + r)
                    y_point = int(r * 1.05 * np.sin(angle) + r)
                    base_points.append([x_point, y_point])
                
                # Create a smaller polygon for the inner fully opaque region
                inner_points = []
                inner_r = r * 0.9  # 90% of the radius for inner polygon
                for angle in angles:
                    x_point = int(inner_r * np.cos(angle) + r)
                    y_point = int(inner_r * np.sin(angle) + r)
                    inner_points.append([x_point, y_point])
                
                # Create masks for inner and outer polygons
                outer_mask = np.zeros((aa_size, aa_size), dtype=np.uint8)
                inner_mask = np.zeros((aa_size, aa_size), dtype=np.uint8)
                
                cv2.fillPoly(outer_mask, [np.array(base_points)], 1)
                cv2.fillPoly(inner_mask, [np.array(inner_points)], 1)
                
                # Calculate transition region
                transition = outer_mask.astype(np.float32) - inner_mask.astype(np.float32)
                
                # Blur the transition region for anti-aliasing
                transition_blurred = cv2.GaussianBlur(
                    transition, 
                    (int(aa_factor * 0.7) | 1, int(aa_factor * 0.7) | 1), 
                    0
                )
                
                # Combine inner (full opacity) with transition (varying opacity)
                mask = inner_mask.astype(np.float32) + transition_blurred
                
                # Ensure center is 1.0 for proper normalization
                if mask[r, r] > 0:
                    mask = mask / mask[r, r]
                
            else:
                # Default to circle
                dist_squared = xx**2 + yy**2
                r_squared = r**2
                
                soft_edge_width = r * 0.15  # Wider edge for smoother transition
                inner_r_squared = (r - soft_edge_width)**2
                outer_r_squared = (r + soft_edge_width)**2
                
                # Create base mask
                mask = np.ones((aa_size, aa_size), dtype=np.float32)
                
                # Apply soft edge with sinusoidal falloff
                edge_region = (dist_squared > inner_r_squared) & (dist_squared < outer_r_squared)
                if np.any(edge_region):
                    t = (np.sqrt(dist_squared[edge_region]) - (r - soft_edge_width)) / (2 * soft_edge_width)
                    edge_values = 0.5 * (1 + np.cos(np.pi * t))
                    mask[edge_region] = np.clip(edge_values, 0, 1)
                
                # Set outside to zero
                outside_region = (dist_squared >= outer_r_squared)
                mask[outside_region] = 0
            
            # Downsample to target size using area interpolation for best anti-aliasing
            if aa_factor > 1:
                mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
            
            # Normalize kernel to maintain image brightness
            kernel_sum = np.sum(mask)
            if kernel_sum > 0:
                mask /= kernel_sum
            
            return mask
        
        # First pass: Create a subtle base blur layer just to prevent dark edges
        # This is a minimal base layer to avoid dark strokes while preserving original blur strength
        base_blur_size = max(3, int(np.mean(blur_sizes) * 0.5))  # Use a more conservative blur
        if base_blur_size > 1:
            base_kernel = create_bokeh_kernel(
                base_blur_size, 
                aperture_shape,
                aa_factor=4
            )
            base_blurred = cv2.filter2D(linear_image, -1, base_kernel)
        else:
            base_blurred = linear_image.copy()
            
        # Initialize the result with this base blur, but with reduced opacity
        # This keeps the original blur strength more apparent while eliminating dark edges
        base_opacity = 0.3  # Lower opacity for subtler effect
        result[:,:,:3] = base_blurred * base_opacity + linear_image * (1 - base_opacity)
        
        # Process each depth layer from far to near
        for depth_val in depth_values:
            # Create layer mask for current depth with some thickness
            layer_thickness = 1.0 / depth_bins
            layer_mask = np.logical_and(
                normalized_depth >= depth_val - layer_thickness,
                normalized_depth < depth_val + layer_thickness
            )
            
            # If no pixels in this layer, skip
            if not np.any(layer_mask):
                continue
            
            # Get average blur size for this layer
            layer_blur_sizes = blur_sizes[layer_mask]
            if len(layer_blur_sizes) == 0:
                continue
                
            # Get blur size for this layer - REVERTED to original max calculation
            layer_blur_size = int(np.max(layer_blur_sizes))
            if layer_blur_size <= 1:
                # For in-focus areas, just use the original image
                result[layer_mask, :3] = linear_image[layer_mask]
                cumulative_opacity[layer_mask] = 1.0
                unfilled_mask[layer_mask] = False
                continue
            
            # Get or create the bokeh kernel for this size
            if layer_blur_size not in bokeh_kernels:
                bokeh_kernels[layer_blur_size] = create_bokeh_kernel(
                    layer_blur_size, 
                    aperture_shape,
                    aa_factor=4
                )
            
            # Apply bokeh blur with disk/aperture kernel for base image
            blurred = cv2.filter2D(linear_image, -1, bokeh_kernels[layer_blur_size])
            
            # Apply chromatic aberration if enabled
            if chromatic_aberration > 0:
                # Apply different blur sizes to R, G, B channels - REVERTED to original values
                r_size = max(1, int(layer_blur_size * (1 + chromatic_aberration)))
                b_size = max(1, int(layer_blur_size * (1 - chromatic_aberration)))
                
                # Get or create kernels
                if r_size not in bokeh_kernels:
                    bokeh_kernels[r_size] = create_bokeh_kernel(r_size, aperture_shape, aa_factor=4)
                if b_size not in bokeh_kernels:
                    bokeh_kernels[b_size] = create_bokeh_kernel(b_size, aperture_shape, aa_factor=4)
                
                # Apply different blur to R and B channels
                r_blurred = cv2.filter2D(linear_image[:,:,0], -1, bokeh_kernels[r_size])
                b_blurred = cv2.filter2D(linear_image[:,:,2], -1, bokeh_kernels[b_size])
                
                # Replace channels
                blurred[:,:,0] = r_blurred
                blurred[:,:,2] = b_blurred
            
            # Apply highlight boost for bokeh effect - REVERTED to original
            if highlight_boost > 1.0:
                # Use a larger kernel for highlights
                highlight_size = min(max_blur_size, layer_blur_size + 4)
                if highlight_size not in bokeh_kernels:
                    bokeh_kernels[highlight_size] = create_bokeh_kernel(highlight_size, aperture_shape, aa_factor=4)
                
                highlights_blurred = cv2.filter2D(highlights, -1, bokeh_kernels[highlight_size])
                
                # Add highlights to blurred image
                blurred = np.maximum(blurred, highlights_blurred)
            
            # Create a soft mask for this layer that extends beyond the hard mask
            # This creates a gradual transition at depth boundaries
            layer_mask_float = layer_mask.astype(np.float32)
            extended_mask = cv2.dilate(
                layer_mask_float, 
                np.ones((3, 3), np.uint8), 
                iterations=2
            )
            extended_mask = cv2.GaussianBlur(extended_mask, (5, 5), 0)
            
            # Add edge-aware modulation
            if edge_smoothness > 0.5:
                # Further smooth transitions at edges
                edge_aware_factor = np.clip(1.0 - edge_influence * 2.0, 0, 1)
                extended_mask = extended_mask * edge_aware_factor + layer_mask_float * (1 - edge_aware_factor)
                extended_mask = cv2.GaussianBlur(extended_mask, (5, 5), 0)
            
            # Copy the blurred pixels using the extended mask as alpha
            layer_alpha = np.expand_dims(extended_mask, axis=2)
            new_alpha = np.minimum(1.0, cumulative_opacity + extended_mask)
            alpha_diff = new_alpha - cumulative_opacity
            
            # Only blend where alpha_diff is positive (new content to add)
            blend_mask = alpha_diff > 0.01
            if np.any(blend_mask):
                # Update colors with this layer's contribution
                for c in range(3):
                    # Blend colors based on the alpha differential
                    result[:,:,c] = np.where(
                        blend_mask,
                        (result[:,:,c] * cumulative_opacity + blurred[:,:,c] * alpha_diff) / np.maximum(0.001, new_alpha),
                        result[:,:,c]
                    )
                
                # Update the cumulative opacity
                cumulative_opacity = new_alpha
                
                # Update unfilled mask
                unfilled_mask = unfilled_mask & ~blend_mask
        
        # Ensure all pixels have been processed
        if np.any(unfilled_mask):
            # Fill any remaining areas with original image
            result[unfilled_mask, :3] = linear_image[unfilled_mask]
            result[unfilled_mask, 3] = 1.0
        
        # Apply final edge-specific processing to smooth transitions
        if edge_smoothness > 0.8:
            # Identify areas with high edge influence for targeted smoothing
            strong_edge_mask = edge_influence > 0.3
            if np.any(strong_edge_mask):
                # Create a version with bilateral filtering for these areas
                edge_filtered = cv2.bilateralFilter(
                    result[:,:,:3], 
                    d=int(5 * edge_smoothness),
                    sigmaColor=0.05,
                    sigmaSpace=int(5 * edge_smoothness)
                )
                
                # Blend only at strong edges with adaptive factor
                blend_factor = np.clip((edge_influence - 0.3) * 2 * edge_smoothness, 0, 0.9)
                
                # Fix: Create a proper 3D blend factor for broadcasting
                for c in range(3):
                    # Use where() with the mask directly, not the extracted values
                    result[:,:,c] = np.where(
                        strong_edge_mask,
                        result[:,:,c] * (1 - blend_factor) + edge_filtered[:,:,c] * blend_factor,
                        result[:,:,c]
                    )
        
        # Convert back from linear light space to perceptual space (sRGB)
        # Handle potential NaN/inf values with np.nan_to_num
        final_result_linear = np.nan_to_num(result[:,:,:3], nan=0.0, posinf=1.0, neginf=0.0)
        final_result_srgb = np.power(np.clip(final_result_linear, 0.0001, 1.0), 1.0/gamma) * 255.0
        
        # Apply final anti-banding dithering
        if enable_dithering:
            # Create targeted dithering based on color gradients
            gray = cv2.cvtColor((final_result_srgb/255.0).astype(np.float32), cv2.COLOR_RGB2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Normalize gradient to 0-1
            if gradient.max() > 0:
                gradient_norm = gradient / gradient.max()
            else:
                gradient_norm = gradient
                
            # Apply stronger dithering to subtle gradients (prone to banding)
            subtle_gradients = (gradient_norm > 0.01) & (gradient_norm < 0.1)
            dither_strength = np.ones_like(gradient_norm) * 0.1
            dither_strength[subtle_gradients] = 0.7
            
            # Generate high-quality blue noise
            noise = np.random.normal(0, 0.5, final_result_srgb.shape)
            blue_noise = cv2.GaussianBlur(noise, (0, 0), 1.5) - cv2.GaussianBlur(noise, (0, 0), 0.5)
            
            # Scale noise by dither strength
            blue_noise = blue_noise * np.expand_dims(dither_strength, axis=2)
            
            # Apply blue noise dithering
            final_result_srgb = np.clip(final_result_srgb + blue_noise, 0, 255)
        
        # Convert back to uint8
        return np.clip(final_result_srgb, 0, 255).astype(np.uint8)
    
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

    def generate_sbs_stereo(self, image_path, output_path=None, shift_factor=0.05, efficient_mode=True, 
                           apply_depth_blur=False, focal_distance=0.5, focal_thickness=0.1, blur_strength=1.0, max_blur_size=21,
                           aperture_shape='circle', highlight_boost=1.5, enable_dithering=True, chromatic_aberration=0.0,
                           edge_smoothness=1.0):
        """Generate side-by-side stereo 3D image from 2D image
        
        Args:
            image_path: Path to the input image or image array
            output_path: Path to save the output image (optional)
            shift_factor: Factor to control stereo separation
            efficient_mode: Whether to use efficient mode for inpainting
            apply_depth_blur: Whether to apply depth-based blur
            focal_distance: Distance to keep in focus (0-1, where 0 is closest, 1 is farthest)
            focal_thickness: Thickness of the in-focus region (0-1)
            blur_strength: Strength of the blur effect (multiplier)
            max_blur_size: Maximum kernel size for blur
            aperture_shape: Shape of the bokeh ('circle', 'hexagon', 'octagon')
            highlight_boost: How much to boost bright areas (1.0 = no boost)
            enable_dithering: Whether to apply dithering to prevent banding
            chromatic_aberration: Amount of color channel separation (0.0 = none)
            edge_smoothness: Controls the smoothness of transitions at depth boundaries (0.5-2.0)
            
        Returns:
            Tuple of (stereo_sbs, depth_map)
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            # Assume it's already an image array
            image = image_path
            
        # Generate depth map
        print("Estimating depth map...")
        depth_map = self.estimate_depth(image)
        
        # Create stereo views
        print(f"Generating stereo views (shift_factor={shift_factor})...")
        left_view, right_view, left_holes, right_holes = self.generate_stereo_views(
            image, 
            depth_map, 
            shift_factor=shift_factor, 
            apply_depth_blur=apply_depth_blur,
            focal_distance=focal_distance,
            focal_thickness=focal_thickness,
            blur_strength=blur_strength,
            max_blur_size=max_blur_size,
            aperture_shape=aperture_shape,
            highlight_boost=highlight_boost,
            enable_dithering=enable_dithering,
            chromatic_aberration=chromatic_aberration,
            edge_smoothness=edge_smoothness
        )
        
        # Fill holes if there are any
        if np.any(left_holes):
            print("Filling holes in left view...")
            if self.use_advanced_infill:
                left_view = self.advanced_infill(left_view, left_holes, efficient_mode=efficient_mode)
            else:
                left_view = self.basic_infill(left_view, left_holes)
        
        if np.any(right_holes):
            print("Filling holes in right view...")
            if self.use_advanced_infill:
                right_view = self.advanced_infill(right_view, right_holes, efficient_mode=efficient_mode)
            else:
                right_view = self.basic_infill(right_view, right_holes)
        
        # Apply post-processing for better quality
        if hasattr(self, 'high_color_quality') and self.high_color_quality:
            left_view = self._enhance_image_quality(left_view)
            right_view = self._enhance_image_quality(right_view)
        
        # Make sure the views have the same height
        h_left, w_left = left_view.shape[:2]
        h_right, w_right = right_view.shape[:2]
        
        if h_left != h_right:
            # Resize to match heights
            if h_left > h_right:
                scale = h_right / h_left
                new_width = int(w_left * scale)
                left_view = cv2.resize(left_view, (new_width, h_right))
            else:
                scale = h_left / h_right
                new_width = int(w_right * scale)
                right_view = cv2.resize(right_view, (new_width, h_left))
        
        # Create side-by-side stereo image
        stereo_sbs = np.hstack((left_view, right_view))
        
        # Apply dithering if enabled
        if hasattr(self, 'apply_dithering') and self.apply_dithering:
            stereo_sbs = self._apply_dithering(stereo_sbs)
        
        # Save output if path is provided
        if output_path is not None:
            print(f"Saving output to {output_path}...")
            cv2.imwrite(output_path, stereo_sbs)
            
        return stereo_sbs, depth_map
    
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
        """
        Apply Stable Diffusion inpainting to the image.
        
        This breaks down large images into patches and processes them efficiently.
        
        Args:
            img_pil: PIL image to inpaint
            mask_pil: PIL mask image (white = areas to inpaint)
            steps: Number of denoising steps for SD inpainting
            guidance_scale: How closely to follow the prompt
            patch_size: Size of patches to process (smaller = less VRAM but lower quality)
            patch_overlap: How much patches should overlap (higher = smoother transitions)
            high_quality: Whether to use higher quality settings
            
        Returns:
            Inpainted PIL image
        """
        try:
            # Implement the patch-based inpainting logic...
            from diffusers import StableDiffusionInpaintPipeline
            import torch
            
            # Convert inputs to RGB (in case mask is grayscale)
            img_rgb = img_pil.convert("RGB")
            mask_rgb = mask_pil.convert("RGB")
            
            # Resize images if they're too large for the model
            orig_size = img_rgb.size
            max_size = 2048  # Maximum size to process
            
            if img_rgb.width > max_size or img_rgb.height > max_size:
                scale = max_size / max(img_rgb.width, img_rgb.height)
                new_width = int(img_rgb.width * scale)
                new_height = int(img_rgb.height * scale)
                img_rgb = img_rgb.resize((new_width, new_height), Image.LANCZOS)
                mask_rgb = mask_rgb.resize((new_width, new_height), Image.NEAREST)
            
            # Create patch-based inpainting
            from core.patch_inpainting import patch_inpaint
            
            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                
            # Adjust patch settings based on image size
            img_size = max(img_rgb.width, img_rgb.height)
            if img_size < 512:
                # For small images, use larger patches for better coherence
                patch_size = min(patch_size, img_size)
                patch_overlap = min(patch_overlap, patch_size // 2)
            
            # Call the patch inpainting function
            result = patch_inpaint(
                self.inpaint_model,
                img_rgb,
                mask_rgb,
                prompt="a realistic image with natural texture",
                device=device,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                high_quality=high_quality
            )
            
            # Resize back to original size if needed
            if result.size != orig_size:
                result = result.resize(orig_size, Image.LANCZOS)
                
            return result
                
        except Exception as e:
            print(f"Critical error in inpainting: {e}")
            import traceback
            traceback.print_exc()
            # If everything fails, return original image
            return np.array(img_pil)

    def set_color_quality(self, high_color_quality=True, apply_dithering=True, dithering_level=1.0):
        """Set parameters for color quality enhancement"""
        self.high_color_quality = high_color_quality
        self.apply_dithering = apply_dithering
        self.dithering_level = dithering_level
        return self

    def smooth_depth_edges(self, depth_map, edge_smoothness=1.0):
        """
        Apply advanced edge smoothing to a depth map to reduce harsh transitions
        
        Args:
            depth_map: Depth map (0-1 range)
            edge_smoothness: Controls the smoothness of transitions at depth boundaries (0.5-2.0)
            
        Returns:
            Smoothed depth map
        """
        # Ensure depth map is in 0-1 range
        if depth_map.max() > 1.0:
            normalized_depth = depth_map / depth_map.max()
        else:
            normalized_depth = depth_map.copy()
        
        # Detect edges in the depth map
        depth_edges = cv2.Canny(
            (normalized_depth * 255).astype(np.uint8), 
            threshold1=10, 
            threshold2=50
        )
        
        # Create a mask around depth edges for special treatment
        edge_mask = cv2.dilate(
            depth_edges, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 
            iterations=int(edge_smoothness * 2)
        ).astype(bool)
        
        # Apply bilateral filtering to preserve depth discontinuities while smoothing
        smoothed_depth = cv2.bilateralFilter(
            normalized_depth, 
            d=int(9 * edge_smoothness),  # Diameter of pixel neighborhood
            sigmaColor=0.1,  # Filter sigma in color space
            sigmaSpace=int(7 * edge_smoothness)  # Filter sigma in coordinate space
        )
        
        # Apply additional Gaussian smoothing with larger kernel at the edges
        if edge_smoothness > 0.5:
            # Create a smoothed version for edge areas
            edge_smoothed = cv2.GaussianBlur(
                normalized_depth, 
                (int(7 * edge_smoothness) | 1, int(7 * edge_smoothness) | 1),  # Ensure odd kernel size
                0
            )
            
            # Blend original with smoothed version at the edges
            blend_factor = np.minimum(1.0, edge_smoothness / 1.5)
            smoothed_depth[edge_mask] = (
                smoothed_depth[edge_mask] * (1 - blend_factor) + 
                edge_smoothed[edge_mask] * blend_factor
            )
        
        # Apply a final gradient-preserving filter
        final_smoothed = cv2.bilateralFilter(
            smoothed_depth,
            d=int(5 * edge_smoothness),
            sigmaColor=0.05,
            sigmaSpace=int(5 * edge_smoothness)
        )
        
        return final_smoothed