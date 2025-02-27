import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
import numpy as np
import cv2
import math

class AdvancedInfillTechniques:
    """Additional advanced background infill techniques for stereo 3D generation"""
    
    def __init__(self, max_resolution=1024, patch_size=512, patch_overlap=64, batch_size=1, inference_steps=30):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_resolution = max_resolution
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size
        self.inference_steps = inference_steps
        print(f"Advanced infill initialized with batch_size={batch_size}, inference_steps={inference_steps}")
        self._init_models()
    
    def _init_models(self):
        # Initialize SDXL inpainting model for higher quality results
        self.sdxl_inpaint = AutoPipelineForInpainting.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        # Disable safety checker to prevent NSFW filtering
        self.sdxl_inpaint.safety_checker = None
        # Set batch size for inference
        self.sdxl_inpaint.batch_size = self.batch_size
        # Use memory-efficient attention if possible
        try:
            if hasattr(self.sdxl_inpaint, 'enable_xformers_memory_efficient_attention'):
                print("Enabling xformers memory efficient attention")
                self.sdxl_inpaint.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers: {str(e)}")
            print("Using standard attention mechanism")
        # Move to device
        self.sdxl_inpaint.to(self.device)
        
        # You could add more specialized models here, such as:
        # - LaMa inpainting model (specifically designed for large masks)
        # - Depth-aware inpainting models
    
    def extract_hole_patches(self, image, hole_mask, context_size=64):
        """Extract patches containing holes with some context around them"""
        # Find connected components in the hole mask
        num_labels, labels = cv2.connectedComponents(hole_mask.astype(np.uint8))
        
        patches = []
        patch_positions = []
        
        print(f"Found {num_labels-1} connected hole regions")
        
        # If too many small components, use grid-based approach instead
        if num_labels > 50:
            print(f"Too many hole components ({num_labels-1}), using grid-based approach")
            return self._extract_hole_grid_patches(image, hole_mask, self.patch_size, self.patch_overlap)
        
        for label in range(1, num_labels):  # Skip 0 (background)
            # Get bounding box for this hole
            hole_component = (labels == label).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(hole_component)
            
            # Calculate patch dimensions to ensure they're at least min_patch_size
            min_patch_size = 128  # Minimum patch size for quality
            
            # If hole is very small, ensure a minimum patch size
            if w < min_patch_size or h < min_patch_size:
                # Center the patch around the small hole
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate patch bounds ensuring minimum size
                half_size = max(min_patch_size // 2, max(w, h) // 2)
                x_start = max(0, center_x - half_size)
                y_start = max(0, center_y - half_size)
                x_end = min(image.shape[1], center_x + half_size)
                y_end = min(image.shape[0], center_y + half_size)
            else:
                # Add context around the hole
                x_start = max(0, x - context_size)
                y_start = max(0, y - context_size)
                x_end = min(image.shape[1], x + w + context_size)
                y_end = min(image.shape[0], y + h + context_size)
            
            # Skip if patch is too small (shouldn't happen with above logic)
            if x_end - x_start < 32 or y_end - y_start < 32:
                continue
                
            # For large holes, split into multiple overlapping patches
            if (x_end - x_start) > self.patch_size or (y_end - y_start) > self.patch_size:
                print(f"Large hole region at ({x},{y}) size {w}x{h}, splitting into patches")
                grid_patches, grid_positions = self._extract_grid_patches(
                    image, hole_mask, x_start, y_start, x_end, y_end, 
                    self.patch_size, self.patch_overlap
                )
                patches.extend(grid_patches)
                patch_positions.extend(grid_positions)
            else:
                # Extract the patch and its mask
                patch = image[y_start:y_end, x_start:x_end].copy()
                patch_mask = hole_mask[y_start:y_end, x_start:x_end].copy()
                
                # Only add if the patch has holes
                if np.any(patch_mask):
                    patches.append((patch, patch_mask))
                    patch_positions.append((x_start, y_start, x_end, y_end))
        
        print(f"Extracted {len(patches)} patches for processing")
        return patches, patch_positions
    
    def _extract_grid_patches(self, image, hole_mask, start_x, start_y, end_x, end_y, patch_size, overlap):
        """Split a large region into a grid of overlapping patches"""
        patches = []
        patch_positions = []
        
        # Calculate step size (with overlap)
        step_size = patch_size - overlap
        
        for y in range(start_y, end_y, step_size):
            for x in range(start_x, end_x, step_size):
                # Calculate patch boundaries
                x_start = x
                y_start = y
                x_end = min(end_x, x + patch_size)
                y_end = min(end_y, y + patch_size)
                
                # Skip undersized patches at edges
                if x_end - x_start < patch_size / 2 or y_end - y_start < patch_size / 2:
                    continue
                
                # Extract the patch
                patch = image[y_start:y_end, x_start:x_end].copy()
                patch_mask = hole_mask[y_start:y_end, x_start:x_end].copy()
                
                # Only add if the patch has holes
                if np.any(patch_mask):
                    patches.append((patch, patch_mask))
                    patch_positions.append((x_start, y_start, x_end, y_end))
        
        return patches, patch_positions
    
    def _extract_hole_grid_patches(self, image, hole_mask, patch_size, overlap):
        """Divide the image into a grid of patches, focusing on areas with holes"""
        height, width = image.shape[:2]
        patches = []
        patch_positions = []
        
        # Only process areas with holes to save computation
        # Get bounding box of all holes
        y_indices, x_indices = np.where(hole_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return [], []
            
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        # Add context around the holes area
        context = 2 * overlap
        min_x = max(0, min_x - context)
        min_y = max(0, min_y - context)
        max_x = min(width, max_x + context)
        max_y = min(height, max_y + context)
        
        # Calculate step size (with overlap)
        step_size = patch_size - overlap
        
        # Process grid over the holes area
        for y in range(min_y, max_y, step_size):
            for x in range(min_x, max_x, step_size):
                # Calculate patch boundaries
                x_start = x
                y_start = y
                x_end = min(width, x + patch_size)
                y_end = min(height, y + patch_size)
                
                # Skip small edge patches
                if x_end - x_start < patch_size / 2 or y_end - y_start < patch_size / 2:
                    continue
                
                # Extract the patch
                patch = image[y_start:y_end, x_start:x_end].copy()
                patch_mask = hole_mask[y_start:y_end, x_start:x_end].copy()
                
                # Only add if the patch has holes
                if np.any(patch_mask):
                    patches.append((patch, patch_mask))
                    patch_positions.append((x_start, y_start, x_end, y_end))
        
        return patches, patch_positions

    def depth_aware_infill(self, image, hole_mask, depth_map, prompt="", efficient_mode=True):
        """Depth-aware infill that considers depth information for more accurate filling"""
        # Handle case where there are no holes
        if not np.any(hole_mask):
            print("No holes found for inpainting")
            return image.copy()
            
        # Analyze hole size and distribution for optimized processing
        num_hole_pixels = np.sum(hole_mask)
        image_size = image.shape[0] * image.shape[1]
        hole_percentage = (num_hole_pixels / image_size) * 100
        
        print(f"Processing image with {num_hole_pixels} hole pixels ({hole_percentage:.2f}% of image)")
        
        # If holes are very small (less than 0.1% of image), use faster basic inpainting
        if hole_percentage < 0.1:
            print("Very small holes detected, using optimized fast inpainting")
            return cv2.inpaint(image, (hole_mask*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            
        # For small to medium holes (0.1-1% of image), use context-aware approach
        if hole_percentage < 1.0:
            print("Small holes detected, using context-aware inpainting")
            # Extract only the regions with holes plus context for processing
            return self._process_only_hole_regions(image, hole_mask, depth_map, prompt)
        
        # If not using efficient mode or if the image is small, process the entire image
        if not efficient_mode or (image.shape[0] <= self.max_resolution and image.shape[1] <= self.max_resolution):
            return self._full_image_depth_aware_infill(image, hole_mask, depth_map, prompt)
        
        # Otherwise, use a patch-based approach
        result = image.copy()
        
        # Extract patches containing holes
        patches, patch_positions = self.extract_hole_patches(image, hole_mask, context_size=self.patch_overlap)
        
        if not patches:
            print("No holes found for inpainting")
            return result
            
        print(f"Processing {len(patches)} patches for efficient inpainting")
        
        # Process patches with lower batch size to save memory
        # Process in smaller batches to reduce memory usage
        max_concurrent_patches = min(5, len(patches))  # Never process more than 5 patches at once
        
        # Process each patch
        for i, ((patch, patch_mask), (x_start, y_start, x_end, y_end)) in enumerate(zip(patches, patch_positions)):
            print(f"  Inpainting patch {i+1}/{len(patches)} at position ({x_start},{y_start}) size {patch.shape}")
            
            # Extract the depth information for this patch
            patch_depth = depth_map[y_start:y_end, x_start:x_end] if depth_map is not None else None
            
            # Skip if patch size is invalid
            if patch.shape[0] == 0 or patch.shape[1] == 0 or not np.any(patch_mask):
                continue
            
            # Store original patch dimensions
            patch_height, patch_width = patch.shape[:2]
            
            # For large patches, resize down to save VRAM
            need_resize = patch_height > self.patch_size or patch_width > self.patch_size
            orig_patch = None
            if need_resize:
                print(f"    Resizing large patch from {patch.shape[:2]} to fit within {self.patch_size}x{self.patch_size}")
                orig_patch = patch.copy()  # Save original for better blending
                scale = self.patch_size / max(patch_height, patch_width)
                new_height = int(patch_height * scale)
                new_width = int(patch_width * scale)
                patch = cv2.resize(patch, (new_width, new_height), interpolation=cv2.INTER_AREA)
                patch_mask = cv2.resize(patch_mask.astype(np.uint8), (new_width, new_height), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                if patch_depth is not None:
                    patch_depth = cv2.resize(patch_depth, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Inpaint the patch with reduced memory usage
            inpainted_patch = self._full_image_depth_aware_infill(patch, patch_mask, patch_depth, prompt)
            
            # Clear CUDA cache immediately after inpainting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Resize back if needed
            if need_resize:
                inpainted_patch = cv2.resize(inpainted_patch, (patch_width, patch_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Blend the edges for smoother transitions (only where there were holes)
                blend_mask = cv2.dilate(cv2.resize(patch_mask.astype(np.uint8), (patch_width, patch_height), 
                                                 interpolation=cv2.INTER_NEAREST), 
                                      np.ones((5,5), np.uint8), iterations=2)
                blend_mask = cv2.GaussianBlur(blend_mask, (15, 15), 0) / 255.0
                blend_mask = np.stack([blend_mask] * 3, axis=2)
                
                # Only blend the inpainted areas with the original
                inpainted_patch = orig_patch * (1 - blend_mask) + inpainted_patch * blend_mask
            
            # Ensure the inpainted patch has the same size as the original patch
            if inpainted_patch.shape[:2] != (patch_height, patch_width):
                print(f"    Resizing inpainted patch from {inpainted_patch.shape[:2]} to {(patch_height, patch_width)}")
                inpainted_patch = cv2.resize(inpainted_patch, (patch_width, patch_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Create feathered blending mask for patch edges to avoid visible seams
            # Create a mask that's 1 inside, 0 outside, with smooth transition at edges
            y_indices, x_indices = np.mgrid[0:patch_height, 0:patch_width]
            
            # Distance from each edge
            left_dist = x_indices
            right_dist = patch_width - x_indices - 1
            top_dist = y_indices
            bottom_dist = patch_height - y_indices - 1
            
            # Convert to normalized weight (higher near center, lower at edges)
            edge_falloff = min(20, int(min(patch_width, patch_height) * 0.1))  # Dynamic falloff based on patch size
            edge_weight = np.minimum(
                np.minimum(left_dist, right_dist),
                np.minimum(top_dist, bottom_dist)
            )
            edge_weight = np.clip(edge_weight, 0, edge_falloff) / edge_falloff
            
            # Apply gaussian blur for smoother transitions
            edge_weight = cv2.GaussianBlur(edge_weight.astype(np.float32), (0, 0), 3)
            
            # Expand to 3 channels and limit to hole regions
            edge_weight_3ch = np.stack([edge_weight] * 3, axis=2)
            hole_mask_region = hole_mask[y_start:y_end, x_start:x_end]
            hole_mask_3ch = np.stack([hole_mask_region] * 3, axis=2)
            
            # Place the inpainted patch back with edge blending
            current_region = result[y_start:y_end, x_start:x_end]
            
            # Only blend at hole regions
            blend_mask = edge_weight_3ch * hole_mask_3ch
            result[y_start:y_end, x_start:x_end] = (
                current_region * (1 - blend_mask) + 
                inpainted_patch * blend_mask
            )
            
            # Explicitly clean up variables to save memory
            del inpainted_patch, patch, patch_mask, patch_depth, edge_weight, edge_weight_3ch, hole_mask_region, hole_mask_3ch, blend_mask
            
            # Clear CUDA cache after each patch if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Add a small delay to allow memory cleanup
            if i % max_concurrent_patches == 0 and i > 0:
                import time
                time.sleep(0.5)  # Small delay to help GPU memory stabilize
        
        return result
    
    def _process_only_hole_regions(self, image, hole_mask, depth_map=None, prompt=""):
        """Process only the regions containing holes with sufficient context"""
        result = image.copy()
        
        # Find connected components in the hole mask
        num_labels, labels = cv2.connectedComponents(hole_mask.astype(np.uint8))
        
        print(f"Found {num_labels-1} separate hole regions to process")
        
        # Process each hole region separately with context
        for label in range(1, num_labels):
            # Extract this hole region
            current_hole = (labels == label).astype(np.uint8)
            
            # Skip tiny holes (less than 10 pixels)
            if np.sum(current_hole) < 10:
                # Use simple inpainting for tiny holes
                result = cv2.inpaint(result, current_hole, 3, cv2.INPAINT_TELEA)
                continue
                
            # Get bounding box with context
            y_indices, x_indices = np.where(current_hole)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            
            # Add context (at least 30 pixels, or proportional to hole size)
            context_size = max(30, int(max(max_y - min_y, max_x - min_x) * 0.5))
            y_start = max(0, min_y - context_size)
            y_end = min(image.shape[0], max_y + context_size)
            x_start = max(0, min_x - context_size)
            x_end = min(image.shape[1], max_x + context_size)
            
            # Extract region with hole
            region = result[y_start:y_end, x_start:x_end].copy()
            region_mask = current_hole[y_start:y_end, x_start:x_end].astype(bool)
            
            # Extract depth for this region if available
            region_depth = depth_map[y_start:y_end, x_start:x_end] if depth_map is not None else None
            
            # Generate adaptive prompt based on hole location and context
            adaptive_prompt = self._generate_adaptive_prompt(region, region_mask, region_depth, prompt)
            
            # Inpaint only this region
            inpainted_region = self._full_image_depth_aware_infill(region, region_mask, region_depth, adaptive_prompt)
            
            # Create a blending mask to avoid hard edges
            # Dilate the mask slightly for blending
            dilated_mask = cv2.dilate(region_mask.astype(np.uint8), np.ones((3,3), np.uint8))
            # Apply gaussian blur for smooth transitions
            blend_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32), (9, 9), 0)
            blend_mask = np.clip(blend_mask, 0, 1)
            blend_mask = np.stack([blend_mask] * 3, axis=2)
            
            # Blend inpainted region back into the result
            result[y_start:y_end, x_start:x_end] = (
                result[y_start:y_end, x_start:x_end] * (1 - blend_mask) + 
                inpainted_region * blend_mask
            )
            
            # Clean up memory
            del region, region_mask, region_depth, inpainted_region, blend_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
    
    def _generate_adaptive_prompt(self, region, mask, depth=None, base_prompt=""):
        """Generate a context-aware prompt based on the image region and depth"""
        # If user provided a prompt, use it as the base
        if base_prompt:
            prompt = base_prompt
        else:
            prompt = "photorealistic seamless background extension"
        
        # Add depth-based context if available
        if depth is not None:
            # Calculate average depth in the surrounding area
            # Ensure mask is boolean
            mask_bool = mask.astype(bool) if not np.issubdtype(mask.dtype, np.bool_) else mask
            inverted_mask = ~mask_bool
            
            if np.any(inverted_mask):
                try:
                    avg_depth = np.mean(depth[inverted_mask])
                    
                    # Adapt prompt based on depth
                    if avg_depth < 0.3:  # Close/foreground objects
                        prompt += ", detailed foreground texture"
                    elif avg_depth > 0.7:  # Far/background
                        prompt += ", distant background landscape"
                    else:  # Mid-range
                        prompt += ", mid-range scene elements"
                except Exception as e:
                    print(f"Error processing depth for prompt adaptation: {str(e)}")
                    # Continue without depth adaptation
        
        # Add quality keywords for better results
        quality_terms = "detailed, sharp, consistent with surroundings"
        if quality_terms not in prompt:
            prompt += f", {quality_terms}"
            
        return prompt

    def _full_image_depth_aware_infill(self, image, hole_mask, depth_map=None, prompt=""):
        """Process a single image or patch with depth-aware inpainting"""
        # Convert image to RGB PIL format
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        
        # Convert hole mask to PIL
        mask_pil = Image.fromarray((hole_mask * 255).astype(np.uint8))
        
        # Check if we need to resize for model constraints
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
            
            # Ensure dimensions are divisible by 8 for diffusion models
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            # Resize image and mask
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
        else:
            # Even for non-resized images, ensure dimensions are divisible by 8
            width, height = img_pil.size
            if width % 8 != 0 or height % 8 != 0:
                new_width = (width // 8) * 8
                new_height = (height // 8) * 8
                img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
                print(f"Adjusted dimensions from {width}x{height} to {new_width}x{new_height} to be divisible by 8")
        
        # For parallax holes, we want to maintain coherence with surroundings
        if not prompt:
            prompt = "consistent seamless extension of surrounding image areas, maintain texture continuity"
        
        # Adapt to common parallax hole types
        hole_percent = float(np.sum(hole_mask)) / (hole_mask.shape[0] * hole_mask.shape[1])
        
        # Edge holes tend to be on the left/right sides from parallax shifting
        is_edge_hole = False
        if hole_mask.shape[1] > 0:
            left_column = hole_mask[:, 0]
            right_column = hole_mask[:, -1]
            if np.sum(left_column) > 0 or np.sum(right_column) > 0:
                is_edge_hole = True
                prompt = f"extend image naturally beyond the edge, {prompt}"
        
        # Add negative prompt to avoid artifacts
        negative_prompt = "blurry, inconsistent edges, seams, distortion"
            
        try:
            # Adjust inference steps based on hole size
            # Smaller holes need fewer steps
            actual_steps = self.inference_steps
            if hole_percent < 0.01:  # Very small holes (typical for parallax)
                actual_steps = max(15, self.inference_steps - 10)
                print(f"Small holes detected ({hole_percent*100:.2f}%), using reduced steps: {actual_steps}")
            
            # For small edge holes especially, use fast inpainting
            if is_edge_hole and hole_percent < 0.005:
                print("Edge holes detected, using optimized edge extension")
            
            # Run inpainting with optimized settings
            guidance_scale = 7.5  # Lower guidance for better blending of small holes
            
            result = self.sdxl_inpaint(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                guidance_scale=guidance_scale,
                num_inference_steps=actual_steps,
                height=img_pil.height,
                width=img_pil.width
            ).images[0]
            
            # Resize back to original dimensions if needed
            if need_resize or img_pil.size != orig_size:
                result = result.resize(orig_size, Image.LANCZOS)
                
            # Convert back to numpy
            result_np = np.array(result)
            if image.shape[2] == 3:
                result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                
            # No sharpening as per user request
        
        except Exception as e:
            print(f"Error in diffusion inpainting: {str(e)}")
            print(f"Falling back to basic inpainting")
            # Fallback to basic inpainting
            result_np = cv2.inpaint(image, (hole_mask*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        # Clear memory
        del img_pil, mask_pil
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result_np
    
    def texture_synthesis_infill(self, image, hole_mask):
        """Use texture synthesis for more coherent background filling"""
        # Convert hole_mask to uint8 format
        hole_mask_uint8 = hole_mask.astype(np.uint8) * 255
        
        # Create a dilated mask to sample textures from nearby regions
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(hole_mask_uint8, kernel, iterations=3)
        sample_region = dilated_mask > 0
        
        # Get sample pixels from nearby regions
        sample_pixels = image[sample_region & ~hole_mask]
        
        # For each hole pixel, find the best matching texture
        result = image.copy()
        hole_coords = np.where(hole_mask)
        
        # Simple implementation - for large holes this is inefficient
        # A more sophisticated PatchMatch algorithm would be better
        for i in range(len(hole_coords[0])):
            y, x = hole_coords[0][i], hole_coords[1][i]
            
            # Find closest non-hole pixel
            min_dist = float('inf')
            best_color = None
            
            for ny in range(max(0, y-10), min(image.shape[0], y+10)):
                for nx in range(max(0, x-10), min(image.shape[1], x+10)):
                    if not hole_mask[ny, nx]:
                        dist = np.sqrt((ny-y)**2 + (nx-x)**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_color = image[ny, nx]
            
            if best_color is not None:
                result[y, x] = best_color
        
        return result
    
    def multimodel_ensemble_infill(self, image, hole_mask, depth_map=None, efficient_mode=True):
        """Combine multiple inpainting techniques for better results"""
        # Get results from multiple methods
        basic_infill = cv2.inpaint(image, (hole_mask*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        # For small holes, basic infill might be sufficient
        hole_count = np.sum(hole_mask)
        if hole_count < 1000:  # Arbitrary threshold
            print(f"Using basic infill only for small holes ({hole_count} pixels)")
            return basic_infill
        
        # For efficient patch-based processing to save VRAM
        if efficient_mode and (image.shape[0] > self.max_resolution or image.shape[1] > self.max_resolution):
            print(f"Using patch-based ensemble infill to reduce VRAM usage")
            result = image.copy()
            
            # Extract patches containing holes
            patches, patch_positions = self.extract_hole_patches(image, hole_mask, context_size=self.patch_overlap)
            
            if not patches:
                print("No holes found for inpainting")
                return result
            
            print(f"Processing {len(patches)} patches for ensemble inpainting")
            
            # Limit the number of patches to process if there are too many
            max_patches = 30
            if len(patches) > max_patches:
                print(f"Too many patches ({len(patches)}), limiting to {max_patches}")
                # Sort patches by size and process the largest ones
                patch_sizes = [patch[0].shape[0] * patch[0].shape[1] for patch in patches]
                sorted_indices = np.argsort(patch_sizes)[::-1][:max_patches]
                patches = [patches[i] for i in sorted_indices]
                patch_positions = [patch_positions[i] for i in sorted_indices]
            
            # Process in smaller batches to reduce memory usage
            max_concurrent_patches = min(3, len(patches))  # Never process more than 3 patches at once
            
            # Process each patch
            for i, ((patch, patch_mask), (x_start, y_start, x_end, y_end)) in enumerate(zip(patches, patch_positions)):
                print(f"  Ensemble inpainting patch {i+1}/{len(patches)}, size: {patch.shape}")
                
                # Store original patch dimensions
                patch_height, patch_width = patch.shape[:2]
                
                # Extract the depth information for this patch
                patch_depth = depth_map[y_start:y_end, x_start:x_end] if depth_map is not None else None
                
                # Apply ensemble infill to this patch
                inpainted_patch = self._process_single_patch_ensemble(patch, patch_mask, patch_depth)
                
                # Ensure the inpainted patch has the same size as the original patch
                if inpainted_patch.shape[:2] != (patch_height, patch_width):
                    print(f"Resizing inpainted patch from {inpainted_patch.shape[:2]} to {(patch_height, patch_width)}")
                    inpainted_patch = cv2.resize(inpainted_patch, (patch_width, patch_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Create a blending mask for smooth transitions
                # Create a mask based on distance from the edge of the patch
                y_indices, x_indices = np.mgrid[0:patch_height, 0:patch_width]
                
                # Distance from each edge (normalized)
                left_dist = x_indices / max(1, patch_width-1)
                right_dist = (patch_width - 1 - x_indices) / max(1, patch_width-1)
                top_dist = y_indices / max(1, patch_height-1)
                bottom_dist = (patch_height - 1 - y_indices) / max(1, patch_height-1)
                
                # Min distance to any edge (0 at edge, 0.5 at center)
                edge_dist = np.minimum(
                    np.minimum(left_dist, right_dist),
                    np.minimum(top_dist, bottom_dist)
                )
                
                # Apply falloff
                edge_falloff = 0.2  # How quickly the weight falls off from center to edge
                edge_weight = np.clip(edge_dist / edge_falloff, 0, 1)
                
                # Apply gaussian blur for smoother transitions
                edge_weight = cv2.GaussianBlur(edge_weight.astype(np.float32), (0, 0), 2)
                
                # Get the current content of the patch in the result image
                current_patch = result[y_start:y_end, x_start:x_end].copy()
                
                # Place the inpainted patch back into the result with edge blending
                # Only modify hole regions
                hole_patch = patch_mask.copy()
                
                # Create blending weight map (3 channels)
                edge_weight_3ch = np.stack([edge_weight] * 3, axis=2)
                hole_patch_3ch = np.stack([hole_patch] * 3, axis=2)
                
                # Apply blending: keep original content outside holes,
                # blend new content inside holes with edge weighting
                blended_patch = current_patch * (1 - hole_patch_3ch) + (
                    current_patch * (1 - edge_weight_3ch) + 
                    inpainted_patch * edge_weight_3ch
                ) * hole_patch_3ch
                
                # Place the blended patch back into the result
                result[y_start:y_end, x_start:x_end] = blended_patch
                
                # Clean up to save memory
                del inpainted_patch, patch, patch_mask, edge_weight, edge_weight_3ch, blended_patch
                
                # Clear CUDA cache after each patch if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Add a small delay to help with memory management
                if i % max_concurrent_patches == 0 and i > 0:
                    import time
                    time.sleep(0.5)  # Small delay to help GPU memory stabilize
                
            return result
        
        # For smaller images, use full image processing
        print(f"Using advanced ensemble infill for larger holes ({hole_count} pixels)")
        sdxl_result = self.depth_aware_infill(image, hole_mask, depth_map, efficient_mode=efficient_mode)
        
        # Ensure sdxl_result has the same shape as the original image
        if sdxl_result.shape != image.shape:
            print(f"Warning: Resizing SDXL result from {sdxl_result.shape} to {image.shape}")
            sdxl_result = cv2.resize(sdxl_result, (image.shape[1], image.shape[0]))
        
        # Blend results based on hole size and location
        result = image.copy()
        
        # Generate a gradient weight mask from hole boundary
        dist_transform = cv2.distanceTransform((hole_mask*255).astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        if max_dist > 0:
            weight_mask = dist_transform / max_dist
        else:
            weight_mask = dist_transform
        
        # Apply Gaussian blur to the weight mask to create smoother transitions
        weight_mask = cv2.GaussianBlur(weight_mask, (5, 5), 0)
        
        # Create a mask for the hole and a small region around it for feathering
        kernel = np.ones((5, 5), np.uint8)
        feather_mask = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=2)
        
        # Apply the mask
        mask_3d = np.stack([weight_mask] * 3, axis=2)
        feather_mask_3d = np.stack([feather_mask] * 3, axis=2)
        
        # Use basic infill for the edge areas and blend to sdxl for deeper parts
        result = np.where(feather_mask_3d, 
                         basic_infill * (1 - mask_3d) + sdxl_result * mask_3d, 
                         image)
        
        # Clear memory
        del sdxl_result, basic_infill, weight_mask, mask_3d, feather_mask_3d
        
        return result

    def _process_single_patch_ensemble(self, patch, patch_mask, depth_map=None):
        """Process a single patch with ensemble methods"""
        # Basic infill is fast and uses minimal VRAM
        basic_infill = cv2.inpaint(patch, (patch_mask*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        # For very small patches, basic infill is sufficient
        if patch.shape[0] < 128 or patch.shape[1] < 128:
            return basic_infill
        
        # Try to use SDXL for larger patches, with error handling
        try:
            # Convert to PIL format
            img_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray((patch_mask * 255).astype(np.uint8))
            
            # Store original dimensions
            orig_height, orig_width = patch.shape[:2]
            
            # Check if we need to resize
            need_resize = img_pil.width > 384 or img_pil.height > 384
            if need_resize:
                # Resize to 384px max dimension to save VRAM (reduced from 512px)
                aspect_ratio = img_pil.width / img_pil.height
                if img_pil.width > img_pil.height:
                    new_width = 384
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = 384
                    new_width = int(new_height * aspect_ratio)
                
                img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
            
            # Use model with reduced precision and steps to save VRAM
            reduced_steps = max(15, self.inference_steps - 10)  # Use fewer steps than main method
            result = self.sdxl_inpaint(
                prompt="natural seamless background",
                image=img_pil,
                mask_image=mask_pil,
                guidance_scale=5.0,  # Lower guidance scale
                num_inference_steps=reduced_steps,
                height=img_pil.height,
                width=img_pil.width
            ).images[0]
            
            # Resize back if needed
            if need_resize:
                result = result.resize((orig_width, orig_height), Image.LANCZOS)
            
            # Convert back to numpy
            sdxl_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            
            # Blend with basic infill for smoother results
            # Generate a weight mask based on distance from hole boundary
            dist_transform = cv2.distanceTransform((patch_mask*255).astype(np.uint8), cv2.DIST_L2, 3)
            max_dist = np.max(dist_transform)
            weight_mask = dist_transform / max_dist if max_dist > 0 else dist_transform
            weight_mask = cv2.GaussianBlur(weight_mask, (3, 3), 0)
            
            # Blend the results
            mask_3d = np.stack([weight_mask] * 3, axis=2)
            final_result = basic_infill * (1 - mask_3d) + sdxl_result * mask_3d
            
            # Clear memory
            del img_pil, mask_pil, sdxl_result, weight_mask, mask_3d
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return final_result
            
        except Exception as e:
            print(f"Error in SDXL inpainting: {str(e)}. Using basic infill.")
            return basic_infill 