"""
Aggressive & Powerful Image Generator

SDXL-based pipeline with LoRA and ControlNet support for generating
ultra-realistic, dominant still images with cinematic intensity.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
from pathlib import Path
import logging
import time
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Intensity Levels
# =============================================================================

class IntensityLevel(Enum):
    """Style intensity levels from calm to ruthless"""
    CALM = "calm"
    INTENSE = "intense"
    DOMINANT = "dominant"
    RUTHLESS = "ruthless"


INTENSITY_MODIFIERS = {
    IntensityLevel.CALM: {
        "facial_tension": 0.2,
        "lighting_contrast": 0.4,
        "expression_strength": 0.3,
        "descriptors": ["confident", "composed", "steady gaze", "subtle power"],
        "lighting": "soft directional light, gentle shadows",
        "negative": "aggressive, angry, harsh"
    },
    IntensityLevel.INTENSE: {
        "facial_tension": 0.5,
        "lighting_contrast": 0.6,
        "expression_strength": 0.6,
        "descriptors": ["intense", "focused", "piercing eyes", "determined"],
        "lighting": "dramatic side lighting, defined shadows, rim light",
        "negative": "soft, gentle, weak"
    },
    IntensityLevel.DOMINANT: {
        "facial_tension": 0.75,
        "lighting_contrast": 0.8,
        "expression_strength": 0.8,
        "descriptors": ["dominant", "commanding", "intimidating presence", "powerful stance"],
        "lighting": "hard rim light, deep moody shadows, high contrast",
        "negative": "submissive, meek, uncertain"
    },
    IntensityLevel.RUTHLESS: {
        "facial_tension": 1.0,
        "lighting_contrast": 0.95,
        "expression_strength": 1.0,
        "descriptors": ["ruthless", "fearless", "controlled rage", "menacing", "apex predator"],
        "lighting": "extreme chiaroscuro, harsh directional light, noir shadows",
        "negative": "friendly, approachable, soft"
    }
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AggressiveImageConfig:
    """Configuration for aggressive image generation"""
    
    # Model settings
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    use_refiner: bool = True
    refiner_strength: float = 0.3
    
    # LoRA settings
    lora_models: List[str] = field(default_factory=list)
    lora_weights: List[float] = field(default_factory=list)
    
    # ControlNet settings
    controlnet_model: Optional[str] = None
    controlnet_conditioning_scale: float = 0.8
    
    # Generation settings
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Quality settings
    high_noise_frac: float = 0.8  # For refiner
    use_karras_sigmas: bool = True
    
    # Face enhancement
    face_enhancement: bool = True
    face_restoration_model: str = "GFPGAN"  # or "CodeFormer"
    face_restoration_weight: float = 0.7
    
    # Intensity
    intensity_level: IntensityLevel = IntensityLevel.DOMINANT
    
    # Output
    output_format: str = "png"
    output_quality: int = 95


@dataclass
class GeneratedImage:
    """Result from image generation"""
    image: Any  # PIL Image or numpy array
    image_path: Optional[Path] = None
    prompt_used: str = ""
    negative_prompt_used: str = ""
    seed: int = -1
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Prompt Templates
# =============================================================================

class AggressivePromptBuilder:
    """Builds optimized prompts for aggressive/powerful image generation"""
    
    # Base quality descriptors
    QUALITY_POSITIVE = [
        "masterpiece", "best quality", "ultra high resolution", "8K UHD",
        "photorealistic", "hyperrealistic", "RAW photo", "professional photography",
        "sharp focus", "intricate details", "extremely detailed face",
        "detailed skin texture", "visible pores", "subsurface scattering"
    ]
    
    QUALITY_NEGATIVE = [
        "blur", "blurry", "out of focus", "soft focus", "motion blur",
        "distortion", "distorted", "warped", "deformed",
        "uncanny valley", "plastic skin", "artificial", "fake",
        "low quality", "low resolution", "jpeg artifacts", "compression artifacts",
        "bad anatomy", "bad proportions", "extra limbs", "missing limbs",
        "disfigured", "mutated", "ugly", "duplicate", "morbid",
        "watermark", "signature", "text", "logo"
    ]
    
    # Face-specific quality
    FACE_QUALITY = [
        "perfect facial symmetry", "anatomically correct face",
        "natural skin texture", "realistic eye reflections",
        "detailed iris", "natural eyelashes", "defined eyebrows",
        "realistic lip texture", "natural teeth", "proper lighting on face"
    ]
    
    # Skin texture details
    SKIN_DETAILS = [
        "hyper-realistic skin", "visible skin pores", "natural skin imperfections",
        "subtle skin texture", "realistic skin sheen", "natural skin tone variations",
        "micro skin details", "realistic stubble texture", "natural aging details"
    ]
    
    # Cinematic camera settings
    CAMERA_SETTINGS = {
        "portrait": "85mm lens, f/1.4 aperture, shallow depth of field, bokeh background",
        "dramatic": "35mm lens, wide angle, dramatic perspective, cinematic framing",
        "closeup": "100mm macro lens, extreme detail, studio lighting, perfect focus",
        "cinematic": "anamorphic lens, 2.39:1 aspect ratio, film grain, cinematic color grading"
    }
    
    def __init__(self):
        self.intensity_modifiers = INTENSITY_MODIFIERS
    
    def build_prompt(
        self,
        subject: str,
        intensity: IntensityLevel = IntensityLevel.DOMINANT,
        emotion: Optional[str] = None,
        camera_style: str = "portrait",
        additional_details: Optional[List[str]] = None,
        custom_lighting: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Build optimized positive and negative prompts
        
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        modifiers = self.intensity_modifiers[intensity]
        
        # Build positive prompt components
        components = []
        
        # Quality foundation
        components.extend(self.QUALITY_POSITIVE[:8])
        
        # Subject with emotion
        if emotion:
            components.append(f"{subject}, {emotion}")
        else:
            components.append(subject)
        
        # Intensity descriptors
        components.extend(modifiers["descriptors"])
        
        # Face and skin quality
        components.extend(self.FACE_QUALITY[:5])
        components.extend(self.SKIN_DETAILS[:4])
        
        # Facial structure emphasis
        components.extend([
            "strong facial structure", "defined jawline", 
            "prominent cheekbones", "facial muscle tension"
        ])
        
        # Lighting
        lighting = custom_lighting or modifiers["lighting"]
        components.append(lighting)
        
        # Camera settings
        if camera_style in self.CAMERA_SETTINGS:
            components.append(self.CAMERA_SETTINGS[camera_style])
        
        # Additional details
        if additional_details:
            components.extend(additional_details)
        
        # Compose positive prompt
        positive_prompt = ", ".join(components)
        
        # Build negative prompt
        negative_components = self.QUALITY_NEGATIVE.copy()
        negative_components.append(modifiers["negative"])
        negative_components.extend([
            "soft features", "round face", "baby face",
            "cartoon", "anime", "illustration", "painting",
            "3D render", "CGI", "artificial lighting"
        ])
        
        negative_prompt = ", ".join(negative_components)
        
        return positive_prompt, negative_prompt
    
    def build_power_portrait_prompt(
        self,
        gender: str = "man",
        age_range: str = "30-40",
        ethnicity: Optional[str] = None,
        intensity: IntensityLevel = IntensityLevel.DOMINANT,
        specific_emotion: str = "controlled intensity"
    ) -> Tuple[str, str]:
        """Build a prompt specifically for power portrait generation"""
        
        # Base subject
        ethnicity_str = f"{ethnicity} " if ethnicity else ""
        subject = f"portrait of a {ethnicity_str}{gender}, age {age_range}"
        
        # Emotion mapping based on intensity
        emotion_map = {
            IntensityLevel.CALM: "confident and composed expression, subtle power",
            IntensityLevel.INTENSE: "intense focused gaze, determined expression",
            IntensityLevel.DOMINANT: "commanding presence, dominant expression, piercing eyes",
            IntensityLevel.RUTHLESS: "ruthless cold stare, predatory gaze, barely contained rage"
        }
        
        emotion = specific_emotion or emotion_map[intensity]
        
        # Additional power descriptors
        power_details = [
            "powerful presence", "alpha energy", "magnetic charisma",
            "battle-worn features", "weathered skin texture",
            "tension in facial muscles", "clenched jaw",
            "sweat beads on skin", "subtle scars"
        ]
        
        return self.build_prompt(
            subject=subject,
            intensity=intensity,
            emotion=emotion,
            camera_style="portrait",
            additional_details=power_details,
            custom_lighting="dramatic Rembrandt lighting, strong key light, deep shadows"
        )


# =============================================================================
# Image Generator Pipeline
# =============================================================================

class AggressiveImageGenerator:
    """
    SDXL-based image generator optimized for aggressive/powerful portraits
    with LoRA and ControlNet support
    """
    
    def __init__(self, config: Optional[AggressiveImageConfig] = None):
        self.config = config or AggressiveImageConfig()
        self.prompt_builder = AggressivePromptBuilder()
        
        # Pipeline references (lazy loaded)
        self._base_pipeline = None
        self._refiner_pipeline = None
        self._controlnet = None
        self._face_enhancer = None
        
        # Device setup
        self.device = self._get_device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Progress callback
        self._progress_callback: Optional[Callable] = None
        
        logger.info(f"AggressiveImageGenerator initialized on {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback if set"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.debug(f"Progress {progress:.1%}: {message}")
    
    def load_models(self):
        """Load all required models"""
        self._report_progress(0.0, "Loading SDXL base model...")
        
        try:
            from diffusers import (
                StableDiffusionXLPipeline,
                StableDiffusionXLImg2ImgPipeline,
                DPMSolverMultistepScheduler,
                ControlNetModel
            )
            
            # Load base pipeline
            self._base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.base_model,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None
            )
            
            # Optimize scheduler
            self._base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self._base_pipeline.scheduler.config,
                use_karras_sigmas=self.config.use_karras_sigmas
            )
            
            self._base_pipeline.to(self.device)
            
            # Enable optimizations
            if self.device.type == "cuda":
                self._base_pipeline.enable_xformers_memory_efficient_attention()
            
            self._report_progress(0.3, "Base model loaded")
            
            # Load refiner if enabled
            if self.config.use_refiner:
                self._report_progress(0.4, "Loading SDXL refiner...")
                self._refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.config.refiner_model,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    variant="fp16" if self.dtype == torch.float16 else None
                )
                self._refiner_pipeline.to(self.device)
                
                if self.device.type == "cuda":
                    self._refiner_pipeline.enable_xformers_memory_efficient_attention()
                
                self._report_progress(0.6, "Refiner loaded")
            
            # Load ControlNet if specified
            if self.config.controlnet_model:
                self._report_progress(0.7, "Loading ControlNet...")
                self._controlnet = ControlNetModel.from_pretrained(
                    self.config.controlnet_model,
                    torch_dtype=self.dtype
                )
                self._controlnet.to(self.device)
                self._report_progress(0.8, "ControlNet loaded")
            
            # Load LoRAs
            if self.config.lora_models:
                self._report_progress(0.85, "Loading LoRA models...")
                self._load_loras()
            
            # Load face enhancer
            if self.config.face_enhancement:
                self._report_progress(0.9, "Loading face enhancer...")
                self._load_face_enhancer()
            
            self._report_progress(1.0, "All models loaded")
            logger.info("All aggressive image generation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_loras(self):
        """Load LoRA models"""
        for lora_path, weight in zip(self.config.lora_models, self.config.lora_weights):
            try:
                self._base_pipeline.load_lora_weights(lora_path)
                logger.info(f"Loaded LoRA: {lora_path} with weight {weight}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA {lora_path}: {e}")
    
    def _load_face_enhancer(self):
        """Load face enhancement model (GFPGAN or CodeFormer)"""
        try:
            if self.config.face_restoration_model == "GFPGAN":
                # Placeholder for GFPGAN loading
                pass
            elif self.config.face_restoration_model == "CodeFormer":
                # Placeholder for CodeFormer loading
                pass
            logger.info(f"Face enhancer {self.config.face_restoration_model} ready")
        except Exception as e:
            logger.warning(f"Face enhancer not available: {e}")
            self._face_enhancer = None
    
    def generate(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        intensity: IntensityLevel = IntensityLevel.DOMINANT,
        subject_description: Optional[str] = None,
        emotion: Optional[str] = None,
        seed: int = -1,
        controlnet_image: Optional[Any] = None,
        num_images: int = 1
    ) -> List[GeneratedImage]:
        """
        Generate aggressive/powerful images
        
        Args:
            prompt: Custom prompt (if None, uses builder with subject_description)
            negative_prompt: Custom negative prompt
            intensity: Intensity level for generation
            subject_description: Description for prompt builder
            emotion: Specific emotion to convey
            seed: Random seed (-1 for random)
            controlnet_image: Optional control image
            num_images: Number of images to generate
            
        Returns:
            List of GeneratedImage results
        """
        if self._base_pipeline is None:
            self.load_models()
        
        start_time = time.time()
        results = []
        
        # Build prompts if not provided
        if prompt is None:
            if subject_description:
                prompt, neg = self.prompt_builder.build_prompt(
                    subject=subject_description,
                    intensity=intensity,
                    emotion=emotion
                )
                negative_prompt = negative_prompt or neg
            else:
                prompt, negative_prompt = self.prompt_builder.build_power_portrait_prompt(
                    intensity=intensity
                )
        
        # Ensure negative prompt
        if negative_prompt is None:
            negative_prompt = ", ".join(AggressivePromptBuilder.QUALITY_NEGATIVE)
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        self._report_progress(0.1, "Starting generation...")
        
        for i in range(num_images):
            try:
                # Generate with base model
                self._report_progress(0.2 + (i * 0.7 / num_images), f"Generating image {i+1}/{num_images}...")
                
                base_output = self._base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=self.config.width,
                    height=self.config.height,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=generator,
                    output_type="latent" if self.config.use_refiner else "pil"
                )
                
                if self.config.use_refiner:
                    # Refine with SDXL refiner
                    self._report_progress(0.5 + (i * 0.4 / num_images), "Refining image...")
                    
                    image = self._refiner_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=base_output.images,
                        num_inference_steps=self.config.num_inference_steps,
                        strength=self.config.refiner_strength,
                        generator=generator
                    ).images[0]
                else:
                    image = base_output.images[0]
                
                # Face enhancement
                if self.config.face_enhancement and self._face_enhancer:
                    self._report_progress(0.85 + (i * 0.1 / num_images), "Enhancing face...")
                    image = self._enhance_face(image)
                
                # Create result
                result = GeneratedImage(
                    image=image,
                    prompt_used=prompt,
                    negative_prompt_used=negative_prompt,
                    seed=seed + i,
                    generation_time=time.time() - start_time,
                    metadata={
                        "intensity": intensity.value,
                        "model": self.config.base_model,
                        "width": self.config.width,
                        "height": self.config.height,
                        "steps": self.config.num_inference_steps,
                        "guidance_scale": self.config.guidance_scale
                    }
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                raise
        
        self._report_progress(1.0, f"Generated {num_images} image(s)")
        return results
    
    def _enhance_face(self, image) -> Any:
        """Apply face enhancement"""
        # Placeholder for face enhancement logic
        return image
    
    def save_image(
        self,
        result: GeneratedImage,
        output_dir: Path,
        filename: Optional[str] = None
    ) -> Path:
        """Save generated image to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"aggressive_{result.seed}_{uuid.uuid4().hex[:8]}.{self.config.output_format}"
        
        output_path = output_dir / filename
        
        if self.config.output_format.lower() == "png":
            result.image.save(output_path, "PNG", optimize=True)
        else:
            result.image.save(output_path, "JPEG", quality=self.config.output_quality)
        
        result.image_path = output_path
        logger.info(f"Image saved to {output_path}")
        return output_path
    
    def unload_models(self):
        """Unload models to free VRAM"""
        if self._base_pipeline:
            del self._base_pipeline
            self._base_pipeline = None
        if self._refiner_pipeline:
            del self._refiner_pipeline
            self._refiner_pipeline = None
        if self._controlnet:
            del self._controlnet
            self._controlnet = None
        if self._face_enhancer:
            del self._face_enhancer
            self._face_enhancer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Models unloaded")


# =============================================================================
# Preset Configurations
# =============================================================================

AGGRESSIVE_PRESETS = {
    "power_portrait": {
        "description": "High-impact power portrait with cinematic lighting",
        "intensity": IntensityLevel.DOMINANT,
        "camera_style": "portrait",
        "lighting": "dramatic Rembrandt lighting, strong shadows"
    },
    "ruthless_closeup": {
        "description": "Extreme closeup with ruthless intensity",
        "intensity": IntensityLevel.RUTHLESS,
        "camera_style": "closeup",
        "lighting": "harsh directional light, noir shadows"
    },
    "intense_cinematic": {
        "description": "Cinematic wide shot with intense presence",
        "intensity": IntensityLevel.INTENSE,
        "camera_style": "cinematic",
        "lighting": "film noir lighting, anamorphic flares"
    },
    "calm_authority": {
        "description": "Calm but authoritative presence",
        "intensity": IntensityLevel.CALM,
        "camera_style": "portrait",
        "lighting": "soft but defined lighting, subtle shadows"
    }
}


def create_generator(preset: str = "power_portrait") -> AggressiveImageGenerator:
    """Create a generator with a preset configuration"""
    config = AggressiveImageConfig()
    
    if preset in AGGRESSIVE_PRESETS:
        preset_config = AGGRESSIVE_PRESETS[preset]
        config.intensity_level = preset_config["intensity"]
    
    return AggressiveImageGenerator(config)
