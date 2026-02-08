"""
Prompt Engine - Intelligent prompt expansion and enhancement

This module handles:
- Parsing raw user prompts
- Expanding prompts with camera, lighting, style information
- Generating negative prompts
- Template-based prompt enhancement
"""

import re
import yaml
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ExpandedPrompt:
    """Structured representation of an expanded prompt"""
    original: str
    expanded: str
    negative: str
    subject: str = ""
    camera_motion: str = "static"
    lighting: str = "natural"
    style: str = "cinematic"
    fps: int = 24
    duration_seconds: float = 6.0
    resolution: tuple = (854, 480)
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "original_prompt": self.original,
            "expanded_prompt": self.expanded,
            "negative_prompt": self.negative,
            "subject": self.subject,
            "camera_motion": self.camera_motion,
            "lighting": self.lighting,
            "style": self.style,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "resolution": list(self.resolution),
            "seed": self.seed,
            "metadata": self.metadata
        }


class PromptEngine:
    """
    Intelligent prompt expansion engine.
    
    Transforms raw user prompts into detailed, model-optimized prompts
    with camera motion, lighting, and style information.
    """
    
    # Keywords for detecting camera motion in prompts
    CAMERA_KEYWORDS = {
        'drone': 'drone',
        'aerial': 'drone',
        'overhead': 'drone',
        'bird': 'drone',
        'pan left': 'pan_left',
        'pan right': 'pan_right',
        'panning': 'pan_right',
        'zoom in': 'zoom_in',
        'zoom out': 'zoom_out',
        'tracking': 'tracking',
        'follow': 'tracking',
        'orbit': 'orbit',
        'rotating': 'orbit',
        'crane': 'crane_up',
        'rising': 'crane_up',
        'descending': 'crane_down',
        'handheld': 'handheld',
        'shaky': 'handheld',
        'dolly': 'dolly_forward',
        'push in': 'dolly_forward',
        'pull back': 'dolly_backward',
        'static': 'static',
        'still': 'static',
        'fixed': 'static'
    }
    
    # Keywords for detecting lighting
    LIGHTING_KEYWORDS = {
        'sunset': 'golden_hour',
        'sunrise': 'golden_hour',
        'golden hour': 'golden_hour',
        'twilight': 'blue_hour',
        'dusk': 'blue_hour',
        'dawn': 'blue_hour',
        'night': 'night',
        'dark': 'night',
        'moonlight': 'moonlight',
        'moon': 'moonlight',
        'neon': 'neon',
        'cyberpunk': 'neon',
        'studio': 'studio',
        'dramatic': 'dramatic',
        'backlit': 'backlit',
        'silhouette': 'backlit',
        'soft light': 'soft',
        'daylight': 'natural',
        'sunny': 'natural'
    }
    
    # Keywords for detecting style
    STYLE_KEYWORDS = {
        'cinematic': 'cinematic',
        'movie': 'cinematic',
        'film': 'cinematic',
        'documentary': 'documentary',
        'realistic': 'photorealistic',
        'photorealistic': 'photorealistic',
        'hyperreal': 'photorealistic',
        'anime': 'anime',
        'animated': 'anime',
        'cartoon': 'anime',
        'artistic': 'artistic',
        'painterly': 'artistic',
        'vintage': 'vintage',
        'retro': 'vintage',
        'old': 'vintage',
        'noir': 'noir',
        'black and white': 'noir',
        'sci-fi': 'sci_fi',
        'futuristic': 'sci_fi',
        'fantasy': 'fantasy',
        'magical': 'fantasy',
        'minimalist': 'minimalist'
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the prompt engine.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.templates = self._load_templates()
        self.defaults = self._load_defaults()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load prompt templates from YAML"""
        template_path = self.config_dir / "prompt_templates.yaml"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default settings from YAML"""
        defaults_path = self.config_dir / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def detect_camera_motion(self, prompt: str) -> str:
        """Detect camera motion from prompt keywords"""
        prompt_lower = prompt.lower()
        for keyword, motion in self.CAMERA_KEYWORDS.items():
            if keyword in prompt_lower:
                return motion
        return "static"
    
    def detect_lighting(self, prompt: str) -> str:
        """Detect lighting style from prompt keywords"""
        prompt_lower = prompt.lower()
        for keyword, lighting in self.LIGHTING_KEYWORDS.items():
            if keyword in prompt_lower:
                return lighting
        return "natural"
    
    def detect_style(self, prompt: str) -> str:
        """Detect visual style from prompt keywords"""
        prompt_lower = prompt.lower()
        for keyword, style in self.STYLE_KEYWORDS.items():
            if keyword in prompt_lower:
                return style
        return "cinematic"
    
    def extract_subject(self, prompt: str) -> str:
        """Extract the main subject from the prompt"""
        # Remove common modifiers and extract core subject
        prompt_clean = prompt.lower()
        
        # Remove common prefixes
        prefixes = [
            r'^a\s+', r'^an\s+', r'^the\s+',
            r'^cinematic\s+', r'^beautiful\s+', r'^stunning\s+'
        ]
        for prefix in prefixes:
            prompt_clean = re.sub(prefix, '', prompt_clean)
        
        # Extract first noun phrase (simplified)
        words = prompt_clean.split()
        if words:
            # Take first 3-5 words as subject
            subject_words = words[:min(5, len(words))]
            return ' '.join(subject_words)
        
        return prompt_clean
    
    def get_camera_tags(self, motion: str) -> List[str]:
        """Get camera motion tags from templates"""
        camera_templates = self.templates.get('camera_motion', {})
        if motion in camera_templates:
            return camera_templates[motion].get('tags', [])
        return []
    
    def get_lighting_tags(self, lighting: str) -> List[str]:
        """Get lighting tags from templates"""
        lighting_templates = self.templates.get('lighting', {})
        if lighting in lighting_templates:
            return lighting_templates[lighting].get('tags', [])
        return []
    
    def get_style_tags(self, style: str) -> List[str]:
        """Get style tags from templates"""
        style_templates = self.templates.get('styles', {})
        if style in style_templates:
            return style_templates[style].get('tags', [])
        return []
    
    def generate_negative_prompt(self, style: str = None) -> str:
        """Generate appropriate negative prompt"""
        # Get default negative prompts
        prompt_config = self.defaults.get('prompt_expansion', {})
        negative_defaults = prompt_config.get('negative_prompt_defaults', [
            "blurry", "low quality", "distorted", "watermark",
            "text", "logo", "artifact", "noise"
        ])
        
        # Add style-specific negative prompts
        style_negatives = {
            'photorealistic': ['cartoon', 'anime', 'drawing', 'painting'],
            'anime': ['realistic', 'photorealistic', '3d render'],
            'cinematic': ['amateur', 'home video', 'low budget'],
            'noir': ['colorful', 'bright', 'saturated']
        }
        
        negatives = negative_defaults.copy()
        if style and style in style_negatives:
            negatives.extend(style_negatives[style])
        
        return ", ".join(negatives)
    
    def expand(
        self,
        prompt: str,
        camera_motion: Optional[str] = None,
        lighting: Optional[str] = None,
        style: Optional[str] = None,
        fps: int = 24,
        duration: float = 6.0,
        resolution: tuple = (854, 480),
        seed: Optional[int] = None,
        enhance_quality: bool = True
    ) -> ExpandedPrompt:
        """
        Expand a raw prompt into a detailed, model-optimized prompt.
        
        Args:
            prompt: Raw user prompt
            camera_motion: Override camera motion (auto-detected if None)
            lighting: Override lighting style (auto-detected if None)
            style: Override visual style (auto-detected if None)
            fps: Frames per second
            duration: Video duration in seconds
            resolution: Output resolution (width, height)
            seed: Random seed for reproducibility
            enhance_quality: Whether to add quality enhancement tags
            
        Returns:
            ExpandedPrompt with all details
        """
        # Auto-detect from prompt if not specified
        detected_camera = self.detect_camera_motion(prompt)
        detected_lighting = self.detect_lighting(prompt)
        detected_style = self.detect_style(prompt)
        
        # Use specified or detected values
        final_camera = camera_motion or detected_camera
        final_lighting = lighting or detected_lighting
        final_style = style or detected_style
        
        # Extract subject
        subject = self.extract_subject(prompt)
        
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Build expanded prompt
        expanded_parts = [prompt]
        
        # Add camera motion tags
        camera_tags = self.get_camera_tags(final_camera)
        if camera_tags:
            expanded_parts.append(camera_tags[0])  # Add primary tag
        
        # Add lighting tags
        lighting_tags = self.get_lighting_tags(final_lighting)
        if lighting_tags:
            expanded_parts.append(lighting_tags[0])
        
        # Add style tags
        style_tags = self.get_style_tags(final_style)
        if style_tags:
            expanded_parts.extend(style_tags[:2])  # Add first 2 style tags
        
        # Add quality enhancement
        if enhance_quality:
            prompt_config = self.defaults.get('prompt_expansion', {})
            quality_tags = prompt_config.get('quality_tags', [
                "high quality", "detailed", "professional"
            ])
            expanded_parts.extend(quality_tags)
        
        # Combine into final prompt
        expanded = ", ".join(expanded_parts)
        
        # Generate negative prompt
        negative = self.generate_negative_prompt(final_style)
        
        # Create expanded prompt object
        return ExpandedPrompt(
            original=prompt,
            expanded=expanded,
            negative=negative,
            subject=subject,
            camera_motion=final_camera,
            lighting=final_lighting,
            style=final_style,
            fps=fps,
            duration_seconds=duration,
            resolution=resolution,
            seed=seed,
            metadata={
                "detected_camera": detected_camera,
                "detected_lighting": detected_lighting,
                "detected_style": detected_style,
                "quality_enhanced": enhance_quality
            }
        )
    
    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Validate a user prompt.
        
        Args:
            prompt: Raw user prompt
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"
        
        if len(prompt) < 3:
            return False, "Prompt is too short (minimum 3 characters)"
        
        if len(prompt) > 1000:
            return False, "Prompt is too long (maximum 1000 characters)"
        
        # Check for potentially problematic content (basic check)
        # In production, you'd want more sophisticated content filtering
        
        return True, ""
    
    def parse_generation_params(self, prompt: str) -> Dict[str, Any]:
        """
        Parse inline generation parameters from prompt.
        
        Supports formats like:
        - "prompt --fps 30 --duration 10"
        - "prompt [720p] [8s]"
        
        Args:
            prompt: Raw prompt potentially containing parameters
            
        Returns:
            Dictionary with parsed parameters and cleaned prompt
        """
        params = {}
        clean_prompt = prompt
        
        # Parse --key value format
        param_pattern = r'--(\w+)\s+(\S+)'
        matches = re.findall(param_pattern, prompt)
        for key, value in matches:
            if key in ['fps', 'steps', 'seed']:
                params[key] = int(value)
            elif key in ['duration', 'cfg', 'guidance']:
                params[key] = float(value)
            else:
                params[key] = value
        
        # Remove parsed parameters from prompt
        clean_prompt = re.sub(param_pattern, '', clean_prompt).strip()
        
        # Parse [resolution] format
        res_pattern = r'\[(\d+)p\]'
        res_match = re.search(res_pattern, clean_prompt)
        if res_match:
            height = int(res_match.group(1))
            width = int(height * 16 / 9)  # Assume 16:9
            params['resolution'] = (width, height)
            clean_prompt = re.sub(res_pattern, '', clean_prompt).strip()
        
        # Parse [duration] format
        dur_pattern = r'\[(\d+)s\]'
        dur_match = re.search(dur_pattern, clean_prompt)
        if dur_match:
            params['duration'] = int(dur_match.group(1))
            clean_prompt = re.sub(dur_pattern, '', clean_prompt).strip()
        
        params['clean_prompt'] = clean_prompt
        return params


def main():
    """Test the prompt engine"""
    engine = PromptEngine()
    
    test_prompts = [
        "A cinematic drone shot over the ocean at night",
        "A cat walking through a neon-lit cyberpunk city",
        "Sunset timelapse of mountains with golden hour lighting",
        "A person running through a forest, tracking shot",
        "Abstract colorful shapes morphing and flowing"
    ]
    
    print("=" * 60)
    print("PROMPT ENGINE TEST")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nOriginal: {prompt}")
        expanded = engine.expand(prompt)
        print(f"Expanded: {expanded.expanded}")
        print(f"Negative: {expanded.negative}")
        print(f"Camera: {expanded.camera_motion}")
        print(f"Lighting: {expanded.lighting}")
        print(f"Style: {expanded.style}")
        print(f"Seed: {expanded.seed}")
        print("-" * 40)


if __name__ == "__main__":
    main()
