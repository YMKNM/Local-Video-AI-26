"""
Temporal Prompt Generator - Advanced Prompt Engineering Engine

This module breaks down user requests into structured video generation prompts with:
- Shot lists and scene decomposition
- Scene actions and transitions
- Emotional cues and mood descriptors
- Camera motion and cinematography
- Sound and music cues
- Character choreography
- Temporal coherence markers
"""

import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ShotType(Enum):
    """Cinematic shot types"""
    EXTREME_WIDE = "extreme_wide_shot"
    WIDE = "wide_shot"
    FULL = "full_shot"
    MEDIUM_WIDE = "medium_wide_shot"
    MEDIUM = "medium_shot"
    MEDIUM_CLOSEUP = "medium_close_up"
    CLOSEUP = "close_up"
    EXTREME_CLOSEUP = "extreme_close_up"
    INSERT = "insert_shot"
    POV = "point_of_view"
    OVER_SHOULDER = "over_the_shoulder"
    TWO_SHOT = "two_shot"


class CameraMovement(Enum):
    """Camera movement types"""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRUCK_LEFT = "truck_left"
    TRUCK_RIGHT = "truck_right"
    PEDESTAL_UP = "pedestal_up"
    PEDESTAL_DOWN = "pedestal_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    CRANE_UP = "crane_up"
    CRANE_DOWN = "crane_down"
    ARC_LEFT = "arc_left"
    ARC_RIGHT = "arc_right"
    HANDHELD = "handheld"
    STEADICAM = "steadicam"
    DRONE_AERIAL = "drone_aerial"
    TRACKING = "tracking"
    WHIP_PAN = "whip_pan"
    DUTCH_ANGLE = "dutch_angle"


class Emotion(Enum):
    """Emotional tone/mood"""
    NEUTRAL = "neutral"
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    TENSE = "tense"
    PEACEFUL = "peaceful"
    DRAMATIC = "dramatic"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    OMINOUS = "ominous"
    TRIUMPHANT = "triumphant"
    NOSTALGIC = "nostalgic"
    ETHEREAL = "ethereal"
    ENERGETIC = "energetic"
    CONTEMPLATIVE = "contemplative"


class LightingType(Enum):
    """Lighting styles"""
    NATURAL = "natural_lighting"
    GOLDEN_HOUR = "golden_hour"
    BLUE_HOUR = "blue_hour"
    OVERCAST = "overcast_diffused"
    DRAMATIC = "dramatic_lighting"
    HIGH_KEY = "high_key"
    LOW_KEY = "low_key"
    SILHOUETTE = "silhouette"
    RIM_LIGHT = "rim_light"
    NEON = "neon_lights"
    CANDLELIGHT = "candlelight"
    MOONLIGHT = "moonlight"
    STUDIO = "studio_lighting"
    CHIAROSCURO = "chiaroscuro"


class AudioCue(Enum):
    """Audio/sound cue types"""
    AMBIENT = "ambient_sound"
    MUSIC_DRAMATIC = "dramatic_score"
    MUSIC_PEACEFUL = "peaceful_melody"
    MUSIC_TENSE = "tense_underscore"
    MUSIC_EPIC = "epic_orchestral"
    MUSIC_MINIMAL = "minimal_electronic"
    DIALOGUE = "dialogue"
    SOUND_EFFECT = "sound_effect"
    NATURE = "nature_sounds"
    URBAN = "urban_ambience"
    SILENCE = "silence"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Shot:
    """Represents a single shot in a sequence"""
    shot_number: int
    shot_type: ShotType
    description: str
    duration_seconds: float
    camera_movement: CameraMovement
    camera_speed: str  # slow, medium, fast
    lighting: LightingType
    emotion: Emotion
    subjects: List[str]
    actions: List[str]
    audio_cue: AudioCue
    audio_description: str
    transition_in: str
    transition_out: str
    guidance_weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Convert shot to a generation prompt"""
        parts = []
        
        # Shot type
        shot_name = self.shot_type.value.replace('_', ' ')
        parts.append(shot_name)
        
        # Main description
        parts.append(self.description)
        
        # Camera
        if self.camera_movement != CameraMovement.STATIC:
            cam = self.camera_movement.value.replace('_', ' ')
            parts.append(f"{cam} camera movement, {self.camera_speed} speed")
        
        # Lighting
        light = self.lighting.value.replace('_', ' ')
        parts.append(light)
        
        # Mood
        parts.append(f"{self.emotion.value} mood")
        
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'shot_number': self.shot_number,
            'shot_type': self.shot_type.value,
            'description': self.description,
            'duration_seconds': self.duration_seconds,
            'camera_movement': self.camera_movement.value,
            'camera_speed': self.camera_speed,
            'lighting': self.lighting.value,
            'emotion': self.emotion.value,
            'subjects': self.subjects,
            'actions': self.actions,
            'audio_cue': self.audio_cue.value,
            'audio_description': self.audio_description,
            'transition_in': self.transition_in,
            'transition_out': self.transition_out,
            'guidance_weight': self.guidance_weight,
            'prompt': self.to_prompt(),
            'metadata': self.metadata
        }


@dataclass
class Character:
    """Character descriptor for choreography"""
    name: str
    description: str
    position: str  # left, center, right, foreground, background
    actions: List[str]
    expression: str
    clothing: str
    movement_path: str
    interactions: List[str]


@dataclass
class Scene:
    """Represents a complete scene with multiple shots"""
    scene_number: int
    scene_name: str
    location: str
    time_of_day: str
    weather: str
    duration_seconds: float
    shots: List[Shot]
    characters: List[Character]
    props: List[str]
    overall_emotion: Emotion
    narrative_purpose: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene_number': self.scene_number,
            'scene_name': self.scene_name,
            'location': self.location,
            'time_of_day': self.time_of_day,
            'weather': self.weather,
            'duration_seconds': self.duration_seconds,
            'shots': [s.to_dict() for s in self.shots],
            'characters': [vars(c) for c in self.characters],
            'props': self.props,
            'overall_emotion': self.overall_emotion.value,
            'narrative_purpose': self.narrative_purpose,
            'metadata': self.metadata
        }


@dataclass
class ShotList:
    """Complete shot list for video generation"""
    title: str
    total_duration_seconds: float
    scenes: List[Scene]
    style_preset: str
    color_palette: List[str]
    aspect_ratio: str
    fps: int
    resolution: Tuple[int, int]
    global_negative_prompt: str
    audio_track: str
    temporal_markers: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'total_duration_seconds': self.total_duration_seconds,
            'scenes': [s.to_dict() for s in self.scenes],
            'style_preset': self.style_preset,
            'color_palette': self.color_palette,
            'aspect_ratio': self.aspect_ratio,
            'fps': self.fps,
            'resolution': list(self.resolution),
            'global_negative_prompt': self.global_negative_prompt,
            'audio_track': self.audio_track,
            'temporal_markers': self.temporal_markers,
            'metadata': self.metadata
        }
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """Get all shot prompts with timing information"""
        prompts = []
        current_time = 0.0
        
        for scene in self.scenes:
            for shot in scene.shots:
                prompts.append({
                    'prompt': shot.to_prompt(),
                    'start_time': current_time,
                    'end_time': current_time + shot.duration_seconds,
                    'duration': shot.duration_seconds,
                    'scene': scene.scene_name,
                    'shot_number': shot.shot_number,
                    'guidance_weight': shot.guidance_weight
                })
                current_time += shot.duration_seconds
        
        return prompts


# =============================================================================
# Temporal Prompt Generator
# =============================================================================

class TemporalPromptGenerator:
    """
    Advanced prompt engineering engine for video generation.
    
    Breaks down natural language requests into structured shot lists with:
    - Cinematic shot composition
    - Camera movements and transitions
    - Lighting and mood descriptors
    - Character choreography
    - Sound cue integration
    - Temporal coherence markers
    """
    
    # Pattern matching for prompt analysis
    CAMERA_PATTERNS = {
        r'\b(drone|aerial|bird\'?s?\s*eye)\b': CameraMovement.DRONE_AERIAL,
        r'\b(pan\s*left|panning\s*left)\b': CameraMovement.PAN_LEFT,
        r'\b(pan\s*right|panning\s*right)\b': CameraMovement.PAN_RIGHT,
        r'\b(zoom\s*in|zooming\s*in)\b': CameraMovement.ZOOM_IN,
        r'\b(zoom\s*out|zooming\s*out)\b': CameraMovement.ZOOM_OUT,
        r'\b(tracking|follow)\b': CameraMovement.TRACKING,
        r'\b(orbit|circl|arc)\b': CameraMovement.ARC_RIGHT,
        r'\b(crane\s*up|rising)\b': CameraMovement.CRANE_UP,
        r'\b(crane\s*down|descend)\b': CameraMovement.CRANE_DOWN,
        r'\b(dolly\s*in|push\s*in)\b': CameraMovement.DOLLY_IN,
        r'\b(dolly\s*out|pull\s*out)\b': CameraMovement.DOLLY_OUT,
        r'\b(handheld|shaky)\b': CameraMovement.HANDHELD,
        r'\b(steadicam|smooth)\b': CameraMovement.STEADICAM,
        r'\b(whip\s*pan)\b': CameraMovement.WHIP_PAN,
        r'\b(dutch|tilted)\b': CameraMovement.DUTCH_ANGLE,
    }
    
    EMOTION_PATTERNS = {
        r'\b(happy|joy|cheerful|bright)\b': Emotion.JOYFUL,
        r'\b(sad|melanchol|somber)\b': Emotion.MELANCHOLIC,
        r'\b(tense|suspense|anxious)\b': Emotion.TENSE,
        r'\b(peace|calm|serene|tranquil)\b': Emotion.PEACEFUL,
        r'\b(dramatic|intense)\b': Emotion.DRAMATIC,
        r'\b(myster|enigmatic|curious)\b': Emotion.MYSTERIOUS,
        r'\b(romantic|love|intimate)\b': Emotion.ROMANTIC,
        r'\b(ominous|foreboding|dark)\b': Emotion.OMINOUS,
        r'\b(triumph|victorious|epic)\b': Emotion.TRIUMPHANT,
        r'\b(nostalgic|remember|past)\b': Emotion.NOSTALGIC,
        r'\b(ethereal|dream|magical)\b': Emotion.ETHEREAL,
        r'\b(energetic|dynamic|exciting)\b': Emotion.ENERGETIC,
    }
    
    LIGHTING_PATTERNS = {
        r'\b(sunset|golden\s*hour|sunrise)\b': LightingType.GOLDEN_HOUR,
        r'\b(blue\s*hour|twilight|dusk)\b': LightingType.BLUE_HOUR,
        r'\b(night|moon|dark)\b': LightingType.MOONLIGHT,
        r'\b(neon|cyberpunk|city\s*lights)\b': LightingType.NEON,
        r'\b(dramatic\s*light|chiaroscuro)\b': LightingType.DRAMATIC,
        r'\b(silhouette|backlit)\b': LightingType.SILHOUETTE,
        r'\b(overcast|cloudy|diffused)\b': LightingType.OVERCAST,
        r'\b(candle|fire|warm\s*glow)\b': LightingType.CANDLELIGHT,
        r'\b(studio|professional)\b': LightingType.STUDIO,
    }
    
    SHOT_PATTERNS = {
        r'\b(extreme\s*wide|establishing)\b': ShotType.EXTREME_WIDE,
        r'\b(wide\s*shot|landscape)\b': ShotType.WIDE,
        r'\b(full\s*shot|full\s*body)\b': ShotType.FULL,
        r'\b(medium\s*shot)\b': ShotType.MEDIUM,
        r'\b(close\s*up|closeup)\b': ShotType.CLOSEUP,
        r'\b(extreme\s*close\s*up|macro)\b': ShotType.EXTREME_CLOSEUP,
        r'\b(pov|point\s*of\s*view|first\s*person)\b': ShotType.POV,
        r'\b(over\s*the\s*shoulder)\b': ShotType.OVER_SHOULDER,
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the temporal prompt generator.
        
        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.templates = self._load_templates()
        self.style_presets = self._load_style_presets()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load prompt templates"""
        template_path = self.config_dir / "prompt_templates.yaml"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_style_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load style presets"""
        return {
            'cinematic': {
                'aspect_ratio': '2.39:1',
                'color_grading': 'filmic',
                'depth_of_field': 'shallow',
                'grain': 'subtle',
                'quality_tags': ['cinematic', 'film grain', 'anamorphic lens flare']
            },
            'documentary': {
                'aspect_ratio': '16:9',
                'color_grading': 'natural',
                'depth_of_field': 'deep',
                'quality_tags': ['documentary style', 'natural lighting', 'authentic']
            },
            'music_video': {
                'aspect_ratio': '16:9',
                'color_grading': 'vibrant',
                'quality_tags': ['music video', 'stylized', 'dynamic cuts']
            },
            'commercial': {
                'aspect_ratio': '16:9',
                'color_grading': 'clean',
                'quality_tags': ['commercial', 'polished', 'product shot']
            },
            'anime': {
                'aspect_ratio': '16:9',
                'style': 'anime',
                'quality_tags': ['anime style', 'cel shaded', 'vibrant colors']
            },
            'noir': {
                'aspect_ratio': '1.85:1',
                'color_grading': 'high contrast',
                'quality_tags': ['film noir', 'black and white', 'dramatic shadows']
            }
        }
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a natural language prompt and extract cinematic elements.
        
        Args:
            prompt: User's natural language prompt
            
        Returns:
            Dictionary with detected elements
        """
        prompt_lower = prompt.lower()
        
        # Detect camera movement
        camera_movement = CameraMovement.STATIC
        for pattern, movement in self.CAMERA_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                camera_movement = movement
                break
        
        # Detect emotion
        emotion = Emotion.NEUTRAL
        for pattern, emot in self.EMOTION_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                emotion = emot
                break
        
        # Detect lighting
        lighting = LightingType.NATURAL
        for pattern, light in self.LIGHTING_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                lighting = light
                break
        
        # Detect shot type
        shot_type = ShotType.MEDIUM
        for pattern, shot in self.SHOT_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                shot_type = shot
                break
        
        # Extract subjects (nouns)
        subjects = self._extract_subjects(prompt)
        
        # Extract actions (verbs)
        actions = self._extract_actions(prompt)
        
        return {
            'camera_movement': camera_movement,
            'emotion': emotion,
            'lighting': lighting,
            'shot_type': shot_type,
            'subjects': subjects,
            'actions': actions,
            'original_prompt': prompt
        }
    
    def _extract_subjects(self, prompt: str) -> List[str]:
        """Extract subject nouns from prompt"""
        # Simple extraction - in production would use NLP
        common_subjects = [
            'person', 'man', 'woman', 'child', 'animal', 'dog', 'cat',
            'car', 'building', 'tree', 'mountain', 'ocean', 'sky',
            'city', 'forest', 'desert', 'beach', 'river', 'street'
        ]
        
        found = []
        prompt_lower = prompt.lower()
        for subject in common_subjects:
            if subject in prompt_lower:
                found.append(subject)
        
        return found if found else ['subject']
    
    def _extract_actions(self, prompt: str) -> List[str]:
        """Extract action verbs from prompt"""
        common_actions = [
            'walking', 'running', 'standing', 'sitting', 'flying',
            'moving', 'dancing', 'playing', 'looking', 'watching',
            'driving', 'swimming', 'climbing', 'falling', 'rising'
        ]
        
        found = []
        prompt_lower = prompt.lower()
        for action in common_actions:
            if action in prompt_lower:
                found.append(action)
        
        return found if found else ['existing']
    
    def generate_shot_list(
        self,
        prompt: str,
        duration_seconds: float = 10.0,
        style_preset: str = 'cinematic',
        num_shots: Optional[int] = None,
        fps: int = 24,
        resolution: Tuple[int, int] = (1920, 1080)
    ) -> ShotList:
        """
        Generate a complete shot list from a natural language prompt.
        
        Args:
            prompt: User's description of the video
            duration_seconds: Total video duration
            style_preset: Visual style preset
            num_shots: Number of shots (auto-calculated if None)
            fps: Frames per second
            resolution: Output resolution
            
        Returns:
            Complete ShotList object
        """
        # Analyze the prompt
        analysis = self.analyze_prompt(prompt)
        
        # Calculate number of shots
        if num_shots is None:
            # Roughly 3-5 seconds per shot for cinematic feel
            num_shots = max(1, int(duration_seconds / 4))
        
        # Calculate duration per shot
        shot_duration = duration_seconds / num_shots
        
        # Generate shots
        shots = []
        for i in range(num_shots):
            shot = self._generate_shot(
                shot_number=i + 1,
                analysis=analysis,
                duration=shot_duration,
                is_first=(i == 0),
                is_last=(i == num_shots - 1),
                style_preset=style_preset
            )
            shots.append(shot)
        
        # Create scene (single scene for simple prompts)
        scene = Scene(
            scene_number=1,
            scene_name="Main Scene",
            location=self._infer_location(prompt),
            time_of_day=self._infer_time_of_day(analysis),
            weather=self._infer_weather(prompt),
            duration_seconds=duration_seconds,
            shots=shots,
            characters=[],
            props=[],
            overall_emotion=analysis['emotion'],
            narrative_purpose="Main narrative"
        )
        
        # Get style config
        style_config = self.style_presets.get(style_preset, self.style_presets['cinematic'])
        
        # Create shot list
        shot_list = ShotList(
            title=self._generate_title(prompt),
            total_duration_seconds=duration_seconds,
            scenes=[scene],
            style_preset=style_preset,
            color_palette=self._generate_color_palette(analysis),
            aspect_ratio=style_config.get('aspect_ratio', '16:9'),
            fps=fps,
            resolution=resolution,
            global_negative_prompt=self._generate_negative_prompt(style_preset),
            audio_track=self._suggest_audio(analysis),
            temporal_markers=self._generate_temporal_markers(shots)
        )
        
        return shot_list
    
    def _generate_shot(
        self,
        shot_number: int,
        analysis: Dict[str, Any],
        duration: float,
        is_first: bool,
        is_last: bool,
        style_preset: str
    ) -> Shot:
        """Generate a single shot"""
        
        # Vary shot type through the sequence
        shot_types = [
            ShotType.WIDE, ShotType.MEDIUM, ShotType.CLOSEUP,
            ShotType.MEDIUM_WIDE, ShotType.MEDIUM_CLOSEUP
        ]
        
        if is_first:
            shot_type = ShotType.WIDE  # Establishing shot
        elif is_last:
            shot_type = analysis['shot_type']
        else:
            shot_type = random.choice(shot_types)
        
        # Vary camera movement
        if is_first:
            camera = CameraMovement.STATIC  # Let viewer orient
        else:
            camera = analysis['camera_movement']
        
        # Transitions
        transition_in = "fade in" if is_first else "cut"
        transition_out = "fade out" if is_last else "cut"
        
        # Audio cue based on emotion
        audio_map = {
            Emotion.JOYFUL: AudioCue.MUSIC_PEACEFUL,
            Emotion.MELANCHOLIC: AudioCue.MUSIC_MINIMAL,
            Emotion.TENSE: AudioCue.MUSIC_TENSE,
            Emotion.PEACEFUL: AudioCue.AMBIENT,
            Emotion.DRAMATIC: AudioCue.MUSIC_DRAMATIC,
            Emotion.TRIUMPHANT: AudioCue.MUSIC_EPIC,
        }
        audio_cue = audio_map.get(analysis['emotion'], AudioCue.AMBIENT)
        
        return Shot(
            shot_number=shot_number,
            shot_type=shot_type,
            description=analysis['original_prompt'],
            duration_seconds=duration,
            camera_movement=camera,
            camera_speed="medium",
            lighting=analysis['lighting'],
            emotion=analysis['emotion'],
            subjects=analysis['subjects'],
            actions=analysis['actions'],
            audio_cue=audio_cue,
            audio_description=f"{audio_cue.value} matching {analysis['emotion'].value} mood",
            transition_in=transition_in,
            transition_out=transition_out,
            guidance_weight=7.5
        )
    
    def _infer_location(self, prompt: str) -> str:
        """Infer location from prompt"""
        locations = {
            'beach': 'beach',
            'ocean': 'ocean',
            'mountain': 'mountains',
            'forest': 'forest',
            'city': 'city',
            'street': 'urban street',
            'room': 'interior room',
            'studio': 'studio',
            'desert': 'desert',
            'snow': 'snowy landscape'
        }
        
        prompt_lower = prompt.lower()
        for key, location in locations.items():
            if key in prompt_lower:
                return location
        
        return "unspecified location"
    
    def _infer_time_of_day(self, analysis: Dict[str, Any]) -> str:
        """Infer time of day from lighting"""
        lighting_to_time = {
            LightingType.GOLDEN_HOUR: "golden hour",
            LightingType.BLUE_HOUR: "blue hour",
            LightingType.MOONLIGHT: "night",
            LightingType.NATURAL: "day",
            LightingType.NEON: "night"
        }
        return lighting_to_time.get(analysis['lighting'], "day")
    
    def _infer_weather(self, prompt: str) -> str:
        """Infer weather from prompt"""
        weather_words = {
            'rain': 'rainy',
            'storm': 'stormy',
            'snow': 'snowy',
            'fog': 'foggy',
            'mist': 'misty',
            'cloud': 'cloudy',
            'sun': 'sunny',
            'clear': 'clear'
        }
        
        prompt_lower = prompt.lower()
        for key, weather in weather_words.items():
            if key in prompt_lower:
                return weather
        
        return "clear"
    
    def _generate_title(self, prompt: str) -> str:
        """Generate a title from prompt"""
        # Take first few words
        words = prompt.split()[:5]
        return ' '.join(words).title()
    
    def _generate_color_palette(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate color palette based on mood"""
        palettes = {
            Emotion.JOYFUL: ['bright yellow', 'warm orange', 'sky blue'],
            Emotion.MELANCHOLIC: ['muted blue', 'grey', 'soft purple'],
            Emotion.TENSE: ['deep red', 'black', 'harsh white'],
            Emotion.PEACEFUL: ['soft green', 'light blue', 'cream'],
            Emotion.DRAMATIC: ['deep contrast', 'rich shadows', 'highlight'],
            Emotion.MYSTERIOUS: ['deep purple', 'dark blue', 'silver'],
            Emotion.ROMANTIC: ['soft pink', 'warm red', 'gold'],
            Emotion.OMINOUS: ['dark grey', 'blood red', 'black'],
            Emotion.TRIUMPHANT: ['gold', 'royal blue', 'bright white'],
        }
        return palettes.get(analysis['emotion'], ['natural', 'balanced', 'neutral'])
    
    def _generate_negative_prompt(self, style_preset: str) -> str:
        """Generate negative prompt"""
        base_negative = [
            "blurry", "low quality", "distorted", "watermark", "text",
            "logo", "artifact", "noise", "grainy", "oversaturated",
            "undersaturated", "overexposed", "underexposed", "jpeg artifacts"
        ]
        
        style_specific = {
            'cinematic': ["amateur", "home video", "shaky cam"],
            'documentary': ["staged", "artificial", "CGI"],
            'anime': ["realistic", "photographic", "3D render"],
            'noir': ["color", "bright", "cheerful"]
        }
        
        negative = base_negative + style_specific.get(style_preset, [])
        return ", ".join(negative)
    
    def _suggest_audio(self, analysis: Dict[str, Any]) -> str:
        """Suggest audio track description"""
        suggestions = {
            Emotion.JOYFUL: "upbeat orchestral with light percussion",
            Emotion.MELANCHOLIC: "soft piano with ambient strings",
            Emotion.TENSE: "building electronic with sharp staccato",
            Emotion.PEACEFUL: "ambient nature sounds with soft melody",
            Emotion.DRAMATIC: "full orchestral with building crescendo",
            Emotion.MYSTERIOUS: "ethereal synth with sparse notes",
            Emotion.TRIUMPHANT: "epic brass and timpani",
        }
        return suggestions.get(analysis['emotion'], "ambient underscore")
    
    def _generate_temporal_markers(self, shots: List[Shot]) -> List[Dict[str, Any]]:
        """Generate temporal coherence markers"""
        markers = []
        current_time = 0.0
        
        for shot in shots:
            markers.append({
                'time': current_time,
                'type': 'shot_start',
                'shot_number': shot.shot_number,
                'description': f"Shot {shot.shot_number} begins"
            })
            current_time += shot.duration_seconds
        
        return markers
    
    def expand_for_generation(
        self,
        shot_list: ShotList,
        include_quality_tags: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Expand shot list into generation-ready prompts.
        
        Args:
            shot_list: Complete shot list
            include_quality_tags: Whether to add quality enhancement tags
            
        Returns:
            List of generation prompts with metadata
        """
        style_config = self.style_presets.get(shot_list.style_preset, {})
        quality_tags = style_config.get('quality_tags', [])
        
        generation_prompts = []
        
        for prompt_info in shot_list.get_all_prompts():
            expanded_prompt = prompt_info['prompt']
            
            if include_quality_tags:
                expanded_prompt = f"{expanded_prompt}, {', '.join(quality_tags)}"
            
            generation_prompts.append({
                'prompt': expanded_prompt,
                'negative_prompt': shot_list.global_negative_prompt,
                'start_time': prompt_info['start_time'],
                'end_time': prompt_info['end_time'],
                'duration': prompt_info['duration'],
                'num_frames': int(prompt_info['duration'] * shot_list.fps),
                'guidance_scale': prompt_info['guidance_weight'],
                'scene': prompt_info['scene'],
                'shot_number': prompt_info['shot_number']
            })
        
        return generation_prompts


# =============================================================================
# Convenience Functions
# =============================================================================

def create_shot_list_from_prompt(
    prompt: str,
    duration: float = 10.0,
    style: str = 'cinematic'
) -> ShotList:
    """
    Quick function to generate a shot list from a prompt.
    
    Args:
        prompt: Natural language description
        duration: Video duration in seconds
        style: Style preset name
        
    Returns:
        Complete ShotList
    """
    generator = TemporalPromptGenerator()
    return generator.generate_shot_list(prompt, duration, style)
