"""
Aggressive & Powerful Generator UI Tab

Dual-stage generation system for ultra-realistic,
dominant images with facial motion animation.
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Any
import logging
import time
import uuid

# Local imports - use relative imports within the package
try:
    from ..generators.aggressive_image import (
        AggressiveImageGenerator,
        AggressiveImageConfig,
        IntensityLevel,
        AggressivePromptBuilder,
        AGGRESSIVE_PRESETS
    )
    from ..generators.image_to_motion import (
        ImageToMotionEngine,
        MotionConfig,
        ExpressionType,
        LipSyncMode,
        MotionIntensity
    )
except ImportError:
    # Fallback for development - define stubs
    AggressiveImageGenerator = None
    AggressiveImageConfig = None
    IntensityLevel = None
    AggressivePromptBuilder = None
    AGGRESSIVE_PRESETS = {}
    ImageToMotionEngine = None
    MotionConfig = None
    ExpressionType = None
    LipSyncMode = None
    MotionIntensity = None

logger = logging.getLogger(__name__)


# =============================================================================
# Global State
# =============================================================================

class GeneratorState:
    """Global state for the generator"""
    image_generator: Optional[Any] = None
    motion_engine: Optional[Any] = None
    current_image: Optional[np.ndarray] = None
    current_image_path: Optional[Path] = None
    generation_history: List[dict] = []


STATE = GeneratorState()


# =============================================================================
# Intensity Mapping
# =============================================================================

INTENSITY_LABELS = {
    "Calm": IntensityLevel.CALM if IntensityLevel else "calm",
    "Intense": IntensityLevel.INTENSE if IntensityLevel else "intense",
    "Dominant": IntensityLevel.DOMINANT if IntensityLevel else "dominant",
    "Ruthless": IntensityLevel.RUTHLESS if IntensityLevel else "ruthless"
}

EXPRESSION_LABELS = {
    "Neutral": "neutral",
    "Intense Focus": "intense_focus",
    "Controlled Anger": "controlled_anger",
    "Cold Stare": "cold_stare",
    "Subtle Threat": "subtle_threat",
    "Confident Smirk": "confident_smirk",
    "Determined": "determined",
    "Ruthless": "ruthless",
    "Predatory": "predatory"
}

CAMERA_STYLES = {
    "Portrait (85mm)": "portrait",
    "Dramatic Wide": "dramatic",
    "Extreme Closeup": "closeup",
    "Cinematic": "cinematic"
}


# =============================================================================
# Image Generation Functions
# =============================================================================

def initialize_image_generator():
    """Initialize the image generator if not already loaded"""
    if STATE.image_generator is None and AggressiveImageGenerator:
        config = AggressiveImageConfig()
        STATE.image_generator = AggressiveImageGenerator(config)
    return STATE.image_generator is not None


def generate_aggressive_image(
    subject_description: str,
    gender: str,
    age_range: str,
    ethnicity: str,
    emotion: str,
    intensity_label: str,
    camera_style: str,
    custom_lighting: str,
    additional_details: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    use_refiner: bool,
    face_enhancement: bool,
    progress=gr.Progress()
) -> Tuple[Optional[Any], str, str]:
    """
    Generate an aggressive/powerful image
    
    Returns:
        Tuple of (image, positive_prompt, negative_prompt)
    """
    try:
        progress(0.0, desc="Initializing generator...")
        
        if not initialize_image_generator():
            return None, "Error: Image generator not available", ""
        
        # Get intensity level
        intensity = INTENSITY_LABELS.get(intensity_label, IntensityLevel.DOMINANT)
        
        # Build subject description
        if not subject_description:
            ethnicity_str = f"{ethnicity} " if ethnicity and ethnicity != "Not specified" else ""
            subject_description = f"portrait of a {ethnicity_str}{gender.lower()}, age {age_range}"
        
        # Configure generator
        STATE.image_generator.config.width = width
        STATE.image_generator.config.height = height
        STATE.image_generator.config.num_inference_steps = steps
        STATE.image_generator.config.guidance_scale = guidance_scale
        STATE.image_generator.config.use_refiner = use_refiner
        STATE.image_generator.config.face_enhancement = face_enhancement
        
        # Set progress callback
        def update_progress(p, msg):
            progress(p * 0.9, desc=msg)
        
        STATE.image_generator.set_progress_callback(update_progress)
        
        # Parse additional details
        details_list = [d.strip() for d in additional_details.split(",") if d.strip()] if additional_details else None
        
        # Get camera style
        camera = CAMERA_STYLES.get(camera_style, "portrait")
        
        progress(0.1, desc="Generating image...")
        
        # Generate
        results = STATE.image_generator.generate(
            subject_description=subject_description,
            intensity=intensity,
            emotion=emotion if emotion else None,
            seed=seed if seed > 0 else -1
        )
        
        if results and len(results) > 0:
            result = results[0]
            STATE.current_image = np.array(result.image)
            
            # Save to outputs
            output_dir = Path("outputs/aggressive")
            output_path = STATE.image_generator.save_image(result, output_dir)
            STATE.current_image_path = output_path
            
            # Add to history
            STATE.generation_history.append({
                'type': 'image',
                'path': str(output_path),
                'prompt': result.prompt_used,
                'seed': result.seed,
                'timestamp': time.time()
            })
            
            progress(1.0, desc="Image generated!")
            
            return result.image, result.prompt_used, result.negative_prompt_used
        else:
            return None, "Generation failed", ""
            
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None, f"Error: {str(e)}", ""


def build_preview_prompt(
    subject_description: str,
    gender: str,
    age_range: str,
    ethnicity: str,
    emotion: str,
    intensity_label: str,
    camera_style: str,
    custom_lighting: str,
    additional_details: str
) -> Tuple[str, str]:
    """Preview the prompt that will be used"""
    try:
        builder = AggressivePromptBuilder()
        intensity = INTENSITY_LABELS.get(intensity_label, IntensityLevel.DOMINANT)
        
        if not subject_description:
            ethnicity_str = f"{ethnicity} " if ethnicity and ethnicity != "Not specified" else ""
            subject_description = f"portrait of a {ethnicity_str}{gender.lower()}, age {age_range}"
        
        camera = CAMERA_STYLES.get(camera_style, "portrait")
        details_list = [d.strip() for d in additional_details.split(",") if d.strip()] if additional_details else None
        
        positive, negative = builder.build_prompt(
            subject=subject_description,
            intensity=intensity,
            emotion=emotion if emotion else None,
            camera_style=camera,
            additional_details=details_list,
            custom_lighting=custom_lighting if custom_lighting else None
        )
        
        return positive, negative
        
    except Exception as e:
        return f"Error: {str(e)}", ""


# =============================================================================
# Motion Animation Functions
# =============================================================================

def initialize_motion_engine():
    """Initialize the motion engine if not already loaded"""
    if STATE.motion_engine is None and ImageToMotionEngine:
        config = MotionConfig()
        STATE.motion_engine = ImageToMotionEngine(config)
    return STATE.motion_engine is not None


def animate_current_image(
    expression_label: str,
    expression_intensity: float,
    lip_sync_mode: str,
    lip_sync_text: str,
    lip_sync_audio: Optional[str],
    duration: float,
    fps: int,
    head_motion: float,
    eye_motion: float,
    motion_intensity_label: str,
    preserve_identity: bool,
    output_resolution: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """
    Animate the currently generated image
    
    Returns:
        Tuple of (video_path, status_message)
    """
    try:
        progress(0.0, desc="Initializing animation engine...")
        
        if STATE.current_image is None:
            return None, "Error: No image to animate. Generate an image first."
        
        if not initialize_motion_engine():
            return None, "Error: Motion engine not available"
        
        # Configure motion engine
        resolution_map = {
            "1080p (1920x1080)": (1920, 1080),
            "720p (1280x720)": (1280, 720),
            "4K (3840x2160)": (3840, 2160)
        }
        
        STATE.motion_engine.config.output_resolution = resolution_map.get(output_resolution, (1920, 1080))
        STATE.motion_engine.config.output_fps = fps
        STATE.motion_engine.config.duration_seconds = duration
        STATE.motion_engine.config.head_motion_scale = head_motion
        STATE.motion_engine.config.eye_motion_scale = eye_motion
        STATE.motion_engine.config.preserve_identity = preserve_identity
        
        # Set progress callback
        def update_progress(p, msg):
            progress(p * 0.9, desc=msg)
        
        STATE.motion_engine.set_progress_callback(update_progress)
        
        # Determine lip sync settings
        if lip_sync_mode == "Text-to-Speech" and lip_sync_text:
            sync_mode = LipSyncMode.TEXT_TO_SPEECH
            sync_input = lip_sync_text
        elif lip_sync_mode == "Audio File" and lip_sync_audio:
            sync_mode = LipSyncMode.AUDIO_FILE
            sync_input = lip_sync_audio
        else:
            sync_mode = LipSyncMode.SILENT_EXPRESSION
            sync_input = None
        
        # Get expression type
        expression_key = EXPRESSION_LABELS.get(expression_label, "intense_focus")
        expression_type = ExpressionType[expression_key.upper()] if hasattr(ExpressionType, expression_key.upper()) else ExpressionType.INTENSE_FOCUS
        
        # Generate output path
        output_path = Path(f"outputs/animations/animation_{uuid.uuid4().hex[:8]}.mp4")
        
        progress(0.1, desc="Starting animation...")
        
        # Animate
        result = STATE.motion_engine.animate(
            source_image=STATE.current_image,
            lip_sync_mode=sync_mode,
            lip_sync_input=sync_input,
            expression=expression_type,
            expression_intensity=expression_intensity,
            output_path=output_path
        )
        
        # Add to history
        STATE.generation_history.append({
            'type': 'animation',
            'path': str(result.video_path),
            'source_image': str(STATE.current_image_path),
            'expression': expression_label,
            'duration': duration,
            'timestamp': time.time()
        })
        
        progress(1.0, desc="Animation complete!")
        
        return str(result.video_path), f"Animation saved to {result.video_path} ({result.generation_time:.1f}s)"
        
    except Exception as e:
        logger.error(f"Animation failed: {e}")
        return None, f"Error: {str(e)}"


def use_uploaded_image(image) -> Tuple[Optional[Any], str]:
    """Use an uploaded image as the source for animation"""
    if image is None:
        return None, "No image uploaded"
    
    STATE.current_image = np.array(image)
    STATE.current_image_path = None
    
    return image, "Image loaded successfully. Ready for animation."


# =============================================================================
# UI Construction
# =============================================================================

def create_aggressive_generator_tab() -> gr.Tab:
    """Create the Aggressive & Powerful Generator tab"""
    
    with gr.Tab("🔥 Aggressive & Powerful Generator") as tab:
        gr.Markdown("""
        # Aggressive & Powerful Generator
        
        **Dual-stage system for ultra-realistic, dominant image generation with facial motion animation.**
        
        1. **Stage 1**: Generate hyper-realistic powerful portraits with cinematic intensity
        2. **Stage 2**: Animate with facial motion, lip sync, and micro-expressions
        """)
        
        with gr.Row():
            # =========================================================
            # Stage 1: Image Generation (Left Column)
            # =========================================================
            with gr.Column(scale=1):
                gr.Markdown("## 🎨 Stage 1: Image Generation")
                
                with gr.Accordion("Subject Details", open=True):
                    subject_input = gr.Textbox(
                        label="Subject Description (optional)",
                        placeholder="Leave blank to use gender/age/ethnicity settings",
                        lines=2
                    )
                    
                    with gr.Row():
                        gender_input = gr.Dropdown(
                            choices=["Man", "Woman"],
                            value="Man",
                            label="Gender"
                        )
                        age_input = gr.Dropdown(
                            choices=["20-30", "30-40", "40-50", "50-60"],
                            value="30-40",
                            label="Age Range"
                        )
                    
                    ethnicity_input = gr.Dropdown(
                        choices=["Not specified", "Caucasian", "African", "Asian", "Latino", "Middle Eastern", "South Asian", "Mixed"],
                        value="Not specified",
                        label="Ethnicity"
                    )
                
                with gr.Accordion("Expression & Intensity", open=True):
                    intensity_slider = gr.Radio(
                        choices=["Calm", "Intense", "Dominant", "Ruthless"],
                        value="Dominant",
                        label="Intensity Level"
                    )
                    
                    emotion_input = gr.Textbox(
                        label="Specific Emotion",
                        placeholder="e.g., controlled rage, piercing gaze, cold determination",
                        value=""
                    )
                
                with gr.Accordion("Camera & Lighting", open=False):
                    camera_input = gr.Dropdown(
                        choices=list(CAMERA_STYLES.keys()),
                        value="Portrait (85mm)",
                        label="Camera Style"
                    )
                    
                    lighting_input = gr.Textbox(
                        label="Custom Lighting (optional)",
                        placeholder="e.g., dramatic Rembrandt lighting, hard rim light",
                        value=""
                    )
                
                with gr.Accordion("Additional Details", open=False):
                    details_input = gr.Textbox(
                        label="Additional Descriptors (comma-separated)",
                        placeholder="battle scars, weathered skin, sweat beads",
                        value=""
                    )
                
                with gr.Accordion("Generation Settings", open=False):
                    with gr.Row():
                        width_input = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                        height_input = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                    
                    with gr.Row():
                        steps_input = gr.Slider(20, 100, value=50, step=5, label="Steps")
                        guidance_input = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                    
                    seed_input = gr.Number(value=-1, label="Seed (-1 for random)")
                    
                    with gr.Row():
                        refiner_check = gr.Checkbox(value=True, label="Use SDXL Refiner")
                        face_enhance_check = gr.Checkbox(value=True, label="Face Enhancement")
                
                with gr.Row():
                    preview_prompt_btn = gr.Button("👁️ Preview Prompt", variant="secondary")
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                
                # Prompt preview
                with gr.Accordion("Prompt Preview", open=False):
                    positive_prompt_output = gr.Textbox(label="Positive Prompt", lines=4, interactive=False)
                    negative_prompt_output = gr.Textbox(label="Negative Prompt", lines=3, interactive=False)
            
            # =========================================================
            # Center: Generated Image
            # =========================================================
            with gr.Column(scale=1):
                gr.Markdown("## 📸 Generated Image")
                
                generated_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=512
                )
                
                # Or upload an image
                gr.Markdown("---")
                gr.Markdown("### Or Upload Your Own Image")
                uploaded_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=200
                )
                use_uploaded_btn = gr.Button("Use Uploaded Image")
                upload_status = gr.Textbox(label="Status", interactive=False)
            
            # =========================================================
            # Stage 2: Motion Animation (Right Column)
            # =========================================================
            with gr.Column(scale=1):
                gr.Markdown("## 🎬 Stage 2: Motion Animation")
                
                with gr.Accordion("Expression Settings", open=True):
                    expression_input = gr.Dropdown(
                        choices=list(EXPRESSION_LABELS.keys()),
                        value="Intense Focus",
                        label="Expression Type"
                    )
                    
                    expression_intensity = gr.Slider(
                        0.0, 1.0, value=0.8, step=0.1,
                        label="Expression Intensity"
                    )
                
                with gr.Accordion("Lip Sync", open=True):
                    lip_sync_mode = gr.Radio(
                        choices=["Silent Expression", "Text-to-Speech", "Audio File"],
                        value="Silent Expression",
                        label="Lip Sync Mode"
                    )
                    
                    lip_sync_text = gr.Textbox(
                        label="Text for TTS",
                        placeholder="Enter text to speak...",
                        lines=3,
                        visible=True
                    )
                    
                    lip_sync_audio = gr.Audio(
                        label="Audio File",
                        type="filepath",
                        visible=True
                    )
                
                with gr.Accordion("Motion Settings", open=False):
                    motion_intensity = gr.Radio(
                        choices=["Subtle", "Natural", "Expressive", "Dramatic"],
                        value="Natural",
                        label="Motion Intensity"
                    )
                    
                    with gr.Row():
                        head_motion = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Head Motion")
                        eye_motion = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Eye Motion")
                    
                    preserve_identity = gr.Checkbox(value=True, label="Preserve Identity (Strong)")
                
                with gr.Accordion("Output Settings", open=False):
                    with gr.Row():
                        duration_input = gr.Slider(3.0, 12.0, value=6.0, step=0.5, label="Duration (seconds)")
                        fps_input = gr.Slider(24, 60, value=30, step=6, label="FPS")
                    
                    resolution_input = gr.Dropdown(
                        choices=["720p (1280x720)", "1080p (1920x1080)", "4K (3840x2160)"],
                        value="1080p (1920x1080)",
                        label="Output Resolution"
                    )
                
                animate_btn = gr.Button("🎬 Animate Image", variant="primary", size="lg")
                
                # Output video
                gr.Markdown("### Generated Animation")
                output_video = gr.Video(label="Output Video")
                animation_status = gr.Textbox(label="Status", interactive=False)
        
        # =========================================================
        # Event Handlers
        # =========================================================
        
        # Preview prompt
        preview_prompt_btn.click(
            fn=build_preview_prompt,
            inputs=[
                subject_input, gender_input, age_input, ethnicity_input,
                emotion_input, intensity_slider, camera_input,
                lighting_input, details_input
            ],
            outputs=[positive_prompt_output, negative_prompt_output]
        )
        
        # Generate image
        generate_btn.click(
            fn=generate_aggressive_image,
            inputs=[
                subject_input, gender_input, age_input, ethnicity_input,
                emotion_input, intensity_slider, camera_input,
                lighting_input, details_input, width_input, height_input,
                steps_input, guidance_input, seed_input,
                refiner_check, face_enhance_check
            ],
            outputs=[generated_image, positive_prompt_output, negative_prompt_output]
        )
        
        # Use uploaded image
        use_uploaded_btn.click(
            fn=use_uploaded_image,
            inputs=[uploaded_image],
            outputs=[generated_image, upload_status]
        )
        
        # Animate image
        animate_btn.click(
            fn=animate_current_image,
            inputs=[
                expression_input, expression_intensity, lip_sync_mode,
                lip_sync_text, lip_sync_audio, duration_input, fps_input,
                head_motion, eye_motion, motion_intensity,
                preserve_identity, resolution_input
            ],
            outputs=[output_video, animation_status]
        )
    
    return tab


# =============================================================================
# Standalone Launch
# =============================================================================

def launch_aggressive_generator_ui(share: bool = False, port: int = 7861):
    """Launch the aggressive generator as a standalone UI"""
    
    with gr.Blocks(
        title="Aggressive & Powerful Generator",
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="slate",
            neutral_hue="slate"
        )
    ) as demo:
        create_aggressive_generator_tab()
    
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch_aggressive_generator_ui()
