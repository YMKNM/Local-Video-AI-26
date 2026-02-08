"""
video_ai.image_motion -- Object-level image-to-video animation (v2).

Architecture v2 pipeline:
  1. SAM 2 segmentation with retry logic (never reduces motion)
  2. DWPose/OpenPose skeletal pose extraction
  3. Procedural biomechanical motion synthesis
  4. AnimateDiff + ControlNet OpenPose pose-conditioned generation
  5. Temporal consistency (latent reuse + optical flow + anti-ghosting)
  6. Physical motion planning + temporal stabilisation

Public API::

    from video_ai.image_motion import (
        # Pipeline
        ObjectVideoPipeline, PipelineConfig, PipelineResult,
        # Action parsing
        ActionParser, ActionIntent,
        # Segmentation
        ObjectSegmenter, SegmentationResult,
        SAM2Segmenter, MultiObjectSegmentation,
        # Pose
        PoseEstimator, PoseResult,
        # Motion
        MotionPlanner, MotionPlan,
        MotionSynthesizer, MotionConfig,
        # Generation
        PoseConditionedPipeline, GenerationConfig, GenerationResult,
        # Temporal
        TemporalStabilizer,
        TemporalConsistencyPass, ConsistencyConfig, ConsistencyResult,
    )
"""

from .action_parser import ActionIntent, ActionParser
from .segmenter import ObjectSegmenter, SegmentationResult
from .sam2_segmenter import SAM2Segmenter, MultiObjectSegmentation
from .pose_estimator import PoseEstimator, PoseResult
from .motion_planner import MotionPlan, MotionPlanner
from .motion_synthesizer import MotionSynthesizer, MotionConfig
from .pose_conditioned_pipeline import (
    PoseConditionedPipeline, GenerationConfig, GenerationResult,
)
from .temporal_stabilizer import TemporalStabilizer
from .temporal_consistency import (
    TemporalConsistencyPass, ConsistencyConfig, ConsistencyResult,
)
from .object_video_pipeline import ObjectVideoPipeline, PipelineConfig, PipelineResult

__all__ = [
    # Pipeline
    "ObjectVideoPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Action parsing
    "ActionParser",
    "ActionIntent",
    # Segmentation
    "ObjectSegmenter",
    "SegmentationResult",
    "SAM2Segmenter",
    "MultiObjectSegmentation",
    # Pose
    "PoseEstimator",
    "PoseResult",
    # Motion planning
    "MotionPlanner",
    "MotionPlan",
    "MotionSynthesizer",
    "MotionConfig",
    # Generation
    "PoseConditionedPipeline",
    "GenerationConfig",
    "GenerationResult",
    # Temporal
    "TemporalStabilizer",
    "TemporalConsistencyPass",
    "ConsistencyConfig",
    "ConsistencyResult",
]
