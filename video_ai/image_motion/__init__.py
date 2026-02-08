"""
video_ai.image_motion -- Object-level image-to-video animation.

This package adds the ability to animate a single still image by:
  1. Segmenting the subject
  2. Parsing a natural-language action prompt
  3. Planning physically-plausible motion
  4. Generating video via diffusion
  5. Stabilising the output temporally

Public API::

    from video_ai.image_motion import (
        ObjectVideoPipeline,
        PipelineConfig,
        PipelineResult,
        ActionParser,
        ActionIntent,
        ObjectSegmenter,
        SegmentationResult,
        MotionPlanner,
        MotionPlan,
        TemporalStabilizer,
    )
"""

from .action_parser import ActionIntent, ActionParser
from .segmenter import ObjectSegmenter, SegmentationResult
from .motion_planner import MotionPlan, MotionPlanner
from .temporal_stabilizer import TemporalStabilizer
from .object_video_pipeline import ObjectVideoPipeline, PipelineConfig, PipelineResult

__all__ = [
    "ActionParser",
    "ActionIntent",
    "ObjectSegmenter",
    "SegmentationResult",
    "MotionPlanner",
    "MotionPlan",
    "TemporalStabilizer",
    "ObjectVideoPipeline",
    "PipelineConfig",
    "PipelineResult",
]
