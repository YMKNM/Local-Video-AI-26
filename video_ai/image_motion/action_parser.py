"""
Action Intent Parser — Stage 2

Parses a user action prompt (e.g. "Make the boy run forward") into a
structured ``ActionIntent`` describing the subject, motion verb, direction,
speed, and camera relationship.

The parser is rule-based with NLP keyword matching; it requires **no**
GPU or large language model.  This keeps it fast and deterministic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────

class ActionVerb(Enum):
    """Supported motion categories."""
    WALK = "walk"
    RUN = "run"
    JUMP = "jump"
    DRIVE = "drive"
    FLY = "fly"
    DANCE = "dance"
    SWIM = "swim"
    WAVE = "wave"
    TURN = "turn"
    SIT = "sit"
    STAND = "stand"
    IDLE = "idle"           # catch-all: subtle breathing / sway


class Direction(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    CLOCKWISE = "clockwise"
    COUNTER_CLOCKWISE = "counter_clockwise"
    STATIONARY = "stationary"


class Speed(Enum):
    VERY_SLOW = "very_slow"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    VERY_FAST = "very_fast"


class CameraMode(Enum):
    STATIC = "static"           # locked camera
    FOLLOW = "follow"           # track the subject
    PAN = "pan"                 # slow horizontal sweep
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class SubjectCategory(Enum):
    HUMAN = "human"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    OBJECT = "object"
    UNKNOWN = "unknown"


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class ActionIntent:
    """Structured representation of a parsed action prompt."""
    action: ActionVerb = ActionVerb.IDLE
    subject: str = ""
    subject_category: SubjectCategory = SubjectCategory.UNKNOWN
    direction: Direction = Direction.FORWARD
    speed: Speed = Speed.MEDIUM
    camera: CameraMode = CameraMode.STATIC
    intensity: float = 0.5          # 0.0 (subtle) → 1.0 (maximum)
    confidence: float = 1.0         # how sure we are about the parse
    raw_prompt: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "subject": self.subject,
            "subject_category": self.subject_category.value,
            "direction": self.direction.value,
            "speed": self.speed.value,
            "camera": self.camera.value,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "raw_prompt": self.raw_prompt,
            "warnings": self.warnings,
        }


# ── Keyword dictionaries ─────────────────────────────────────────────

_VERB_MAP: Dict[str, ActionVerb] = {}
_VERB_SYNONYMS: Dict[str, str] = {
    # walk
    "walk": "walk", "walks": "walk", "walking": "walk", "stroll": "walk",
    "strolling": "walk", "stride": "walk", "striding": "walk", "pace": "walk",
    "pacing": "walk", "march": "walk", "marching": "walk", "step": "walk",
    "stepping": "walk", "wander": "walk", "wandering": "walk", "amble": "walk",
    # run
    "run": "run", "runs": "run", "running": "run", "sprint": "run",
    "sprinting": "run", "jog": "run", "jogging": "run", "dash": "run",
    "dashing": "run", "rush": "run", "rushing": "run", "bolt": "run",
    # jump
    "jump": "jump", "jumps": "jump", "jumping": "jump", "leap": "jump",
    "leaping": "jump", "hop": "jump", "hopping": "jump", "bounce": "jump",
    "bouncing": "jump", "vault": "jump", "vaulting": "jump",
    # drive
    "drive": "drive", "drives": "drive", "driving": "drive",
    "accelerate": "drive", "accelerating": "drive", "cruise": "drive",
    "cruising": "drive", "speed": "drive", "speeding": "drive",
    # fly
    "fly": "fly", "flies": "fly", "flying": "fly", "soar": "fly",
    "soaring": "fly", "glide": "fly", "gliding": "fly", "hover": "fly",
    "hovering": "fly",
    # dance
    "dance": "dance", "dances": "dance", "dancing": "dance",
    "sway": "dance", "swaying": "dance", "groove": "dance",
    "grooving": "dance", "twirl": "dance", "twirling": "dance",
    # swim
    "swim": "swim", "swims": "swim", "swimming": "swim", "paddle": "swim",
    "paddling": "swim", "float": "swim", "floating": "swim",
    # wave
    "wave": "wave", "waves": "wave", "waving": "wave",
    "beckon": "wave", "beckoning": "wave", "gesture": "wave",
    "gesturing": "wave",
    # turn
    "turn": "turn", "turns": "turn", "turning": "turn",
    "rotate": "turn", "rotating": "turn", "spin": "turn",
    "spinning": "turn", "pivot": "turn", "pivoting": "turn",
    # sit / stand
    "sit": "sit", "sits": "sit", "sitting": "sit",
    "stand": "stand", "stands": "stand", "standing": "stand",
    "rise": "stand", "rising": "stand",
}
for _syn, _canon in _VERB_SYNONYMS.items():
    _VERB_MAP[_syn] = ActionVerb(_canon)

_DIRECTION_KEYWORDS: Dict[str, Direction] = {
    "forward": Direction.FORWARD, "forwards": Direction.FORWARD,
    "ahead": Direction.FORWARD, "onward": Direction.FORWARD,
    "backward": Direction.BACKWARD, "backwards": Direction.BACKWARD,
    "back": Direction.BACKWARD, "reverse": Direction.BACKWARD,
    "left": Direction.LEFT, "right": Direction.RIGHT,
    "up": Direction.UP, "upward": Direction.UP, "upwards": Direction.UP,
    "down": Direction.DOWN, "downward": Direction.DOWN, "downwards": Direction.DOWN,
    "clockwise": Direction.CLOCKWISE,
    "counterclockwise": Direction.COUNTER_CLOCKWISE,
    "counter-clockwise": Direction.COUNTER_CLOCKWISE,
}

_SPEED_KEYWORDS: Dict[str, Speed] = {
    "very slowly": Speed.VERY_SLOW, "very slow": Speed.VERY_SLOW,
    "slowly": Speed.SLOW, "slow": Speed.SLOW, "gently": Speed.SLOW,
    "casually": Speed.SLOW, "leisurely": Speed.SLOW,
    "quickly": Speed.FAST, "quick": Speed.FAST, "fast": Speed.FAST,
    "briskly": Speed.FAST, "rapidly": Speed.FAST,
    "very fast": Speed.VERY_FAST, "very quickly": Speed.VERY_FAST,
    "full speed": Speed.VERY_FAST,
}

_CAMERA_KEYWORDS: Dict[str, CameraMode] = {
    "follow": CameraMode.FOLLOW, "tracking": CameraMode.FOLLOW,
    "track": CameraMode.FOLLOW,
    "pan": CameraMode.PAN, "panning": CameraMode.PAN,
    "zoom in": CameraMode.ZOOM_IN, "close up": CameraMode.ZOOM_IN,
    "zoom out": CameraMode.ZOOM_OUT, "wide shot": CameraMode.ZOOM_OUT,
    "static": CameraMode.STATIC, "fixed": CameraMode.STATIC,
    "locked": CameraMode.STATIC,
}

_HUMAN_SUBJECTS = {
    "man", "woman", "boy", "girl", "person", "child", "kid", "baby",
    "athlete", "runner", "dancer", "soldier", "worker", "he", "she",
    "human", "people", "figure", "character", "player", "adult",
    "teenager", "teen", "lady", "gentleman", "guy", "dude",
}

_ANIMAL_SUBJECTS = {
    "dog", "cat", "horse", "bird", "fish", "elephant", "lion", "tiger",
    "bear", "deer", "rabbit", "fox", "wolf", "monkey", "eagle", "hawk",
    "dolphin", "whale", "snake", "chicken", "duck", "goose", "cow",
    "pig", "sheep", "goat", "puppy", "kitten",
}

_VEHICLE_SUBJECTS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "bike", "train",
    "plane", "airplane", "helicopter", "boat", "ship", "van", "taxi",
    "ambulance", "tank", "drone", "scooter", "skateboard",
}


# ── Default action-verb implied motion ────────────────────────────────

_ACTION_DEFAULTS: Dict[ActionVerb, Dict] = {
    ActionVerb.WALK: {"direction": Direction.FORWARD, "speed": Speed.SLOW, "intensity": 0.4},
    ActionVerb.RUN:  {"direction": Direction.FORWARD, "speed": Speed.FAST, "intensity": 0.7},
    ActionVerb.JUMP: {"direction": Direction.UP, "speed": Speed.FAST, "intensity": 0.8},
    ActionVerb.DRIVE: {"direction": Direction.FORWARD, "speed": Speed.MEDIUM, "intensity": 0.5},
    ActionVerb.FLY:  {"direction": Direction.FORWARD, "speed": Speed.MEDIUM, "intensity": 0.6},
    ActionVerb.DANCE: {"direction": Direction.STATIONARY, "speed": Speed.MEDIUM, "intensity": 0.6},
    ActionVerb.SWIM: {"direction": Direction.FORWARD, "speed": Speed.SLOW, "intensity": 0.4},
    ActionVerb.WAVE: {"direction": Direction.STATIONARY, "speed": Speed.SLOW, "intensity": 0.3},
    ActionVerb.TURN: {"direction": Direction.CLOCKWISE, "speed": Speed.SLOW, "intensity": 0.3},
    ActionVerb.SIT:  {"direction": Direction.DOWN, "speed": Speed.SLOW, "intensity": 0.2},
    ActionVerb.STAND: {"direction": Direction.UP, "speed": Speed.SLOW, "intensity": 0.2},
    ActionVerb.IDLE: {"direction": Direction.STATIONARY, "speed": Speed.VERY_SLOW, "intensity": 0.1},
}


# ── Parser ────────────────────────────────────────────────────────────

class ActionParser:
    """
    Deterministic rule-based parser that converts a free-text action prompt
    into an ``ActionIntent``.

    No neural network required — fast and reproducible.
    """

    def parse(self, prompt: str) -> ActionIntent:
        """
        Parse a user prompt into a structured ActionIntent.

        Examples::

            parser = ActionParser()
            intent = parser.parse("Make the boy run forward quickly")
            # ActionIntent(action=RUN, subject="boy", direction=FORWARD,
            #              speed=FAST, ...)
        """
        if not prompt or not prompt.strip():
            return ActionIntent(
                action=ActionVerb.IDLE,
                confidence=0.0,
                raw_prompt=prompt or "",
                warnings=["Empty prompt — defaulting to idle animation"],
            )

        raw = prompt.strip()
        text = raw.lower()

        # Strip common prefixes
        for prefix in ("make the", "make a", "make", "animate the",
                       "animate a", "animate", "let the", "have the",
                       "have a", "let a"):
            if text.startswith(prefix + " "):
                text = text[len(prefix) + 1:]
                break

        intent = ActionIntent(raw_prompt=raw)

        # 1. Extract subject — first noun-like token before the verb
        intent.subject, intent.subject_category = self._extract_subject(text)

        # 2. Extract action verb
        intent.action = self._extract_verb(text)

        # 3. Apply verb defaults
        defaults = _ACTION_DEFAULTS.get(intent.action, {})
        intent.direction = defaults.get("direction", Direction.FORWARD)
        intent.speed = defaults.get("speed", Speed.MEDIUM)
        intent.intensity = defaults.get("intensity", 0.5)

        # 4. Override with explicit direction / speed / camera from prompt
        dir_found = self._extract_direction(text)
        if dir_found is not None:
            intent.direction = dir_found

        spd_found = self._extract_speed(text)
        if spd_found is not None:
            intent.speed = spd_found
            # adjust intensity from speed
            speed_intensity: Dict[Speed, float] = {
                Speed.VERY_SLOW: 0.15,
                Speed.SLOW: 0.3,
                Speed.MEDIUM: 0.5,
                Speed.FAST: 0.7,
                Speed.VERY_FAST: 0.9,
            }
            intent.intensity = max(intent.intensity, speed_intensity.get(spd_found, 0.5))

        cam_found = self._extract_camera(text)
        if cam_found is not None:
            intent.camera = cam_found

        # 5. Confidence heuristic
        intent.confidence = self._compute_confidence(intent)

        # 6. Safety clamps
        intent = self._apply_safety(intent)

        logger.info(
            "Parsed action: %s | subject=%s (%s) | dir=%s | speed=%s | "
            "intensity=%.2f | camera=%s | conf=%.2f",
            intent.action.value, intent.subject, intent.subject_category.value,
            intent.direction.value, intent.speed.value, intent.intensity,
            intent.camera.value, intent.confidence,
        )

        return intent

    # ── private helpers ───────────────────────────────────────────

    def _extract_subject(self, text: str) -> Tuple[str, SubjectCategory]:
        """Return (subject_word, category) from the text."""
        words = re.findall(r"[a-z]+", text)
        for w in words:
            if w in _HUMAN_SUBJECTS:
                return w, SubjectCategory.HUMAN
            if w in _ANIMAL_SUBJECTS:
                return w, SubjectCategory.ANIMAL
            if w in _VEHICLE_SUBJECTS:
                return w, SubjectCategory.VEHICLE
        # Fallback: take first non-stop word
        stop = {"the", "a", "an", "make", "let", "have", "my", "this", "that"}
        for w in words:
            if w not in stop and w not in _VERB_MAP:
                return w, SubjectCategory.UNKNOWN
        return "object", SubjectCategory.UNKNOWN

    def _extract_verb(self, text: str) -> ActionVerb:
        words = re.findall(r"[a-z]+", text)
        for w in words:
            if w in _VERB_MAP:
                return _VERB_MAP[w]
        return ActionVerb.IDLE

    def _extract_direction(self, text: str) -> Optional[Direction]:
        # Check multi-word keys first
        for phrase, d in sorted(_DIRECTION_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                return d
        return None

    def _extract_speed(self, text: str) -> Optional[Speed]:
        for phrase, s in sorted(_SPEED_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                return s
        return None

    def _extract_camera(self, text: str) -> Optional[CameraMode]:
        for phrase, c in sorted(_CAMERA_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                return c
        return None

    def _compute_confidence(self, intent: ActionIntent) -> float:
        """Heuristic confidence: higher if we matched more fields."""
        score = 0.3  # base
        if intent.action != ActionVerb.IDLE:
            score += 0.3
        if intent.subject and intent.subject != "object":
            score += 0.2
        if intent.subject_category != SubjectCategory.UNKNOWN:
            score += 0.2
        return min(score, 1.0)

    def _apply_safety(self, intent: ActionIntent) -> ActionIntent:
        """Clamp unreasonable motion requests."""
        # Cap intensity
        intent.intensity = max(0.05, min(intent.intensity, 1.0))

        # If confidence is very low, reduce intensity
        if intent.confidence < 0.4:
            intent.intensity = min(intent.intensity, 0.3)
            intent.warnings.append(
                "Low confidence parse — motion intensity capped at 0.3"
            )

        # Vehicles should not jump/dance
        if intent.subject_category == SubjectCategory.VEHICLE:
            if intent.action in (ActionVerb.JUMP, ActionVerb.DANCE, ActionVerb.WAVE):
                intent.action = ActionVerb.DRIVE
                intent.warnings.append(
                    f"Vehicle cannot {intent.action.value}; falling back to drive"
                )

        return intent
