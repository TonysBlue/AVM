"""Model Core public surface.

Only tensor code is exported here.  Serving and training systems depend on
this package boundary, so they cannot accidentally acquire device or reward
dependencies.
"""

from model.attention import SlowMemory
from model.core import NativeSpeechHead, StreamingLatentLoop
from model.losses import compute_jepa_loss, compute_losses
from model.perceiver import JEPAHead, Perceiver
from model.types import (
    AudioVideoInput,
    Episode,
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SpeechLocalState,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
    TrainingStepOutput,
    training_output,
)
from model.value import ValueHead

__all__ = [
    "Episode",
    "NativeSpeechHead",
    "Perceiver", "JEPAHead", "SlowMemory",
    "AudioVideoInput",
    "GenerationOutput", "LayerKV", "RecurrentState", "SpeechLocalState",
    "SpeechSamplingConfig", "StepOutput", "StreamUnit", "StreamingLatentLoop",
    "TrainingStepOutput", "training_output",
    "ValueHead",
    "compute_jepa_loss", "compute_losses",
]
