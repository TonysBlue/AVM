"""Model Core public surface.

Only tensor code is exported here.  Serving and training systems depend on
this package boundary, so they cannot accidentally acquire device or reward
dependencies.
"""

from model.attention import SlowMemory
from model.core import FactorizedSpeechHead, StreamingLatentLoop
from model.encoders import DeltaTimeEncoder
from model.losses import compute_jepa_loss, compute_losses
from model.perceiver import JEPAHead, Perceiver
from model.types import (
    Episode,
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SpeechLocalState,
    SpeechMode,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
    TrainingStepOutput,
    training_output,
)
from model.value import ValueHead

__all__ = [
    "Episode",
    "FactorizedSpeechHead",
    "Perceiver", "JEPAHead", "SlowMemory",
    "DeltaTimeEncoder",
    "GenerationOutput", "LayerKV", "RecurrentState", "SpeechLocalState", "SpeechMode",
    "SpeechSamplingConfig", "StepOutput", "StreamUnit", "StreamingLatentLoop",
    "TrainingStepOutput", "training_output",
    "ValueHead",
    "compute_jepa_loss", "compute_losses",
]
