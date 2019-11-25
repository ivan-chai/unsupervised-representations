"""Base models."""
from .aggregator import GRUAggregator
from .audio_cnn import AudioCNN


MODELS = {
    "audio_cnn": AudioCNN
}


AGGREGATORS = {
    "gru": GRUAggregator
}
