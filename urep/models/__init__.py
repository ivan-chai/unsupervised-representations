"""Base models."""
from .aggregator import GRUAggregator
from .audio_cnn import AudioCNN
from .fully_connected import FullyConnected


MODELS = {
    "audio_cnn": AudioCNN,
    "fully_connected": FullyConnected
}


AGGREGATORS = {
    "gru": GRUAggregator
}
