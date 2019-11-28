"""Data loading tools."""
import os
import random
from collections import namedtuple, OrderedDict

import librosa
import numpy as np
import torch
from sklearn.datasets import make_spd_matrix

from .config import prepare_config
from .utils import to_tuple


class LibrispeechDataset(torch.utils.data.Dataset):
    AudioFile = namedtuple("AudioFile", ["id", "path", "duration", "speaker"])
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("sampling_rate", 16000),
            ("chunk_size", chunk_size)
        ])

    def __init__(self, librispeech_root, config=None):
        self._config = prepare_config(config, self.get_default_config())
        self._audio_files, self._speakers = self._get_audiofiles(librispeech_root)

    def __len__(self):
        return len(self._audio_files)

    def __getitem__(self, idx):
        sampling_rate = self._config["sampling_rate"]
        chunk_size = self._config["chunk_size"]
        output_duration = chunk_size / sampling_rate
        audio_file = None
        while (audio_file is None) or (audio_file.duration < output_duration):
            audio_file = random.choice(self._audio_files)
        offset = random.random() * (audio_file.duration - output_duration)
        waveform, _ = librosa.load(audio_file.path, sr=sampling_rate,
                                   offset=offset, duration=output_duration)
        waveform = np.expand_dims(waveform, -1)
        return waveform, audio_file.speaker

    @property
    def out_channels(self):
        return 1

    @property
    def speakers(self):
        return self._speakers

    @staticmethod
    def _get_audiofiles(root):
        """Get dictionary of all audio files in the directory."""
        audio_files = []
        speaker_map = OrderedDict()
        speakers = []
        for dir, dirs, files in os.walk(root):
            relroot = os.path.relpath(dir, root)
            for filename in files:
                _, ext = os.path.splitext(filename)
                if not ext:
                    continue
                if ext.lower() != ".flac":
                    continue
                audio_id = os.path.join(relroot, filename)
                audio_path = os.path.join(dir, filename)
                duration = librosa.get_duration(filename=audio_path)
                original_speaker = relroot.split(os.sep)[0]
                if original_speaker not in speaker_map:
                    speaker_map[original_speaker] = len(speaker_map)
                    speakers.append(original_speaker)
                speaker = speaker_map[original_speaker]
                audio_files.append(LibrispeechDataset.AudioFile(audio_id, audio_path, duration, speaker))
        return audio_files, speakers


class MultivariateNormalDataset(torch.utils.data.Dataset):
    """Simple generator of normally distributed samples for testing."""
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("dim", 64),
            ("num_classes", 100),
            ("num_samples", 10000),
            ("sequence_length", None),  # If None, output batches will not have time dimension.
            ("seed", 0)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(config, self.get_default_config())
        np.random.seed(self._config["seed"])
        self._centroid_mean = np.random.random(self._config["dim"]) * 10 - 5
        self._centroid_cov = make_spd_matrix(self._config["dim"])
        self._class_cov = np.eye(self._config["dim"])
        self._class_mean = []
        for _ in range(self._config["num_classes"]):
            self._class_mean.append(np.random.multivariate_normal(self._centroid_mean, self._centroid_cov))

    def __len__(self):
        return self._config["num_samples"]

    def __getitem__(self, idx):
        cls = np.random.randint(0, self._config["num_classes"])
        # (time, dim).
        sampler_params = {}
        if self._config["sequence_length"] is not None:
            sampler_params["size"] = self._config["sequence_length"]
        sample = np.random.multivariate_normal(self._class_mean[cls], self._class_cov,
                                               **sampler_params)
        return sample.astype(np.float32), cls

    @property
    def out_channels(self):
        return self._config["dim"]


def make_dataloader(dataset, batch_size, num_steps=None, **kwargs):
    dataset = to_tuple(dataset)
    if len(dataset) == 0:
        raise ValueError("At least one dataset should be provided")
    dataset = torch.utils.data.ConcatDataset(dataset) if len(dataset) > 1 else dataset[0]
    if num_steps is not None:
        num_elements = num_steps * batch_size
        if num_elements < len(dataset):
            indices = random.sample(list(range(len(dataset))), num_elements)
            dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              **kwargs)
    return data_loader
