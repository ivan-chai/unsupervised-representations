"""Data loading tools."""
import os
import random
from collections import namedtuple, OrderedDict

import librosa
import torch


class LibrispeechDataset(torch.utils.data.Dataset):
    AudioFile = namedtuple("AudioFile", ["id", "path", "duration", "speaker"])

    def __init__(self, librispeech_root, sampling_rate=16000, chunk_size=20480):
        self._audio_files, self._speakers = self._get_audiofiles(librispeech_root)
        self._sampling_rate = sampling_rate
        self._chunk_size = 20480

    def __len__(self):
        return len(self._audio_files)

    def __getitem__(self, idx):
        output_duration = self._chunk_size / self._sampling_rate
        audio_file = None
        while (audio_file is None) or (audio_file.duration < output_duration):
            audio_file = random.choice(self._audio_files)
        offset = random.random() * (audio_file.duration - output_duration)
        waveform, _ = librosa.load(audio_file.path, sr=self._sampling_rate,
                                   offset=offset, duration=output_duration)
        return waveform, audio_file.speaker

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
