from data_utils import get_melspectrogram
from scipy.io.wavfile import read
import os
import torch

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

class AccentDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, hparams):
        self.audio_files = []
        self.labels = []
        self.hparams = hparams
        with open(file_path, 'r') as f:
            for line in f:
                filename, label = line.split(',')
                self.audio_files.append(os.path.join(hparams.audio_dir, filename))
                self.labels.append(label.replace("\n", ""))

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        mel = get_melspectrogram(audio_file, self.hparams)
        mel = torch.from_numpy(mel)
        # pad mel len to 300 if necessary
        if mel.size(0) < 300:
            pad_mel = torch.zeros((300, 257), dtype=mel.dtype)
            pad_mel[:mel.size(0), :] = mel
            mel = pad_mel
        elif mel.size(0) > 300:
            mel = mel[:300, :]
        label = self.labels[index]
        label_id = self.hparams.labels.index(label)
        return (mel, label_id)

    def __len__(self):
        return len(self.audio_files)
