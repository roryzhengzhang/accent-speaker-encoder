import glob
import os
import re
import librosa
import argparse
import numpy as np
from pathlib import Path
from pydub import AudioSegment

'''
    get_filtered_filelist: filter out the accents with less than 30 speakers and prepare the training filelist
'''
def get_filtered_filelist(dir_path, out_path):
    accent = None
    count = 0
    accents = []
    filepaths = glob.glob(os.path.join(dir_path, '*.mp3'))
    for filepath in sorted(filepaths):
        cur_accent = re.match(r'([A-Za-z]+)', Path(filepath).stem)
        # print(f"matched: {cur_accent.groups()}, filename: {Path(filepath).stem}")
        cur_accent = cur_accent.groups(0)[0]
        if accent is None:
            accent = cur_accent
            count += 1
        elif accent != cur_accent:
            if count >= 30:
                accents.append(accent)
            accent = cur_accent
            count = 1
        else:
            count += 1

    with open(out_path, 'w') as f:
        for filepath in glob.iglob(os.path.join(dir_path, '*.mp3')):
            cur_accent = re.match(r'([A-Za-z]+)', Path(filepath).stem)
            cur_accent = cur_accent.groups(0)[0]
            if cur_accent in accents:
                f.write(f"{filepath.split('/')[-1]},{cur_accent}\n")

    print(f"filtered accents: {accents}")
    return accents

def get_filtered_filelist_by_labels(dir_path, out_path, accent_labels):
    with open(out_path, 'w') as f:
        for filepath in glob.iglob(os.path.join(dir_path, '*.mp3')):
            cur_accent = re.match(r'([A-Za-z]+)', Path(filepath).stem)
            cur_accent = cur_accent.groups(0)[0]
            if cur_accent in accent_labels:
                f.write(f"{filepath.split('/')[-1]},{cur_accent}\n")


def get_melspectrogram(filepath, hparams):
    # original sr is 8kHz
    y, sr = librosa.load(filepath, sr=hparams.init_sampling_rate)
    # resample to 16kHz``
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=hparams.target_sampling_rate)
    mel = librosa.feature.melspectrogram(y=y_16k, sr=hparams.target_sampling_rate, n_mels=hparams.num_mels, fmax=hparams.fmax, n_fft=hparams.n_fft, hop_length=hparams.hop_size)
    mel = np.transpose(mel)
    # print(f"mel shape: {mel.shape}")
    return mel

def segment_audio(audio_dir, out_dir, chunk_len):
    filepaths = glob.glob(os.path.join(audio_dir, '*.mp3'))
    for filepath in sorted(filepaths):
        print(f"processing {filepath}")
        audio = AudioSegment.from_mp3(filepath)
        start = 0
        audio_len = len(audio)
        index = 1
        while start < audio_len:
            if audio_len - start > chunk_len:
                seg = audio[start: start+chunk_len]
            else:
                seg = audio[start:]
            seg.export(os.path.join(out_dir, Path(filepath).stem+"_"+str(index)+".mp3"), format="mp3")
            start += chunk_len
            index += 1
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filtered_filelist', type=bool)
    parser.add_argument('--filtered_filelist_by_labels', type=bool)
    parser.add_argument('--melspectrogram', type=bool)
    parser.add_argument('--chunk', type=bool)
    parser.add_argument('--audio_dir', default="data/recordings")
    parser.add_argument('--out_path', default="data/filelist.txt")
    parser.add_argument('--mel_input_file', default="data/recordings/agni1.mp3")
    parser.add_argument('--chunk_size', default=3000)
    parser.add_argument('--out_chunk_path', default="data/recording_chunks")

    a = parser.parse_args()

    if a.filtered_filelist:
        get_filtered_filelist(a.audio_dir, a.out_path)
    
    if a.melspectrogram:
        get_melspectrogram(a.mel_input_file)

    if a.chunk:
        segment_audio(a.audio_dir, a.out_chunk_path, a.chunk_size)

    if a.filtered_filelist_by_labels:
        get_filtered_filelist_by_labels(a.audio_dir, a.out_path, ["arabic", "dutch", "english", "french", "german", "italian", "korean", "mandarin", "polish", "portuguese", "russian", "spanish", "turkish"])
