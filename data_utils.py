import glob
import os
import re
import torch
import pandas as pd
import librosa
import argparse
import numpy as np
from pathlib import Path
from pydub import AudioSegment


# refer to: https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
'''
    Calculate the topk accuracy (only for single classification)
    topk: list of k we consider
'''
def cal_topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        _, y_pred = output.topk(k=maxk, dim=1)
        # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        y_pred = y_pred.t()
        # [B] -> [B, 1] -> [maxk, B]
        target_reshaped = target.view(1, -1).expand_as(y_pred)
        # [maxk, B] were for each example we know which topk prediction matched truth
        correct = (y_pred == target_reshaped)

        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc.float())
        return list_topk_accs


'''
    get_filtered_filelist: filter out the accents with less than 30 speakers and prepare the training filelist
'''
def get_filtered_filelist(dir_path, out_path, format):
    accent = None
    count = 0
    accents = []
    filepaths = glob.glob(os.path.join(dir_path, f'*.{format}'))
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
        for filepath in glob.iglob(os.path.join(dir_path, f'*.{format}')):
            cur_accent = re.match(r'([A-Za-z]+)', Path(filepath).stem)
            cur_accent = cur_accent.groups(0)[0]
            if cur_accent in accents:
                f.write(f"{filepath.split('/')[-1]},{cur_accent}\n")

    print(f"filtered accents: {accents}")
    return accents

def get_filtered_filelist_by_labels(dir_path, out_path, accent_labels, format):
    with open(out_path, 'w') as f:
        for filepath in glob.iglob(os.path.join(dir_path, f'*.{format}')):
            cur_accent = re.match(r'([A-Za-z]+)', Path(filepath).stem)
            cur_accent = cur_accent.groups(0)[0]
            if cur_accent in accent_labels:
                f.write(f"{filepath.split('/')[-1]},{cur_accent}\n")


def get_melspectrogram(filepath, hparams):
    y, sr = librosa.load(filepath, sr=hparams.init_sampling_rate)
    if hparams.init_sampling_rate != hparams.target_sampling_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=hparams.target_sampling_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=hparams.target_sampling_rate, n_mels=hparams.num_mels, fmax=hparams.fmax, n_fft=hparams.n_fft, hop_length=hparams.hop_size)
    mel = np.transpose(mel)
    # print(f"mel shape: {mel.shape}")
    return mel

def chunk_audios(audio_dir, out_dir, chunk_len, format, speaker=False):
    filepaths = glob.glob(os.path.join(audio_dir, f'*.{format}'))
    for filepath in sorted(filepaths):
        print(f"processing {filepath}")
        if format == 'mp3':
            audio = AudioSegment.from_mp3(filepath)
        elif format == 'wav':
            audio = AudioSegment.from_wav(filepath)
        start = 0
        audio_len = len(audio)
        index = 1
        while start < audio_len:
            if audio_len - start > chunk_len:
                seg = audio[start: start+chunk_len]
            else:
                seg = audio[start:]
            if not speaker:
                seg.export(os.path.join(out_dir, Path(filepath).stem+"_"+str(index)+f".{format}"), format=format)
            else:
                path_split = filepath.split('/')
                seg.export(os.path.join(out_dir, path_split[-3]+"_"+path_split[-2]+"_"+Path(filepath).stem+"_"+str(index)+f".{format}"), format=format)
            start += chunk_len
            index += 1
    return

def chunk_audio(audio_dir, out_dir, chunk_len, format, speaker=True, filelist_path=None, speaker_id=None):
    if format == 'mp3':
        audio = AudioSegment.from_mp3(audio_dir)
    elif format == 'wav':
        audio = AudioSegment.from_wav(audio_dir)
    start = 0
    audio_len = len(audio)
    index = 1
    while start < audio_len:
        if audio_len - start > chunk_len:
            seg = audio[start: start+chunk_len]
        else:
            seg = audio[start:]
        if not speaker:
            seg.export(os.path.join(out_dir, Path(audio_dir).stem+"_"+str(index)+f".{format}"), format=format)
        else:
            path_split = audio_dir.split('/')
            filename = path_split[-3]+"_"+path_split[-2]+"_"+Path(audio_dir).stem+"_"+str(index)+f".{format}"
            seg.export(os.path.join(out_dir, filename), format=format)
            with open(filelist_path, 'a') as f:
                f.write(f"{filename},{speaker_id}\n")
        start += chunk_len
        index += 1

def prepare_speaker_data(audio_root_dir, filelist_path, chunk_output_path, meta_path, chunk_len ):
    meta = pd.read_csv(meta_path)
    filepaths = glob.glob(os.path.join(audio_root_dir, '*', '*', '*.wav'))
    for file in filepaths:
        speaker = file.split('/')[-3]
        speaker_id = meta.index[meta['ID'] == speaker][0]
        chunk_audio(file, chunk_output_path, chunk_len, "wav", filelist_path=filelist_path, speaker_id=speaker_id)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filtered_filelist', type=bool)
    parser.add_argument('--filtered_filelist_by_labels', type=bool)
    parser.add_argument('--melspectrogram', type=bool)
    parser.add_argument('--prepare_speaker_data', type=bool)
    parser.add_argument('--chunk', type=bool)
    parser.add_argument('--format', default="mp3")
    parser.add_argument('--audio_dir', default="data/recordings")
    parser.add_argument('--out_path', default="data/filelist.txt")
    parser.add_argument('--mel_input_file', default="data/recordings/agni1.mp3")
    parser.add_argument('--chunk_size', default=3000)
    parser.add_argument('--out_chunk_path', default="data/recording_chunks")

    a = parser.parse_args()

    if a.prepare_speaker_data:
        prepare_speaker_data("speaker_data/wav", "speaker_data/filelist.txt", "speaker_data/recording_chunks", "speaker_data/vox1_meta.csv", a.chunk_size)

    if a.filtered_filelist:
        get_filtered_filelist(a.audio_dir, a.out_path, a.format)
    
    if a.melspectrogram:
        get_melspectrogram(a.mel_input_file)

    if a.chunk:
        chunk_audios(a.audio_dir, a.out_chunk_path, a.chunk_size, a.format)

    if a.filtered_filelist_by_labels:
        get_filtered_filelist_by_labels(a.audio_dir, a.out_path, ["arabic", "dutch", "english", "french", "german", "italian", "korean", "mandarin", "polish", "portuguese", "russian", "spanish", "turkish"], a.format)
