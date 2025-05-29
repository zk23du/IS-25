import os
import numpy as np
import pandas as pd
import librosa
from multiprocessing import Pool
import torch
import torch.nn as nn
import pickle as pk
import torch.utils as torch_utils
from torch.utils.data import Dataset


SPLIT_MAP = {
    "train": "Train",
    "dev": "Development",
    "test1": "Test1",
    "test2": "Test2",
    "test3": "Test3"
}

# Load label
def load_utts(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    
    return cur_utts

def load_adv_emo_label(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoAct", "EmoDom", "EmoVal"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_arousal(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoAct"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_valence(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoVal"]].to_numpy()

    return cur_utts, cur_labs

def load_adv_dominance(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoDom"]].to_numpy()

    return cur_utts, cur_labs

def load_cat_emo_label(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]].to_numpy()

    return cur_utts, cur_labs

def load_spk_id(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[(label_df["Split_Set"] == SPLIT_MAP[dtype])]
    cur_df = cur_df[(cur_df["SpkrID"] != "Unknown")]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_spk_ids = cur_df["SpkrID"].to_numpy().astype(np.int)
    # Cleanining speaker id
    uniq_spk_id = list(set(cur_spk_ids))
    uniq_spk_id.sort()
    for new_id, old_id in enumerate(uniq_spk_id):
        cur_spk_ids[cur_spk_ids == old_id] = new_id
    total_spk_num = len(uniq_spk_id)

    return cur_utts, cur_spk_ids, total_spk_num


# Load audio
def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav
def load_audio(audio_path, utts, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        wavs = list(p.imap(extract_wav, wav_paths))
    return wavs


def collate_fn_wav_lab_mask(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask, total_utt


def collate_fn_wav_test3(batch):
    total_wav = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        total_wav.append(torch.Tensor(wav))
        total_dur.append(dur)
        total_utt.append(wav_data[1])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, attention_mask, total_utt


"""
All dataset should have the same order based on the utt_list
"""
def load_norm_stat(norm_stat_file):
    with open(norm_stat_file, 'rb') as f:
        wav_mean, wav_std = pk.load(f)
    return wav_mean, wav_std


class CombinedSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CombinedSet, self).__init__()
        self.datasets = kwargs.get("datasets", args[0]) 
        self.data_len = len(self.datasets[0])
        for cur_dataset in self.datasets:
            assert len(cur_dataset) == self.data_len, "All dataset should have the same order based on the utt_list"
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        result = []
        for cur_dataset in self.datasets:
            result.append(cur_dataset[idx])
        return result


class WavSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.upper_bound_max_dur = kwargs.get("max_dur", 12)
        self.sampling_rate = kwargs.get("sr", 16000)

        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), self.upper_bound_max_dur*self.sampling_rate])
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = get_norm_stat_for_wav(self.wav_list)
    
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        
        result = (cur_wav, cur_dur)
        return result

class ADV_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(ADV_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
        self.max_score = kwargs.get("max_score", 7)
        self.min_score = kwargs.get("min_score", 1)
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        cur_lab = (cur_lab - self.min_score) / (self.max_score-self.min_score)
        result = cur_lab
        return result
    
class CAT_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CAT_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        result = cur_lab
        return result

class SpkSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(SpkSet, self).__init__()
        self.spk_list = kwargs.get("spk_list", args[0])
    
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        cur_lab = self.spk_list[idx]
        result = cur_lab
        return result    

class UttSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(UttSet, self).__init__()
        self.utt_list = kwargs.get("utt_list", args[0])
    
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        cur_lab = self.utt_list[idx]
        result = cur_lab
        return result

def get_norm_stat_for_wav(wav_list, verbose=False):
    count = 0
    wav_sum = 0
    wav_sqsum = 0
    
    for cur_wav in wav_list:
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav**2)
        count += len(cur_wav)
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)

    return wav_mean, wav_std


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)    
    def __len__(self):
        return len(self.labels)    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]