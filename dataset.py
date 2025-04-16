import os

import librosa
import torchaudio
from sympy import false
from torchaudio import transforms
from torch.utils.data import Dataset
import pandas as pd
import dsdtools

# DSD100 cite
# @inproceedings{
#   SiSEC16,
#   Title = {The 2016 Signal Separation Evaluation Campaign},
#   Address = {Cham},
#   Author = {Liutkus, Antoine and St{\"o}ter, Fabian-Robert and Rafii, Zafar and Kitamura, Daichi and Rivet, Bertrand and Ito, Nobutaka and Ono, Nobutaka and Fontecave, Julie},
#   Editor = {Tichavsk{\'y}, Petr and Babaie-Zadeh, Massoud and Michel, Olivier J.J. and Thirion-Moreau, Nad{\`e}ge},
#   Pages = {323--332},
#   Publisher = {Springer International Publishing},
#   Year = {2017},
#   booktitle = {Latent Variable Analysis and Signal Separation - 12th International Conference, {LVA/ICA} 2015, Liberec, Czech Republic, August 25-28, 2015, Proceedings},
# }

### Dataset class for DSD100
### Hardware limitation: some of the audio files are too big for my pc to load in, so I have limited them to 30 seconds
### This allows the program to actually run, and parsing & reading & analyzing songs should now take less time
class DSD100Data(Dataset):
    def __init__(self, test=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if test:
            data = []
            self.mix_path = ".\\DSD100\\Mixtures\\Test"
            for s in os.listdir(".\\DSD100\\Sources\\Test"):
                accompaniment = get_accompaniment(os.path.join(".\\DSD100\\Sources\\Test", s))
                vocals = torchaudio.load(os.path.join(".\\DSD100\\Sources\\Test", s, "vocals.wav"), num_frames=44100 * 30)[
                    0]
                if transform:
                    vocals = transform(vocals)
                    accompaniment = transform(accompaniment)
                data.append({"name": s,
                             "vocals": vocals,
                             "accompaniment": accompaniment})
        else:
            data = []
            self.mix_path = ".\\DSD100\\Mixtures\\Dev"
            for s in os.listdir(".\\DSD100\\Sources\\Dev"):
                accompaniment = get_accompaniment(os.path.join(".\\DSD100\\Sources\\Dev", s))
                vocals = torchaudio.load(os.path.join(".\\DSD100\\Sources\\Dev", s, "vocals.wav"), num_frames=44100 * 30)[0]
                if transform:
                    vocals = transform(vocals)
                    accompaniment = transform(accompaniment)
                data.append({"name": s,
                             "vocals": vocals,
                             "accompaniment": accompaniment})
        self.tracks = pd.DataFrame(data)

    def __len__(self):
        return len(self.tracks)

    ### Shape of data: tensor(channel, dunno, dunno)
    def __getitem__(self, idx):
        ### Find the mixture, load it, transform it, return it
        song, rate = torchaudio.load(os.path.join(self.mix_path, self.tracks.iloc[idx,0], "mixture.wav"),
                                                  num_frames=44100 * 30)

        ind_files = self.tracks.iloc[idx,1]
        if self.transform is not None:
            song = self.transform(song)
        if self.target_transform is not None:
            song = self.target_transform(song)

        return song, ind_files

# Get the accompaniment based off of the bass, drum, other tensors
def get_accompaniment(path):
    bass = torchaudio.load(os.path.join(path, "bass.wav"), num_frames=44100 * 30)[0]
    drum = torchaudio.load(os.path.join(path, "drums.wav"), num_frames=44100 * 30)[0]
    other = torchaudio.load(os.path.join(path, "other.wav"), num_frames=44100 * 30)[0]

    accompaniment = bass + drum + other
    return accompaniment