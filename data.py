import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from ml_collections import ConfigDict

emotion_mapping = {
    "disgust": 0,
    "contempt": 1,
    "anger": 2,
    "neutral": 3,
    "joy": 4,
    "sadness": 5,
    "fear": 6,
    "surprise": 7,
}

cfg = ConfigDict()
cfg.batch_size = 32


class DialogueDataset(Dataset):
    def __init__(self, path, mode):
        super().__init__()
        self.mode = mode
        self.dataframe = pd.read_csv(path, sep="\t")
        self.label_mappings = emotion_mapping
        if self.mode != "inference":
            self.dataframe["Label"] = self.dataframe.emotions.map(self.label_mappings)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if self.mode == "inference":
            utterance = self.dataframe["utterances"].to_list()
            return utterance[index]
        else:
            utterance = self.dataframe["utterances"].to_list()
            label = self.dataframe["Label"].to_list()
            return utterance[index], label[index]

    def create_loader(self):
        if self.mode == "train":
            return DataLoader(self, batch_size=cfg.batch_size, shuffle=True)
        else:
            return DataLoader(self, batch_size=cfg.batch_size, shuffle=False)
