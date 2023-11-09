import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-trp", "--train_path", type=str, help="Path of training data")
parser.add_argument("-vp", "--val_path", type=str, help="Path of validation data")
parser.add_argument("-tsp", "--test_path", type=str, help="Path of test data")
parser.add_argument("-bs", "--batch_size", type=int, help="Batch Size")
parser.add_argument("-nw", "--num_workers", type=int, help="Number of Workers")
parser.add_argument(
    "-wrs", "--use_weighted_sampler", type=bool, help="Use Weighted Random Sampler"
)
args = parser.parse_args()
"""

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


class DialogueDataset(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataframe = pd.read_csv(path=self.args.train_path, sep="\t")
        elif self.mode == "val":
            self.dataframe = pd.read_csv(path=self.args.val_path, sep="\t")
        elif self.mode == "test":
            self.dataframe = pd.read_csv(path=self.args.test_path, sep="\t")
        self.label_mappings = emotion_mapping
        self.dataframe["labels"] = self.dataframe.emotions.map(self.label_mappings)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        utterance = self.df["utterances"].to_list()
        label = self.df["labels"].to_list()
        return utterance[index], label[index]

    def make_weights_for_balanced_classes(self):
        count = self.df["labels"].value_counts().to_list()
        class_weights = [1 / c for c in count]
        labels = self.df["labels"].to_list()
        sample_weights = [0] * len(labels)
        for idx, lbl in enumerate(labels):
            sample_weights[idx] = class_weights[lbl]
        return sample_weights

    def create_loader(self):
        if self.args.mode == "train":
            if self.args.use_weighted_sampler:
                weights = self.make_weights_for_balanced_classes()
                weights = torch.DoubleTensor(weights)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights, len(weights)
                )
                return DataLoader(
                    self,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    sampler=sampler,
                )
            else:
                return DataLoader(
                    self,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    shuffle=True,
                )
        elif self.args.mode == "val" or self.args.mode == "test":
            return DataLoader(
                self,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )
