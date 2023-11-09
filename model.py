import torch
import warnings
import random
import wandb
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-mn", "--model_name", type=str, help="Name of backbone")
parser.add_argument("-nc", "--num_classes", type=int, help="Number of classes")
parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
parser.add_argument("-wd", "--weight_decay", type=float, help="Weight decay")
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")

args = parser.parse_args()
"""

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class SemEvalNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = self.args.num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.backbone = AutoModel.from_pretrained(self.args.model_name)
        self.linear_layer = nn.Linear(768, self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.push_to_device()

    def forward(self, batch):
        x = self.transformer(**batch).pooler_output
        x = self.lin(x)
        return x

    def push_to_device(self):
        self.backbone.to(self.device)
        self.linear_layer.to(self.device)

    def calculate_f1(self, labels, predictions):
        return classification_report(
            torch.concat(labels, dim=0).cpu(),
            torch.concat(predictions, dim=0).cpu(),
            digits=4,
        )

    def accuracy(self, labels, preds):
        return np.mean(labels == preds)

    def fit(self, train_loader, val_loader):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0
        for epoch in range(self.args.epochs):
            train_loss = []
            train_preds = []
            train_labels = []

            val_loss = []
            val_preds = []
            val_labels = []
            self.train()
            print(f"Epoch - {epoch+1}/{self.args.epochs}")
            for batch in tqdm(train_loader):
                batch[0] = self.tokenizer(
                    text=list(batch[0]),
                    return_attention_mask=True,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = batch[1].to(self.device)
                scores = self(text)
                loss = criterion(scores, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.detach().cpu().numpy())
                train_labels.append(batch[1])
                train_preds.append(scores.argmax(dim=-1))
            print(f"Train loss - {sum(train_loss)/len(train_loss)}")
            train_acc = self.accuracy(
                torch.concat(train_labels, dim=0).cpu(),
                torch.concat(train_preds, dim=0).cpu(),
            )
            train_f1 = self.calc_f1(train_labels, train_preds)
            self.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    batch[0] = self.tokenizer(
                        text=list(batch[0]),
                        return_attention_mask=True,
                        max_length=256,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    text = {k: v.to(self.device) for k, v in batch[0].items()}
                    labels = batch[1].to(self.device)
                    scores = self(text)
                    loss = criterion(scores, labels)
                    val_loss.append(loss.detach().cpu().numpy())
                    val_labels.append(batch[1])
                    val_preds.append(scores.argmax(dim=-1))
                print(f"Validation loss - {sum(val_loss)/len(val_loss)}")
                val_acc = self.accuracy(
                    torch.concat(val_labels, dim=0).cpu(),
                    torch.concat(val_preds, dim=0).cpu(),
                )
                val_f1 = self.calc_f1(val_labels, val_preds)
            print(self.calculate_f1(train_labels, train_preds))
            print(f"Training Accuracy - {train_acc}")
            print(f"Training F1 - {train_f1}")
            print(self.calculate_f1(val_labels, val_preds))
            print(f"Validation Accuracy - {val_acc}")
            print(f"Validation F1 - {val_f1}")
            wandb.log(
                {
                    "Train_loss": np.mean(train_loss),
                    "Val_loss": np.mean(val_loss),
                    "Train_acc": train_acc,
                    "Val_acc": val_acc,
                    "Train_F1": train_f1,
                    "Val_F1": val_f1,
                }
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                print("Saved")
                torch.save(
                    self.state_dict(),
                    f"../{self.args.model_name}-{self.args.wrs}-{epoch}.pt",
                )
        wandb.finish()

    def test(self, loader):
        labels = []
        preds = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                img = batch[0].to(device=self.device)
                labels = batch[1].to(device=self.device)
                scores = self(img)
                preds.append(scores.argmax(dim=-1))
                labels.append(batch[1])
            print(self.calculate_f1(labels, preds))
