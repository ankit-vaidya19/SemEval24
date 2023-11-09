import argparse
import wandb
from datetime import datetime
from ml_collections import ConfigDict
from data import DialogueDataset
from model import SemEvalNet


parser = argparse.ArgumentParser()
"""data.py args"""
parser.add_argument("-trp", "--train_path", type=str, help="Path of training data")
parser.add_argument("-vp", "--val_path", type=str, help="Path of validation data")
parser.add_argument("-tsp", "--test_path", type=str, help="Path of test data")
parser.add_argument("-bs", "--batch_size", type=int, help="Batch Size")
parser.add_argument("-nw", "--num_workers", type=int, help="Number of Workers")
parser.add_argument(
    "-wrs", "--use_weighted_sampler", type=bool, help="Use Weighted Random Sampler"
)
"""model.py args"""
parser.add_argument("-mn", "--model_name", type=str, help="Name of backbone")
parser.add_argument("-nc", "--num_classes", type=int, help="Number of classes")
parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
parser.add_argument("-wd", "--weight_decay", type=float, help="Weight decay")
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
args = parser.parse_args()

cfg = ConfigDict()
cfg.epochs = args.epochs
cfg.learning_rate = args.learning_rate
cfg.weight_decay = args.weight_decay
cfg.batch_size = args.batch_size
cfg.used_sampler = args.wrs

now = datetime.now()
timestr = now.strftime("%d_%m_%Hh%Mm%Ss")
wandb.login(key="dedc08f0b13e84705452beca60fc2b5f46430ff9")
wandb.init(
    project="SemEval Task1",
    config=cfg,
    entity="ankxanity19",
    name="_".join([args.model_name, str(args.wrs), timestr]),
)

train_data = DialogueDataset(args, mode="train")
train_loader = train_data.create_loader()

val_data = DialogueDataset(args, mode="val")
val_loader = val_data.create_loader()

test_data = DialogueDataset(args, mode="test")
test_loader = test_data.create_loader()


model = SemEvalNet(args)
model.fit(train_loader, val_loader)
model.test(test_loader)
