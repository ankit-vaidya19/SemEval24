from model import SemEvalNet
from data import DialogueDataset


train_ds = DialogueDataset("train.csv", "train")
val_ds = DialogueDataset("val.csv", "val")
test_ds = DialogueDataset("test.csv", "test")
inference_ds = DialogueDataset("inference.csv", "inference")

train_loader = train_ds.create_loader()
val_loader = val_ds.create_loader()
test_loader = test_ds.create_loader()
inference_loader = inference_ds.create_loader()


net = SemEvalNet()

net.fit(train_loader, test_loader)
net.test(test_loader)
net.predict(inference_loader, "inference_path.csv")
