import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import batcher
from model import LexicalModel
from logger import INIT_LOG, LOG_INFO

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--max_vocab_size", default=25000, type=int, help="vocabulary size.")
parser.add_argument("--n_labels", default=5, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--conv_channels", default=256, type=int, help="Number of 1-D convolutional channels.")
parser.add_argument("--kernel_size", default=5, type=int, help="Convolution kernel size.")
parser.add_argument("--embedding_dim", default=300, type=int, help="Size of word embedding.")
parser.add_argument("--output_dim", default=128, type=int, help="Model output dim = LSTM hidden dim.")
parser.add_argument("--context_size", default=3, type=int, help="Total number of sentences to consider.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=50, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--log_file", default='', type=str, help="Log file")

args = parser.parse_args()
print(args)

INIT_LOG(args.log_file)


# folders,data_folders = utils.folders_info()
# vocabulary,data_dict,labels_dict,frames_dict = utils.init_dictionaries(folders,data_folders)

# loaders = batcher.initialize_tensors(folders,data_folders,vocabulary,data_dict,labels_dict,frames_dict,100,shuffle = True)
# train_loader = loaders[0]

# for batch,(audio,text,label) in enumerate(train_loader):
#     if batch<10:
#         print(batch)
#         print(audio.shape)
#         print(text.shape)
#         print(label)


device = 'cuda'
# ---------- Model Definition -----------
vocab_size = args.max_vocab_size 
output_dim = args.output_dim
num_labels = args.num_labels

model = nn.Sequential([
    LexicalModel(vocab_size = vocab_size, output_dim=output_dim),
    nn.Linear(output_dim,num_labels)
]).to(device)

criterion = nn.CrossEntropyLoss()
# ----------------------------------------

def train(epoch, model, iterator, optimizer, criterion):
    loss_list = []
    acc_list = []

    model.train()

    for i, (audio,text,label) in enumerate(iterator):
        optimizer.zero_grad()
        predictions = model(text)

        loss = criterion(predictions, label.long())
        loss.backward()
        optimizer.step()

        acc = (predictions.max(1)[1] == label.long()).float().mean()
        loss_list.append(loss.item())
        acc_list.append(acc.item())

        if i % args.display_freq == 0:
            msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f, train acc = %.4f" % (
                epoch, i, len(iterator), np.mean(loss_list), np.mean(acc_list)
            )
            LOG_INFO(msg)
            loss_list.clear()
            acc_list.clear()


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label.long())

            acc = (predictions.max(1)[1] == batch.label.long()).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
