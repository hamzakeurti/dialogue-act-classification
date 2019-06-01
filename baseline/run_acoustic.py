import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import utils
import batcher
from model import AcousticModel
from logger import INIT_LOG, LOG_INFO

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", default=5, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--data_balancing",default=0,type=int,help="1 if balancing the training data")
parser.add_argument("--dropout",default=0.5,type=float,help="dropout rate for final layer, default 0.5, put 0 for no dropout")
parser.add_argument("--optimizer",default="SGD",type=str,help="optimizer, Adam or SGD")

parser.add_argument("--num_frames", default=500, type=int, help="Number of frames per sentence.")
parser.add_argument("--conv_channels", default=256, type=int, help="Number of 1-D convolutional channels.")
parser.add_argument("--kernel_size", default=5, type=int, help="Convolution kernel size.")
parser.add_argument("--mfcc", default=13, type=int, help="Number of MFCC components.")
parser.add_argument("--output_dim", default=128, type=int, help="Model output dim = FC output dim.")

parser.add_argument("--batch_size", default=50, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=250, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--log_file", default='', type=str, help="Log file")
args = parser.parse_args()
print(args)

INIT_LOG(args.log_file)


# ------- Data Loaders -----------------------------------
folders,data_folders = utils.folders_info()

batch_size = args.batch_size
datasets = batcher.initialize_datasets(folders,data_folders)
class_sample_count = [2940, 10557,5361,5618,50224]
class_weights = 1/torch.Tensor(class_sample_count)

train_dataset,test_dataset,valid_dataset = datasets[0],datasets[1],datasets[2]


sampler = None
if args.data_balancing==1:
    train_weights = [class_weights[train_dataset.__getitem__(i)[2].item()] for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,len(train_dataset))

train_iterator = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,sampler = sampler)
test_iterator = torch.utils.data.DataLoader(test_dataset)
valid_iterator = torch.utils.data.DataLoader(valid_dataset)
# --------------------------------------------------------------

device = 'cuda'
# ---------- Model Definition -----------


model = nn.Sequential(
    AcousticModel(
        num_frames = args.num_frames,
        mfcc = args.mfcc,
        conv_channels = args.conv_channels,
        kernel_size = args.kernel_size,
        output_dim = args.output_dim,
        dropout=args.dropout
    ),
    nn.Dropout(p=args.dropout),
    nn.ReLU(),
    nn.Linear(args.output_dim,args.n_labels)).to(device)

 
criterion = nn.CrossEntropyLoss()
if args.optimizer=='Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
# ----------------------------------------

def train(epoch, model, iterator, optimizer, criterion):
    loss_list = []
    acc_list = []

    model.train()

    for i, (audio,text,label) in enumerate(iterator):
        audio = audio.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predictions = model(audio)

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
        for (audio,text,label) in iterator:
            audio = audio.to(device)
            label = label.to(device)
            predictions = model(audio)
            loss = criterion(predictions, label.long())

            acc = (predictions.max(1)[1] == label.long()).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



best_acc = 0
best_epoch = -1
for epoch in range(1, args.epochs + 1):
    train(epoch, model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    msg = '...Epoch %02d, val loss = %.4f, val acc = %.4f' % (
        epoch, valid_loss, valid_acc
    )
    LOG_INFO(msg)

    if valid_acc > best_acc:
        best_acc = valid_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best-model_audio.pth')

LOG_INFO('Test best model @ Epoch %02d' % best_epoch)
model.load_state_dict(torch.load('best-model_audio.pth'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
LOG_INFO('Finally, test loss = %.4f, test acc = %.4f' % (test_loss, test_acc))
