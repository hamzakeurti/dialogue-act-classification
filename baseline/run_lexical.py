import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from model import LexicalModel
from logger import INIT_LOG, LOG_INFO
import batcher
import torch.utils.data
import json

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", default=5, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--data_balancing",default=0,type=int,help="1 if balancing the training data")
parser.add_argument("--dropout",default=0.5,type=float,help="dropout rate for final layer, default 0.5, put 0 for no dropout")
parser.add_argument("--optimizer",default="SGD",type=str,help="optimizer, Adam or SGD")

parser.add_argument("--embedding",default=1,type=int,help="1 if using pretrained embedding")
parser.add_argument("--conv_channels", default=100, type=int, help="Number of 1-D convolutional channels.")
parser.add_argument("--kernel_size", default=5, type=int, help="Convolution kernel size.")
parser.add_argument("--embedding_dim", default=300, type=int, help="Size of word embedding.")
parser.add_argument("--output_dim", default=128, type=int, help="Model output dim = LSTM hidden dim.")
parser.add_argument("--context_size", default=3, type=int, help="Total number of sentences to consider.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=292, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--log_file", default='', type=str, help="Log file")

args = parser.parse_args()
print(args)

INIT_LOG(args.log_file)
device = 'cuda'

# ------- Data Loaders -----------------------------------
folders,data_folders = utils.folders_info()

# Initialize vocabulary
vocabulary_filename = 'data/vocabulary.json'
vocabulary = {}
if args.embedding==1:
    try:
        vocabulary_file = open(vocabulary_filename,'r')
        vocabulary = json.loads(vocabulary_file.read())
    except:
        vocabulary,_,_,_ = utils.init_dictionaries(folders,data_folders)
        with open(vocabulary_filename,'w') as vocabulary_file:
                vocabulary_file.write(json.dumps(vocabulary))

# getting the pretrained embeddings
pretrained_embeddings = utils.pretrain_embedding(vocabulary).to(device)

# Initialize dataset loaders
batch_size = args.batch_size
datasets = batcher.initialize_datasets(folders,data_folders)
class_sample_count = [2940, 10557,5361,5618,50224]
class_weights = 1/torch.Tensor(class_sample_count)

train_dataset,test_dataset,valid_dataset = datasets[0],datasets[1],datasets[2]
sampler = None
if args.data_balancing==1:
    train_weights = [class_weights[train_dataset.__getitem__(i)[2].item()] for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,len(train_dataset))

train_iterator = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,sampler=sampler)
# train_iterator = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size)
test_iterator = torch.utils.data.DataLoader(test_dataset)
valid_iterator = torch.utils.data.DataLoader(valid_dataset)
# --------------------------------------------------------------


# ---------- Model Definition -----------
vocab_size = pretrained_embeddings.shape[0]
output_dim = args.output_dim
num_labels = args.n_labels

model = nn.Sequential(
    LexicalModel(vocab_size = vocab_size, output_dim=output_dim,init_embedding = pretrained_embeddings,dropout=args.dropout),
    nn.Dropout(p=args.dropout),
    nn.ReLU(),
    nn.Linear(output_dim,num_labels)).to(device)


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
        audio,text,label = audio.to(device),text.to(device),label.to(device)
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
        for (audio,text,label) in iterator:
            
            audio = audio.to(device)
            text = text.to(device)
            label = label.to(device)


            predictions = model(text)
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
        torch.save(model.state_dict(), 'best-model_lexical.pth')

LOG_INFO('Test best model @ Epoch %02d' % best_epoch)
model.load_state_dict(torch.load('best-model_lexical.pth'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
LOG_INFO('Finally, test loss = %.4f, test acc = %.4f' % (test_loss, test_acc))
