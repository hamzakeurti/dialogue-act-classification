from model import LexicalModel, AcousticModel, LexicalAcousticModel
import torch
import torch.nn as nn
import utils
from logger import INIT_LOG, LOG_INFO
import batcher

criterion = nn.CrossEntropyLoss()


batch_size = args.batch_size
datasets = batcher.initialize_datasets(folders,data_folders)
_,test_dataset,_ = datasets

vocabulary = {}
pretrained_embeddings = utils.pretrain_embedding(vocabulary).to(device)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (audio,text,label) in iterator:
            
            audio = audio.to(device)
            text = text.to(device)
            label = label.to(device)


            predictions = model(text,audio)
            loss = criterion(predictions, label.long())

            acc = (predictions.max(1)[1] == label.long()).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

LOG_INFO('Test best model @ Epoch %02d' % best_epoch)
model.load_state_dict(torch.load('best-model.pth'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
LOG_INFO('Finally, test loss = %.4f, test acc = %.4f' % (test_loss, test_acc))