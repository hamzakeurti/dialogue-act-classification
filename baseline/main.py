import utils
import batcher

folders,data_folders = utils.folders_info()
vocabulary,data_dict,labels_dict,frames_dict = utils.init_dictionaries(folders,data_folders)

loaders = batcher.initialize_tensors(folders,data_folders,vocabulary,data_dict,labels_dict,frames_dict,100)
train_loader = loaders[0]

labels = []
for batch,(audio,text,label) in enumerate(train_loader):
    labels.append(label)
