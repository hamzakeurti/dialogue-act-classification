import utils
import batcher

folders,data_folders = utils.folders_info()
vocabulary,data_dict,labels_dict,frames_dict = utils.init_dictionaries(folders,data_folders)

loaders = batcher.initialize_tensors(folders,data_folders,vocabulary,data_dict,labels_dict,frames_dict,100,shuffle = True)
train_loader = loaders[0]

for batch,(audio,text,label) in enumerate(train_loader):
    if batch<10:
        print(batch)
        print(audio.shape)
        print(text.shape)
        print(label)