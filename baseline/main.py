import utils
import batcher
import torch.utils.data

folders,data_folders = utils.folders_info()

batch_size = 100
datasets = batcher.initialize_datasets(folders,data_folders)
class_sample_count = [2940, 10557,5361,5618,50224]
weights = 1/torch.Tensor(class_sample_count)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,batch_size)

print(len(datasets))
print(datasets)
train_loader = torch.utils.data.DataLoader(datasets[0],batch_size = batch_size)

labels = []
for batch,(audio,text,label) in enumerate(train_loader):
    print(batch)
    if batch < 10:
        print(label)
