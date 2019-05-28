import utils
import batcher
import torch.utils.data

folders,data_folders = utils.folders_info()

batch_size = 100
datasets = batcher.initialize_datasets(folders,data_folders)
class_sample_count = [2940, 10557,5361,5618,50224]
class_weights = 1/torch.Tensor(class_sample_count)

train_dataset = datasets[0]
train_weights = [class_weights[train_dataset.__getitem__(i)[2].item()] for i in range(len(train_dataset))]
sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,len(train_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,sampler = sampler)