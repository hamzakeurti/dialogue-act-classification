import os
import numpy as np
import lexical
import utils
import torch
import torch.utils.data

def initialize_datasets(folders,data_folders):
    # With folders = ['train','dev','test'] and data_folders a dictionary which links the folder names
    # to the names of the files that are attributed to each of them,
    # Generate pytorch datasets for each of the folders, generating the tensor files if they don't already exist or loading them otherwise
    sent_len = 50
    audio_len = 500
    labels_encoding = {'%':0, 'b':1, 'f':2, 'q':3, 's':4}
    datasets = []

    for folder in folders:
        try:
            folder_audio = torch.load('data/dataset/audio_' + folder + '.pt')
            folder_text = torch.load('data/dataset/text_' + folder + '.pt')
            folder_labels = torch.load('data/dataset/labels_' + folder + '.pt')
        except:
            vocabulary,data_dict,labels_dict,frames_dict = utils.init_dictionaries(folders,data_folders)
            audio_tensors = []
            text_tensors = []
            labels_tensor = []
            for name in data_folders[folder]:
                audio = torch.load('data/' + folder +'/' + name + '.pt')
                data = data_dict[name]
                frames = frames_dict[name]
                labels = labels_dict[name]
                for i in range(len(data)):
                    begin_f = frames[i][0]
                    end_f = frames[i][1]
                    if end_f - begin_f < 501 and labels[i]!='z' and max(len(data[i][k]) for k in range(3)) < 51:
                        audio_tensors.append(utils.padding_audio(audio[:,begin_f:end_f],500))
                        text_tensors.append(utils.text_to_torch(data[i],sent_len))
                        labels_tensor.append(labels_encoding[labels[i]])
            folder_audio = torch.stack(audio_tensors)
            folder_text = torch.stack(text_tensors)
            folder_labels = torch.tensor(labels_tensor)

            torch.save(folder_audio,'data/dataset/audio_' + folder + '.pt')
            torch.save(folder_text,'data/dataset/text_' + folder + '.pt')
            torch.save(folder_labels,'data/dataset/labels_' + folder + '.pt')
        dataset = torch.utils.data.TensorDataset(folder_audio,folder_text,folder_labels)
        datasets.append(dataset)
    return datasets
