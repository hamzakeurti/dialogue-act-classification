import librosa.core
import torch

def parse_dadb(filename):
    data = []
    labels = []
    frames = []
    
    with open(filename,'r') as file:
        line = file.readline()
        k=0
        while line:
            line_split = line.split(',')
            begin_time,end_time = float(line_split[0]),float(line_split[1])
            label = line_split[-9]
            words = [x.split('+')[-1] for x in line_split[4].split('|')]


            data.append(words)
            labels.append(label)
            frames.append(librosa.core.time_to_frames((begin_time,end_time)))

            line = file.readline()
    return data,labels,frames

def mapping_labels(label):
    if '|' in label:
        for i in range(len(label)):
            if label[i]=='|':
                return label[i+1]
    if label=='' or label[0].isdigit():
        return 'z'
    if label[0] in ['s','q','b','%','f']:
        return label[0]
    if label[0] == 'h':
        return 'f'
    if label[0] == 'x':
        return '%'
    return label

def folders_info():
    training_set = []
    dev_set = []
    test_set = []
    with open('train_test_split.txt','r') as file:
        line = file.readline()
        train = True
        line = file.readline()
        while line[:3]!='Dev':
            if len(line)>1:
                training_set.append(line[:-1])
            line = file.readline()
        line = file.readline()
        while line[:3]!='Tes':
            if len(line)>1:
                dev_set.append(line[:-1])
            line = file.readline()
        line = file.readline()
        while line[:3]!='Not':
            if len(line)>1:
                test_set.append(line[:-1])
            line = file.readline()
    data_folders = {'train':training_set,'dev':dev_set,'test':test_set}
    folders = ['train','dev','test']
    return folders,data_folders

def init_dictionaries(folders,data_folders):
    sent_len = 50
    audio_len = 500
    frames_dict = {}
    data_dict = {}
    labels_dict = {}
    vocabulary = {}
    dict_index = 1
    for folder in folders:
        for name in data_folders[folder]:
            data,labels,frames = parse_dadb('data\\dadb\\' + name + '.dadb')                 

            #update dictionary
            for sentence in data:
                for word in sentence:
                    try:
                        vocabulary[word]
                    except:
                        vocabulary[word] = dict_index
                        dict_index+=1
                        
            indexed_data = [[vocabulary[word] for word in sentence] for sentence in data]
            indexed_data = [[],[]] + indexed_data

            data_dict[name] = [[indexed_data[k],indexed_data[k+1],indexed_data[k+2]] for k in range(len(data))]
            labels_dict[name] = [mapping_labels(label) for label in labels]
            frames_dict[name] = frames
            
    return vocabulary,data_dict,labels_dict,frames_dict

def padding_audio(tensor,length):
    i,j = tensor.shape
    objective = torch.zeros(i,length)
    objective[:,:j] = tensor
    return objective

def text_to_torch(text_list,sent_len):
    sentences = [torch.zeros(sent_len) for i in range(3)]
    for i in range(3):
        text = text_list[i]
        sentences[i][:len(text)] = torch.tensor(text)
    return torch.stack(sentences)