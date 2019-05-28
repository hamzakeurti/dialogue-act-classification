import torch.nn as nn
import torch

class LexicalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 300, conv_channels=256, output_dim=128, kernel_size=5, context_size=3, device='cpu'):
        super(LexicalModel, self).__init__()
        self.vocab_size = vocab_size          # sentence_length
        self.context_size = context_size    # Number of sentences provided as context including current sentence.
        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels
        self.output_dim = output_dim 
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv = nn.Conv1d(self.embedding_dim,self.conv_channels,self .kernel_size)

        self.lstm = nn.LSTM(
            input_size = self.conv_channels,
            hidden_size = self.output_dim,
            batch_first = False) 

        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, text):
        # 1. get embdding vectors
        # embedded: [num sent, batch size, emb dim, sent len]
        embedded = self.embedding(text.long()).permute(0,1,3,2)        
        
        # 2. convolution over each sentence
        conv_outputs = [] 
        for i in range(self.context_size):
            conv_outputs.append(self.conv(embedded[i]))
        # conv_output: [num sent, batch size, num channels, sent len]
        conv_output = torch.stack(conv_outputs)
        
        # 3. MaxPool the whole sentence into a single vector of dim num channels
        # max_output: [num sent, batch size, num_channels]
        max_output,_ = torch.max(conv_output,dim = 3)  #max over sentence length

        # 4. LSTM the 3 sentences to determine attention 
        # lstm_output: [num sent, batch size, output dim]
        lstm_output, _ = self.lstm(max_output)

        # 5. Sum the resulting vectors
        return torch.sum(lstm_output,dim=0)

 
class AcousticModel(nn.Module):
    def __init__(self, num_frames = 500, mfcc=13, conv_channels=128, kernel_size = 5,output_dim = 128, device='cpu'):
        super(AcousticModel, self).__init__()
        self.num_frames = num_frames
        self.mfcc = mfcc
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.device = device

        self.conv = nn.Conv1d(self.mfcc,self.conv_channels,self.kernel_size)

        self.fc = nn.Linear(self.conv_channels, self.output_dim)
        self.device = device

    def forward(self, input):
        # input: [bacth size, mfcc, num frames]

        # 1. convolution
        # conv_output: [batch size, num channels, num frames]
        conv_output = self.conv(input)

        # 2. MaxPool over all frames into a single vector of dim num channels
        # max_output: [batch size, num_channels]
        max_output,_ = torch.max(conv_output,dim = 2)  #max over sentence length

        # 3. Fully connected layer
        # fc_output: [num sent, batch size, num_channels]
        fc_output = self.fc(max_output)

        return fc_output
