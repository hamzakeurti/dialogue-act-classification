import torch.nn as nn
import torch

class LexicalModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, conv_channels, lstm_hidden_dim, kernel_size, context_size, device):
        super(LexicalModel, self).__init__()
        self.input_dim = input_dim          # sentence_length
        self.context_size = context_size    # Number of sentences provided as context including current sentence.
        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels
        self.lstm_hidden_dim = lstm_hidden_dim 
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.conv = nn.Conv1d(self.embedding_dim,self.conv_channels,self .kernel_size)

        self.lstm = nn.LSTM(
            input_size = self.conv_channels,
            hidden_size = self.lstm_hidden_dim,
            batch_first = True) 

        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, text):
        # TODO: your codes here
        # text: [num sent, batch size, sent len]
        
        # 1. get embdding vectors
        # embedded: [num sent, batch size, sent len, emb dim]
        embedded = self.embedding(text)

        # 2. convolution over each sentence
        conv_outputs = [] 

        for i in range(self.context_size):
            conv_outputs.append(self.conv(embedded[i]))
        # conv_output: [num sent, batch size, sent len, num_channels]
        conv_output = torch.stack(conv_outputs)

        # 3. MaxPool the whole sentence into a single vector of dim num channels
        # max_output: [num sent, batch size, num_channels]
        max_output = torch.max(conv_output,dim = 3)  #max over sentence length

        # 4. LSTM the 3 sentences to determine attention 
        # lstm_output: [num sent, batch size, num_channels]
        lstm_output, _ = self.lstm(max_output)

        # 5. Sum the resulting vectors
        return torch.sum(lstm_output,dim=0)

 
